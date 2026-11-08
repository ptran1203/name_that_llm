"""LoRA fine-tune a frozen decoder model (Qwen2.5-3B by default) for 18-way classification.

Why not AutoModelForSequenceClassification: its default head pools the LAST layer via the last
real token -- found to be the WORST frozen layer for this task (TRIED.md's layer sweep: 0.2893
at the last layer vs 0.3978 at layer 7). LoRA's low-rank updates are deliberately small, so
fine-tuning alone isn't trusted to fully relocate that signal to the last layer in a few epochs
on ~2700 rows. Instead this wraps the base model directly, pools from a CONFIGURABLE layer
(default 7, the empirically-best frozen layer) via mean-pooling over the attention mask -- same
pooling already validated in experiment_decoder_embedding.py -- then a fresh linear head.

LoRA (peft): freezes the base model, inserts small trainable low-rank adapters into attention
projections. Only the adapters + the new classifier head get gradients/optimizer state --
full fine-tuning a 3B model's optimizer state alone wouldn't fit a single consumer/Kaggle GPU.
Gradients still flow back through every frozen layer up to the pooling layer, so activation
memory is controlled via gradient checkpointing + small batch + gradient accumulation instead.

GroupKFold by prompt_id, single GPU, same leakage discipline as train_transformer.py. Test
--max_folds 1 first before committing to a full run -- this is new, heavier infrastructure.

Usage (Kaggle): pip install -q peft  (if not already present in the image), then:
    python src/train_qwen_lora.py --max_folds 1 --batch_size 4 --grad_accum_steps 4
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainerCallback

from contrastive_losses import ArcMarginProduct, PKBatchSampler, batch_hard_triplet_loss, supcon_loss
from exp_log import ExperimentRun
from experiment_multi_embed_concat import load_external_data
from paths import data_dir
from prompt_cluster_assign import resolve_cluster_assignments, resolve_cluster_assignments_hungarian
from train import build_vectorizers, fit_transform
from train_transformer import build_training_args

# Adjacent-version clusters flagged (TRIED.md, error_analysis.ipynb) as the hardest same-family
# confusions -- oversampled as deliberate hard negatives by PKBatchSampler when triplet/supcon
# loss is active, instead of leaving negative sampling to chance (IDEAS.md #6, WhosAI).
HARD_CLUSTERS = [
    {"claude-opus-4-6", "claude-opus-4-8", "claude-opus-5"},
    {"gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"},
    {"gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"},
    {"glm-5.3", "glm-5.3-flash"},
]

# Under accelerate/DDP each process has its own assigned GPU -- PartialState().device resolves
# to the right one for THIS process regardless of launch method (accelerate launch, torchrun,
# plain python), unlike hand-parsing LOCAL_RANK ourselves. A bare "cuda" always means cuda:0,
# which would put every process's model on the same GPU and break DDP.
DEVICE = PartialState().device


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, extra_features=None):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)
        # extra_features: optional [n, d] dense array (e.g. SVD-reduced TF-IDF) concatenated to
        # the pooled embedding in the model -- fixed-size per row, so no padding needed.
        self.extra_features = extra_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        if self.extra_features is not None:
            item["extra_features"] = self.extra_features[idx]
        return item


class ExtraFeaturesCollator:
    """Wraps DataCollatorWithPadding: pads the tokenizer fields as usual, and separately stacks
    "extra_features" (already fixed-size per row, nothing to pad) instead of relying on
    tokenizer.pad()'s undocumented handling of non-tokenizer keys."""

    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        has_extra = "extra_features" in features[0]
        extra = None
        if has_extra:
            extra = torch.stack([torch.as_tensor(f.pop("extra_features"), dtype=torch.float32)
                                  for f in features])
        batch = self.base(features)
        if has_extra:
            batch["extra_features"] = extra
        return batch


class LayerPoolClassifier(nn.Module):
    """base_model (frozen + LoRA) -> hidden_states[layer] -> mean-pool over attention mask ->
    linear head -> logits. The classifier head is left in float32 (standard practice for
    numerical stability even when the backbone runs in bf16/fp16) -- pooled activations are
    cast to match before the final matmul."""

    def __init__(self, base_model, hidden_size, num_labels, layer, center_loss_weight=0.0,
                 center_momentum=0.9, triplet_weight=0.0, triplet_margin=0.3,
                 supcon_weight=0.0, supcon_temperature=0.07, use_arcface_head=False,
                 arcface_scale=30.0, arcface_margin=0.3, extra_dim=0,
                 token_mask_prob=0.0, dropout_p=0.0, multi_sample_dropout_k=1, mixup_alpha=0.0):
        super().__init__()
        self.base_model = base_model
        self.layer = layer
        self.extra_dim = extra_dim
        combined_dim = hidden_size + extra_dim
        self.classifier = (
            ArcMarginProduct(combined_dim, num_labels, arcface_scale, arcface_margin)
            if use_arcface_head else nn.Linear(combined_dim, num_labels)
        )
        self.center_loss_weight = center_loss_weight
        self.center_momentum = center_momentum
        self.triplet_weight = triplet_weight
        self.triplet_margin = triplet_margin
        self.supcon_weight = supcon_weight
        self.supcon_temperature = supcon_temperature
        # CV-trick analogs, all 0/1 (off) by default -- exact backward compatibility with every
        # run before these existed:
        self.token_mask_prob = token_mask_prob  # "cutout": randomly hide real tokens (attention
        # mask only, not input_ids -- Qwen has no [MASK] token) from both the backbone and the
        # pooling step, training-time only.
        self.dropout = nn.Dropout(p=dropout_p)
        self.multi_sample_dropout_k = multi_sample_dropout_k  # Inoue 2019: K independent dropout
        # passes through the head per batch, CE loss averaged over them -- implicit ensembling
        # within one forward pass, applied to the CE term only (not center/triplet/supcon, which
        # need one consistent embedding per sample).
        self.mixup_alpha = mixup_alpha  # manifold mixup between two SAME-class embeddings in a
        # batch (label unchanged, since both share it already) -- the "safer" variant IDEAS.md
        # #1 flagged as untried; the TF-IDF-space version already tried and failed (TRIED.md).
        # EMA per-class embedding center (not a trainable nn.Parameter -- updated directly from
        # batch statistics, not via gradient descent, so it tracks the class mean independent of
        # the aux loss's own gradient).
        self.register_buffer("centers", torch.zeros(num_labels, combined_dim))

    def _maybe_mask_tokens(self, attention_mask):
        """Training-time-only "cutout for text": randomly zero out a fraction of the REAL
        (attention_mask==1) positions, applied to both the backbone's self-attention and the
        pooling mask (same tensor controls both) -- forces the model not to rely on any single
        token/phrase. Never drops every real token in a row (would make pooling divide by zero)."""
        if not (self.training and self.token_mask_prob > 0):
            return attention_mask
        real = attention_mask.bool()
        drop = (torch.rand_like(attention_mask, dtype=torch.float) < self.token_mask_prob) & real
        out = attention_mask.clone()
        out[drop] = 0
        emptied = out.sum(dim=1) == 0
        if emptied.any():
            first_real_idx = real.float().argmax(dim=1)
            out[emptied, first_real_idx[emptied]] = 1
        return out

    def pool(self, input_ids, attention_mask):
        """The embedding this model is actually using right now -- same op the probe callback
        below reuses to check whether that embedding is improving epoch over epoch, independent
        of how the classifier head on top of it is doing."""
        attention_mask = self._maybe_mask_tokens(attention_mask)
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
        hidden = out.hidden_states[self.layer]
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    def _same_class_mixup(self, pooled, labels):
        """Manifold mixup between two embeddings of the SAME class in this batch (if one
        exists) -- label stays the anchor's own label, since the partner already shares it.
        Rows with no same-class partner in the batch (common without PK sampling) pass through
        unchanged. Training-time only."""
        if not (self.training and self.mixup_alpha > 0):
            return pooled
        n = labels.size(0)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        same &= ~torch.eye(n, dtype=torch.bool, device=labels.device)
        has_partner = same.any(dim=1)
        if not has_partner.any():
            return pooled
        rand_scores = torch.rand(n, n, device=labels.device).masked_fill(~same, -1.0)
        partner_idx = rand_scores.argmax(dim=1)
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample((n,)).to(pooled.device)
        lam = lam.unsqueeze(1).to(pooled.dtype)
        mixed = pooled.clone()
        mixed[has_partner] = (lam[has_partner] * pooled[has_partner]
                               + (1 - lam[has_partner]) * pooled[partner_idx[has_partner]])
        return mixed

    def _head_logits(self, x, labels):
        if isinstance(self.classifier, ArcMarginProduct):
            return self.classifier(x, labels if (self.training and labels is not None) else None)
        return self.classifier(x.to(self.classifier.weight.dtype))

    def forward(self, input_ids=None, attention_mask=None, labels=None, extra_features=None):
        pooled = self.pool(input_ids, attention_mask)
        if extra_features is not None:
            # Per-block LayerNorm (no learnable affine -- just mean/var equalization) BEFORE
            # concatenating: the Qwen pooled embedding's raw L2 norm (~44) swamps the TF-IDF-SVD
            # block's (~0.77, a ~57x gap, measured empirically) once anything L2-normalizes or
            # takes squared-L2 distance over the combined vector (ArcFace's cosine head, the
            # nearest-centroid readout, center/triplet/supcon's distance terms) -- without this,
            # the TF-IDF block is contributing almost nothing regardless of how informative it
            # actually is. No new trainable params, and stays correct as the Qwen embedding's
            # own scale drifts during LoRA fine-tuning (unlike a fixed, precomputed scale factor).
            pooled = nn.functional.layer_norm(pooled.float(), pooled.shape[-1:])
            extra_features = nn.functional.layer_norm(extra_features.float(), extra_features.shape[-1:])
            pooled = torch.cat([pooled, extra_features.to(pooled.dtype)], dim=1)
        loss = None
        if labels is not None:
            # Mixup and multi-sample dropout both only affect the CE term -- center/triplet/
            # supcon below keep using the CLEAN `pooled`, since they need one consistent
            # per-sample embedding, not a blended/dropped-out view of it.
            ce_input = self._same_class_mixup(pooled, labels)
            if self.training and self.multi_sample_dropout_k > 1:
                ce_terms = []
                for _ in range(self.multi_sample_dropout_k):
                    logits = self._head_logits(self.dropout(ce_input), labels)
                    ce_terms.append(nn.functional.cross_entropy(logits, labels))
                loss = torch.stack(ce_terms).mean()
            else:
                logits = self._head_logits(self.dropout(ce_input), labels)
                loss = nn.functional.cross_entropy(logits, labels)
            if self.center_loss_weight > 0:
                loss = loss + self.center_loss_weight * self._center_loss(pooled, labels)
            if self.triplet_weight > 0:
                loss = loss + self.triplet_weight * batch_hard_triplet_loss(
                    pooled, labels, self.triplet_margin)
            if self.supcon_weight > 0:
                loss = loss + self.supcon_weight * supcon_loss(pooled, labels, self.supcon_temperature)
        else:
            logits = self._head_logits(self.dropout(pooled), None)
        return {"loss": loss, "logits": logits}

    def _center_loss(self, pooled, labels):
        """Pulls each sample's embedding toward its class's EMA center (L2), independent of
        batch composition -- unlike a pairwise same-class loss, this still gives a signal even
        when a batch (size 4 here, 18 classes) contains at most one sample of a given class."""
        pooled_f = pooled.float()
        centers = self.centers[labels]
        dist = (pooled_f - centers).pow(2).sum(dim=1).mean()
        with torch.no_grad():
            for y in labels.unique():
                mask = labels == y
                batch_mean = pooled_f[mask].mean(dim=0)
                self.centers[y] = self.center_momentum * self.centers[y] + (1 - self.center_momentum) * batch_mean
        return dist


def embed_texts(model, tokenizer, texts, max_length, batch_size):
    """Pooled embedding for a list of texts from the CURRENT model state -- shared by
    ProbeCallback (fold-local, per-epoch) and embedding_readout (final, per-fold)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True,
                             return_tensors="pt").to(DEVICE)
            pooled = model.pool(enc["input_ids"], enc["attention_mask"])
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def embedding_readout(model, tokenizer, train_texts, train_labels, val_texts, val_labels,
                       max_length, batch_size, log, train_extra=None, val_extra=None):
    """Nearest-centroid (cosine) + SVC readout on the model's final pooled embedding -- the
    evaluation the metric-learning literature (WhosAI arXiv 2407.09364, etc.) actually uses for
    triplet/contrastive-trained embeddings, instead of just the jointly-trained classifier head
    (plain linear, or ArcFace margin). The head optimizes its OWN decision boundary; triplet/
    SupCon instead optimize the embedding's distance structure, so the head isn't necessarily
    the best way to read out what those two losses actually learned. Logged alongside the head's
    own eval_balanced_accuracy for every loss variant (center/triplet/supcon/arcface/plain), not
    just triplet/supcon, so they're all directly comparable.

    train_extra/val_extra: optional [n, d] SVD-TFIDF arrays (same ones fed into the model during
    training via --tfidf_svd_dim) -- concatenated here too, so this readout sees the same
    [qwen embedding | TF-IDF] vector the classifier head was actually trained on."""
    train_emb = embed_texts(model, tokenizer, list(train_texts), max_length, batch_size)
    val_emb = embed_texts(model, tokenizer, list(val_texts), max_length, batch_size)
    if train_extra is not None:
        # Same per-row scale-equalization as the model's forward() (see its comment) -- without
        # it the nearest-centroid cosine step below is dominated by whichever block has the
        # larger raw norm, which was empirically the Qwen embedding by ~57x.
        def row_normalize(x):
            return (x - x.mean(axis=1, keepdims=True)) / np.clip(x.std(axis=1, keepdims=True), 1e-5, None)
        train_emb = np.concatenate([row_normalize(train_emb), row_normalize(train_extra)], axis=1)
        val_emb = np.concatenate([row_normalize(val_emb), row_normalize(val_extra)], axis=1)
    train_labels = np.asarray(train_labels)
    val_labels = np.asarray(val_labels)

    classes = np.unique(train_labels)
    centroids = np.stack([train_emb[train_labels == c].mean(axis=0) for c in classes])
    centroid_norm = centroids / np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8, None)
    val_norm = val_emb / np.clip(np.linalg.norm(val_emb, axis=1, keepdims=True), 1e-8, None)
    centroid_pred = classes[(val_norm @ centroid_norm.T).argmax(axis=1)]
    centroid_score = balanced_accuracy_score(val_labels, centroid_pred)

    svc = make_pipeline(StandardScaler(), SVC(class_weight="balanced"))
    svc.fit(train_emb, train_labels)
    svc_score = balanced_accuracy_score(val_labels, svc.predict(val_emb))

    log(f"  embedding readout: nearest_centroid_balanced_accuracy={centroid_score:.4f}  "
        f"svc_balanced_accuracy={svc_score:.4f}")
    model.train()
    return {"nearest_centroid": centroid_score, "svc": svc_score}


class ProbeCallback(TrainerCallback):
    """Fold-local, cheap diagnostic (NOT an OOF estimate -- see TRIED.md): after each epoch,
    snapshot the CURRENT model's pooled embedding (same layer/pooling as the live head) for
    this fold's train and val rows, fit a fresh plain classifier (LogisticRegression or
    LightGBM) on train, score balanced accuracy on val. Tracks whether the REPRESENTATION at
    this layer is actually improving epoch over epoch, independent of how the live linear head
    (trained jointly via gradient descent) happens to be doing -- if the probe plateaus while
    the live head keeps climbing, that's the head compensating/overfitting, not the backbone
    learning better features."""

    def __init__(self, train_texts, train_labels, val_texts, val_labels, tokenizer, max_length,
                 batch_size, probe_classifier, log):
        self.train_texts, self.train_labels = list(train_texts), np.asarray(train_labels)
        self.val_texts, self.val_labels = list(val_texts), np.asarray(val_labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.probe_classifier = probe_classifier
        self.log = log

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        train_emb = embed_texts(model, self.tokenizer, self.train_texts, self.max_length, self.batch_size)
        val_emb = embed_texts(model, self.tokenizer, self.val_texts, self.max_length, self.batch_size)
        model.train()

        if self.probe_classifier == "lgbm":
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(n_estimators=200, verbosity=-1)
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=5.0,
                                                                       class_weight="balanced"))
        clf.fit(train_emb, self.train_labels)
        pred = clf.predict(val_emb)
        score = balanced_accuracy_score(self.val_labels, pred)
        self.log(f"  [probe, epoch {state.epoch:.0f}] fold-local {self.probe_classifier} on "
                 f"layer-{model.layer} embedding: balanced_accuracy={score:.4f} (not OOF)")


def build_model(model_name, num_labels, layer, lora_r, lora_alpha, lora_dropout,
                 gradient_checkpointing, precision, load_in_4bit, log,
                 center_loss_weight=0.0, center_momentum=0.9, triplet_weight=0.0,
                 triplet_margin=0.3, supcon_weight=0.0, supcon_temperature=0.07,
                 use_arcface_head=False, arcface_scale=30.0, arcface_margin=0.3, extra_dim=0,
                 token_mask_prob=0.0, dropout_p=0.0, multi_sample_dropout_k=1, mixup_alpha=0.0):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

    if load_in_4bit:
        # QLoRA: the frozen base stays in 4-bit (~1/4 the memory of bf16), LoRA adapters still
        # train in `dtype`. Needed once the base model itself is too big to fit alongside LoRA's
        # activation memory otherwise (e.g. 7B+ on a single 16GB GPU) -- not needed at 3B.
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
        # device_map must point at THIS process's own assigned GPU (DEVICE, derived from
        # LOCAL_RANK) -- a hardcoded device would put every DDP process's model on the same GPU.
        base = AutoModel.from_pretrained(model_name, quantization_config=bnb_config,
                                          device_map={"": str(DEVICE)})
        # use_reentrant=False: the default reentrant checkpoint re-runs part of the forward
        # pass during backward, which conflicts with DDP's parameter-ready tracking under
        # multi-GPU training ("marked as ready twice") -- non-reentrant is the mode PyTorch
        # itself now recommends, and avoids this class of bug entirely.
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    else:
        base = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(DEVICE)
    hidden_size = base.config.hidden_size

    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="FEATURE_EXTRACTION",
    )
    base = get_peft_model(base, lora_config)
    if gradient_checkpointing and not load_in_4bit:  # prepare_model_for_kbit_training already enabled it
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base.parameters())
    log(f"LoRA base: {trainable:,}/{total:,} trainable ({100 * trainable / total:.3f}%)")

    model = LayerPoolClassifier(base, hidden_size, num_labels, layer,
                                 center_loss_weight, center_momentum, triplet_weight,
                                 triplet_margin, supcon_weight, supcon_temperature,
                                 use_arcface_head, arcface_scale, arcface_margin,
                                 extra_dim, token_mask_prob, dropout_p,
                                 multi_sample_dropout_k, mixup_alpha).to(DEVICE)
    return model


def save_fold_model(model, tokenizer, out_dir, model_name, pooling_layer, max_length, log):
    """Save the LoRA adapter (small -- frozen base weights aren't re-saved) + tokenizer + head +
    pooling metadata, so a finished fold's model can be reloaded later (e.g. via
    embed_lora_cv.py's load_lora_model, which reads the same pooling_metadata.json convention)
    instead of being discarded when the process exits. Always called -- a multi-GPU-hour LoRA
    fine-tune with nothing saved afterward is exactly the mistake this fixes."""
    model.base_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save({"classifier": model.classifier.state_dict(), "centers": model.centers},
               f"{out_dir}/head_state.pt")
    with open(f"{out_dir}/pooling_metadata.json", "w") as f:
        json.dump({"model_name": model_name, "pooling_layer": pooling_layer,
                    "max_length": max_length}, f, indent=2)
    log(f"  saved adapter + tokenizer + head -> {out_dir}")


class PKSamplerTrainer(Trainer):
    """Trainer that batches the TRAIN split via a PKBatchSampler instead of random shuffling --
    needed for triplet/supcon loss, which require same-class positive pairs co-present in a
    batch. Eval always uses the default (order doesn't matter, no pairing needed there)."""

    def __init__(self, *args, pk_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pk_sampler = pk_sampler

    def get_train_dataloader(self):
        if self.pk_sampler is None:
            return super().get_train_dataloader()
        return DataLoader(self.train_dataset, batch_sampler=self.pk_sampler,
                           collate_fn=self.data_collator)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"balanced_accuracy": balanced_accuracy_score(labels, preds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--pooling_layer", type=int, default=7,
                     help="hidden_states index to pool from -- default is the empirically-best "
                          "frozen layer for Qwen2.5-3B (TRIED.md), NOT the last layer")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4,
                     help="per-device batch size -- kept small since activation memory for a "
                          "3B model's full forward/backward graph is the real VRAM constraint "
                          "under LoRA, not optimizer state")
    ap.add_argument("--grad_accum_steps", type=int, default=4,
                     help="effective batch size = batch_size * grad_accum_steps")
    ap.add_argument("--epochs", type=float, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=None,
                     help="stop after this many folds -- test with 1 before a full run, this "
                          "is new infrastructure")
    ap.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"],
                     help="bf16 needs no loss scaler (unlike fp16) -- safer given this project's "
                          "documented fp16 GradScaler crash history. Use fp32 on a GPU without "
                          "bf16 support (e.g. T4/P100).")
    ap.add_argument("--no_gradient_checkpointing", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true",
                     help="QLoRA: quantize the frozen base to 4-bit (bitsandbytes) before "
                          "attaching LoRA adapters -- needed once the base model itself doesn't "
                          "fit alongside LoRA's activation memory (e.g. 7B+ on a 16GB GPU). Not "
                          "needed at 3B. Requires `pip install bitsandbytes`.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe", default=None, choices=[None, "lr", "lgbm"],
                     help="fold-local (NOT OOF, see TRIED.md), cheap diagnostic: after each "
                          "epoch, fit this classifier on the current model's layer embedding "
                          "for this fold's train rows, score on val -- tracks whether the "
                          "representation itself is improving, independent of the live "
                          "classifier head's own training dynamics. None (default) = off.")
    ap.add_argument("--external_paths", default=None,
                     help="comma-separated parquet path(s) of external data (e.g. the "
                          "name-that-llm-open-pool dataset) to add to every fold's TRAINING set. "
                          "Unlike the frozen-embedding concat experiment -- where this was tried "
                          "and found flat-to-negative (TRIED.md) -- LoRA actually updates "
                          "weights on this data instead of just feeding a downstream classifier, "
                          "so it has a real mechanism to help here.")
    ap.add_argument("--external_cap_per_class", type=int, default=300)
    ap.add_argument("--final_official_only_epochs", type=float, default=1.0,
                     help="when --external_paths is set, reserve the LAST this-many epochs to "
                          "fine-tune on official training data only (external rows excluded) -- "
                          "lets the model first learn from the larger/noisier external pool, "
                          "then polish on the clean official distribution it's actually scored "
                          "on. Ignored when --external_paths is not set.")
    ap.add_argument("--center_loss_weight", type=float, default=0.0,
                     help="weight on an auxiliary L2 'center loss' (Wen et al. 2016) that pulls "
                          "each sample's pooled embedding toward an EMA-tracked per-class center "
                          "-- encourages within-class embeddings to cluster. 0.0 (default) = off.")
    ap.add_argument("--center_momentum", type=float, default=0.9,
                     help="EMA momentum for the per-class center loss's running centers")
    ap.add_argument("--triplet_weight", type=float, default=0.0,
                     help="weight on an auxiliary batch-hard triplet loss (Hermans et al. 2017; "
                          "WhosAI, arXiv 2407.09364, uses this exact loss for LLM-generator "
                          "attribution) -- pulls the hardest same-class pair together, pushes "
                          "the hardest different-class pair apart. Needs --pk_classes/"
                          "--pk_per_class > 1 to have same-class pairs in-batch at all. "
                          "0.0 (default) = off.")
    ap.add_argument("--triplet_margin", type=float, default=0.3)
    ap.add_argument("--supcon_weight", type=float, default=0.0,
                     help="weight on an auxiliary supervised contrastive loss (Khosla et al. "
                          "2020) -- every other same-class row in the batch is a positive, "
                          "validated specifically for few-labels-per-class text classification "
                          "(arXiv 2211.15202, 2205.01308). Same PK-sampling requirement as "
                          "--triplet_weight. 0.0 (default) = off.")
    ap.add_argument("--supcon_temperature", type=float, default=0.07)
    ap.add_argument("--use_arcface_head", action="store_true",
                     help="replace the plain linear classifier head with an ArcFace (Deng et "
                          "al. 2019) additive-angular-margin cosine head. EXPLORATORY: no NLP/"
                          "text precedent found when researching this, every reference is face/"
                          "vision -- lower confidence than triplet/supcon.")
    ap.add_argument("--arcface_scale", type=float, default=30.0)
    ap.add_argument("--arcface_margin", type=float, default=0.3,
                     help="lower than the 0.5 common in face recognition -- untested on text, "
                          "being conservative given our much smaller per-class N")
    ap.add_argument("--pk_classes", type=int, default=6,
                     help="P: number of distinct classes per training batch when "
                          "triplet/supcon loss is active (ignored otherwise). Overrides "
                          "--batch_size for the train split -- effective batch = pk_classes * "
                          "pk_per_class.")
    ap.add_argument("--pk_per_class", type=int, default=2,
                     help="K: samples per class per training batch -- must be >=2 for "
                          "triplet/supcon to ever see a same-class pair.")
    ap.add_argument("--hard_cluster_prob", type=float, default=0.5,
                     help="probability a training batch's classes are forced to include one "
                          "full adjacent-version cluster (HARD_CLUSTERS, top of this file) "
                          "instead of drawing classes uniformly at random -- deliberate hard-"
                          "negative sampling (IDEAS.md #6, WhosAI/HALvest-Contrastive).")
    ap.add_argument("--tfidf_svd_dim", type=str, default="0",
                     help="POST-HOC readout only -- does NOT affect training or the classifier "
                          "head at all (the model trains ONCE per fold regardless of this flag). "
                          "After training, concatenate SVD-reduced word+char TF-IDF features "
                          "(same build_vectorizers/fit_transform as train.py, SVD_DIM=150 "
                          "convention from frozen_embeddings.ipynb) onto the trained embedding "
                          "and report nearest-centroid/SVC balanced accuracy -- same readout "
                          "embedding_readout() already does, just with TF-IDF added in. "
                          "Comma-separated list sweeps several dims in one run cheaply (each is "
                          "just a TF-IDF/SVD fit + sklearn classifier, no GPU/retraining), e.g. "
                          "'0,50,150,300'. 0 = no TF-IDF (the plain embedding_readout numbers), "
                          "and '0' alone (the default) matches every run before this flag existed.")
    ap.add_argument("--token_mask_prob", type=float, default=0.0,
                     help="CV-cutout analog: randomly hide this fraction of REAL tokens per "
                          "training row from both the backbone's attention and the pooling mask "
                          "(not input_ids -- Qwen has no [MASK] token). Training-time only, "
                          "never drops every real token in a row. 0.0 (default) = off.")
    ap.add_argument("--dropout_p", type=float, default=0.0,
                     help="dropout applied to the pooled embedding right before the classifier "
                          "head (and before every multi-sample-dropout pass, if that's also "
                          "on). 0.0 (default) = off, matching every run before this existed.")
    ap.add_argument("--multi_sample_dropout_k", type=int, default=1,
                     help="Inoue 2019 multi-sample dropout: run K independent --dropout_p "
                          "passes through the head per batch and average their CE losses -- "
                          "implicit ensembling within one forward pass. Needs --dropout_p > 0 "
                          "to do anything. 1 (default) = off (single pass, matching every run "
                          "before this existed).")
    ap.add_argument("--mixup_alpha", type=float, default=0.0,
                     help="manifold mixup (Beta(alpha,alpha) blend) between two SAME-class "
                          "pooled embeddings in a batch, when one exists (label unchanged, "
                          "since the partner already shares it) -- affects only the CE term, "
                          "not center/triplet/supcon. The embedding-space variant IDEAS.md #1 "
                          "flagged as untried; the TF-IDF-space version already tried and failed "
                          "(TRIED.md). Works best with --pk_per_class >= 2 so every class in a "
                          "batch actually has a same-class partner to mix with. 0.0 (default) = "
                          "off.")
    args = ap.parse_args()
    svd_dims = [int(x) for x in args.tfidf_svd_dim.split(",")]

    if torch.cuda.device_count() > 1:
        print("WARNING: >1 GPU visible. transformers.Trainer silently wraps in DataParallel in "
              "that case, which has caused multiple confirmed crash bugs on this project "
              "(TRIED.md). Set CUDA_VISIBLE_DEVICES=0 before running this script.")

    run = ExperimentRun(f"qwen_lora_{args.model_name.split('/')[-1]}", config=vars(args))
    df = pd.read_csv(data_dir() / "train.csv")
    class_labels = sorted(df["model"].unique())
    label2id = {l: i for i, l in enumerate(class_labels)}
    df["label"] = df["model"].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ext_df = None
    if args.external_paths:
        ext_df = load_external_data(args.external_paths, args.external_cap_per_class, label2id, run.log)
        ext_df["label"] = ext_df["model"].map(label2id)

    use_pk_sampling = args.triplet_weight > 0 or args.supcon_weight > 0
    if use_pk_sampling and args.pk_per_class < 2:
        run.log("WARNING: --pk_per_class < 2 means triplet/supcon loss never sees a same-class "
                "pair -- it will silently contribute ~0 every batch. Use >= 2.")
    hard_cluster_label_ids = [
        {label2id[name] for name in cluster if name in label2id} for cluster in HARD_CLUSTERS
    ]

    gkf = GroupKFold(n_splits=args.n_splits)
    scores = []
    centroid_scores = {d: [] for d in svd_dims}
    svc_scores = {d: [] for d in svd_dims}
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["model"], groups=df["prompt_id"])):
        if args.max_folds is not None and fold >= args.max_folds:
            break
        run.log(f"=== fold {fold} ===")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        official_train_df = train_df[["text", "label"]].reset_index(drop=True)

        if ext_df is not None:
            combined_train_df = pd.concat([official_train_df, ext_df[["text", "label"]]],
                                           ignore_index=True)
            phase2_epochs = min(args.final_official_only_epochs, args.epochs)
            phase1_epochs = args.epochs - phase2_epochs
        else:
            combined_train_df = official_train_df
            phase1_epochs, phase2_epochs = args.epochs, 0

        # TF-IDF is NEVER fed into training -- it changes the classifier head's input size,
        # which would force a full LoRA retrain per dim. The model trains ONCE per fold here;
        # --tfidf_svd_dim is swept purely as a cheap POST-HOC readout below (fit TF-IDF/SVD +
        # nearest-centroid/SVC on the one trained embedding, no GPU retraining at all).
        val_ds = TextDataset(val_df["text"], val_df["label"], tokenizer, args.max_length)

        model = build_model(args.model_name, len(class_labels), args.pooling_layer,
                             args.lora_r, args.lora_alpha, args.lora_dropout,
                             not args.no_gradient_checkpointing, args.precision,
                             args.load_in_4bit, run.log,
                             args.center_loss_weight, args.center_momentum,
                             args.triplet_weight, args.triplet_margin,
                             args.supcon_weight, args.supcon_temperature,
                             args.use_arcface_head, args.arcface_scale, args.arcface_margin,
                             0, args.token_mask_prob, args.dropout_p,
                             args.multi_sample_dropout_k, args.mixup_alpha)

        def run_phase(phase_name, phase_train_df, phase_epochs):
            phase_train_ds = TextDataset(phase_train_df["text"], phase_train_df["label"],
                                          tokenizer, args.max_length)
            train_batch_size = (args.pk_classes * args.pk_per_class if use_pk_sampling
                                 else args.batch_size)
            training_args = build_training_args(dict(
                output_dir=str(run.artifact_path(f"fold{fold}_{phase_name}_ckpt")),
                per_device_train_batch_size=train_batch_size,
                per_device_eval_batch_size=args.batch_size * 2,
                gradient_accumulation_steps=args.grad_accum_steps,
                num_train_epochs=phase_epochs,
                learning_rate=args.lr,
                eval_strategy="epoch",
                save_strategy="no",
                logging_steps=50,
                bf16=(args.precision == "bf16"),
                fp16=(args.precision == "fp16"),
                seed=args.seed,
                report_to=[],
            ))
            callbacks = []
            if args.probe:
                callbacks.append(ProbeCallback(
                    phase_train_df["text"], phase_train_df["label"], val_df["text"], val_df["label"],
                    tokenizer, args.max_length, args.batch_size * 2, args.probe, run.log,
                ))
            pk_sampler = None
            if use_pk_sampling:
                pk_sampler = PKBatchSampler(
                    phase_train_df["label"].values, args.pk_classes, args.pk_per_class,
                    hard_clusters=hard_cluster_label_ids, hard_cluster_prob=args.hard_cluster_prob,
                    seed=args.seed + fold,
                )
                run.log(f"  PK sampling: p={args.pk_classes} k={args.pk_per_class} "
                        f"({len(pk_sampler)} batches/epoch, batch_size={train_batch_size})")
            phase_trainer = PKSamplerTrainer(
                model=model, args=training_args, train_dataset=phase_train_ds, eval_dataset=val_ds,
                data_collator=ExtraFeaturesCollator(tokenizer), compute_metrics=compute_metrics,
                callbacks=callbacks, pk_sampler=pk_sampler,
            )
            phase_trainer.train()
            return phase_trainer

        trainer = None
        if phase1_epochs > 0:
            run.log(f"  phase 1: {phase1_epochs:.2g} epoch(s) on "
                     f"{'official+external' if ext_df is not None else 'official'} data "
                     f"({len(combined_train_df)} rows)")
            trainer = run_phase("phase1", combined_train_df, phase1_epochs)
        if phase2_epochs > 0:
            run.log(f"  phase 2: {phase2_epochs:.2g} epoch(s) on official data only "
                     f"({len(official_train_df)} rows)")
            trainer = run_phase("phase2", official_train_df, phase2_epochs)

        eval_result = trainer.evaluate()
        score = eval_result["eval_balanced_accuracy"]
        scores.append(score)
        run.log(f"fold {fold}: balanced_accuracy={score:.4f}")

        # Oracle cluster-assignment check (ideas list item 30): uses the REAL prompt_id, which
        # is available on official val data but NOT at real test time -- an upper bound on what
        # blind TF-IDF-detected clusters (predict.py's actual mechanism) could achieve here, not
        # a number achievable as-is. On the frozen-embedding baseline this was large (+0.14
        # oracle Hungarian) -- worth checking whether the fine-tuned head's errors are similarly
        # cluster-exploitable.
        val_logits = trainer.predict(val_ds).predictions
        val_proba = np.exp(val_logits - val_logits.max(axis=1, keepdims=True))
        val_proba /= val_proba.sum(axis=1, keepdims=True)
        for name, resolver in [("greedy", resolve_cluster_assignments),
                                ("hungarian", resolve_cluster_assignments_hungarian)]:
            pred_idx = resolver(val_proba, val_df["prompt_id"].values)
            pred = [class_labels[i] for i in pred_idx]
            cluster_score = balanced_accuracy_score(val_df["model"], pred)
            run.log(f"  fold {fold} oracle cluster-assign ({name}): "
                    f"balanced_accuracy={cluster_score:.4f} (delta={cluster_score - score:+.4f})")

        save_fold_model(model, tokenizer, run.artifact_path(f"fold{fold}_adapter"),
                         args.model_name, args.pooling_layer, args.max_length, run.log)

        for dim in svd_dims:
            official_extra = val_extra = None
            if dim > 0:
                # Fresh TF-IDF+SVD fit per (fold, dim) -- seconds, not GPU time. Fit on this
                # fold's official training text only (same leakage discipline as train.py).
                word_vec, char_vec, _, _ = build_vectorizers()
                X_fit = fit_transform(official_train_df["text"], word_vec, char_vec, fit=True)
                X_val = fit_transform(val_df["text"], word_vec, char_vec, fit=False)
                svd = TruncatedSVD(n_components=dim, random_state=args.seed + fold)
                official_extra = svd.fit_transform(X_fit).astype(np.float32)
                val_extra = svd.transform(X_val).astype(np.float32)

            readout = embedding_readout(
                model, tokenizer, official_train_df["text"], official_train_df["label"],
                val_df["text"], val_df["label"], args.max_length, args.batch_size * 2, run.log,
                train_extra=official_extra, val_extra=val_extra,
            )
            run.log(f"  fold {fold} readout @ tfidf_svd_dim={dim}: "
                    f"nearest_centroid={readout['nearest_centroid']:.4f} svc={readout['svc']:.4f}")
            centroid_scores[dim].append(readout["nearest_centroid"])
            svc_scores[dim].append(readout["svc"])

        del model, trainer
        torch.cuda.empty_cache()

    run.log_scores("balanced_accuracy", scores)
    for dim in svd_dims:
        run.log_scores(f"balanced_accuracy_nearest_centroid_svd{dim}", centroid_scores[dim])
        run.log_scores(f"balanced_accuracy_svc_embedding_svd{dim}", svc_scores[dim])


if __name__ == "__main__":
    main()
