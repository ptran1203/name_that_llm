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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainerCallback

from exp_log import ExperimentRun
from experiment_multi_embed_concat import load_external_data
from paths import data_dir
from train_transformer import build_training_args

# Under accelerate/DDP each process has its own assigned GPU -- PartialState().device resolves
# to the right one for THIS process regardless of launch method (accelerate launch, torchrun,
# plain python), unlike hand-parsing LOCAL_RANK ourselves. A bare "cuda" always means cuda:0,
# which would put every process's model on the same GPU and break DDP.
DEVICE = PartialState().device


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


class LayerPoolClassifier(nn.Module):
    """base_model (frozen + LoRA) -> hidden_states[layer] -> mean-pool over attention mask ->
    linear head -> logits. The classifier head is left in float32 (standard practice for
    numerical stability even when the backbone runs in bf16/fp16) -- pooled activations are
    cast to match before the final matmul."""

    def __init__(self, base_model, hidden_size, num_labels, layer, center_loss_weight=0.0,
                 center_momentum=0.9):
        super().__init__()
        self.base_model = base_model
        self.layer = layer
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.center_loss_weight = center_loss_weight
        self.center_momentum = center_momentum
        # EMA per-class embedding center (not a trainable nn.Parameter -- updated directly from
        # batch statistics, not via gradient descent, so it tracks the class mean independent of
        # the aux loss's own gradient).
        self.register_buffer("centers", torch.zeros(num_labels, hidden_size))

    def pool(self, input_ids, attention_mask):
        """The embedding this model is actually using right now -- same op the probe callback
        below reuses to check whether that embedding is improving epoch over epoch, independent
        of how the classifier head on top of it is doing."""
        out = self.base_model(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
        hidden = out.hidden_states[self.layer]
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        pooled = self.pool(input_ids, attention_mask)
        logits = self.classifier(pooled.to(self.classifier.weight.dtype))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            if self.center_loss_weight > 0:
                loss = loss + self.center_loss_weight * self._center_loss(pooled, labels)
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

    def _embed(self, model, texts):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self.tokenizer(batch, truncation=True, max_length=self.max_length,
                                      padding=True, return_tensors="pt").to(DEVICE)
                pooled = model.pool(enc["input_ids"], enc["attention_mask"])
                out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        train_emb = self._embed(model, self.train_texts)
        val_emb = self._embed(model, self.val_texts)
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
                 center_loss_weight=0.0, center_momentum=0.9):
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
                                 center_loss_weight, center_momentum).to(DEVICE)
    return model


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
    args = ap.parse_args()

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

    gkf = GroupKFold(n_splits=args.n_splits)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["model"], groups=df["prompt_id"])):
        if args.max_folds is not None and fold >= args.max_folds:
            break
        run.log(f"=== fold {fold} ===")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        official_train_df = train_df[["text", "label"]].reset_index(drop=True)
        val_ds = TextDataset(val_df["text"], val_df["label"], tokenizer, args.max_length)

        if ext_df is not None:
            combined_train_df = pd.concat([official_train_df, ext_df[["text", "label"]]],
                                           ignore_index=True)
            phase2_epochs = min(args.final_official_only_epochs, args.epochs)
            phase1_epochs = args.epochs - phase2_epochs
        else:
            combined_train_df = official_train_df
            phase1_epochs, phase2_epochs = args.epochs, 0

        model = build_model(args.model_name, len(class_labels), args.pooling_layer,
                             args.lora_r, args.lora_alpha, args.lora_dropout,
                             not args.no_gradient_checkpointing, args.precision,
                             args.load_in_4bit, run.log,
                             args.center_loss_weight, args.center_momentum)

        def run_phase(phase_name, phase_train_df, phase_epochs):
            phase_train_ds = TextDataset(phase_train_df["text"], phase_train_df["label"],
                                          tokenizer, args.max_length)
            training_args = build_training_args(dict(
                output_dir=str(run.artifact_path(f"fold{fold}_{phase_name}_ckpt")),
                per_device_train_batch_size=args.batch_size,
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
            phase_trainer = Trainer(
                model=model, args=training_args, train_dataset=phase_train_ds, eval_dataset=val_ds,
                data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
                callbacks=callbacks,
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

        del model, trainer
        torch.cuda.empty_cache()

    run.log_scores("balanced_accuracy", scores)


if __name__ == "__main__":
    main()
