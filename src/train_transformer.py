"""Fine-tune a transformer encoder (default: DeBERTa-v3-base) for 18-way model attribution.

GroupKFold on prompt_id (never split a prompt across train/val) to get an honest CV estimate.
Saves per-fold OOF predictions and prints mean balanced accuracy. Optionally fits a final
model on all data for later inference / ensembling.
"""
import argparse
import inspect
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from paths import data_dir
from exp_log import ExperimentRun

# Mirrors analyze_errors.py's FAMILY_PREFIXES -- kept duplicated (not imported) so this module
# doesn't pull in analyze_errors.py's matplotlib dependency just for a 6-item list.
FAMILY_PREFIXES = ["claude", "gpt", "gemini", "glm", "qwen", "kimi"]


def family_of(label):
    for p in FAMILY_PREFIXES:
        if label.startswith(p):
            return p
    return label


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, weights=None, family_labels=None):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = None if labels is None else list(labels)
        self.weights = None if weights is None else list(weights)
        self.family_labels = None if family_labels is None else list(family_labels)

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        if self.weights is not None:
            item["sample_weight"] = self.weights[idx]
        if self.family_labels is not None:
            item["family_label"] = self.family_labels[idx]
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"balanced_accuracy": balanced_accuracy_score(labels, preds)}


_NOISE_KEYS = {"eval_runtime", "eval_samples_per_second", "eval_steps_per_second"}


def build_curve(log_history):
    """Merges Trainer.state.log_history (separate train-step and eval entries) into one
    row per epoch: {"epoch": ..., "train_loss": ..., "eval_loss": ..., "eval_balanced_accuracy": ...}."""
    by_epoch = {}
    for entry in log_history:
        if "train_runtime" in entry:
            continue  # final train() summary row, not a per-epoch point
        epoch = round(entry.get("epoch", -1), 4)
        row = by_epoch.setdefault(epoch, {"epoch": epoch})
        if "loss" in entry:
            row["train_loss"] = entry["loss"]
        for k, v in entry.items():
            if k.startswith("eval_") and k not in _NOISE_KEYS:
                row[k] = v
    return sorted(by_epoch.values(), key=lambda r: r["epoch"])


def freeze_layers(model, unfreeze_last_n):
    """Freezes embeddings + all encoder transformer layers except the last `unfreeze_last_n`;
    pooler/classifier head always stays trainable. None = full fine-tune (no freezing).
    Rationale: with ~2160 train rows/fold, full fine-tuning a 184M-param model visibly overfits
    (train_loss << eval_loss within a few epochs). Freezing early layers -- which tend to carry
    generic/syntactic features, not task-specific ones -- cuts trainable params and should
    overfit less, at the cost of less task adaptation. Worth testing empirically, not assuming."""
    if unfreeze_last_n is None:
        return None, None
    layer_re = re.compile(r"encoder\.layer\.(\d+)\.")
    max_layer = max(
        (int(m.group(1)) for name, _ in model.named_parameters() if (m := layer_re.search(name))),
        default=None,
    )
    assert max_layer is not None, "could not find 'encoder.layer.<i>.' params to freeze"
    cutoff = max_layer - unfreeze_last_n + 1

    trainable, frozen = 0, 0
    for name, param in model.named_parameters():
        m = layer_re.search(name)
        if m:
            param.requires_grad = int(m.group(1)) >= cutoff
        elif "embeddings" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True  # pooler, classifier, any layers outside the numbered stack
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return trainable, frozen


_CRITICAL_ARGS = {"eval_strategy", "evaluation_strategy", "save_strategy", "load_best_model_at_end",
                   "metric_for_best_model", "greater_is_better"}
_RENAMES = {"eval_strategy": "evaluation_strategy"}  # older transformers used the longer name


def build_training_args(wanted):
    """Builds TrainingArguments from `wanted`, filtered to whatever this installed transformers
    version actually supports. Different environments (local/Kaggle/Colab) can have meaningfully
    different transformers versions, and a hard crash on one unsupported kwarg (e.g. a renamed
    scheduler option) shouldn't take down the whole run. Loudly warns if anything load-bearing
    for correctness (best-checkpoint selection) gets dropped instead of silently losing it."""
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    kwargs = {}
    dropped = []
    for key, value in wanted.items():
        if key in supported:
            kwargs[key] = value
        elif key in _RENAMES and _RENAMES[key] in supported:
            kwargs[_RENAMES[key]] = value
        else:
            dropped.append(key)
    if dropped:
        bad = _CRITICAL_ARGS & set(dropped)
        print(f"TrainingArguments on this transformers version doesn't support: {dropped} -- dropping them.")
        if bad:
            print(f"WARNING: dropped args affecting correctness (best-checkpoint selection): {bad}")
    return TrainingArguments(**kwargs)


class MultiTaskTrainer(Trainer):
    """Custom compute_loss for: (a) per-example sample weighting (down-weights external-data rows
    relative to official rows -- the same discipline as the frozen-embeddings notebook), and
    (b) an optional family-level auxiliary loss. The aux loss marginalizes the existing 18-way
    softmax into FAMILY_PREFIXES groups and penalizes getting the *family* wrong -- zero new
    parameters (same classifier head), so it doesn't inherit the p>>n overfitting that sank the
    mixture-of-experts gate classifiers in TRIED.md. Targets cross-family confusions (claude-vs-
    glm); does nothing for within-family sibling confusions since family is already correct there.
    label_smoothing=0 and family_aux_weight=0 reduce this to plain unweighted cross-entropy."""

    def __init__(self, *args, label_smoothing=0.0, family_aux_weight=0.0, family_matrix=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
        self.family_aux_weight = family_aux_weight
        self.family_matrix = family_matrix  # [num_labels, num_families] one-hot, or None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weight = inputs.pop("sample_weight", None)
        family_label = inputs.pop("family_label", None)
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        per_example = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing, reduction="none")

        if self.family_aux_weight > 0 and family_label is not None and self.family_matrix is not None:
            probs = F.softmax(logits, dim=-1)
            family_probs = probs @ self.family_matrix.to(device=probs.device, dtype=probs.dtype)
            family_loss = F.nll_loss(torch.log(family_probs.clamp_min(1e-8)), family_label, reduction="none")
            per_example = per_example + self.family_aux_weight * family_loss

        if sample_weight is not None:
            sample_weight = sample_weight.to(per_example.dtype)
            loss = (per_example * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
        else:
            loss = per_example.mean()

        return (loss, outputs) if return_outputs else loss


def make_trainer(model_name, train_ds, val_ds, tokenizer, out_dir, epochs, bs, lr, seed=42,
                  unfreeze_last_n=None, precision="fp16", max_steps=None,
                  label_smoothing=0.0, family_aux_weight=0.0, family_matrix=None):
    # Must seed BEFORE from_pretrained: the classification head (classifier/pooler) is randomly
    # initialized right here, and TrainingArguments(seed=...) only takes effect once the Trainer
    # is constructed afterward -- too late to control this init.
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=18)
    trainable, frozen = freeze_layers(model, unfreeze_last_n)
    if trainable is not None:
        print(f"unfreeze_last_n={unfreeze_last_n}: trainable={trainable:,} frozen={frozen:,} "
              f"({100 * trainable / (trainable + frozen):.1f}% trainable)")

    if precision == "fp16":
        # Diagnostic for the known Kaggle (transformers 5.x) crash: "Attempting to unscale FP16
        # gradients" during grad-clipping. Expected/correct state for autocast-based fp16 mixed
        # precision is fp32 master weights; if from_pretrained() itself produced fp16 weights,
        # that's the actual bug, and this will say so immediately instead of failing opaquely
        # mid-training (and only on whichever rank/step happens to print first under DDP).
        model_dtype = next(model.parameters()).dtype
        if model_dtype != torch.float32:
            print(f"WARNING: model loaded as {model_dtype}, not float32 -- fp16 mixed precision "
                  f"expects fp32 master weights and will likely crash with 'Attempting to unscale "
                  f"FP16 gradients' during grad-clipping. This confirms from_pretrained() itself is "
                  f"producing non-fp32 weights on this environment, not an autocast config issue.")

    # max_steps: for smoke/correctness tests -- train.py's --max_steps overrides epoch-based
    # training entirely, so e.g. --max_steps 1 --batch_size 2 runs exactly one batch, not a full
    # epoch's worth of steps (epochs=1 still means ~270 steps on the full fold, not "one batch").
    eval_save_kwargs = (
        dict(eval_strategy="steps", eval_steps=max_steps, save_strategy="steps", save_steps=max_steps)
        if max_steps else
        dict(eval_strategy="epoch", save_strategy="epoch")
    )
    wanted = dict(
        output_dir=out_dir,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps else -1,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=(precision == "fp16" and torch.cuda.is_available()),
        bf16=(precision == "bf16" and torch.cuda.is_available()),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        logging_strategy="steps" if max_steps else "epoch",
        logging_steps=max_steps if max_steps else 500,
        report_to="none",
        seed=seed,
        dataloader_pin_memory=torch.cuda.is_available(),
        # our TextDataset is a plain torch Dataset, not a datasets.Dataset, so column-pruning
        # normally no-ops anyway -- set explicitly so sample_weight/family_label can never get
        # silently dropped before reaching compute_loss.
        remove_unused_columns=False,
        **eval_save_kwargs,
    )
    args = build_training_args(wanted)
    collator = DataCollatorWithPadding(tokenizer)
    return MultiTaskTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        label_smoothing=label_smoothing,
        family_aux_weight=family_aux_weight,
        family_matrix=family_matrix,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--max_steps", type=int, default=None,
                     help="overrides --epochs; for smoke tests, e.g. --max_steps 1 --batch_size 2")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_folds", type=int, default=None, help="only run first N folds (quick timing check)")
    ap.add_argument("--fit_final", action="store_true", help="also fit on all data and save as an artifact")
    ap.add_argument("--run_name", default=None, help="defaults to the model name, slash-free")
    ap.add_argument("--unfreeze_last_n", type=int, default=None,
                     help="only fine-tune the last N encoder layers + head; default None = full fine-tune")
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"],
                     help="default fp32: confirmed working on Kaggle (transformers 5.x), where "
                          "fp16 hits 'Attempting to unscale FP16 gradients' during grad-clipping. "
                          "Root cause not fully pinned down (needs live CUDA debugging I can't do "
                          "locally); fp32 is slower but reliable. Pass fp16 explicitly if your "
                          "environment is confirmed to support it (e.g. matches transformers 4.57.6).")
    ap.add_argument("--external_paths", default=None,
                     help="comma-separated parquet paths (schema: text, model[, prompt_id, "
                          "source_dataset] -- matches src/prepare_external_data.py's output, e.g. "
                          "data/external/prvsiyan.parquet,data/external/darragh1walsh.parquet). "
                          "Rows always join the TRAIN side of every fold, never val -- CV stays an "
                          "honest estimate on official data only. ntl_open_pool.parquet needs its "
                          "own flag-filtering/capping first (see the frozen_embeddings notebook); "
                          "not done here.")
    ap.add_argument("--external_weight", type=float, default=0.25,
                     help="sample_weight for external rows relative to official rows (1.0)")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--family_aux_weight", type=float, default=0.0,
                     help="weight of the family-level auxiliary loss (0 = off). Marginalizes the "
                          "18-way softmax into FAMILY_PREFIXES groups; targets cross-family "
                          "confusions like claude-vs-glm. Try e.g. 0.3 -- untested, quick-check "
                          "with --max_folds 1 before trusting.")
    args = ap.parse_args()

    run_name = args.run_name or args.model_name.replace("/", "_")
    run = ExperimentRun(run_name, config=vars(args))

    df = pd.read_csv(data_dir() / "train.csv")
    labels_sorted = sorted(df["model"].unique())
    label2id = {l: i for i, l in enumerate(labels_sorted)}
    df["label"] = df["model"].map(label2id)
    df["sample_weight"] = 1.0

    families_sorted = sorted(set(family_of(l) for l in labels_sorted))
    family2id = {f: i for i, f in enumerate(families_sorted)}
    id2family_idx = torch.tensor([family2id[family_of(l)] for l in labels_sorted], dtype=torch.long)
    family_matrix = F.one_hot(id2family_idx, num_classes=len(families_sorted)).float()  # [18, n_fam]
    df["family_label"] = df["label"].map(lambda i: family2id[family_of(labels_sorted[i])])
    run.log(f"families: {family2id}")

    ext_df = None
    if args.external_paths:
        frames = []
        for p in args.external_paths.split(","):
            edf = pd.read_parquet(p.strip())
            before = len(edf)
            edf = edf[edf["model"].isin(label2id)].copy()
            if len(edf) < before:
                run.log(f"{p}: dropped {before - len(edf)} rows with unknown model labels")
            frames.append(edf[["text", "model"]])
        ext_df = pd.concat(frames, ignore_index=True)
        ext_df["label"] = ext_df["model"].map(label2id)
        ext_df["family_label"] = ext_df["label"].map(lambda i: family2id[family_of(labels_sorted[i])])
        ext_df["sample_weight"] = args.external_weight
        run.log(f"external data: {len(ext_df)} rows, weight={args.external_weight}")
        run.log(f"external by model:\n{ext_df['model'].value_counts()}")

    # DeBERTa-v3's sentencepiece tokenizer has a known fast-tokenizer conversion bug (byte-fallback
    # isn't implemented in the fast path), which silently produces extra unknown tokens and can tank
    # fine-tuning accuracy. Force the slow tokenizer to avoid it.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    gkf = GroupKFold(n_splits=args.n_splits)
    oof_pred = np.full(len(df), -1)
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(df, df["label"], groups=df["prompt_id"])
    ):
        if args.max_folds is not None and fold >= args.max_folds:
            break
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        # external rows always join the TRAIN side only -- val stays pure official so CV remains
        # an honest estimate of generalization to competition data.
        if ext_df is not None:
            train_df = pd.concat([train_df, ext_df], ignore_index=True)
        train_ds = TextDataset(train_df["text"], train_df["label"].tolist(), tokenizer, args.max_length,
                                weights=train_df["sample_weight"].tolist(),
                                family_labels=train_df["family_label"].tolist())
        val_ds = TextDataset(val_df["text"], val_df["label"].tolist(), tokenizer, args.max_length)

        trainer = make_trainer(
            args.model_name, train_ds, val_ds, tokenizer,
            out_dir=str(run.artifact_path(f"fold{fold}_ckpt")),
            epochs=args.epochs, bs=args.batch_size, lr=args.lr, seed=args.seed,
            unfreeze_last_n=args.unfreeze_last_n, precision=args.precision, max_steps=args.max_steps,
            label_smoothing=args.label_smoothing, family_aux_weight=args.family_aux_weight,
            family_matrix=family_matrix,
        )
        trainer.train()
        run.log_curve(f"fold{fold}", build_curve(trainer.state.log_history))
        preds = trainer.predict(val_ds)
        pred_labels = np.argmax(preds.predictions, axis=-1)
        oof_pred[val_idx] = pred_labels
        score = balanced_accuracy_score(val_df["label"], pred_labels)
        fold_scores.append(score)
        run.log(f"fold {fold}: balanced_accuracy={score:.4f}")

        del trainer
        torch.cuda.empty_cache()

    run.log_scores("balanced_accuracy", fold_scores)
    np.save(run.artifact_path("oof_pred.npy"), oof_pred)

    if args.fit_final:
        out_dir = run.artifact_path("final_model")
        full_df = pd.concat([df, ext_df], ignore_index=True) if ext_df is not None else df
        full_ds = TextDataset(full_df["text"], full_df["label"].tolist(), tokenizer, args.max_length,
                               weights=full_df["sample_weight"].tolist(),
                               family_labels=full_df["family_label"].tolist())
        trainer = make_trainer(
            args.model_name, full_ds, full_ds, tokenizer,
            out_dir=str(out_dir), epochs=args.epochs, bs=args.batch_size, lr=args.lr, seed=args.seed,
            unfreeze_last_n=args.unfreeze_last_n, precision=args.precision, max_steps=args.max_steps,
            label_smoothing=args.label_smoothing, family_aux_weight=args.family_aux_weight,
            family_matrix=family_matrix,
        )
        trainer.train()
        run.log_curve("final_fit", build_curve(trainer.state.log_history))  # note: eval here is in-sample (val_ds == train_ds)
        os.makedirs(out_dir, exist_ok=True)
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        import json
        json.dump(labels_sorted, open(out_dir / "labels.json", "w"))
        run.log(f"saved final model to {out_dir}")


if __name__ == "__main__":
    main()


