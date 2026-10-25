"""Idea 2 (feature-level fusion variant): per GroupKFold fold, fine-tune a transformer for the
18-way task, then fuse [its mean-pooled embedding | SVD-compressed TF-IDF | stylometric features]
into one dense matrix and train LightGBM on top. Compares against the transformer's own plain
classifier head on the same fold, so we can see whether fusion actually adds anything.

Rationale for SVD instead of raw TF-IDF: LightGBM is tree-based, which splits one feature at a
time and doesn't exploit very high-dimensional sparse bag-of-words features nearly as well as a
linear model does -- that's why the baseline uses LogisticRegression, not trees, on raw TF-IDF.
Compressing TF-IDF to a dense ~150-dim space via TruncatedSVD first plays to LightGBM's strength
(dense tabular features) instead of against it.
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from exp_log import ExperimentRun
from paths import data_dir
from stylometric import stylometric_matrix
from train import build_vectorizers, fit_transform
from train_transformer import TextDataset, build_curve, make_trainer


@torch.no_grad()
def infer(model, tokenizer, texts, max_length, batch_size=16):
    """One forward pass per batch -> (mean-pooled last-hidden-state embedding, softmax probs
    from the classification head). Avoids running the model twice for embeddings + predictions."""
    model.eval()
    device = next(model.parameters()).device
    embs, probs = [], []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        embs.append(pooled.float().cpu().numpy())
        probs.append(torch.softmax(out.logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(embs, axis=0), np.concatenate(probs, axis=0)


def load_checkpoint_model(checkpoint_root, fold):
    """Loads the best checkpoint a previous run saved for this fold, instead of retraining."""
    hits = sorted(glob.glob(os.path.join(checkpoint_root, "artifacts", f"fold{fold}_ckpt", "checkpoint-*")))
    assert hits, f"no checkpoint found under {checkpoint_root}/artifacts/fold{fold}_ckpt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return AutoModelForSequenceClassification.from_pretrained(hits[-1]).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--svd_dim", type=int, default=150)
    ap.add_argument("--checkpoint_dir", default=None,
                     help="reuse a previous run's fold checkpoints (experiments/<run_id>) instead of retraining")
    args = ap.parse_args()

    run = ExperimentRun("fusion_lgbm", config=vars(args))

    df = pd.read_csv(data_dir() / "train.csv")
    labels_sorted = sorted(df["model"].unique())
    label2id = {l: i for i, l in enumerate(labels_sorted)}
    df["label"] = df["model"].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.save_pretrained(run.artifact_path("tokenizer"))

    # OOF containers, aligned to df's row order, filled in as each fold's val split is scored.
    n, k = len(df), len(labels_sorted)
    oof_fusion_proba = np.full((n, k), np.nan, dtype=np.float32)
    oof_transformer_proba = np.full((n, k), np.nan, dtype=np.float32)

    gkf = GroupKFold(n_splits=args.n_splits)
    fusion_scores, transformer_scores = [], []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["label"], groups=df["prompt_id"])):
        if args.max_folds is not None and fold >= args.max_folds:
            break
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        trainer = None
        if args.checkpoint_dir:
            model = load_checkpoint_model(args.checkpoint_dir, fold)
        else:
            train_ds = TextDataset(train_df["text"], train_df["label"].tolist(), tokenizer, args.max_length)
            val_ds = TextDataset(val_df["text"], val_df["label"].tolist(), tokenizer, args.max_length)
            trainer = make_trainer(
                args.model_name, train_ds, val_ds, tokenizer,
                out_dir=str(run.artifact_path(f"fold{fold}_ckpt")),
                epochs=args.epochs, bs=args.batch_size, lr=args.lr, seed=args.seed,
            )
            trainer.train()
            run.log_curve(f"fold{fold}", build_curve(trainer.state.log_history))
            model = trainer.model

        emb_train, _ = infer(model, tokenizer, train_df["text"].tolist(), args.max_length, args.batch_size)
        emb_val, proba_val = infer(model, tokenizer, val_df["text"].tolist(), args.max_length, args.batch_size)
        oof_transformer_proba[val_idx] = proba_val

        t_score = balanced_accuracy_score(val_df["label"], np.argmax(proba_val, axis=-1))
        transformer_scores.append(t_score)
        run.log(f"fold {fold}: transformer_only={t_score:.4f}")

        word_vec, char_vec = build_vectorizers()
        X_tfidf_train = fit_transform(train_df["text"], word_vec, char_vec, fit=True)
        X_tfidf_val = fit_transform(val_df["text"], word_vec, char_vec, fit=False)
        svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed)
        svd_train = svd.fit_transform(X_tfidf_train)
        svd_val = svd.transform(X_tfidf_val)

        stylo_train = stylometric_matrix(train_df["text"])
        stylo_val = stylometric_matrix(val_df["text"])

        X_fusion_train = np.concatenate([emb_train, svd_train, stylo_train], axis=1).astype(np.float32)
        X_fusion_val = np.concatenate([emb_val, svd_val, stylo_val], axis=1).astype(np.float32)

        sample_weight = compute_sample_weight("balanced", train_df["label"])
        lgbm = LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
            random_state=args.seed, verbose=-1, n_jobs=-1,
        )
        lgbm.fit(X_fusion_train, train_df["label"], sample_weight=sample_weight)
        fold_proba = np.zeros((len(val_df), k), dtype=np.float32)
        fold_proba[:, lgbm.classes_] = lgbm.predict_proba(X_fusion_val)
        oof_fusion_proba[val_idx] = fold_proba

        joblib.dump(
            {"word_vec": word_vec, "char_vec": char_vec, "svd": svd, "lgbm": lgbm},
            run.artifact_path(f"fold{fold}_fusion.joblib"),
        )

        f_score = balanced_accuracy_score(val_df["label"], np.argmax(fold_proba, axis=-1))
        fusion_scores.append(f_score)
        run.log(
            f"fold {fold}: fusion_lgbm={f_score:.4f}  "
            f"(embed={emb_train.shape[1]}d, svd={args.svd_dim}d, stylo={stylo_train.shape[1]}d)"
        )

        del model
        if trainer is not None:
            del trainer
        torch.cuda.empty_cache()

    run.log_scores("transformer_only", transformer_scores)
    run.log_scores("fusion_lgbm", fusion_scores)

    covered = ~np.isnan(oof_fusion_proba).any(axis=1)
    oof = df.loc[covered, ["id", "prompt_id", "text", "model"]].rename(columns={"model": "true_label"}).copy()
    oof["fusion_pred"] = [labels_sorted[i] for i in oof_fusion_proba[covered].argmax(axis=1)]
    oof["transformer_pred"] = [labels_sorted[i] for i in oof_transformer_proba[covered].argmax(axis=1)]
    oof.to_csv(run.artifact_path("oof.csv"), index=False)
    np.save(run.artifact_path("oof_fusion_proba.npy"), oof_fusion_proba)
    np.save(run.artifact_path("oof_transformer_proba.npy"), oof_transformer_proba)
    json.dump(labels_sorted, open(run.artifact_path("labels.json"), "w"))
    run.log(f"saved OOF predictions for {covered.sum()}/{n} rows -> oof.csv, oof_*_proba.npy")


if __name__ == "__main__":
    main()
