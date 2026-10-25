"""Inference: ensembles the 5 fold models from a fusion_lgbm run (transformer embedding +
SVD-TFIDF + stylometric -> LightGBM, per fold) by averaging their predicted probabilities,
and writes submission.parquet in the competition's required format.

Works both locally and inside the Kaggle competition environment: paths.py resolves
data/model locations for either case, so this script doesn't need editing per environment.

Usage:
    python src/predict.py --run_dir experiments/<run_id>
    python src/predict.py --run_dir /kaggle/input/<dataset-slug>   # on Kaggle, internet off
"""
import argparse
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from paths import data_dir
from stylometric import stylometric_matrix
from train_fusion_lgbm import infer


def find_run_dir(run_dir):
    """A run_dir may be an experiments/<run_id> folder (artifacts/ subfolder) or, once
    packaged for Kaggle, the dataset root itself (artifacts/ flattened into the root)."""
    if os.path.isdir(os.path.join(run_dir, "artifacts")):
        return os.path.join(run_dir, "artifacts")
    return run_dir  # already flattened (Kaggle dataset layout)


def load_fold(artifacts_dir, fold, device):
    ckpt_hits = sorted(glob.glob(os.path.join(artifacts_dir, f"fold{fold}_ckpt", "checkpoint-*")))
    ckpt_path = ckpt_hits[-1] if ckpt_hits else os.path.join(artifacts_dir, f"fold{fold}_ckpt")
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)
    fusion = joblib.load(os.path.join(artifacts_dir, f"fold{fold}_fusion.joblib"))
    return model, fusion


def predict_fold(model, fusion, tokenizer, texts, max_length, batch_size, k):
    emb, _ = infer(model, tokenizer, texts, max_length, batch_size)

    word_vec, char_vec, svd, lgbm = fusion["word_vec"], fusion["char_vec"], fusion["svd"], fusion["lgbm"]
    X_tfidf = hstack([word_vec.transform(texts), char_vec.transform(texts)]).tocsr()
    svd_feat = svd.transform(X_tfidf)
    stylo_feat = stylometric_matrix(texts)

    X_fusion = np.concatenate([emb, svd_feat, stylo_feat], axis=1).astype(np.float32)

    fusion_proba = np.zeros((len(texts), k), dtype=np.float32)
    fusion_proba[:, lgbm.classes_] = lgbm.predict_proba(X_fusion)
    return fusion_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="experiments/<run_id> from train_fusion_lgbm.py, or a packaged Kaggle dataset dir")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="submission.parquet")
    args = ap.parse_args()

    artifacts_dir = find_run_dir(args.run_dir)
    labels_sorted = json.load(open(os.path.join(artifacts_dir, "labels.json")))
    k = len(labels_sorted)

    test = pd.read_csv(data_dir() / "test.csv")
    test["text"] = test["text"].fillna("").astype(str)
    texts = test["text"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(artifacts_dir, "tokenizer"), use_fast=False)

    avg_proba = np.zeros((len(test), k), dtype=np.float32)
    n_used = 0
    for fold in range(args.n_folds):
        fold_dir = os.path.join(artifacts_dir, f"fold{fold}_ckpt")
        if not os.path.isdir(fold_dir):
            print(f"fold {fold}: no checkpoint found, skipping")
            continue
        model, fusion = load_fold(artifacts_dir, fold, device)
        proba = predict_fold(model, fusion, tokenizer, texts, args.max_length, args.batch_size, k)
        avg_proba += proba
        n_used += 1
        print(f"fold {fold}: predicted {len(test)} rows")
        del model
        torch.cuda.empty_cache()

    assert n_used > 0, f"no fold checkpoints found under {artifacts_dir}"
    avg_proba /= n_used
    pred_idx = avg_proba.argmax(axis=1)
    pred_labels = [labels_sorted[i] for i in pred_idx]

    sub = pd.DataFrame({"id": test["id"], "model": pred_labels})
    assert sub["id"].is_unique and sub["model"].isin(labels_sorted).all() and len(sub) == len(test)
    sub.to_parquet(args.out, index=False)
    print(f"wrote {args.out} ({len(sub)} rows, ensembled over {n_used} fold(s))")
    print(sub["model"].value_counts().to_string())


if __name__ == "__main__":
    main()
