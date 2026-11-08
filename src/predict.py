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
import tarfile
import tempfile

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import hstack
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from paths import data_dir
from prompt_cluster_assign import (
    DEFAULT_THRESHOLD, detect_clusters, resolve_cluster_assignments,
    resolve_cluster_assignments_hungarian,
)
from stylometric import stylometric_matrix
from train_fusion_lgbm import infer


def find_run_dir(run_dir):
    """A run_dir may be an experiments/<run_id> folder (artifacts/ subfolder) or, once
    packaged for Kaggle, the dataset root itself (artifacts/ flattened into the root)."""
    if os.path.isdir(os.path.join(run_dir, "artifacts")):
        return os.path.join(run_dir, "artifacts")
    return run_dir  # already flattened (Kaggle dataset layout)


def ensure_dir(path):
    """package_for_kaggle.py uploads directories with --dir-mode tar; it's not confirmed
    whether Kaggle auto-extracts .tar the way it does .zip when a dataset is attached, so
    handle both: use the directory if it's already there, else extract the sibling .tar
    (input datasets are read-only, so extract into a writable temp dir)."""
    if os.path.isdir(path):
        return path
    tar_path = path + ".tar"
    assert os.path.isfile(tar_path), f"neither {path} nor {tar_path} found"
    out_dir = os.path.join(tempfile.gettempdir(), "name_that_llm_extracted", os.path.basename(path))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(out_dir)
    if os.path.isfile(os.path.join(out_dir, "config.json")):
        return out_dir
    subdirs = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
    return os.path.join(out_dir, subdirs[0]) if len(subdirs) == 1 else out_dir


def load_fold(artifacts_dir, fold, device):
    fold_dir = ensure_dir(os.path.join(artifacts_dir, f"fold{fold}_ckpt"))
    ckpt_hits = sorted(glob.glob(os.path.join(fold_dir, "checkpoint-*")))
    ckpt_path = ckpt_hits[-1] if ckpt_hits else fold_dir
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)
    fusion = joblib.load(os.path.join(artifacts_dir, f"fold{fold}_fusion.joblib"))
    return model, fusion


def predict_fold(model, fusion, tokenizer, texts, max_length, batch_size, k):
    emb, _ = infer(model, tokenizer, texts, max_length, batch_size)

    word_vec, char_vec, svd, lgbm = fusion["word_vec"], fusion["char_vec"], fusion["svd"], fusion["lgbm"]
    X_tfidf = hstack([word_vec.transform(texts), char_vec.transform(texts)]).tocsr()
    svd_feat = svd.transform(X_tfidf)
    stylo_feat = stylometric_matrix(texts)

    # stylometric.py has been extended over time (11 -> 16 features, see git history) -- a
    # checkpoint saved before such a change expects fewer stylo columns than the CURRENT
    # stylometric_matrix() produces. Truncate/pad to match so old checkpoints keep working
    # instead of a cryptic LightGBM feature-count ValueError, but say so loudly: this means
    # results reflect that checkpoint's OLDER feature set, not the current one. Retrain to
    # actually pick up newer stylometric features.
    expected_stylo_dim = lgbm.n_features_ - emb.shape[1] - svd_feat.shape[1]
    if stylo_feat.shape[1] != expected_stylo_dim:
        print(f"WARNING: stylometric_matrix() produces {stylo_feat.shape[1]} features, but this "
              f"saved LGBM model expects {expected_stylo_dim} (trained with a different "
              f"stylometric.py version). Truncating/padding to match -- retrain this fold to "
              f"actually use the current stylometric features.")
        if stylo_feat.shape[1] > expected_stylo_dim:
            stylo_feat = stylo_feat[:, :expected_stylo_dim]
        else:
            pad = np.zeros((stylo_feat.shape[0], expected_stylo_dim - stylo_feat.shape[1]), dtype=stylo_feat.dtype)
            stylo_feat = np.concatenate([stylo_feat, pad], axis=1)

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
    ap.add_argument("--cluster_assign", action="store_true",
                     help="post-process predictions with prompt_cluster_assign.py: detect "
                          "passages likely sharing a hidden prompt via TF-IDF similarity, then "
                          "resolve within-cluster label conflicts by confidence rank instead of "
                          "plain per-row argmax. Calibrated on train (see TRIED.md): a real but "
                          "small, fragile gain on the fusion_lgbm model (~+0.004-0.005 at the "
                          "best threshold, sign-unstable nearby) -- low risk, modest upside, off "
                          "by default.")
    ap.add_argument("--cluster_threshold", type=float, default=DEFAULT_THRESHOLD,
                     help="TF-IDF cosine similarity threshold for prompt_cluster_assign.py's "
                          "cluster detection -- see its module docstring / TRIED.md for the "
                          "calibration sweep behind this default.")
    ap.add_argument("--cluster_resolve", default="greedy", choices=["greedy", "hungarian"],
                     help="greedy (default, calibrated in TRIED.md) resolves conflicts by "
                          "confidence rank, one (row, class) pair at a time. hungarian (ideas "
                          "list item 30) solves the whole cluster's assignment JOINTLY/optimally "
                          "-- TRIED.md's oracle test (true prompt_id groups) found this scores "
                          "higher than greedy (0.4453 vs 0.3905), but that was only tested with "
                          "ground-truth groups, never with DETECTED clusters end to end -- try "
                          "both and compare before trusting either on the real submission.")
    args = ap.parse_args()

    artifacts_dir = find_run_dir(args.run_dir)
    labels_sorted = json.load(open(os.path.join(artifacts_dir, "labels.json")))
    k = len(labels_sorted)

    test = pd.read_csv(data_dir() / "test.csv")
    test["text"] = test["text"].fillna("").astype(str)
    texts = test["text"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_dir = ensure_dir(os.path.join(artifacts_dir, "tokenizer"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)

    avg_proba = np.zeros((len(test), k), dtype=np.float32)
    n_used = 0
    for fold in range(args.n_folds):
        fold_path = os.path.join(artifacts_dir, f"fold{fold}_ckpt")
        if not (os.path.isdir(fold_path) or os.path.isfile(fold_path + ".tar")):
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

    if args.cluster_assign:
        print(f"detecting prompt clusters (threshold={args.cluster_threshold}) and resolving "
              f"within-cluster conflicts by rank...")
        cluster_labels = detect_clusters(texts, threshold=args.cluster_threshold)
        n_multi = sum(1 for c in np.unique(cluster_labels) if (cluster_labels == c).sum() > 1)
        print(f"  {n_multi} multi-row clusters detected out of {len(texts)} rows "
              f"(resolve={args.cluster_resolve})")
        resolver = (resolve_cluster_assignments_hungarian if args.cluster_resolve == "hungarian"
                    else resolve_cluster_assignments)
        pred_idx = resolver(avg_proba, cluster_labels, max_cluster_size=k)
    else:
        pred_idx = avg_proba.argmax(axis=1)
    pred_labels = [labels_sorted[i] for i in pred_idx]

    sub = pd.DataFrame({"id": test["id"], "model": pred_labels})
    assert sub["id"].is_unique and sub["model"].isin(labels_sorted).all() and len(sub) == len(test)
    print(sub.head())
    sub.to_parquet(args.out, index=False)
    print(f"wrote {args.out} ({len(sub)} rows, ensembled over {n_used} fold(s))")
    print(sub["model"].value_counts().to_string())


if __name__ == "__main__":
    main()
