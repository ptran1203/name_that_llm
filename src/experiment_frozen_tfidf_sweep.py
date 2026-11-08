"""Frozen (no fine-tuning) Qwen2.5-3B layer-7 embedding + SVD-reduced TF-IDF, swept over several
SVD dims, full GroupKFold CV -- the fast way to check whether TF-IDF adds anything ON TOP of the
frozen baseline (0.3978 CV, TRIED.md's layer sweep) without spending any GPU time on fine-tuning.
Reuses experiment_decoder_embedding.py's embedding cache (data/embed_cache/) if the frozen
embedding was already extracted there -- this is a near-instant CPU-only check in that case,
just a TF-IDF/SVD fit + a plain classifier per fold per dim.

dim=0 in the sweep reproduces the plain frozen-embedding baseline exactly (same StandardScaler +
LogisticRegression as experiment_decoder_embedding.py's make_clf_factory) -- a built-in sanity
check that the sweep itself isn't silently changing anything for the no-TF-IDF case.

Per-block row-normalization before concatenation (only applied when dim > 0): the frozen
embedding's raw L2 norm (~44, measured) is ~57x the TF-IDF-SVD block's (~0.77) -- StandardScaler
already handles per-FEATURE scale fairly for LogisticRegression/SVC, but this keeps the two
BLOCKS on a comparable footing before that, same fix as train_qwen_lora.py's embedding_readout.

Usage (quick check, 1 fold):
    python src/experiment_frozen_tfidf_sweep.py --tfidf_svd_dim 0,50,150,300 --max_folds 1

Usage (full 5-fold):
    python src/experiment_frozen_tfidf_sweep.py --tfidf_svd_dim 0,50,150,300
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from exp_log import ExperimentRun
from experiment_decoder_embedding import extract_decoder_embeddings
from paths import data_dir
from train import build_vectorizers, fit_transform


def make_clf(name):
    if name == "svc":
        return make_pipeline(StandardScaler(), SVC(class_weight="balanced"))
    return make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1))


def row_normalize(x):
    return (x - x.mean(axis=1, keepdims=True)) / np.clip(x.std(axis=1, keepdims=True), 1e-5, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--layer", type=int, default=7,
                     help="empirically-best frozen layer for this model (TRIED.md)")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--load_in_4bit", action="store_true",
                     help="only matters on an embedding-cache miss -- see "
                          "experiment_decoder_embedding.py")
    ap.add_argument("--classifier", default="lr", choices=["lr", "svc"])
    ap.add_argument("--tfidf_svd_dim", default="0,50,150,300",
                     help="comma-separated SVD dims to sweep, e.g. '0,50,150,300'. 0 = no "
                          "TF-IDF (reproduces the plain frozen-embedding baseline as a sanity "
                          "check on the sweep itself).")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=None,
                     help="stop after this many folds, for a quick single-fold check")
    ap.add_argument("--cache_dir", default="data/embed_cache",
                     help="reuses experiment_decoder_embedding.py's embedding cache if present")
    args = ap.parse_args()

    svd_dims = [int(x) for x in args.tfidf_svd_dim.split(",")]
    run = ExperimentRun(f"frozen_tfidf_sweep_{args.model_name.split('/')[-1]}", config=vars(args))

    df = pd.read_csv(data_dir() / "train.csv")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_4bit" if args.load_in_4bit else ""
    key = hashlib.md5(
        f"{args.model_name}_causal_{args.max_length}_mean_{args.layer}{suffix}".encode()
    ).hexdigest()[:10]
    cache_path = cache_dir / f"{key}.npy"
    if cache_path.exists():
        run.log(f"loading cached frozen embedding -> {cache_path}")
        emb = np.load(cache_path)
    else:
        run.log(f"cache miss -- extracting frozen embeddings from {args.model_name} "
                f"layer {args.layer} (needs GPU)...")
        emb = extract_decoder_embeddings(
            args.model_name, df["text"].tolist(), max_length=args.max_length,
            pooling="mean", layers=(args.layer,), architecture="causal",
            batch_size=args.batch_size, log=run.log, load_in_4bit=args.load_in_4bit,
        )[args.layer]
        np.save(cache_path, emb)
        run.log(f"saved embedding {emb.shape} -> {cache_path}")

    gkf = GroupKFold(n_splits=args.n_splits)
    folds = list(gkf.split(df, df["model"], groups=df["prompt_id"]))
    if args.max_folds is not None:
        folds = folds[:args.max_folds]

    for dim in svd_dims:
        scores = []
        for fold, (train_idx, val_idx) in enumerate(folds):
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
            X_train, X_val = emb[train_idx], emb[val_idx]

            if dim > 0:
                word_vec, char_vec, _, _ = build_vectorizers()
                X_tfidf_fit = fit_transform(train_df["text"], word_vec, char_vec, fit=True)
                X_tfidf_val = fit_transform(val_df["text"], word_vec, char_vec, fit=False)
                svd = TruncatedSVD(n_components=dim, random_state=fold)
                tfidf_train = svd.fit_transform(X_tfidf_fit).astype(np.float32)
                tfidf_val = svd.transform(X_tfidf_val).astype(np.float32)
                X_train = np.concatenate([row_normalize(X_train), row_normalize(tfidf_train)], axis=1)
                X_val = np.concatenate([row_normalize(X_val), row_normalize(tfidf_val)], axis=1)

            clf = make_clf(args.classifier)
            clf.fit(X_train, train_df["model"].values)
            pred = clf.predict(X_val)
            score = balanced_accuracy_score(val_df["model"].values, pred)
            scores.append(score)
            run.log(f"[svd{dim}] fold {fold}: balanced_accuracy={score:.4f}")
        run.log_scores(f"balanced_accuracy_svd{dim}", scores)


if __name__ == "__main__":
    main()
