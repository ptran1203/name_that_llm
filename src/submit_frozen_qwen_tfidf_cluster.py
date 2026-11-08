"""Builds a submission from the best-validated pipeline this project has found:

  frozen Qwen2.5-3B layer-7 embedding + LR  \
                                              -- temperature-calibrate each --> weighted blend
  combined word+char TF-IDF + LR            /

  --> Hungarian cluster-assignment (TF-IDF-similarity-detected prompt clusters) on the blend

Full-CV validated balanced accuracy on train (nested-CV blend, oracle-free/realistic cluster
detection): 0.5309 -- see experiment history below for each step's isolated contribution:
  frozen Qwen L7 alone:                     0.3978
  + temperature calibration, weighted blend: 0.4019 (nested CV, trustworthy)
  + Hungarian cluster assignment:            0.5309

Why temperature-calibrate before blending: frozen-Qwen's OOF probabilities were drastically
overconfident (mean max-proba 0.83) vs TF-IDF's underconfident ones (0.17) -- blending raw,
uncalibrated probabilities let the weaker, differently-scaled model dominate the fitted weights
for no good reason (confirmed: calibrated blend+cluster = 0.531 vs uncalibrated = 0.489).

Usage (Kaggle, needs GPU for the one embedding extraction step):
    python src/submit_frozen_qwen_tfidf_cluster.py --out submission.parquet
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD  # noqa: F401 (kept for parity with other scripts)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ensemble_utils import fit_temperature, fit_weighted_average
from exp_log import ExperimentRun
from experiment_decoder_embedding import extract_decoder_embeddings
from paths import data_dir
from prompt_cluster_assign import detect_clusters, resolve_cluster_assignments_hungarian
from train import build_vectorizers, fit_transform


def make_embedding_clf():
    # dense embedding, unscaled raw magnitudes -- needs StandardScaler to converge
    # (experiment_decoder_embedding.py's own finding).
    return make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1))


def make_tfidf_clf():
    # sparse TF-IDF -- StandardScaler can't center a sparse matrix (and train.py's own
    # established baseline never scales TF-IDF anyway, it's already a sane numeric range).
    return LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1)


def get_qwen_embedding(texts, model_name, layer, max_length, batch_size, cache_dir, log,
                        cache_suffix=""):
    """cache_suffix="" matches experiment_decoder_embedding.py's own cache key EXACTLY (it
    always embeds the full train.csv with no suffix) -- pass "" for train so this hits that
    pre-existing cache, and something else (e.g. "_test") for any other text set, so the two
    never collide and train's embedding is only ever computed/cached once."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{model_name}_causal_{max_length}_mean_{layer}{cache_suffix}".encode()).hexdigest()[:10]
    cache_path = cache_dir / f"{key}.npy"
    if cache_path.exists():
        log(f"  loading cached embedding -> {cache_path}")
        return np.load(cache_path)
    log(f"  extracting embeddings from {model_name} layer {layer} for {len(texts)} rows "
        f"(needs GPU)...")
    emb = extract_decoder_embeddings(
        model_name, texts, max_length=max_length, pooling="mean", layers=(layer,),
        architecture="causal", batch_size=batch_size, log=log,
    )[layer]
    np.save(cache_path, emb)
    return emb


def get_train_qwen_oof(train_df, emb, log):
    """Train's OOF frozen-embedding probabilities, needed to fit the temperature/blend weights.
    Takes the already-extracted train embedding (so it's never computed twice)."""
    class_labels = sorted(train_df["model"].unique())
    label2id = {l: i for i, l in enumerate(class_labels)}
    oof_proba = np.full((len(train_df), len(class_labels)), np.nan, dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(train_df, train_df["model"], groups=train_df["prompt_id"]):
        clf = make_embedding_clf()
        clf.fit(emb[train_idx], train_df["model"].values[train_idx])
        fold_proba = np.zeros((len(val_idx), len(class_labels)), dtype=np.float32)
        fold_proba[:, [label2id[c] for c in clf.classes_]] = clf.predict_proba(emb[val_idx])
        oof_proba[val_idx] = fold_proba
    return oof_proba, class_labels


def get_train_tfidf_oof(train_df, log):
    class_labels = sorted(train_df["model"].unique())
    label2id = {l: i for i, l in enumerate(class_labels)}
    oof_proba = np.full((len(train_df), len(class_labels)), np.nan, dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(train_df, train_df["model"], groups=train_df["prompt_id"]):
        fold_train, fold_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
        word_vec, char_vec, _, _ = build_vectorizers()
        X_train = fit_transform(fold_train["text"], word_vec, char_vec, fit=True)
        X_val = fit_transform(fold_val["text"], word_vec, char_vec, fit=False)
        clf = make_tfidf_clf()
        clf.fit(X_train, fold_train["model"].values)
        fold_proba = np.zeros((len(val_idx), len(class_labels)), dtype=np.float32)
        fold_proba[:, [label2id[c] for c in clf.classes_]] = clf.predict_proba(X_val)
        oof_proba[val_idx] = fold_proba
    return oof_proba, class_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--cache_dir", default="data/embed_cache")
    ap.add_argument("--cluster_threshold", type=float, default=0.24)
    ap.add_argument("--out", default="submission.parquet")
    args = ap.parse_args()

    run = ExperimentRun(f"submit_frozen_qwen_tfidf_cluster_{args.model_name.split('/')[-1]}",
                         config=vars(args))
    train_df = pd.read_csv(data_dir() / "train.csv")
    test_df = pd.read_csv(data_dir() / "test.csv")
    class_labels = sorted(train_df["model"].unique())

    # -- step 1: OOF probabilities for each base model, to fit calibration + blend weights --
    run.log("extracting train embedding (used for both OOF and the final refit below)...")
    train_emb = get_qwen_embedding(train_df["text"].tolist(), args.model_name, args.layer,
                                    args.max_length, args.batch_size, args.cache_dir, run.log)
    run.log("computing train OOF for frozen Qwen embedding...")
    qwen_oof, _ = get_train_qwen_oof(train_df, train_emb, run.log)
    run.log("computing train OOF for TF-IDF...")
    tfidf_oof, _ = get_train_tfidf_oof(train_df, run.log)

    y_true = train_df["model"].values
    qwen_score = balanced_accuracy_score(y_true, np.array(class_labels)[qwen_oof.argmax(axis=1)])
    tfidf_score = balanced_accuracy_score(y_true, np.array(class_labels)[tfidf_oof.argmax(axis=1)])
    run.log(f"solo OOF: frozen_qwen={qwen_score:.4f} tfidf={tfidf_score:.4f}")

    # -- step 2: temperature-calibrate each, fit blend weights on calibrated OOF --
    run.log("fitting temperature calibration...")
    T_qwen, qwen_oof_cal = fit_temperature(qwen_oof, y_true, class_labels, log=run.log)
    T_tfidf, tfidf_oof_cal = fit_temperature(tfidf_oof, y_true, class_labels, log=run.log)

    run.log("fitting blend weights on calibrated OOF...")
    weights, _ = fit_weighted_average(
        {"qwen": qwen_oof_cal, "tfidf": tfidf_oof_cal}, y_true, class_labels, log=run.log)

    blend_oof = weights["qwen"] * qwen_oof_cal + weights["tfidf"] * tfidf_oof_cal
    blend_score = balanced_accuracy_score(y_true, np.array(class_labels)[blend_oof.argmax(axis=1)])
    detected_train = detect_clusters(train_df["text"].tolist(), threshold=args.cluster_threshold)
    cluster_pred = np.array(class_labels)[
        resolve_cluster_assignments_hungarian(blend_oof, detected_train)]
    cluster_score = balanced_accuracy_score(y_true, cluster_pred)
    run.log(f"train OOF: blend={blend_score:.4f} blend+hungarian={cluster_score:.4f} "
            f"(this run's reproduction of the validated ~0.53 pipeline number)")

    # -- step 3: fit FINAL models on ALL train data, predict test --
    run.log("fitting final models on all train data...")
    qwen_clf = make_embedding_clf()
    qwen_clf.fit(train_emb, train_df["model"].values)
    test_emb = get_qwen_embedding(test_df["text"].tolist(), args.model_name, args.layer,
                                   args.max_length, args.batch_size, args.cache_dir, run.log,
                                   cache_suffix="_test")
    qwen_test_proba = np.zeros((len(test_df), len(class_labels)))
    label2id = {l: i for i, l in enumerate(class_labels)}
    qwen_test_proba[:, [label2id[c] for c in qwen_clf.classes_]] = qwen_clf.predict_proba(test_emb)

    word_vec, char_vec, _, _ = build_vectorizers()
    X_train = fit_transform(train_df["text"], word_vec, char_vec, fit=True)
    X_test = fit_transform(test_df["text"], word_vec, char_vec, fit=False)
    tfidf_clf = make_tfidf_clf()
    tfidf_clf.fit(X_train, train_df["model"].values)
    tfidf_test_proba = np.zeros((len(test_df), len(class_labels)))
    tfidf_test_proba[:, [label2id[c] for c in tfidf_clf.classes_]] = tfidf_clf.predict_proba(X_test)

    # -- step 4: apply the SAME calibration temperatures + blend weights fit on train OOF --
    def calibrate(proba, T):
        logits = np.log(np.clip(proba, 1e-12, None)) / T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    qwen_test_cal = calibrate(qwen_test_proba, T_qwen)
    tfidf_test_cal = calibrate(tfidf_test_proba, T_tfidf)
    test_blend = weights["qwen"] * qwen_test_cal + weights["tfidf"] * tfidf_test_cal

    # -- step 5: Hungarian cluster assignment on the test blend --
    detected_test = detect_clusters(test_df["text"].tolist(), threshold=args.cluster_threshold)
    n_multi = sum(1 for c in np.unique(detected_test) if (detected_test == c).sum() > 1)
    run.log(f"test: {n_multi} multi-row clusters detected out of {len(test_df)} rows")
    final_pred = np.array(class_labels)[
        resolve_cluster_assignments_hungarian(test_blend, detected_test)]

    sub = pd.DataFrame({"id": test_df["id"], "model": final_pred})
    sub.to_parquet(args.out, index=False)
    run.log(f"wrote {args.out} ({len(sub)} rows)")
    run.log(sub["model"].value_counts().to_string())


if __name__ == "__main__":
    main()
