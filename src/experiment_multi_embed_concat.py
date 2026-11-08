"""Concatenate frozen embeddings from several DIVERSE models (different architecture/pretraining
objective, and for decoders, a middle layer rather than the last) into one feature vector, then
fit a plain classifier. Unlike the earlier failed ensemble (averaging 3 TF-IDF tokenizer
variants' OOF probabilities, which gained nothing because all 3 shared the same char n-gram
block and classifier), these embeddings are genuinely decorrelated: a decoder's early-middle
layer, an encoder's last layer, and a second, architecturally different decoder's middle layer
each capture a different slice of signal (see TRIED.md's layer-sweep entry for why those specific
layers were picked over each model's last layer).

Reuses cached embeddings from experiment_decoder_embedding.py (data/embed_cache/) -- run that
script first for each (model, architecture, layer) combo this script references, or pass
--recompute to extract fresh.

Two optional extra ingredients, both reusing the SAME fusion recipe already validated in
train_fusion_lgbm.py:
- External data (--external_paths / --external_cap_per_class): train-fold-only, down-weighted
  via sample_weight -- same discipline as train_transformer.py. Capped by default since
  extracting frozen embeddings from 3 sizeable models over the full ~17k-row external pool is
  expensive; the cap controls that cost, not a correctness concern.
- SVD-compressed custom-tokenizer TF-IDF (--tfidf_tokenizer): fit PER FOLD (unlike the frozen
  embeddings, which don't depend on fold assignment) on official-train-fold + external text,
  SVD-compressed to args.svd_dim before concatenating -- raw/sparse TF-IDF concatenated directly
  with dense embeddings was the thing that failed elsewhere this project (IDEAS.md #1); SVD-
  compressing first is the documented reason the original fusion recipe worked instead.
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from exp_log import ExperimentRun
from experiment_decoder_embedding import extract_decoder_embeddings
from paths import data_dir, external_data_dir
from train import build_tokenizer_from_spec, build_vectorizers, fit_transform, tokenizer_needs_refit

# (model_name, architecture, pooling, layer) -- the best-scoring layer found per model in
# TRIED.md's sweep, not each model's last layer.
DEFAULT_SOURCES = [
    ("Qwen/Qwen2.5-3B", "causal", "mean", 7),
    ("microsoft/deberta-v3-base", "encoder", "mean", -1),
    ("microsoft/Phi-3.5-mini-instruct", "causal", "mean", 16),
]


def cache_path_for(cache_dir, tag, model_name, architecture, max_length, pooling, layer,
                    load_in_4bit=False):
    """tag distinguishes official-train embeddings (tag='official', matching
    experiment_decoder_embedding.py's own cache key exactly, so official rows reuse that
    script's cache) from external-data embeddings (tag='external_<n_rows>', a different row
    set -> must be its own cache entry, never collide with the official one). load_in_4bit adds
    a suffix (matching experiment_decoder_embedding.py's own convention) so a 4-bit extraction
    never collides with -- and gets wrongly reused as -- a full-precision one of the same model."""
    suffix = "_4bit" if load_in_4bit else ""
    if tag == "official":
        key = hashlib.md5(f"{model_name}_{architecture}_{max_length}_{pooling}_{layer}{suffix}".encode()).hexdigest()[:10]
    else:
        key = hashlib.md5(f"{tag}_{model_name}_{architecture}_{max_length}_{pooling}_{layer}{suffix}".encode()).hexdigest()[:10]
    return Path(cache_dir) / f"{key}.npy"


def load_or_extract(texts, tag, model_name, architecture, pooling, layer, max_length, cache_dir,
                     recompute, log, load_in_4bit=False, batch_size=16):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path_for(cache_dir, tag, model_name, architecture, max_length, pooling, layer,
                           load_in_4bit)
    if path.exists() and not recompute:
        log(f"loading cached [{tag}] {model_name} L{layer} ({pooling}) -> {path.name}")
        return np.load(path)
    log(f"extracting [{tag}] {model_name} L{layer} ({pooling}) for {len(texts)} rows "
        f"(4bit={load_in_4bit}, batch_size={batch_size})...")
    emb = extract_decoder_embeddings(
        model_name, texts, max_length=max_length, pooling=pooling,
        layers=(layer,), architecture=architecture, log=log, load_in_4bit=load_in_4bit,
        batch_size=batch_size,
    )[layer]
    np.save(path, emb)
    return emb


def load_external_data(paths, cap_per_class, label2id, log):
    def resolve_path(p):
        p = p.strip()
        if os.path.exists(p):
            return p
        candidate = external_data_dir() / os.path.basename(p)
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"external data file not found: {p!r} (also tried {candidate})")

    frames = []
    for p in paths.split(","):
        resolved = resolve_path(p)
        edf = pd.read_parquet(resolved)
        before = len(edf)
        edf = edf[edf["model"].isin(label2id)].copy()
        if len(edf) < before:
            log(f"{p}: dropped {before - len(edf)} rows with unknown model labels")
        cols = ["text", "model"] + (["match_weight"] if "match_weight" in edf.columns else [])
        frames.append(edf[cols])
    ext_df = pd.concat(frames, ignore_index=True)
    if cap_per_class is not None:
        # name-that-llm-open-pool's own ablation: sampling weighted by match_weight (matches
        # each row's typography/length to real train passages) gave +0.026 balanced accuracy
        # alone vs uniform sampling -- use it when the source provides it.
        def sample_group(g):
            n = min(len(g), cap_per_class)
            if "match_weight" in g.columns and g["match_weight"].notna().any():
                weights = g["match_weight"].fillna(1.0).clip(lower=1e-6)
                return g.sample(n=n, weights=weights, random_state=0)
            return g.sample(n=n, random_state=0)

        ext_df = (
            ext_df.groupby("model", group_keys=False).apply(sample_group).reset_index(drop=True)
        )
    log(f"external data: {len(ext_df)} rows after cap={cap_per_class}/class\n{ext_df['model'].value_counts()}")
    return ext_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=None,
                     help="stop after this many folds, for a quick single-fold check")
    ap.add_argument("--cache_dir", default="data/embed_cache")
    ap.add_argument("--recompute", action="store_true", help="ignore cache, re-extract embeddings")
    ap.add_argument("--external_paths", default=None,
                     help="comma-separated external parquet paths (same contract as "
                          "train_transformer.py's --external_paths)")
    ap.add_argument("--external_cap_per_class", type=int, default=300,
                     help="cap external rows per class before extracting embeddings for them -- "
                          "controls extraction cost (3 sizeable models), not a leakage concern")
    ap.add_argument("--external_weight", type=float, default=0.25)
    ap.add_argument("--tfidf_tokenizer", default=None,
                     help="subword tokenizer spec for an SVD-compressed TF-IDF block, via "
                          "build_tokenizer_from_spec() -- e.g. microsoft/deberta-v3-base "
                          "(already confirmed +0.03 balanced accuracy as a standalone TF-IDF "
                          "feature). None (default) = no TF-IDF block added.")
    ap.add_argument("--svd_dim", type=int, default=150)
    ap.add_argument("--sources", default=None,
                     help="override DEFAULT_SOURCES -- semicolon-separated "
                          "'model_name:architecture:pooling:layer' entries, e.g. "
                          "'Qwen/Qwen2.5-3B:causal:mean:7' for a single-model fast check "
                          "instead of the 3-model ensemble. None (default) = DEFAULT_SOURCES.")
    args = ap.parse_args()

    if args.sources:
        sources = []
        for entry in args.sources.split(";"):
            model_name, architecture, pooling, layer = entry.split(":")
            sources.append((model_name, architecture, pooling, int(layer)))
    else:
        sources = DEFAULT_SOURCES

    run = ExperimentRun("multi_embed_concat", config={"sources": sources, **vars(args)})
    df = pd.read_csv(data_dir() / "train.csv")
    class_labels = sorted(df["model"].unique())
    label2id = {l: i for i, l in enumerate(class_labels)}

    embeddings = []
    for model_name, architecture, pooling, layer in sources:
        emb = load_or_extract(df["text"].tolist(), "official", model_name, architecture, pooling,
                               layer, args.max_length, args.cache_dir, args.recompute, run.log)
        run.log(f"  official {model_name} L{layer}: shape {emb.shape}")
        embeddings.append(emb)
    emb_concat = np.concatenate(embeddings, axis=1).astype(np.float32)
    run.log(f"official concatenated embedding shape: {emb_concat.shape}")

    ext_df = None
    ext_emb_concat = None
    if args.external_paths:
        ext_df = load_external_data(args.external_paths, args.external_cap_per_class, label2id, run.log)
        ext_tag = f"external_{len(ext_df)}"
        ext_embeddings = []
        for model_name, architecture, pooling, layer in sources:
            emb = load_or_extract(ext_df["text"].tolist(), ext_tag, model_name, architecture,
                                   pooling, layer, args.max_length, args.cache_dir,
                                   args.recompute, run.log)
            run.log(f"  external {model_name} L{layer}: shape {emb.shape}")
            ext_embeddings.append(emb)
        ext_emb_concat = np.concatenate(ext_embeddings, axis=1).astype(np.float32)

    refit_per_fold = tokenizer_needs_refit(args.tfidf_tokenizer)
    shared_tokenizer = None if refit_per_fold else build_tokenizer_from_spec(args.tfidf_tokenizer)
    if shared_tokenizer is not None:
        shared_tokenizer.fit(df["text"])

    gkf = GroupKFold(n_splits=args.n_splits)
    oof_proba = np.full((len(df), len(class_labels)), np.nan, dtype=np.float32)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["model"], groups=df["prompt_id"])):
        if args.max_folds is not None and fold >= args.max_folds:
            break
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        # Row-stack (axis=0): official-train-fold rows + external rows -- same embedding
        # feature set, just more samples. External rows down-weighted, never in val.
        if ext_df is not None:
            X_train_emb = np.concatenate([emb_concat[train_idx], ext_emb_concat], axis=0)
            y_train = np.concatenate([train_df["model"].values, ext_df["model"].values])
            sw = np.concatenate([
                np.ones(len(train_idx), dtype=np.float32),
                np.full(len(ext_df), args.external_weight, dtype=np.float32),
            ])
        else:
            X_train_emb = emb_concat[train_idx]
            y_train = train_df["model"].values
            sw = np.ones(len(train_idx), dtype=np.float32)
        X_val_emb = emb_concat[val_idx]

        # Feature-concatenate (axis=1): embedding block + TF-IDF/SVD block, for the SAME rows.
        feature_blocks_train = [X_train_emb]
        feature_blocks_val = [X_val_emb]

        if args.tfidf_tokenizer:
            tokenizer = (build_tokenizer_from_spec(args.tfidf_tokenizer).fit(train_df["text"])
                         if refit_per_fold else shared_tokenizer)
            word_vec, char_vec, _p, _f = build_vectorizers(tokenizer)
            # Row order must match X_train_emb exactly: official-train-fold rows, then external.
            tfidf_fit_texts = (pd.concat([train_df["text"], ext_df["text"]], ignore_index=True)
                                if ext_df is not None else train_df["text"])
            X_tfidf_fit = fit_transform(tfidf_fit_texts, word_vec, char_vec, fit=True)
            X_tfidf_val = fit_transform(val_df["text"], word_vec, char_vec, fit=False)
            svd = TruncatedSVD(n_components=args.svd_dim, random_state=fold)
            svd_fit = svd.fit_transform(X_tfidf_fit).astype(np.float32)
            svd_val = svd.transform(X_tfidf_val).astype(np.float32)
            feature_blocks_train.append(svd_fit)
            feature_blocks_val.append(svd_val)

        X_train = np.concatenate(feature_blocks_train, axis=1)
        X_val = np.concatenate(feature_blocks_val, axis=1)

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1),
        )
        clf.fit(X_train, y_train, logisticregression__sample_weight=sw)

        fold_proba = np.zeros((len(val_idx), len(class_labels)), dtype=np.float32)
        fold_proba[:, [label2id[c] for c in clf.classes_]] = clf.predict_proba(X_val)
        oof_proba[val_idx] = fold_proba

        pred = [class_labels[i] for i in np.argmax(fold_proba, axis=-1)]
        score = balanced_accuracy_score(val_df["model"], pred)
        scores.append(score)
        run.log(f"fold {fold}: balanced_accuracy={score:.4f} (X_train={X_train.shape}, X_val={X_val.shape})")

    run.log_scores("balanced_accuracy", scores)
    np.save(run.artifact_path("oof_proba.npy"), oof_proba)
    run.log("saved OOF predictions -> oof_proba.npy")


if __name__ == "__main__":
    main()
