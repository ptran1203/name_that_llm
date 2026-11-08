"""Submission pipeline, v2: accepts SEVERAL frozen decoder models (e.g. multiple Qwen3 sizes),
concatenates their embeddings into one feature vector, fits SVC/LR/LightGBM on that combined
embedding (3 classifiers, not just LR), adds TF-IDF+LR as a 4th member, then:

  temperature-calibrate each of the 4 OOF sources -> weighted blend (nested-CV validated)
  -> Hungarian cluster-assignment (TF-IDF-similarity-detected shared-prompt groups)

This replaces submit_frozen_qwen_tfidf_cluster.py's single-model version. The calibrate-before-
blend step matters a lot here too (see that script's docstring/TRIED.md for why): each
classifier family produces very differently-scaled probabilities (SVC with Platt scaling, LR,
LightGBM's leaf-count-driven scores) -- blending them raw would repeat the same confound already
found and fixed once.

Per-model layer choice: TRIED.md's layer sweep found ~19% depth is the best middle-decoder-layer
heuristic for Qwen2.5-3B specifically (L7/36), and noted public top solutions pick similar
relative depths on OTHER models too -- used here as the default for any NEW model via
AutoConfig.num_hidden_layers, but this is an untested-per-model GUESS, not a confirmed-best
layer the way L7 is for Qwen2.5-3B. Override with --layers (comma list, same order/count as
--model_names) once you've actually swept a given model.

Usage (Kaggle):
    python src/submit_multi_qwen_ensemble.py \
        --model_names /kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1,/kaggle/input/models/qwen-lm/qwen-3/transformers/14b/1 \
        --load_in_4bit_for 14b \
        --out submission.parquet
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD  # noqa: F401 (parity with sibling scripts)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import AutoConfig

from ensemble_utils import fit_temperature, fit_weighted_average, fit_weighted_average_nested_cv
from exp_log import ExperimentRun
from experiment_multi_embed_concat import cache_path_for, load_or_extract
from paths import data_dir
from prompt_cluster_assign import detect_clusters, resolve_cluster_assignments_hungarian
from train import build_vectorizers, fit_transform

LAYER_FRACTION_DEFAULT = 0.19  # TRIED.md: Qwen2.5-3B's empirically-best layer was ~19% depth


def validate_model_names(model_names):
    """A local path that doesn't actually exist (e.g. the Kaggle Model wasn't attached as an
    input to THIS notebook) falls through transformers' from_pretrained into treating the
    string as a HuggingFace Hub repo id instead -- which then fails with a confusing
    'Repo id must be in the form...' error that has nothing to do with the real problem. Catch
    it here with a clear message instead."""
    for name in model_names:
        if name.startswith("/") and not Path(name).is_dir():
            raise FileNotFoundError(
                f"Local model path does not exist: {name!r} -- if this is a Kaggle Model, check "
                f"it's attached as an input to THIS notebook (Add Input > Models), and that the "
                f"path matches exactly what's mounted (try `!ls {Path(name).parent}` to see "
                f"what's actually there)."
            )


def default_layer_for(model_name, fraction=LAYER_FRACTION_DEFAULT):
    """UNTESTED-per-model guess -- see module docstring. Sweep with
    experiment_decoder_embedding.py and pass --layers explicitly once you know better."""
    cfg = AutoConfig.from_pretrained(model_name)
    n_layers = cfg.num_hidden_layers
    return max(1, round(fraction * n_layers))


def get_concat_embedding(texts, tag, model_names, layers, max_length, batch_size, cache_dir, log,
                          needs_4bit):
    parts = []
    for model_name, layer, use_4bit in zip(model_names, layers, needs_4bit):
        emb = load_or_extract(texts, tag, model_name, "causal", "mean", layer, max_length,
                               cache_dir, recompute=False, log=log, load_in_4bit=use_4bit,
                               batch_size=batch_size)
        parts.append(emb)
    return np.concatenate(parts, axis=1)


def make_embedding_clf(name):
    if name == "svc":
        # probability=True (Platt scaling) -- needed since the ensembling step blends
        # PROBABILITIES, not just argmax predictions.
        return make_pipeline(StandardScaler(), SVC(class_weight="balanced", probability=True))
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return make_pipeline(StandardScaler(), LGBMClassifier(n_estimators=300, verbosity=-1))
    return make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1))


def make_tfidf_clf():
    # sparse TF-IDF -- no StandardScaler (can't center sparse; train.py's own convention is to
    # not scale TF-IDF at all, it's already a sane numeric range).
    return LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1)


def get_oof(fit_fn, df, class_labels, log, n_splits=5):
    """fit_fn(train_idx, val_idx) -> (val_predict_proba, val_classes_). Shared GroupKFold-by-
    prompt_id loop, used identically for every classifier/feature-set combo below."""
    label2id = {l: i for i, l in enumerate(class_labels)}
    oof_proba = np.full((len(df), len(class_labels)), np.nan, dtype=np.float32)
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(df, df["model"], groups=df["prompt_id"]):
        val_proba, val_classes = fit_fn(train_idx, val_idx)
        fold_proba = np.zeros((len(val_idx), len(class_labels)), dtype=np.float32)
        fold_proba[:, [label2id[c] for c in val_classes]] = val_proba
        oof_proba[val_idx] = fold_proba
    return oof_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_names", required=True,
                     help="comma-separated list of frozen causal-LM model names/paths (HF hub "
                          "id or a local directory, e.g. a Kaggle-attached model dataset). Each "
                          "gets its own cached embedding, concatenated together.")
    ap.add_argument("--layers", default=None,
                     help="comma-separated layer index per model, same order/count as "
                          "--model_names. Omit to use the untested ~19%%-depth default per model "
                          "(see module docstring) -- sweep with experiment_decoder_embedding.py "
                          "first if you want a validated choice instead.")
    ap.add_argument("--load_in_4bit_for", default="",
                     help="comma-separated SUBSTRINGS -- any --model_names entry containing one "
                          "of these loads in 4-bit (QLoRA-style) instead of fp16/bf16, for "
                          "models too big to fit otherwise (e.g. '14b').")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--cache_dir", default="data/embed_cache")
    ap.add_argument("--classifiers", default="lr,svc,lgbm",
                     help="comma-separated classifiers to fit on the CONCATENATED embedding -- "
                          "each becomes its own OOF source for the ensemble below.")
    ap.add_argument("--cluster_threshold", type=float, default=0.24)
    ap.add_argument("--out", default="submission.parquet")
    ap.add_argument("--extract_only", action="store_true",
                     help="extract+cache train and test embeddings for --model_names, then "
                          "exit -- no classifier fitting, no blend, no submission. Lets you "
                          "use 2 GPUs safely: run this twice in parallel, each with a DIFFERENT "
                          "--model_names subset and its own CUDA_VISIBLE_DEVICES, each writing "
                          "to the SAME --cache_dir. Then one final normal (no --extract_only) "
                          "run with the FULL --model_names list hits the cache for all of them "
                          "-- no GPU-heavy work repeated, and no Trainer/DataParallel/DDP "
                          "involved anywhere (this script is plain forward-pass extraction, "
                          "none of TRIED.md's multi-GPU Trainer bugs apply here).")
    args = ap.parse_args()

    model_names = args.model_names.split(",")
    validate_model_names(model_names)
    bit4_substrings = [s for s in args.load_in_4bit_for.split(",") if s]
    needs_4bit = [any(s in m for s in bit4_substrings) for m in model_names]
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
        assert len(layers) == len(model_names), "--layers must match --model_names count"
    else:
        layers = [default_layer_for(m) for m in model_names]

    run = ExperimentRun("submit_multi_qwen_ensemble", config=vars(args) | {"resolved_layers": layers})
    run.log(f"models={model_names} layers={layers} (default-guessed where not passed explicitly) "
            f"4bit={needs_4bit}")

    train_df = pd.read_csv(data_dir() / "train.csv")
    test_df = pd.read_csv(data_dir() / "test.csv")
    class_labels = sorted(train_df["model"].unique())

    if args.extract_only:
        run.log("--extract_only: caching train+test embeddings for this model subset, then exiting")
        get_concat_embedding(train_df["text"].tolist(), "official", model_names, layers,
                              args.max_length, args.batch_size, args.cache_dir, run.log, needs_4bit)
        get_concat_embedding(test_df["text"].tolist(), "submission_test", model_names, layers,
                              args.max_length, args.batch_size, args.cache_dir, run.log, needs_4bit)
        run.log("done -- rerun without --extract_only, with the FULL --model_names list, to "
                "actually fit classifiers and write a submission.")
        return

    # -- step 1: concatenated embedding (train + test), extracted/cached once per model --
    run.log("extracting/loading per-model embeddings (train)...")
    train_emb = get_concat_embedding(train_df["text"].tolist(), "official", model_names, layers,
                                      args.max_length, args.batch_size, args.cache_dir, run.log,
                                      needs_4bit)
    run.log(f"concatenated train embedding shape: {train_emb.shape}")

    # -- step 2: OOF for each embedding-classifier + TF-IDF+LR, to fit calibration/blend --
    clf_names = args.classifiers.split(",")
    oof_sources = {}
    y_true = train_df["model"].values
    for clf_name in clf_names:
        run.log(f"computing train OOF for embedding + {clf_name}...")
        def fit_fn(train_idx, val_idx, clf_name=clf_name):
            clf = make_embedding_clf(clf_name)
            clf.fit(train_emb[train_idx], y_true[train_idx])
            return clf.predict_proba(train_emb[val_idx]), clf.classes_
        oof = get_oof(fit_fn, train_df, class_labels, run.log)
        score = balanced_accuracy_score(y_true, np.array(class_labels)[oof.argmax(axis=1)])
        run.log(f"  embedding+{clf_name} solo OOF balanced_accuracy={score:.4f}")
        oof_sources[f"embedding_{clf_name}"] = oof

    run.log("computing train OOF for TF-IDF+LR...")
    def tfidf_fit_fn(train_idx, val_idx):
        fold_train, fold_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
        word_vec, char_vec, _, _ = build_vectorizers()
        X_train = fit_transform(fold_train["text"], word_vec, char_vec, fit=True)
        X_val = fit_transform(fold_val["text"], word_vec, char_vec, fit=False)
        clf = make_tfidf_clf()
        clf.fit(X_train, fold_train["model"].values)
        return clf.predict_proba(X_val), clf.classes_
    tfidf_oof = get_oof(tfidf_fit_fn, train_df, class_labels, run.log)
    tfidf_score = balanced_accuracy_score(y_true, np.array(class_labels)[tfidf_oof.argmax(axis=1)])
    run.log(f"  tfidf+lr solo OOF balanced_accuracy={tfidf_score:.4f}")
    oof_sources["tfidf_lr"] = tfidf_oof

    # -- step 3: calibrate each source, validate the blend with nested CV, then fit final weights --
    run.log("temperature-calibrating each OOF source...")
    temperatures, calibrated = {}, {}
    for name, oof in oof_sources.items():
        T, cal = fit_temperature(oof, y_true, class_labels, log=run.log)
        temperatures[name], calibrated[name] = T, cal

    run.log("validating the blend with nested CV (trustworthy estimate)...")
    fit_weighted_average_nested_cv(calibrated, y_true, class_labels, train_df["prompt_id"].values,
                                    log=run.log)

    run.log("fitting FINAL blend weights on all calibrated OOF (for shipping)...")
    weights, _ = fit_weighted_average(calibrated, y_true, class_labels, log=run.log)
    blend_oof = sum(weights[n] * calibrated[n] for n in calibrated)
    blend_score = balanced_accuracy_score(y_true, np.array(class_labels)[blend_oof.argmax(axis=1)])
    detected_train = detect_clusters(train_df["text"].tolist(), threshold=args.cluster_threshold)
    cluster_pred = np.array(class_labels)[
        resolve_cluster_assignments_hungarian(blend_oof, detected_train)]
    cluster_score = balanced_accuracy_score(y_true, cluster_pred)
    run.log(f"train OOF: blend={blend_score:.4f} blend+hungarian={cluster_score:.4f}")

    # -- step 4: fit FINAL classifiers on ALL train data, predict test --
    run.log("extracting/loading per-model embeddings (test)...")
    test_emb = get_concat_embedding(test_df["text"].tolist(), "submission_test", model_names,
                                     layers, args.max_length, args.batch_size, args.cache_dir,
                                     run.log, needs_4bit)

    label2id = {l: i for i, l in enumerate(class_labels)}
    test_proba_sources = {}
    for clf_name in clf_names:
        clf = make_embedding_clf(clf_name)
        clf.fit(train_emb, y_true)
        proba = np.zeros((len(test_df), len(class_labels)))
        proba[:, [label2id[c] for c in clf.classes_]] = clf.predict_proba(test_emb)
        test_proba_sources[f"embedding_{clf_name}"] = proba

    word_vec, char_vec, _, _ = build_vectorizers()
    X_train = fit_transform(train_df["text"], word_vec, char_vec, fit=True)
    X_test = fit_transform(test_df["text"], word_vec, char_vec, fit=False)
    tfidf_clf = make_tfidf_clf()
    tfidf_clf.fit(X_train, y_true)
    tfidf_test_proba = np.zeros((len(test_df), len(class_labels)))
    tfidf_test_proba[:, [label2id[c] for c in tfidf_clf.classes_]] = tfidf_clf.predict_proba(X_test)
    test_proba_sources["tfidf_lr"] = tfidf_test_proba

    # -- step 5: apply the SAME per-source temperatures + blend weights fit on train OOF --
    def calibrate(proba, T):
        logits = np.log(np.clip(proba, 1e-12, None)) / T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    test_calibrated = {n: calibrate(test_proba_sources[n], temperatures[n]) for n in test_proba_sources}
    test_blend = sum(weights[n] * test_calibrated[n] for n in test_calibrated)

    # -- step 6: Hungarian cluster assignment on the test blend --
    detected_test = detect_clusters(test_df["text"].tolist(), threshold=args.cluster_threshold)
    n_multi = sum(1 for c in np.unique(detected_test) if (detected_test == c).sum() > 1)
    run.log(f"test: {n_multi} multi-row clusters detected out of {len(test_df)} rows")
    final_pred = np.array(class_labels)[resolve_cluster_assignments_hungarian(test_blend, detected_test)]

    sub = pd.DataFrame({"id": test_df["id"], "model": final_pred})
    sub.to_parquet(args.out, index=False)
    run.log(f"wrote {args.out} ({len(sub)} rows)")
    run.log(sub["model"].value_counts().to_string())


if __name__ == "__main__":
    main()
