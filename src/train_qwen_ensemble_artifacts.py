"""Training-phase half of the frozen-embedding ensemble pipeline (split out of
submit_multi_qwen_ensemble.py per the user's 2-notebook Kaggle workflow): does everything that
needs TRAIN data -- embedding extraction, OOF, calibration, blend-weight fitting, and fitting
the FINAL classifiers -- then saves all of it to --artifacts_dir. The matching
predict_qwen_ensemble_artifacts.py loads that bundle and only needs to touch TEST data, so it's
fast and never re-does any of this.

Artifact bundle (under --artifacts_dir):
  config.json            -- model_names/layers/needs_4bit/max_length/batch_size/classifiers/
                             cluster_threshold, so inference extracts test embeddings the
                             SAME way these were extracted
  class_labels.json
  embedding_clf_<name>.joblib   -- one per --classifiers entry, fit on the concatenated embedding
  tfidf_word_vec.joblib, tfidf_char_vec.joblib, tfidf_clf.joblib
  temperatures.json      -- per-source fitted temperature (item 33)
  blend_weights.json     -- per-source fitted blend weight (item 31)
  train_embedding.npy    -- ONLY if --save_train_embeddings (not needed for inference as-is;
                             kept only if you want to later add pseudo-labeling, which would
                             need to re-concatenate pseudo-labeled test rows with real train
                             embeddings before refitting -- not implemented here yet)

Usage (Kaggle):
    python src/train_qwen_ensemble_artifacts.py \
        --model_names /kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1,/kaggle/input/models/qwen-lm/qwen-3/transformers/14b/1 \
        --load_in_4bit_for 14b \
        --artifacts_dir /kaggle/working/qwen_ensemble_artifacts
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from ensemble_utils import fit_temperature, fit_weighted_average, fit_weighted_average_nested_cv
from exp_log import ExperimentRun
from paths import data_dir
from prompt_cluster_assign import detect_clusters, resolve_cluster_assignments_hungarian
from submit_multi_qwen_ensemble import (
    default_layer_for, get_concat_embedding, get_oof, make_embedding_clf, make_tfidf_clf,
    validate_model_names,
)
from train import build_vectorizers, fit_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_names", required=True)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--load_in_4bit_for", default="")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--cache_dir", default="data/embed_cache")
    ap.add_argument("--classifiers", default="lr,svc,lgbm")
    ap.add_argument("--cluster_threshold", type=float, default=0.24)
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--save_train_embeddings", action="store_true",
                     help="also save train_embedding.npy -- not needed for ordinary inference, "
                          "only if you plan to refit later (e.g. pseudo-labeling, not "
                          "implemented yet).")
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

    run = ExperimentRun("train_qwen_ensemble_artifacts", config=vars(args) | {"resolved_layers": layers})
    run.log(f"models={model_names} layers={layers} (default-guessed where not passed explicitly) "
            f"4bit={needs_4bit}")

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir() / "train.csv")
    class_labels = sorted(train_df["model"].unique())
    y_true = train_df["model"].values

    # -- embeddings --
    run.log("extracting/loading per-model embeddings (train)...")
    train_emb = get_concat_embedding(train_df["text"].tolist(), "official", model_names, layers,
                                      args.max_length, args.batch_size, args.cache_dir, run.log,
                                      needs_4bit)
    run.log(f"concatenated train embedding shape: {train_emb.shape}")
    if args.save_train_embeddings:
        np.save(artifacts_dir / "train_embedding.npy", train_emb)

    # -- OOF per embedding-classifier + TF-IDF+LR --
    clf_names = args.classifiers.split(",")
    oof_sources = {}
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

    # -- calibrate, validate the blend (nested CV), fit FINAL blend weights --
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
    run.log(f"train OOF: blend={blend_score:.4f} blend+hungarian={cluster_score:.4f} "
            f"-- expected inference-time quality if test behaves like train")

    # -- fit and save FINAL models on ALL train data --
    run.log("fitting + saving final classifiers on all train data...")
    for clf_name in clf_names:
        clf = make_embedding_clf(clf_name)
        clf.fit(train_emb, y_true)
        joblib.dump(clf, artifacts_dir / f"embedding_clf_{clf_name}.joblib")

    word_vec, char_vec, _, _ = build_vectorizers()
    X_train = fit_transform(train_df["text"], word_vec, char_vec, fit=True)
    tfidf_clf = make_tfidf_clf()
    tfidf_clf.fit(X_train, y_true)
    joblib.dump(word_vec, artifacts_dir / "tfidf_word_vec.joblib")
    joblib.dump(char_vec, artifacts_dir / "tfidf_char_vec.joblib")
    joblib.dump(tfidf_clf, artifacts_dir / "tfidf_clf.joblib")

    with open(artifacts_dir / "config.json", "w") as f:
        json.dump({
            "model_names": model_names, "layers": layers, "needs_4bit": needs_4bit,
            "max_length": args.max_length, "batch_size": args.batch_size,
            "classifiers": clf_names, "cluster_threshold": args.cluster_threshold,
        }, f, indent=2)
    with open(artifacts_dir / "class_labels.json", "w") as f:
        json.dump(class_labels, f)
    with open(artifacts_dir / "temperatures.json", "w") as f:
        json.dump(temperatures, f, indent=2)
    with open(artifacts_dir / "blend_weights.json", "w") as f:
        json.dump(weights, f, indent=2)

    run.log(f"saved artifact bundle -> {artifacts_dir}")


if __name__ == "__main__":
    main()
