"""Inference-phase half of the frozen-embedding ensemble pipeline -- loads the artifact bundle
train_qwen_ensemble_artifacts.py saved, extracts embeddings for TEST ONLY (fast -- the expensive
part, fitting on train, already happened), and writes a submission. Never touches train.csv.

Pseudo-labeling (if you want to add it later): after this script's blend step, high-confidence
test predictions could be concatenated with train and the artifact-training script rerun on the
combined set. Not implemented here -- the current artifact bundle doesn't carry raw train
embeddings unless --save_train_embeddings was passed when creating it, and even with those,
appending pseudo-labeled test rows and refitting is a new step this script doesn't do yet.

Usage (Kaggle):
    python src/predict_qwen_ensemble_artifacts.py \
        --artifacts_dir /kaggle/input/datasets/ptran1203/name-that-llm-qwen-ensemble-artifacts \
        --out submission.parquet
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ensemble_utils import apply_temperature
from exp_log import ExperimentRun
from experiment_multi_embed_concat import load_or_extract
from paths import data_dir
from prompt_cluster_assign import detect_clusters, resolve_cluster_assignments_hungarian
from train import fit_transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", required=True)
    ap.add_argument("--cache_dir", default="data/embed_cache")
    ap.add_argument("--cluster_threshold", type=float, default=None,
                     help="override the threshold saved in config.json, if you want to try a "
                          "different one without retraining anything.")
    ap.add_argument("--out", default="submission.parquet")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    config = json.loads((artifacts_dir / "config.json").read_text())
    class_labels = json.loads((artifacts_dir / "class_labels.json").read_text())
    temperatures = json.loads((artifacts_dir / "temperatures.json").read_text())
    weights = json.loads((artifacts_dir / "blend_weights.json").read_text())
    cluster_threshold = args.cluster_threshold if args.cluster_threshold is not None \
        else config["cluster_threshold"]

    run = ExperimentRun("predict_qwen_ensemble_artifacts", config=vars(args) | {"loaded_config": config})
    run.log(f"loaded artifact bundle from {artifacts_dir}: {config}")

    test_df = pd.read_csv(data_dir() / "test.csv")
    label2id = {l: i for i, l in enumerate(class_labels)}

    # -- test-only embedding extraction (the one GPU-heavy step left) --
    run.log("extracting per-model embeddings (test only)...")
    parts = []
    for model_name, layer, use_4bit in zip(config["model_names"], config["layers"], config["needs_4bit"]):
        emb = load_or_extract(test_df["text"].tolist(), "submission_test", model_name, "causal",
                               "mean", layer, config["max_length"], args.cache_dir, recompute=False,
                               log=run.log, load_in_4bit=use_4bit, batch_size=config["batch_size"])
        parts.append(emb)
    test_emb = np.concatenate(parts, axis=1)
    run.log(f"concatenated test embedding shape: {test_emb.shape}")

    # -- load fitted classifiers, predict --
    test_proba_sources = {}
    for clf_name in config["classifiers"]:
        clf = joblib.load(artifacts_dir / f"embedding_clf_{clf_name}.joblib")
        proba = np.zeros((len(test_df), len(class_labels)))
        proba[:, [label2id[c] for c in clf.classes_]] = clf.predict_proba(test_emb)
        test_proba_sources[f"embedding_{clf_name}"] = proba

    word_vec = joblib.load(artifacts_dir / "tfidf_word_vec.joblib")
    char_vec = joblib.load(artifacts_dir / "tfidf_char_vec.joblib")
    tfidf_clf = joblib.load(artifacts_dir / "tfidf_clf.joblib")
    X_test = fit_transform(test_df["text"], word_vec, char_vec, fit=False)
    tfidf_proba = np.zeros((len(test_df), len(class_labels)))
    tfidf_proba[:, [label2id[c] for c in tfidf_clf.classes_]] = tfidf_clf.predict_proba(X_test)
    test_proba_sources["tfidf_lr"] = tfidf_proba

    # -- apply the SAME temperatures + blend weights fit at training time --
    test_calibrated = {n: apply_temperature(test_proba_sources[n], temperatures[n])
                        for n in test_proba_sources}
    test_blend = sum(weights[n] * test_calibrated[n] for n in test_calibrated)

    # -- Hungarian cluster assignment --
    detected_test = detect_clusters(test_df["text"].tolist(), threshold=cluster_threshold)
    n_multi = sum(1 for c in np.unique(detected_test) if (detected_test == c).sum() > 1)
    run.log(f"test: {n_multi} multi-row clusters detected out of {len(test_df)} rows "
            f"(threshold={cluster_threshold})")
    final_pred = np.array(class_labels)[resolve_cluster_assignments_hungarian(test_blend, detected_test)]

    sub = pd.DataFrame({"id": test_df["id"], "model": final_pred})
    sub.to_parquet(args.out, index=False)
    run.log(f"wrote {args.out} ({len(sub)} rows)")
    run.log(sub["model"].value_counts().to_string())


if __name__ == "__main__":
    main()
