"""Detect likely-mislabeled rows in an external data pool using an independent oracle: a
classifier trained on the OFFICIAL train.csv embedding (ground truth, never user-claimed). For
each external row, this oracle gives a predicted probability for the row's CLAIMED label --
rows where the oracle strongly disagrees with the claim are likely mislabeled (or at least
atypical for that label), independent of the dataset's own `style_score` column, which measures
genre match to the competition's prompts, not label correctness (the dataset's own docs:
"nothing re-verifies which model produced a passage").

Caveat: the oracle itself is imperfect (full 5-fold CV balanced_accuracy ~0.40 for this
embedding), so this is a noisy filter, not ground truth -- it should raise average label
reliability, not guarantee it. Expect it to also flag some correctly-labeled-but-atypical rows.

Usage (Kaggle):
    python src/clean_external_data.py \
        --external_paths /kaggle/input/name-that-llm-open-pool/ntl_open_pool.parquet \
        --cap_per_class 3000 --keep_frac 0.6 \
        --output_path ntl_open_pool_v13_cleaned.parquet
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exp_log import ExperimentRun
from experiment_decoder_embedding import extract_decoder_embeddings
from experiment_multi_embed_concat import load_external_data
from paths import data_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external_paths", required=True)
    ap.add_argument("--cap_per_class", type=int, default=3000,
                     help="rows sampled per class from the external pool before scoring -- "
                          "generous, since --keep_frac filters it down afterward")
    ap.add_argument("--oracle_model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--oracle_layer", type=int, default=7,
                     help="the empirically-best frozen layer for this model (TRIED.md)")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--keep_frac", type=float, default=0.6,
                     help="keep the top this fraction of each class's rows, ranked by the "
                          "oracle's predicted probability for the claimed label")
    ap.add_argument("--output_path", required=True,
                     help="where to save the cleaned parquet")
    args = ap.parse_args()

    run = ExperimentRun("clean_external_data", config=vars(args))

    official_df = pd.read_csv(data_dir() / "train.csv")
    class_labels = sorted(official_df["model"].unique())
    label2id = {l: i for i, l in enumerate(class_labels)}

    run.log(f"extracting oracle embeddings for {len(official_df)} official rows...")
    official_emb = extract_decoder_embeddings(
        args.oracle_model, official_df["text"].tolist(), max_length=args.max_length,
        pooling="mean", layers=(args.oracle_layer,), architecture="causal",
        batch_size=args.batch_size, log=run.log,
    )[args.oracle_layer]
    official_labels = official_df["model"].map(label2id).values

    # Fit on ALL official rows, no held-out split -- this is a labeling oracle, not a model
    # being evaluated, so there's no leakage concern in using every official row to train it.
    oracle = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1))
    oracle.fit(official_emb, official_labels)
    run.log(f"oracle trained on {len(official_df)} official rows, {len(class_labels)} classes")

    ext_df = load_external_data(args.external_paths, args.cap_per_class, label2id, run.log)
    ext_df["label"] = ext_df["model"].map(label2id)

    run.log(f"extracting oracle embeddings for {len(ext_df)} external rows...")
    ext_emb = extract_decoder_embeddings(
        args.oracle_model, ext_df["text"].tolist(), max_length=args.max_length,
        pooling="mean", layers=(args.oracle_layer,), architecture="causal",
        batch_size=args.batch_size, log=run.log,
    )[args.oracle_layer]

    proba = oracle.predict_proba(ext_emb)
    claimed_idx = ext_df["label"].values
    ext_df["oracle_claimed_label_proba"] = proba[np.arange(len(ext_df)), claimed_idx]
    ext_df["oracle_predicted_label"] = oracle.classes_[proba.argmax(axis=1)]
    ext_df["oracle_agrees"] = ext_df["oracle_predicted_label"] == claimed_idx

    overall_agree_rate = ext_df["oracle_agrees"].mean()
    run.log(f"oracle agrees with claimed label on {overall_agree_rate:.1%} of external rows overall")
    per_class = ext_df.groupby("model").agg(
        n=("label", "size"),
        agree_rate=("oracle_agrees", "mean"),
        mean_claimed_proba=("oracle_claimed_label_proba", "mean"),
    ).sort_values("agree_rate")
    run.log(f"per-class oracle agreement (worst first):\n{per_class}")

    cleaned = (
        ext_df.groupby("model", group_keys=False)
        .apply(lambda g: g.nlargest(int(len(g) * args.keep_frac), "oracle_claimed_label_proba"))
        .reset_index(drop=True)
    )
    run.log(f"kept {len(cleaned)}/{len(ext_df)} rows ({args.keep_frac:.0%} per class, by "
            f"oracle-claimed-label confidence)")
    cleaned.to_parquet(args.output_path, index=False)
    run.log(f"saved cleaned pool -> {args.output_path}")


if __name__ == "__main__":
    main()
