"""Scores external data (default: ntl_open_pool.parquet) with our best model (a fusion_lgbm
run's 5-fold ensemble) and flags rows where the model CONFIDENTLY disagrees with the row's
claimed label -- likely mislabeling, not just a hard example.

Deliberately NOT "any disagreement" / "low confidence in the claimed label" -- that would also
catch genuinely hard-but-correctly-labeled boundary cases (e.g. a real gpt-5.6-sol passage that
looks like gpt-5.6-luna), which are exactly the examples most useful for fixing the known
sibling-model confusions, not noise to discard. Confident disagreement specifically requires the
model to be CONFIDENT in a DIFFERENT class, not merely uncertain about the claimed one -- the
"confident learning" (cleanlab-style) criterion for label-noise detection.

Usage:
    python src/filter_external_data.py --run_dir experiments/<fusion_lgbm run> --cap_per_class 500
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from paths import data_dir, external_pool_path
from predict import ensure_dir, find_run_dir, load_fold, predict_fold


def load_pool_quality_filtered(cap_per_class=None):
    """Same flag-based cleaning as train_transformer.py's load_external_pool, but WITHOUT its
    per-class cap by default -- here we want a broad candidate set to apply confidence-based
    selection to, not n_sources-based selection. cap_per_class (if given) limits how many rows
    get SCORED per class, for a quicker pilot run, not how many survive filtering."""
    pool = pd.read_parquet(external_pool_path())
    pool = pool[~pool["flag_near_train_prompt"] & ~pool["flag_train_prompt_regen"]
                & ~pool["flag_gemini37_routing_risk"]].copy()
    if cap_per_class is not None:
        pool = pool.sort_values("n_sources", ascending=False)
        pool = pool.groupby("model", group_keys=False).head(cap_per_class)
    return pool.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="experiments/<run_id> from train_fusion_lgbm.py")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--cap_per_class", type=int, default=None,
                     help="limit rows SCORED per class (for a quick pilot); None = score everything "
                          "that survives the existing quality flags (slow at full ~120k+ scale)")
    ap.add_argument("--mismatch_gap", type=float, default=0.3,
                     help="flag a row if (model's top other-class prob) - (model's prob for the "
                          "claimed label) exceeds this")
    ap.add_argument("--min_other_conf", type=float, default=0.4,
                     help="...AND the model's top other-class prob is at least this confident "
                          "(not just 'slightly more than an already-low claimed-label prob')")
    ap.add_argument("--out", default=None,
                     help="where to save the scored+flagged parquet; defaults next to the run's artifacts")
    args = ap.parse_args()

    artifacts_dir = find_run_dir(args.run_dir)
    labels_sorted = json.load(open(os.path.join(artifacts_dir, "labels.json")))
    label2id = {l: i for i, l in enumerate(labels_sorted)}
    k = len(labels_sorted)

    pool = load_pool_quality_filtered(args.cap_per_class)
    pool = pool[pool["model"].isin(label2id)].reset_index(drop=True)
    print(f"scoring {len(pool)} external rows (after quality-flag filtering"
          + (f", capped at {args.cap_per_class}/class" if args.cap_per_class else "") + ")")
    texts = pool["text"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_dir = ensure_dir(os.path.join(artifacts_dir, "tokenizer"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)

    avg_proba = np.zeros((len(pool), k), dtype=np.float32)
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
        print(f"fold {fold}: scored {len(pool)} rows")
        del model
        torch.cuda.empty_cache()
    assert n_used > 0, f"no fold checkpoints found under {artifacts_dir}"
    avg_proba /= n_used

    claimed_idx = pool["model"].map(label2id).values
    claimed_prob = avg_proba[np.arange(len(pool)), claimed_idx]
    other_proba = avg_proba.copy()
    other_proba[np.arange(len(pool)), claimed_idx] = -1  # exclude the claimed class from the "other" max
    max_other_prob = other_proba.max(axis=1)
    max_other_label = [labels_sorted[i] for i in other_proba.argmax(axis=1)]
    gap = max_other_prob - claimed_prob

    confident_mismatch = (gap > args.mismatch_gap) & (max_other_prob > args.min_other_conf)

    pool["claimed_prob"] = claimed_prob
    pool["top_other_label"] = max_other_label
    pool["top_other_prob"] = max_other_prob
    pool["confidence_gap"] = gap
    pool["confident_mismatch"] = confident_mismatch

    out_path = args.out or os.path.join(args.run_dir, "ntl_open_pool_scored.parquet")
    pool.to_parquet(out_path, index=False)

    print(f"\nflagged {confident_mismatch.sum()}/{len(pool)} rows "
          f"({confident_mismatch.mean():.1%}) as confident mismatches "
          f"(gap > {args.mismatch_gap}, top-other-prob > {args.min_other_conf})")
    print("\nflagged rows by claimed model:")
    print(pool.loc[confident_mismatch, "model"].value_counts().to_string())
    print("\nflagged rows: claimed -> top predicted other class (most common pairs):")
    pairs = pool.loc[confident_mismatch].groupby(["model", "top_other_label"]).size().sort_values(ascending=False)
    print(pairs.head(15).to_string())
    print(f"\nsaved scored data -> {out_path}")


if __name__ == "__main__":
    main()
