"""Cheap, no-training recon checks (user's list, items 1-4) -- run these BEFORE investing in any
new modeling, since a positive hit on any of them could change the whole approach:

1. cluster_size_recon: does blind TF-IDF-similarity clustering (no real prompt_id) recover
   ~18-member groups on train? If so, the same detector should work on test, validating
   prompt_cluster_assign.py's whole premise. Reuses detect_clusters() from that module.
2. duplicate_check: any train/test near-duplicate gets the train row's label for free.
3. id_structure_check: does row order or the `id` column encode anything about the label?
4. class_balance_check: compare train's class counts against a given set of test predictions,
   to see whether the test-time prior differs from train's (exactly-balanced, 150/class).

Note: data/test.csv locally is a 12-row SAMPLE, not the real hidden eval set -- the test-side
checks here are code-correct but will only be informative once run against the real test file
(e.g. on Kaggle). The train-side checks (1, 3) are fully informative now (2699 real rows).
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import normalize

from prompt_cluster_assign import cosine_sim_matrix, detect_clusters


def cluster_size_recon(texts, true_groups=None, thresholds=(0.20, 0.22, 0.24, 0.28), log=print):
    """Runs the blind (no ground truth) cluster detector at several thresholds and reports the
    size distribution. If true_groups (e.g. train's real prompt_id) is given, also reports the
    TRUE group-size distribution for comparison -- the thing we're hoping blind detection
    recovers. On train this validates the detector itself (same spirit as TRIED.md's threshold
    calibration, but reporting the full size histogram rather than just precision/recall at one
    threshold); on test, this is the actual question item #1 asks."""
    if true_groups is not None:
        true_sizes = pd.Series(true_groups).value_counts()
        log(f"TRUE group sizes: n_groups={len(true_sizes)} "
            f"min={true_sizes.min()} max={true_sizes.max()} "
            f"mean={true_sizes.mean():.1f} mode={true_sizes.mode().iloc[0]}")
        log(f"  size distribution: {true_sizes.value_counts().sort_index().to_dict()}")

    for t in thresholds:
        labels = detect_clusters(texts, threshold=t)
        sizes = pd.Series(labels).value_counts()
        non_singleton = sizes[sizes > 1]
        log(f"threshold={t}: {len(sizes)} clusters total, {len(non_singleton)} non-singleton, "
            f"singleton_rows={len(labels) - non_singleton.sum()}/{len(labels)}")
        if len(non_singleton):
            log(f"  non-singleton size distribution: "
                f"{non_singleton.value_counts().sort_index().to_dict()}")
            near_18 = non_singleton[(non_singleton >= 15) & (non_singleton <= 18)]
            log(f"  clusters sized 15-18 (the signal item #1 is actually asking about): "
                f"{len(near_18)}")


def duplicate_check(train_df, test_df, threshold=0.95, log=print):
    """Cross-similarity (TF-IDF cosine, fit on train+test text jointly -- unsupervised, no
    labels involved, so no leakage) between every test row and every train row. Returns a
    DataFrame of (test_id, best_train_idx, similarity, copied_label) for every test row whose
    best match clears `threshold` -- these are candidate exact/near-duplicates where copying the
    train label is likely safe. Rows below threshold are left for the normal classifier."""
    from train import build_vectorizers, fit_transform

    word_vec, char_vec, _punct_vec, _func_vec = build_vectorizers()
    all_text = pd.concat([train_df["text"], test_df["text"]], ignore_index=True)
    X = fit_transform(all_text, word_vec, char_vec, fit=True).tocsr()
    Xn = normalize(X, norm="l2")
    X_train, X_test = Xn[: len(train_df)], Xn[len(train_df):]

    sim = (X_test @ X_train.T).toarray()
    best_idx = sim.argmax(axis=1)
    best_sim = sim.max(axis=1)

    hits = best_sim >= threshold
    out = pd.DataFrame({
        "test_id": test_df["id"].values,
        "best_train_idx": best_idx,
        "similarity": best_sim,
        "copied_label": np.where(hits, train_df["model"].values[best_idx], None),
    })
    log(f"duplicate_check: {hits.sum()}/{len(test_df)} test rows have a train match "
        f">= {threshold} similarity")
    if hits.any():
        log(out[hits].to_string(index=False))
    return out


def id_structure_check(df, id_col="id", label_col=None, log=print):
    """Checks whether row order or the `id` column encodes anything about the label (train) or
    just reports the id column's apparent structure (test, no label to correlate against).
    Purely diagnostic -- a hit here would be a modeling shortcut, a miss just rules it out."""
    ids = df[id_col].astype(str)
    all_hex = ids.str.match(r"^[0-9a-f]+$").all()
    lengths = ids.str.len()
    log(f"id column: n={len(ids)} all-lowercase-hex={all_hex} "
        f"length_unique_values={sorted(lengths.unique().tolist())}")
    if all_hex and lengths.nunique() == 1:
        log("  looks like a random hex/UUID-style id -- unlikely to encode row order or label "
            "(no further correlation check meaningful)")

    if label_col is not None and label_col in df.columns:
        label_codes = pd.Categorical(df[label_col]).codes
        row_pos = np.arange(len(df))
        r, p = pearsonr(row_pos, label_codes)
        log(f"row-position vs label correlation: r={r:.4f} p={p:.4f} "
            f"({'SIGNIFICANT -- investigate further' if p < 0.01 else 'no evidence of a shortcut'})")


def class_balance_check(train_labels, test_pred_labels, log=print):
    """Compares train's true class distribution against a given set of TEST predictions (from
    whatever model you're currently scoring) -- a large shift suggests the test-time prior
    differs from train's exactly-balanced 150/class, which is when prior-adjustment (item #34,
    TRIED.md's already-negative-on-balanced-data calibration) would actually have a chance of
    helping, unlike on the current balanced-trained setup."""
    train_counts = pd.Series(train_labels).value_counts()
    test_counts = pd.Series(test_pred_labels).value_counts()
    both = pd.DataFrame({"train_count": train_counts, "test_pred_count": test_counts}).fillna(0)
    both["train_frac"] = both["train_count"] / both["train_count"].sum()
    both["test_pred_frac"] = both["test_pred_count"] / both["test_pred_count"].sum()
    both["frac_diff"] = both["test_pred_frac"] - both["train_frac"]
    log(both.sort_values("frac_diff", key=abs, ascending=False).to_string())
    return both


if __name__ == "__main__":
    import argparse

    from paths import data_dir

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="all",
                     choices=["all", "clusters", "duplicates", "id_structure"])
    ap.add_argument("--duplicate_threshold", type=float, default=0.95)
    args = ap.parse_args()

    train_df = pd.read_csv(data_dir() / "train.csv")
    test_path = data_dir() / "test.csv"
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    if args.check in ("all", "clusters"):
        print("=== item 1: cluster size recon (train, blind vs true prompt_id) ===")
        cluster_size_recon(train_df["text"].tolist(), true_groups=train_df["prompt_id"].tolist())
        if test_df is not None:
            print(f"\n=== item 1: cluster size recon (test, {len(test_df)} rows) ===")
            if len(test_df) < 18:
                print(f"  WARNING: only {len(test_df)} test rows locally -- this is a sample, "
                      f"not the real hidden eval set. Run on Kaggle for a real answer.")
            cluster_size_recon(test_df["text"].tolist())

    if args.check in ("all", "duplicates") and test_df is not None:
        print("\n=== item 2: train/test duplicate check ===")
        duplicate_check(train_df, test_df, threshold=args.duplicate_threshold)

    if args.check in ("all", "id_structure"):
        print("\n=== item 3: id/row-order structure check (train) ===")
        id_structure_check(train_df, id_col="id", label_col="model")
        if test_df is not None:
            print("\n=== item 3: id/row-order structure check (test, no label to correlate) ===")
            id_structure_check(test_df, id_col="id")
