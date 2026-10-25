"""Error analysis on a fusion_lgbm run's OOF predictions: confusion matrix (plot + top
confusion pairs), within- vs cross-family error breakdown, and example misclassified passages.
"""
import argparse
import glob
import os

from exp_log import EXPERIMENTS_ROOT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

FAMILY_PREFIXES = ["claude", "gpt", "gemini", "glm", "qwen", "kimi"]


def family_of(label):
    for p in FAMILY_PREFIXES:
        if label.startswith(p):
            return p
    return label


def latest_run(name_glob="*fusion_lgbm*"):
    pattern = str(EXPERIMENTS_ROOT / name_glob)
    hits = sorted(glob.glob(pattern), key=os.path.getmtime)
    assert hits, f"no run directories matching {pattern}"
    return hits[-1]


def plot_confusion(cm_norm, labels, out_path):
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Fusion LGBM -- confusion matrix (row-normalized = per-class recall)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("fraction of true-class rows")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, help="defaults to the most recent fusion_lgbm run")
    ap.add_argument("--pred_col", default="fusion_pred", choices=["fusion_pred", "transformer_pred"])
    ap.add_argument("--n_examples", type=int, default=8)
    args = ap.parse_args()

    run_dir = args.run_dir or latest_run()
    oof = pd.read_csv(os.path.join(run_dir, "artifacts", "oof.csv"))
    labels = sorted(oof["true_label"].unique())
    print(f"run_dir={run_dir}  rows={len(oof)}  pred_col={args.pred_col}")

    y_true, y_pred = oof["true_label"], oof[args.pred_col]
    acc = (y_true == y_pred).mean()
    print(f"overall accuracy (unbalanced): {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    out_path = os.path.join(run_dir, "artifacts", f"confusion_matrix_{args.pred_col}.png")
    plot_confusion(cm_norm, labels, out_path)
    print(f"saved confusion matrix plot -> {out_path}")

    # top confusion pairs (true -> predicted), excluding the diagonal
    pairs = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pairs.append((cm_norm[i, j], cm[i, j], a, b))
    pairs.sort(reverse=True)
    print("\nTop confusions (true -> predicted, rate, count):")
    for rate, cnt, a, b in pairs[:15]:
        same_family = "same-family" if family_of(a) == family_of(b) else "cross-family"
        print(f"  {a:28s} -> {b:28s}  rate={rate:.2f}  n={cnt:3d}  ({same_family})")

    # within- vs cross-family error rate
    errors = oof[y_true != y_pred].copy()
    errors["true_family"] = errors["true_label"].map(family_of)
    errors["pred_family"] = errors[args.pred_col].map(family_of)
    within = (errors["true_family"] == errors["pred_family"]).mean()
    print(f"\nOf {len(errors)} errors: {within:.1%} stay within the same provider family, "
          f"{1 - within:.1%} cross families.")

    # a few concrete misclassified examples to read
    print(f"\n{args.n_examples} example misclassifications:")
    sample = errors.sample(min(args.n_examples, len(errors)), random_state=0)
    for _, row in sample.iterrows():
        snippet = row["text"][:220].replace("\n", " ")
        print(f"  true={row['true_label']:24s} pred={row[args.pred_col]:24s} | {snippet}...")


if __name__ == "__main__":
    main()
