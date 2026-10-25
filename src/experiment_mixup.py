"""Idea 1 from IDEAS.md: feature-space mixup augmentation.

For each GroupKFold-by-prompt_id training fold, synthesize extra same-class rows by
interpolating two TF-IDF vectors from the SAME model (label stays the same model -- this
is intra-class interpolation for augmentation, not the standard cross-class mixup that
blends labels). Compares LogisticRegression CV with vs without the synthetic rows.
"""
import numpy as np
import pandas as pd
from scipy.sparse import diags, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

from paths import data_dir
from train import build_vectorizers, fit_transform
from exp_log import ExperimentRun


def mixup_augment(X, y, multiplier=1.0, alpha=0.4, seed=0):
    """Add synthetic same-class rows: lambda*X[i] + (1-lambda)*X[j], i,j same class.
    multiplier=1.0 -> add as many synthetic rows as original rows."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    n_new = int(len(y) * multiplier)
    idx_a = np.empty(n_new, dtype=int)
    idx_b = np.empty(n_new, dtype=int)
    classes = np.unique(y)
    class_indices = {c: np.flatnonzero(y == c) for c in classes}
    # sample rows proportionally to class frequency, then pair with another row of the same class
    all_idx = rng.choice(len(y), size=n_new, replace=True)
    for k, i in enumerate(all_idx):
        same_class = class_indices[y[i]]
        j = same_class[rng.randint(len(same_class))]
        idx_a[k] = i
        idx_b[k] = j

    lam = rng.beta(alpha, alpha, size=n_new)
    lam = np.clip(lam, 0.15, 0.85)  # avoid near-degenerate combos close to 0 or 1

    X_a, X_b = X[idx_a], X[idx_b]
    X_mix = diags(lam) @ X_a + diags(1 - lam) @ X_b
    X_mix = normalize(X_mix, norm="l2")  # TF-IDF rows are L2-normalized; keep that invariant

    y_mix = y[idx_a]  # same class as both parents by construction
    X_aug = vstack([X, X_mix]).tocsr()
    y_aug = np.concatenate([y, y_mix])
    return X_aug, y_aug


def run_experiment(multiplier, alpha=0.4, n_splits=5):
    exp = ExperimentRun("mixup_augmentation", config={"multiplier": multiplier, "alpha": alpha, "n_splits": n_splits})

    df = pd.read_csv(data_dir() / "train.csv")
    gkf = GroupKFold(n_splits=n_splits)
    plain_scores, aug_scores = [], []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["model"], groups=df["prompt_id"])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        word_vec, char_vec = build_vectorizers()
        X_train = fit_transform(train_df["text"], word_vec, char_vec, fit=True)
        X_val = fit_transform(val_df["text"], word_vec, char_vec, fit=False)
        y_train = train_df["model"].values

        clf = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1)
        clf.fit(X_train, y_train)
        plain_score = balanced_accuracy_score(val_df["model"], clf.predict(X_val))
        plain_scores.append(plain_score)

        X_aug, y_aug = mixup_augment(X_train, y_train, multiplier=multiplier, alpha=alpha, seed=fold)
        clf_aug = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1)
        clf_aug.fit(X_aug, y_aug)
        aug_score = balanced_accuracy_score(val_df["model"], clf_aug.predict(X_val))
        aug_scores.append(aug_score)

        exp.log(f"fold {fold}: plain={plain_score:.4f}  mixup(x{multiplier})={aug_score:.4f}")

    exp.log_scores("plain", plain_scores)
    exp.log_scores(f"mixup_x{multiplier}", aug_scores)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--multiplier", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.4)
    args = ap.parse_args()
    run_experiment(args.multiplier, args.alpha)
