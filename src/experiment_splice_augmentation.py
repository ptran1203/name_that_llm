"""Idea 1 from IDEAS.md / user-proposed: text-level splicing augmentation.

For each GroupKFold-by-prompt_id training fold, synthesize extra same-class rows by
concatenating two same-model passages' raw TEXT (label stays the same model). Distinct from
mixup_augment (experiment_mixup.py), which interpolates TF-IDF VECTORS instead of concatenating
text -- splicing risks introducing an artificial splice-boundary artifact (a sentence-length/
punctuation discontinuity a real single-author passage would never have), which mixup's
feature-space blend avoids. Compares LogisticRegression CV with vs without the synthetic rows.
"""
import random

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from exp_log import ExperimentRun
from paths import data_dir
from train import build_vectorizers, fit_transform


def splice_augment_texts(texts, labels, multiplier=1.0, seed=0):
    """Returns (aug_texts, aug_labels) = original rows + synthetic same-class concatenations."""
    rng = random.Random(seed)
    texts = list(texts)
    labels = list(labels)
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)

    n_new = int(len(texts) * multiplier)
    new_texts, new_labels = [], []
    all_idx = list(range(len(texts)))
    for _ in range(n_new):
        i = rng.choice(all_idx)
        same_class = by_class[labels[i]]
        j = rng.choice(same_class)
        new_texts.append(texts[i] + " " + texts[j])
        new_labels.append(labels[i])

    return texts + new_texts, labels + new_labels


def run_experiment(multiplier=1.0, n_splits=5):
    run = ExperimentRun("splice_augmentation", config={"multiplier": multiplier, "n_splits": n_splits})

    df = pd.read_csv(data_dir() / "train.csv")
    gkf = GroupKFold(n_splits=n_splits)
    plain_scores, aug_scores = [], []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["model"], groups=df["prompt_id"])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        word_vec, char_vec, _punct_vec, _func_vec = build_vectorizers()
        X_train = fit_transform(train_df["text"], word_vec, char_vec, fit=True)
        X_val = fit_transform(val_df["text"], word_vec, char_vec, fit=False)
        y_train = train_df["model"].values

        clf = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1)
        clf.fit(X_train, y_train)
        plain_score = balanced_accuracy_score(val_df["model"], clf.predict(X_val))
        plain_scores.append(plain_score)

        aug_texts, aug_labels = splice_augment_texts(
            train_df["text"].values, y_train, multiplier=multiplier, seed=fold)
        word_vec2, char_vec2, _p2, _f2 = build_vectorizers()
        X_aug = fit_transform(aug_texts, word_vec2, char_vec2, fit=True)
        X_val_aug = fit_transform(val_df["text"], word_vec2, char_vec2, fit=False)
        clf_aug = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1)
        clf_aug.fit(X_aug, aug_labels)
        aug_score = balanced_accuracy_score(val_df["model"], clf_aug.predict(X_val_aug))
        aug_scores.append(aug_score)

        run.log(f"fold {fold}: plain={plain_score:.4f}  splice(x{multiplier})={aug_score:.4f}")

    run.log_scores("plain", plain_scores)
    run.log_scores(f"splice_x{multiplier}", aug_scores)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--multiplier", type=float, default=1.0)
    args = ap.parse_args()
    run_experiment(args.multiplier)
