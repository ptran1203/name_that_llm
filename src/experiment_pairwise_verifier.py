"""User-proposed pair of related ideas:
1. Concatenate two passages into one synthetic text, as either a same-model augmentation
   example (IDEAS.md #1's untried "text-level splicing" variant) or a different-model example
   for idea 2 below.
2. Train a classifier on the concatenated text to predict whether the two passages came from
   the SAME model -- closer in spirit to IDEAS.md #6 (pairwise verification) but using a TRAINED
   classifier rather than IDEAS.md #6 Step 1's zero-training raw-embedding-cosine baseline
   (which failed, 0.17-0.20 on the 18-way reduction). A trained classifier is a materially
   different, more powerful approach, worth testing on its own rather than treating Step 1's
   failure as a hard gate on this.

Evaluated here as its own binary task (same-model vs different-model), not yet converted into
an 18-way classifier via nearest-prototype matching -- that would be the natural next step if
this shows real signal (IDEAS.md #6's staged plan).
"""
import argparse
import itertools
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from exp_log import ExperimentRun
from paths import data_dir
from train import build_vectorizers, fit_transform

# Known hard confusion clusters (IDEAS.md #6) -- oversample different-model pairs drawn from
# WITHIN these clusters, so the classifier has to learn real fine-grained signal instead of
# trivially separating different provider families.
HARD_CLUSTERS = [
    {"claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
     "glm-5.3", "glm-5.3-flash"},
    {"gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"},
    {"gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"},
]


def cluster_of(model):
    for c in HARD_CLUSTERS:
        if model in c:
            return c
    return None


def sample_pairs(labels, n_pos_per_class, hard_frac=0.5, seed=0):
    """labels: array of model names, aligned to LOCAL positions 0..len(labels)-1 within
    whichever fold/split this is called for. Returns [(pos_a, pos_b, same), ...]."""
    rng = random.Random(seed)
    by_class = {}
    for pos, m in enumerate(labels):
        by_class.setdefault(m, []).append(pos)

    pos_pairs = []
    for m, positions in by_class.items():
        if len(positions) < 2:
            continue
        combos = list(itertools.combinations(positions, 2))
        rng.shuffle(combos)
        pos_pairs.extend(combos[:n_pos_per_class])

    n_neg = len(pos_pairs)
    n_hard = int(n_neg * hard_frac)
    neg_pairs = []
    all_positions = list(range(len(labels)))
    tries = 0
    while len(neg_pairs) < n_hard and tries < n_hard * 30:
        tries += 1
        a, b = rng.sample(all_positions, 2)
        ma, mb = labels[a], labels[b]
        if ma == mb:
            continue
        ca, cb = cluster_of(ma), cluster_of(mb)
        if ca is not None and ca == cb:
            neg_pairs.append((a, b))
    while len(neg_pairs) < n_neg:
        a, b = rng.sample(all_positions, 2)
        if labels[a] != labels[b]:
            neg_pairs.append((a, b))

    pairs = [(i, j, 1) for i, j in pos_pairs] + [(i, j, 0) for i, j in neg_pairs]
    rng.shuffle(pairs)
    return pairs


def concat_texts(texts, pairs):
    return [texts[i] + " " + texts[j] for i, j, _ in pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pos_per_class", type=int, default=150)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = ExperimentRun("pairwise_same_model_verifier", config=vars(args))
    df = pd.read_csv(data_dir() / "train.csv").reset_index(drop=True)

    gkf = GroupKFold(n_splits=args.n_splits)
    aucs, accs = [], []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df["model"], groups=df["prompt_id"])):
        train_texts = df["text"].values[train_idx]
        val_texts = df["text"].values[val_idx]
        train_labels = df["model"].values[train_idx]
        val_labels = df["model"].values[val_idx]

        train_pairs = sample_pairs(train_labels, args.n_pos_per_class, seed=args.seed + fold)
        val_pairs = sample_pairs(val_labels, max(40, args.n_pos_per_class // 3), seed=args.seed + 1000 + fold)

        train_concat = concat_texts(train_texts, train_pairs)
        val_concat = concat_texts(val_texts, val_pairs)
        y_train = np.array([y for _, _, y in train_pairs])
        y_val = np.array([y for _, _, y in val_pairs])

        word_vec, char_vec, _punct_vec, _func_vec = build_vectorizers()
        X_train = fit_transform(train_concat, word_vec, char_vec, fit=True)
        X_val = fit_transform(val_concat, word_vec, char_vec, fit=False)

        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_val, proba)
        acc = balanced_accuracy_score(y_val, pred)
        aucs.append(auc)
        accs.append(acc)
        run.log(f"fold {fold}: pair_auc={auc:.4f} pair_balanced_accuracy={acc:.4f} "
                f"(train_pairs={len(train_pairs)}, val_pairs={len(val_pairs)})")

    run.log_scores("pair_auc", aucs)
    run.log_scores("pair_balanced_accuracy", accs)


if __name__ == "__main__":
    main()
