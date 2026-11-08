"""Within-cluster specialist models for the 4 adjacent-version clusters confirmed to dominate
errors (error_analysis.ipynb: 42.6% of errors whose true label is in one of these land on
another member of the SAME cluster, ~7x the rate expected by chance). This is a MORE SPECIFIC
hypothesis than the already-tried (and negative, TRIED.md) family-level mixture-of-experts --
that one gated by broad vendor family (Claude/GPT/Gemini/GLM); this one targets exactly the
tight sibling groups the error analysis found, nothing coarser.

items 24/25 (specialist models + blending): train a small classifier PER cluster, using only
that cluster's own rows, then for any row where the base model's prediction falls inside a
cluster, redistribute probability mass using the specialist's prediction restricted to just that
cluster's members (base model's own mass over those classes gets replaced, not added to, so
this is a true redistribution, not a boost).

item 26 (discriminative tokens per confused pair): chi-square-ranked TF-IDF tokens between the
two members of each confusion pair, surfaced as a diagnostic -- a lightweight precursor to
turning them into explicit features, not the fully-automated version of that yet.

Base model: the cached frozen Qwen2.5-3B layer-7 embedding (0.3978 CV, the best single-embedding
result this project has) + LogisticRegression -- reuses the existing data/embed_cache/ entry, no
GPU needed. Full GroupKFold CV, same leakage discipline as everywhere else: the specialist for
a fold is trained ONLY on that fold's train rows, restricted to the cluster.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exp_log import ExperimentRun
from paths import data_dir
from train import build_vectorizers, fit_transform

HARD_CLUSTERS = [
    {"claude-opus-4-6", "claude-opus-4-8", "claude-opus-5"},
    {"gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"},
    {"gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash"},
    {"glm-5.3", "glm-5.3-flash"},
]


def make_base_clf():
    return make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1))


def cluster_of(label):
    for cluster in HARD_CLUSTERS:
        if label in cluster:
            return frozenset(cluster)
    return None


def redistribute_with_specialists(base_proba, class_labels, train_emb, train_labels, val_emb, log):
    """base_proba: [n_val, n_classes] from the base model, already fit on this fold. For every
    row whose base-argmax lands inside a hard cluster, replaces that row's probability mass over
    the cluster's member classes with a specialist (trained on only that cluster's train rows)
    prediction, renormalized -- mass OUTSIDE the cluster is left untouched."""
    label2id = {l: i for i, l in enumerate(class_labels)}
    out = base_proba.copy()
    base_pred = np.array(class_labels)[base_proba.argmax(axis=1)]

    for cluster in HARD_CLUSTERS:
        cluster_order = list(cluster)  # fixed order, reused for both columns and the specialist
        cluster_ids = [label2id[c] for c in cluster_order]
        in_cluster_val = np.array([p in cluster for p in base_pred])
        if not in_cluster_val.any():
            continue
        cluster_train_mask = np.isin(train_labels, cluster_order)
        if cluster_train_mask.sum() < len(cluster_order) * 2:
            log(f"  cluster {sorted(cluster)}: too few train rows "
                f"({cluster_train_mask.sum()}), skipping specialist")
            continue

        specialist = make_base_clf()
        specialist.fit(train_emb[cluster_train_mask], train_labels[cluster_train_mask])
        spec_proba = specialist.predict_proba(val_emb[in_cluster_val])
        spec_classes = specialist.named_steps["logisticregression"].classes_

        mass_in_cluster = out[np.ix_(in_cluster_val, cluster_ids)].sum(axis=1, keepdims=True)
        col_map = {c: cluster_order.index(c) for c in spec_classes}
        redistributed = np.zeros((in_cluster_val.sum(), len(cluster_order)))
        for j, c in enumerate(spec_classes):
            redistributed[:, col_map[c]] = spec_proba[:, j]
        redistributed = redistributed * mass_in_cluster  # keep the base model's total mass on
        # this cluster fixed, just redistribute WHO within it gets how much
        out[np.ix_(in_cluster_val, cluster_ids)] = redistributed

    return out


def discriminative_tokens(train_df, pair, top_n=15, log=print):
    """item 26: chi-square-ranked TF-IDF tokens distinguishing the two classes in `pair`
    (word 1-2grams + char 3-5grams, same vectorizers as the main pipeline). Diagnostic only --
    prints the top tokens, doesn't (yet) wire them into a feature block."""
    sub = train_df[train_df["model"].isin(pair)]
    word_vec, char_vec, _, _ = build_vectorizers()
    X = fit_transform(sub["text"], word_vec, char_vec, fit=True)
    y = (sub["model"].values == sorted(pair)[0]).astype(int)

    chi2_scores, _ = chi2(X, y)
    names = np.array(word_vec.get_feature_names_out().tolist()
                      + char_vec.get_feature_names_out().tolist())
    order = np.argsort(chi2_scores)[::-1][:top_n]
    log(f"  top {top_n} discriminative tokens for {sorted(pair)}:")
    for i in order:
        log(f"    {names[i]!r:20s} chi2={chi2_scores[i]:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--cache_dir", default="data/embed_cache")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--discriminative_tokens", action="store_true",
                     help="also run item 26's chi-square token diagnostic for one pair per "
                          "cluster (the two members with the worst mutual confusion in "
                          "error_analysis.ipynb)")
    args = ap.parse_args()

    run = ExperimentRun(f"hard_cluster_specialists_{args.model_name.split('/')[-1]}",
                         config=vars(args))
    df = pd.read_csv(data_dir() / "train.csv")

    cache_dir = Path(args.cache_dir)
    key = hashlib.md5(
        f"{args.model_name}_causal_{args.max_length}_mean_{args.layer}".encode()
    ).hexdigest()[:10]
    cache_path = cache_dir / f"{key}.npy"
    if cache_path.exists():
        run.log(f"loading cached frozen embedding -> {cache_path}")
        emb = np.load(cache_path)
    else:
        from experiment_decoder_embedding import extract_decoder_embeddings
        run.log(f"cache miss -- extracting frozen embeddings (needs GPU)...")
        emb = extract_decoder_embeddings(
            args.model_name, df["text"].tolist(), max_length=args.max_length,
            pooling="mean", layers=(args.layer,), architecture="causal",
            batch_size=args.batch_size, log=run.log,
        )[args.layer]
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)

    if args.discriminative_tokens:
        # the single worst confusion pair per cluster, from error_analysis.ipynb's printed table
        worst_pairs = [
            ("gemini-3.8-flash", "gemini-3.7-flash"),
            ("glm-5.3", "glm-5.3-flash"),
            ("gpt-5.6-sol", "gpt-5.6-terra"),
            ("claude-opus-4-8", "claude-sonnet-5"),  # not a hard-cluster pair itself, but the
            # worst claude-opus-involving confusion in the printed table -- included for
            # completeness since claude-opus's worst pairs are mostly cross-cluster.
        ]
        for pair in worst_pairs:
            discriminative_tokens(df, pair, log=run.log)

    gkf = GroupKFold(n_splits=args.n_splits)
    folds = list(gkf.split(df, df["model"], groups=df["prompt_id"]))
    if args.max_folds is not None:
        folds = folds[:args.max_folds]

    class_labels = sorted(df["model"].unique())
    base_scores, specialist_scores = [], []
    for fold, (train_idx, val_idx) in enumerate(folds):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        train_emb, val_emb = emb[train_idx], emb[val_idx]
        train_labels = train_df["model"].values

        base = make_base_clf()
        base.fit(train_emb, train_labels)
        base_proba = np.zeros((len(val_df), len(class_labels)))
        label2id = {l: i for i, l in enumerate(class_labels)}
        base_proba[:, [label2id[c] for c in base.named_steps["logisticregression"].classes_]] = \
            base.predict_proba(val_emb)
        base_pred = [class_labels[i] for i in base_proba.argmax(axis=1)]
        base_score = balanced_accuracy_score(val_df["model"], base_pred)
        base_scores.append(base_score)

        redistributed = redistribute_with_specialists(
            base_proba, class_labels, train_emb, train_labels, val_emb, run.log)
        spec_pred = [class_labels[i] for i in redistributed.argmax(axis=1)]
        spec_score = balanced_accuracy_score(val_df["model"], spec_pred)
        specialist_scores.append(spec_score)

        run.log(f"fold {fold}: base={base_score:.4f} specialist_redistributed={spec_score:.4f} "
                f"(delta={spec_score - base_score:+.4f})")

    run.log_scores("balanced_accuracy_base", base_scores)
    run.log_scores("balanced_accuracy_specialist_redistributed", specialist_scores)


if __name__ == "__main__":
    main()
