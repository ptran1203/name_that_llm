"""Reusable ensembling/calibration utilities (ideas list items 31, 33) -- generic over any set
of OOF probability arrays, not tied to a specific experiment script, so any of this project's
existing *_proba.npy artifacts can be combined without rerunning anything that produced them.

item 32 (stacking: a meta-LR on OOF probabilities) already has a PARTIAL result in
frozen_embeddings.ipynb (2/5 folds, interrupted, val~0.42) -- not duplicated here since it's not
in the user's stated minimum-viable list; this file covers 31 and 33 specifically, which are
each simpler/cheaper alternatives to full stacking worth having on their own.
"""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import GroupKFold


def fit_weighted_average(oof_probas, y_true, class_labels, log=print):
    """item 31: fit non-negative weights (summing to 1) over several models' OOF probability
    arrays to maximize balanced accuracy of the weighted-average blend -- simpler and cheaper
    than stacking (item 32, a full meta-LR), and a reasonable first thing to try before reaching
    for that heavier option.

    oof_probas: {model_name: [n_rows, n_classes] array}, all aligned to the same row order and
    class_labels. y_true: [n_rows] true label strings. Returns (weights dict, blended balanced
    accuracy) -- also logs each individual model's own balanced accuracy for comparison, so you
    can see whether the blend actually beats its best single member."""
    names = list(oof_probas.keys())
    stacked = np.stack([oof_probas[n] for n in names], axis=0)  # [n_models, n_rows, n_classes]
    label2id = {l: i for i, l in enumerate(class_labels)}
    y_idx = np.array([label2id[l] for l in y_true])

    for n in names:
        pred = np.array(class_labels)[oof_probas[n].argmax(axis=1)]
        log(f"  {n}: balanced_accuracy={balanced_accuracy_score(y_true, pred):.4f} (solo)")

    def neg_balanced_acc(raw_weights):
        w = np.abs(raw_weights)
        w = w / w.sum()
        blend = (stacked * w[:, None, None]).sum(axis=0)
        pred = blend.argmax(axis=1)
        return -balanced_accuracy_score(y_idx, pred)

    x0 = np.ones(len(names)) / len(names)
    # balanced_accuracy is piecewise-constant (argmax-based) -- not smooth, so a gradient method
    # would stall; Nelder-Mead (derivative-free) handles that, at the cost of needing multiple
    # restarts to avoid a bad local optimum on a handful of weights (cheap, len(names) is small).
    best_w, best_score = x0, -neg_balanced_acc(x0)
    rng = np.random.RandomState(0)
    for _ in range(10):
        start = rng.dirichlet(np.ones(len(names)))
        res = minimize(neg_balanced_acc, start, method="Nelder-Mead")
        if -res.fun > best_score:
            best_score, best_w = -res.fun, np.abs(res.x) / np.abs(res.x).sum()

    weights = dict(zip(names, best_w.tolist()))
    log(f"  fitted weights: {weights}")
    log(f"  weighted-average blend: balanced_accuracy={best_score:.4f}")
    return weights, best_score


def fit_weighted_average_nested_cv(oof_probas, y_true, class_labels, groups, n_splits=5, log=print):
    """The trustworthy version of fit_weighted_average: fitting weights AND scoring the blend on
    the exact same rows (what fit_weighted_average does) can report an inflated number, since a
    non-smooth, hand-picked-weight optimizer is free to fit noise in that specific set with
    nothing held out to catch it. This does an OUTER GroupKFold (same grouping as the original
    CV, e.g. prompt_id) around fit_weighted_average: fit weights on the other folds' rows, score
    the blend (with those weights) on the held-out fold, repeat. Returns (per-fold held-out
    scores, mean) -- that mean, not fit_weighted_average's single number, is what should be
    trusted as the blend's real expected balanced accuracy."""
    names = list(oof_probas.keys())
    y_true = np.asarray(y_true)
    groups = np.asarray(groups)
    gkf = GroupKFold(n_splits=n_splits)

    held_out_scores = []
    for fold, (fit_idx, held_idx) in enumerate(gkf.split(y_true, y_true, groups=groups)):
        fit_probas = {n: oof_probas[n][fit_idx] for n in names}
        weights, _ = fit_weighted_average(fit_probas, y_true[fit_idx], class_labels,
                                           log=lambda *a: None)
        w = np.array([weights[n] for n in names])

        held_stack = np.stack([oof_probas[n][held_idx] for n in names], axis=0)
        blend = (held_stack * w[:, None, None]).sum(axis=0)
        pred = np.array(class_labels)[blend.argmax(axis=1)]
        score = balanced_accuracy_score(y_true[held_idx], pred)
        held_out_scores.append(score)
        log(f"  fold {fold}: weights={({n: round(v, 3) for n, v in weights.items()})} "
            f"held_out_balanced_accuracy={score:.4f}")

    mean_score = float(np.mean(held_out_scores))
    log(f"  nested-CV blend: mean={mean_score:.4f} +/- {np.std(held_out_scores):.4f} "
        f"(folds={held_out_scores})")
    return held_out_scores, mean_score


def fit_temperature(oof_proba, y_true, class_labels, log=print):
    """item 33: fit a single scalar temperature T (Guo et al. 2017) minimizing log-loss on OOF
    predictions -- T > 1 softens overconfident probabilities, T < 1 sharpens underconfident
    ones. Matters most for a log-loss metric (explicitly noted in the ideas list); this
    competition is scored by balanced accuracy, which is argmax-based and invariant to any
    *monotonic* rescaling of a single row's probabilities, so temperature scaling CANNOT change
    balanced accuracy on its own -- only relevant here if probabilities feed into something
    order-sensitive downstream (e.g. the cluster-assignment resolvers in prompt_cluster_assign.py,
    which rank/compare probabilities ACROSS rows, not just within one row's argmax)."""
    label2id = {l: i for i, l in enumerate(class_labels)}
    y_idx = np.array([label2id[l] for l in y_true])
    logits = np.log(np.clip(oof_proba, 1e-12, None))

    def nll(T):
        scaled = logits / T
        scaled -= scaled.max(axis=1, keepdims=True)
        probs = np.exp(scaled)
        probs /= probs.sum(axis=1, keepdims=True)
        return log_loss(y_idx, probs, labels=list(range(len(class_labels))))

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    T = res.x
    log(f"  fitted temperature T={T:.3f} (log_loss before={nll(1.0):.4f}, after={nll(T):.4f})")

    calibrated = apply_temperature(oof_proba, T)
    pred_before = oof_proba.argmax(axis=1)
    pred_after = calibrated.argmax(axis=1)
    log(f"  balanced_accuracy unchanged by design: before={balanced_accuracy_score(y_idx, pred_before):.4f} "
        f"after={balanced_accuracy_score(y_idx, pred_after):.4f}")
    return T, calibrated


def apply_temperature(proba, T):
    """Applies an ALREADY-FITTED temperature T (from fit_temperature) to new probabilities --
    e.g. test-time predictions, which must use the SAME T fit on train OOF, not a freshly-fit
    one (fitting on test would need labels we don't have)."""
    logits = np.log(np.clip(proba, 1e-12, None)) / T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    import argparse

    import pandas as pd

    from exp_log import EXPERIMENTS_ROOT

    ap = argparse.ArgumentParser(description=(
        "Loads two or more runs' oof.csv + oof_proba.npy (same convention train.py/"
        "train_fusion_lgbm.py already save) and fits a weighted-average blend + temperature."))
    ap.add_argument("--run_dirs", required=True,
                     help="comma-separated run directory names under experiments/")
    args = ap.parse_args()

    oof_probas, y_true, class_labels = {}, None, None
    for run_name in args.run_dirs.split(","):
        run_dir = EXPERIMENTS_ROOT / run_name
        oof = pd.read_csv(run_dir / "artifacts" / "oof.csv")
        proba = np.load(run_dir / "artifacts" / "oof_proba.npy")
        if y_true is None:
            y_true = oof["true_label"].tolist()
            class_labels = sorted(set(y_true))
        oof_probas[run_name] = proba

    print("=== item 31: weighted average ===")
    fit_weighted_average(oof_probas, y_true, class_labels)

    print("\n=== item 33: temperature scaling (first run only) ===")
    first = next(iter(oof_probas.values()))
    fit_temperature(first, y_true, class_labels)
