"""Baseline authorship-attribution model for name-that-llm.

Cross-validates a TF-IDF (word + char n-gram) + Logistic Regression pipeline
using GroupKFold on prompt_id, so validation prompts never leak into train.
Prints per-fold and mean balanced accuracy, then fits on all data and saves
the pipeline to models/baseline.joblib.
"""
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

from paths import data_dir
from exp_log import ExperimentRun


def build_vectorizers():
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        analyzer="char_wb",
    )
    return word_vec, char_vec


def fit_transform(texts, word_vec, char_vec, fit=True):
    if fit:
        xw = word_vec.fit_transform(texts)
        xc = char_vec.fit_transform(texts)
    else:
        xw = word_vec.transform(texts)
        xc = char_vec.transform(texts)
    return hstack([xw, xc]).tocsr()


def cross_validate(df, n_splits=5, log=print):
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(df, df["model"], groups=df["prompt_id"])
    ):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        word_vec, char_vec = build_vectorizers()
        X_train = fit_transform(train_df["text"], word_vec, char_vec, fit=True)
        X_val = fit_transform(val_df["text"], word_vec, char_vec, fit=False)

        clf = LogisticRegression(
            max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1
        )
        clf.fit(X_train, train_df["model"])
        preds = clf.predict(X_val)

        score = balanced_accuracy_score(val_df["model"], preds)
        scores.append(score)
        log(f"fold {fold}: balanced_accuracy={score:.4f}")

    return scores


def fit_final(df, artifact_path, log=print):
    word_vec, char_vec = build_vectorizers()
    X = fit_transform(df["text"], word_vec, char_vec, fit=True)
    clf = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1)
    clf.fit(X, df["model"])
    joblib.dump({"word_vec": word_vec, "char_vec": char_vec, "clf": clf}, artifact_path)
    log(f"saved final model to {artifact_path}")


def main():
    config = {"ngram_word": [1, 2], "ngram_char": [3, 5], "C": 5.0, "n_splits": 5}
    run = ExperimentRun("tfidf_lr_baseline", config=config)

    df = pd.read_csv(data_dir() / "train.csv")
    scores = cross_validate(df, n_splits=config["n_splits"], log=run.log)
    run.log_scores("balanced_accuracy", scores)
    fit_final(df, run.artifact_path("baseline.joblib"), log=run.log)


if __name__ == "__main__":
    main()
