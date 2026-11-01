"""Baseline authorship-attribution model for name-that-llm.

Cross-validates a TF-IDF (word + char n-gram) + Logistic Regression pipeline
using GroupKFold on prompt_id, so validation prompts never leak into train.
Prints per-fold and mean balanced accuracy, then fits on all data and saves
the pipeline to models/baseline.joblib.
"""
import argparse

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


def identity_preprocessor(text):
    # Disables TfidfVectorizer's default lowercasing when a subword tokenizer is in use: HF
    # tokenizers expect natural-case input (vocab often distinguishes cased forms), and casing
    # itself may carry stylistic signal (title-case habits, emphasis) worth preserving.
    return text


class SubwordTokenizer:
    """Picklable wrapper around a HF tokenizer's .tokenize(), for use as TfidfVectorizer's
    tokenizer= callable. A closure or lambda would NOT survive fit_final's joblib.dump (standard
    pickle can't serialize nested functions by reference) -- this is a proper top-level class,
    and __getstate__/__setstate__ drop the loaded tokenizer before pickling and reload it lazily
    by name afterward, instead of trying to pickle the whole HF tokenizer object."""

    def __init__(self, model_name):
        self.model_name = model_name
        self._tok = None

    def _ensure_loaded(self):
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name)

    def __call__(self, text):
        self._ensure_loaded()
        return self._tok.tokenize(text)

    def __getstate__(self):
        return {"model_name": self.model_name}

    def __setstate__(self, state):
        self.model_name = state["model_name"]
        self._tok = None


def build_vectorizers(subword_tokenizer=None):
    if subword_tokenizer is not None:
        # Subword vocabulary comes from large external pretraining (BPE/SentencePiece), not fit
        # from scratch on our ~2699 rows -- a form of transfer that should reduce overfitting risk
        # vs. a from-scratch word vocabulary, and decomposes words morphologically (shares
        # "believ" across "believe"/"unbelievable"/"believed") instead of treating each whole
        # word as an opaque, unrelated token.
        word_vec = TfidfVectorizer(
            tokenizer=subword_tokenizer,
            preprocessor=identity_preprocessor,
            token_pattern=None,  # required by sklearn when a custom tokenizer is supplied
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
    else:
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


def cross_validate(df, n_splits=5, log=print, subword_tokenizer_name=None):
    # Loaded once, reused every fold -- AutoTokenizer.from_pretrained has real cost, no need to
    # pay it 5x. SubwordTokenizer lazy-loads on first call, so this is cheap until actually used.
    subword_tokenizer = SubwordTokenizer(subword_tokenizer_name) if subword_tokenizer_name else None

    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(df, df["model"], groups=df["prompt_id"])
    ):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        word_vec, char_vec = build_vectorizers(subword_tokenizer)
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


def fit_final(df, artifact_path, log=print, subword_tokenizer_name=None):
    subword_tokenizer = SubwordTokenizer(subword_tokenizer_name) if subword_tokenizer_name else None
    word_vec, char_vec = build_vectorizers(subword_tokenizer)
    X = fit_transform(df["text"], word_vec, char_vec, fit=True)
    clf = LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1)
    clf.fit(X, df["model"])
    joblib.dump({"word_vec": word_vec, "char_vec": char_vec, "clf": clf}, artifact_path)
    log(f"saved final model to {artifact_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subword_tokenizer", default=None,
                     help="HF model name whose tokenizer replaces the default regex word-"
                          "splitter for the word-level TF-IDF vectorizer, e.g. "
                          "microsoft/deberta-v3-base. None (default) = original regex word "
                          "tokenization.")
    args = ap.parse_args()

    config = {"ngram_word": [1, 2], "ngram_char": [3, 5], "C": 5.0, "n_splits": 5,
              "subword_tokenizer": args.subword_tokenizer}
    run_name = "tfidf_lr_baseline" if not args.subword_tokenizer else "tfidf_lr_subword"
    run = ExperimentRun(run_name, config=config)

    df = pd.read_csv(data_dir() / "train.csv")
    scores = cross_validate(df, n_splits=config["n_splits"], log=run.log,
                             subword_tokenizer_name=args.subword_tokenizer)
    run.log_scores("balanced_accuracy", scores)
    fit_final(df, run.artifact_path("baseline.joblib"), log=run.log,
              subword_tokenizer_name=args.subword_tokenizer)


if __name__ == "__main__":
    main()
