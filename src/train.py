"""Baseline authorship-attribution model for name-that-llm.

Cross-validates a TF-IDF (word + char n-gram) + Logistic Regression pipeline
using GroupKFold on prompt_id, so validation prompts never leak into train.
Prints per-fold and mean balanced accuracy, then fits on all data and saves
the pipeline to models/baseline.joblib.
"""
import argparse
import re

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


class NBSVM:
    """Multi-class NBSVM (Wang & Manning 2012, "Baselines and Bigrams"), adapted to K-class
    classification via one-vs-rest: for each class, scale TF-IDF features by that class's
    Naive-Bayes log-count ratio (log((class-c counts + alpha) / (not-c counts + alpha)), each
    L1-normalized first) before fitting a binary linear classifier; predict by argmax over the K
    binary decision scores. Distinct from OneVsRestClassifier(LogisticRegression()), which would
    reuse the SAME unscaled X for every binary sub-problem -- the whole point of NBSVM is that the
    scaling itself is class-specific, computed from each class's own word/feature frequencies.
    (IDEAS.md #7.)"""

    def __init__(self, C=1.0, alpha=1.0, max_iter=2000):
        self.C = C
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        X = X.tocsr() if hasattr(X, "tocsr") else X
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._ratios = {}
        self._clfs = {}
        for c in self.classes_:
            is_c = (y == c)
            p = np.asarray(X[is_c].sum(axis=0)).ravel() + self.alpha
            q = np.asarray(X[~is_c].sum(axis=0)).ravel() + self.alpha
            r = np.log((p / p.sum()) / (q / q.sum())).reshape(1, -1)
            self._ratios[c] = r
            Xc = X.multiply(r).tocsr()
            clf = LogisticRegression(C=self.C, max_iter=self.max_iter)
            clf.fit(Xc, is_c.astype(int), sample_weight=sample_weight)
            self._clfs[c] = clf
        return self

    def decision_function(self, X):
        X = X.tocsr() if hasattr(X, "tocsr") else X
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            Xc = X.multiply(self._ratios[c]).tocsr()
            scores[:, i] = self._clfs[c].decision_function(Xc)
        return scores

    def predict_proba(self, X):
        # softmax over the K binary decision scores -- not calibrated probabilities in a strict
        # sense (each score comes from an independent binary classifier, not a shared softmax
        # objective), but good enough for argmax prediction and for OOF-proba-based ensembling
        # alongside our other models' outputs.
        scores = self.decision_function(X)
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


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

    def fit(self, texts):
        # No-op (the vocabulary is fixed/pretrained, not learned from texts) -- exists only so
        # callers can treat every tokenizer type uniformly via .fit(corpus) before use, same as
        # CustomBPETokenizer, without needing to know which kind they have.
        self._ensure_loaded()
        return self

    def __call__(self, text):
        self._ensure_loaded()
        return self._tok.tokenize(text)

    def __getstate__(self):
        return {"model_name": self.model_name}

    def __setstate__(self, state):
        self.model_name = state["model_name"]
        self._tok = None


class CustomBPETokenizer:
    """BPE tokenizer trained FROM SCRATCH on a given text corpus (IDEAS.md #8), instead of
    reusing a pretrained HF tokenizer's fixed, externally-learned vocabulary (SubwordTokenizer
    above). Learns a vocabulary specific to whatever corpus is handed to .fit() -- unsupervised,
    so including unlabeled/external text carries no leakage risk the way using it for actual
    classifier training would. Must call .fit(texts) before use; picklable via tokenizers'
    built-in to_str()/from_str() JSON (de)serialization, so this survives fit_final's joblib.dump
    once fitted."""

    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self._tok = None

    def fit(self, texts):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[UNK]"])
        tok.train_from_iterator(list(texts), trainer=trainer)
        self._tok = tok
        return self

    def __call__(self, text):
        assert self._tok is not None, "CustomBPETokenizer.fit() must be called before use"
        return self._tok.encode(text).tokens

    def __getstate__(self):
        return {"vocab_size": self.vocab_size, "tok_json": self._tok.to_str() if self._tok else None}

    def __setstate__(self, state):
        self.vocab_size = state["vocab_size"]
        self._tok = None
        if state["tok_json"] is not None:
            from tokenizers import Tokenizer
            self._tok = Tokenizer.from_str(state["tok_json"])


def build_tokenizer_from_spec(spec):
    """Single entry point for picking a word-level tokenizer by a plain string, so every script
    (train.py, train_fusion_lgbm.py, ...) and any later ensemble sweep selects one the same way
    instead of each wiring its own if/else. Adding a new tokenizer type later = one more branch
    here, not a new CLI flag/code path per script.

    spec:
      None / '' -> None (default: original regex word tokenization, no custom tokenizer)
      'bpe:<vocab_size>' -> CustomBPETokenizer(vocab_size) -- must .fit(corpus) before use (its
        vocabulary is learned from texts, not fixed), so needs_refit(spec) is True for this case
      anything else -> SubwordTokenizer(spec), a pretrained HF tokenizer by model name (e.g.
        'microsoft/deberta-v3-base') -- backward compatible with the original --subword_tokenizer
        contract, which only ever took a bare HF model name.
    """
    if not spec:
        return None
    if spec.startswith("bpe:"):
        return CustomBPETokenizer(vocab_size=int(spec.split(":", 1)[1]))
    return SubwordTokenizer(spec)


def tokenizer_needs_refit(spec):
    """True if build_tokenizer_from_spec(spec) returns an object whose vocabulary must be
    .fit() per-fold on that fold's own train text (CustomBPETokenizer) rather than loaded once
    and reused across folds (SubwordTokenizer's pretrained vocab is fold-independent)."""
    return bool(spec) and spec.startswith("bpe:")


PUNCTUATION_ONLY = re.compile(r"[^\w\s]", re.UNICODE)


def extract_punctuation_stream(text):
    """Just the punctuation characters from a passage, in their original order -- loses all word
    content but keeps punctuation SEQUENCE, unlike stylometric.py's rate-based features (no order
    info) or char_vec's char_wb n-grams below (punctuation gets drowned out in a 3-5 char window
    dominated by letters). PAN authorship-attribution research found this adds signal beyond both
    (IDEAS.md #9) -- directly relevant to the claude<->glm confusion, since punctuation
    (semicolons, curly quotes) is already our strongest known signal there."""
    return "".join(PUNCTUATION_ONLY.findall(text))


def build_punctuation_vectorizer():
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        min_df=2,
        sublinear_tf=True,
    )


def build_vectorizers(subword_tokenizer=None, use_punctuation_ngrams=False):
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
    punct_vec = build_punctuation_vectorizer() if use_punctuation_ngrams else None
    return word_vec, char_vec, punct_vec


def fit_transform(texts, word_vec, char_vec, punct_vec=None, fit=True):
    if fit:
        xw = word_vec.fit_transform(texts)
        xc = char_vec.fit_transform(texts)
    else:
        xw = word_vec.transform(texts)
        xc = char_vec.transform(texts)
    parts = [xw, xc]
    if punct_vec is not None:
        punct_texts = [extract_punctuation_stream(t) for t in texts]
        xp = punct_vec.fit_transform(punct_texts) if fit else punct_vec.transform(punct_texts)
        parts.append(xp)
    return hstack(parts).tocsr()


def default_clf_factory():
    return LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1)


def cross_validate(df, n_splits=5, log=print, tokenizer_spec=None, use_punctuation_ngrams=False,
                    clf_factory=default_clf_factory):
    """Returns (scores, oof_proba, class_labels) -- oof_proba is [n_rows, n_classes], aligned to
    df's row order (np.nan where a row was never in a val fold, shouldn't happen with the full
    GroupKFold split). Saved by main() so different tokenizer_spec/clf_factory runs produce
    comparable, stackable OOF artifacts for later ensembling across variants, not just a score.
    clf_factory: zero-arg callable returning a fresh, unfitted classifier each call (fit(X, y,
    sample_weight=None) + predict_proba(X)) -- same plug-and-play pattern as tokenizer_spec, so
    swapping classifiers (e.g. NBSVM) reuses this harness instead of a one-off script."""
    # Pretrained tokenizers (SubwordTokenizer) have a fold-independent vocabulary -- load once,
    # reuse every fold (AutoTokenizer.from_pretrained has real cost, no need to pay it 5x).
    # CustomBPETokenizer's vocabulary is LEARNED from texts, so it must be rebuilt per fold from
    # that fold's own train text only -- same leakage discipline as word_vec/char_vec already
    # being refit per fold.
    refit_per_fold = tokenizer_needs_refit(tokenizer_spec)
    shared_tokenizer = None if refit_per_fold else build_tokenizer_from_spec(tokenizer_spec)
    if shared_tokenizer is not None:
        shared_tokenizer.fit(df["text"])

    class_labels = sorted(df["model"].unique())
    label2id = {l: i for i, l in enumerate(class_labels)}
    oof_proba = np.full((len(df), len(class_labels)), np.nan, dtype=np.float32)

    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(df, df["model"], groups=df["prompt_id"])
    ):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        tokenizer = (build_tokenizer_from_spec(tokenizer_spec).fit(train_df["text"])
                     if refit_per_fold else shared_tokenizer)

        word_vec, char_vec, punct_vec = build_vectorizers(tokenizer, use_punctuation_ngrams)
        X_train = fit_transform(train_df["text"], word_vec, char_vec, punct_vec, fit=True)
        X_val = fit_transform(val_df["text"], word_vec, char_vec, punct_vec, fit=False)

        clf = clf_factory()
        clf.fit(X_train, train_df["model"])
        fold_proba = np.zeros((len(val_df), len(class_labels)), dtype=np.float32)
        fold_proba[:, [label2id[c] for c in clf.classes_]] = clf.predict_proba(X_val)
        oof_proba[val_idx] = fold_proba

        pred_labels = [class_labels[i] for i in np.argmax(fold_proba, axis=-1)]
        score = balanced_accuracy_score(val_df["model"], pred_labels)
        scores.append(score)
        log(f"fold {fold}: balanced_accuracy={score:.4f}")

    return scores, oof_proba, class_labels


def fit_final(df, artifact_path, log=print, tokenizer_spec=None, use_punctuation_ngrams=False,
              clf_factory=default_clf_factory):
    tokenizer = build_tokenizer_from_spec(tokenizer_spec)
    if tokenizer is not None:
        tokenizer.fit(df["text"])
    word_vec, char_vec, punct_vec = build_vectorizers(tokenizer, use_punctuation_ngrams)
    X = fit_transform(df["text"], word_vec, char_vec, punct_vec, fit=True)
    clf = clf_factory()
    clf.fit(X, df["model"])
    joblib.dump({"word_vec": word_vec, "char_vec": char_vec, "punct_vec": punct_vec, "clf": clf}, artifact_path)
    log(f"saved final model to {artifact_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subword_tokenizer", default=None, dest="tokenizer_spec",
                     help="word-level tokenizer for TF-IDF, via build_tokenizer_from_spec(): a "
                          "bare HF model name reuses its pretrained tokenizer (e.g. "
                          "microsoft/deberta-v3-base, confirmed +0.03 balanced accuracy over the "
                          "default), 'bpe:<vocab_size>' trains a BPE vocabulary from scratch on "
                          "this run's own data (e.g. bpe:5000 -- tested negative on fold 0, see "
                          "TRIED.md). None (default) = original regex word tokenization. Saves "
                          "OOF probabilities (oof_proba.npy) either way, so runs with different "
                          "tokenizer_spec values produce comparable, stackable artifacts for "
                          "later ensembling across tokenizer variants.")
    ap.add_argument("--punctuation_ngrams", action="store_true",
                     help="add a char n-gram (1-3) TF-IDF block over JUST the punctuation stream "
                          "of each passage (order preserved, word content stripped) -- distinct "
                          "from stylometric.py's punctuation RATE features (no order) and from "
                          "char_vec's char_wb n-grams (punctuation drowned out by surrounding "
                          "letters). PAN research found this adds signal for authorship "
                          "attribution specifically (IDEAS.md #9). Untested here.")
    ap.add_argument("--classifier", default="lr", choices=["lr", "nbsvm"],
                     help="lr (default) = LogisticRegression(C=5.0, class_weight='balanced'). "
                          "nbsvm = NBSVM (IDEAS.md #7): scales TF-IDF features by each class's "
                          "Naive-Bayes log-count ratio before a one-vs-rest linear classifier, "
                          "instead of raw/plain TF-IDF weights. Untested here.")
    args = ap.parse_args()
    clf_factory = (lambda: NBSVM(C=1.0, max_iter=2000)) if args.classifier == "nbsvm" else default_clf_factory

    config = {"ngram_word": [1, 2], "ngram_char": [3, 5], "C": 5.0, "n_splits": 5,
              "tokenizer_spec": args.tokenizer_spec, "punctuation_ngrams": args.punctuation_ngrams,
              "classifier": args.classifier}
    run_name = "tfidf_lr_baseline" if not args.tokenizer_spec else "tfidf_lr_" + args.tokenizer_spec.split(":")[0]
    if args.punctuation_ngrams:
        run_name += "_punct"
    if args.classifier != "lr":
        run_name += "_" + args.classifier
    run = ExperimentRun(run_name, config=config)

    df = pd.read_csv(data_dir() / "train.csv")
    scores, oof_proba, class_labels = cross_validate(
        df, n_splits=config["n_splits"], log=run.log, tokenizer_spec=args.tokenizer_spec,
        use_punctuation_ngrams=args.punctuation_ngrams, clf_factory=clf_factory,
    )
    run.log_scores("balanced_accuracy", scores)
    np.save(run.artifact_path("oof_proba.npy"), oof_proba)
    oof = df[["id", "prompt_id", "model"]].rename(columns={"model": "true_label"}).copy()
    oof["pred"] = [class_labels[i] for i in np.argmax(oof_proba, axis=-1)]
    oof.to_csv(run.artifact_path("oof.csv"), index=False)
    run.log(f"saved OOF predictions -> oof.csv, oof_proba.npy ({len(class_labels)} classes)")

    fit_final(df, run.artifact_path("baseline.joblib"), log=run.log,
              tokenizer_spec=args.tokenizer_spec, use_punctuation_ngrams=args.punctuation_ngrams,
              clf_factory=clf_factory)


if __name__ == "__main__":
    main()
