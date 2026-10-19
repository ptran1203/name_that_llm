"""Quick model comparison + confusion analysis on top of the baseline CV."""
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

from paths import data_dir

df = pd.read_csv(data_dir() / "train.csv")


def build_vectorizers():
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    char_vec = TfidfVectorizer(ngram_range=(3, 5), min_df=2, sublinear_tf=True, analyzer="char_wb")
    return word_vec, char_vec


def fit_transform(texts, word_vec, char_vec, fit=True):
    if fit:
        xw = word_vec.fit_transform(texts)
        xc = char_vec.fit_transform(texts)
    else:
        xw = word_vec.transform(texts)
        xc = char_vec.transform(texts)
    return hstack([xw, xc]).tocsr()


def run(clf_factory, name, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    all_val_labels, all_val_preds = [], []
    for train_idx, val_idx in gkf.split(df, df["model"], groups=df["prompt_id"]):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        word_vec, char_vec = build_vectorizers()
        X_train = fit_transform(train_df["text"], word_vec, char_vec, fit=True)
        X_val = fit_transform(val_df["text"], word_vec, char_vec, fit=False)
        clf = clf_factory()
        clf.fit(X_train, train_df["model"])
        preds = clf.predict(X_val)
        scores.append(balanced_accuracy_score(val_df["model"], preds))
        all_val_labels.extend(val_df["model"].tolist())
        all_val_preds.extend(preds.tolist())
    print(f"{name}: mean balanced_accuracy={np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return all_val_labels, all_val_preds


labels, preds = run(lambda: LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced", n_jobs=-1), "LogisticRegression C=5")
run(lambda: LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced", n_jobs=-1), "LogisticRegression C=10")
run(lambda: LinearSVC(C=1.0, class_weight="balanced"), "LinearSVC C=1")
run(lambda: LinearSVC(C=0.5, class_weight="balanced"), "LinearSVC C=0.5")

# Confusion matrix from the first run (LogisticRegression C=5)
classes = sorted(set(labels))
cm = confusion_matrix(labels, preds, labels=classes)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

# print biggest off-diagonal confusions
pairs = []
for i, a in enumerate(classes):
    for j, b in enumerate(classes):
        if i != j and cm[i, j] > 0:
            pairs.append((cm_norm[i, j], a, b, cm[i, j]))
pairs.sort(reverse=True)
print("\nTop confusions (true -> predicted, rate, count):")
for rate, a, b, cnt in pairs[:20]:
    print(f"  {a:28s} -> {b:28s}  rate={rate:.2f}  n={cnt}")
