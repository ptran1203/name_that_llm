"""Normalizes external datasets into the same schema as train.csv (id, prompt_id, text, model),
with a TF-IDF cosine-similarity leakage check against official train.csv (any external row too
similar to an official passage is dropped -- keeps CV honest if this data ever gets used in training).

Works in two environments:
- Local: run after downloading raw files via the Kaggle API into data/external/raw/<source>/.
- Kaggle: attach the raw source datasets (see RAW_CANDIDATES below for the exact slugs) and run
  this directly -- no download step needed, it finds them under /kaggle/input.

Saves one <source>.parquet per source to data/external/ locally, or /kaggle/working/data/external/
on Kaggle (train_transformer.py's --external_paths / paths.external_data_dir() finds both).
"""
import glob
import pathlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from paths import data_dir

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Local candidate = where `kaggle datasets download` extracts to; Kaggle candidates = where the
# raw source dataset gets mounted once attached to a notebook (owner/slug, as Kaggle reports it
# when multiple datasets share a slug; falls back to the no-owner-prefix form too, in case a
# future Kaggle mount doesn't need it).
RAW_CANDIDATES = {
    "prvsiyan": [
        _REPO_ROOT / "data/external/raw/prvsiyan",
        pathlib.Path("/kaggle/input/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset"),
        pathlib.Path("/kaggle/input/prvsiyans-extended-name-that-llm-dataset"),
    ],
    "darragh1walsh": [
        _REPO_ROOT / "data/external/raw/darragh1walsh",
        pathlib.Path("/kaggle/input/datasets/darragh1walsh/ntl-kaggle-ai-gen-run-1"),
        pathlib.Path("/kaggle/input/ntl-kaggle-ai-gen-run-1"),
    ],
}


def resolve_raw_dir(name):
    for p in RAW_CANDIDATES[name]:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no raw dir found for {name!r}. Looked in: {[str(p) for p in RAW_CANDIDATES[name]]}"
    )


def out_dir():
    p = (pathlib.Path("/kaggle/working/data/external") if pathlib.Path("/kaggle/working").exists()
         else _REPO_ROOT / "data/external")
    p.mkdir(parents=True, exist_ok=True)
    return p


LEAKAGE_SIM_THRESHOLD = 0.5


def check_leakage(texts, official_texts):
    """Max TF-IDF cosine similarity of each text to any official train passage."""
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    vectorizer.fit(list(texts) + list(official_texts))
    X = vectorizer.transform(texts)
    X_off = vectorizer.transform(official_texts)
    max_sim = np.zeros(len(texts))
    chunk = 500
    for i in range(0, len(texts), chunk):
        sim = (X[i:i + chunk] @ X_off.T).toarray()
        max_sim[i:i + chunk] = sim.max(axis=1)
    return max_sim


def prep_prvsiyan():
    """passes_funnel + requested==served already guaranteed by the source's own methodology
    (README: mismatched-route calls are quarantined, not published); still apply the funnel flag
    and the source's own n-gram-overlap-with-competition-data check for extra safety."""
    raw = resolve_raw_dir("prvsiyan")
    df = pd.read_csv(raw / "passages_clean.csv")
    df = df[(df["passes_funnel"] == True) & (df["max_shared_8grams_comp"] <= 3)].copy()
    return pd.DataFrame({
        "id": "prvsiyan_" + df["id"].astype(str),
        "prompt_id": "prvsiyan_" + df["prompt_id"].astype(str),
        "text": df["text"].astype(str),
        "model": df["model"],
        "source_dataset": "prvsiyan",
    })


def prep_darragh():
    """passed_filter + finish=='stop' (drop token-limit-truncated rows, which aren't representative
    complete passages)."""
    raw = resolve_raw_dir("darragh1walsh")
    files = sorted(glob.glob(str(raw / "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[(df["passed_filter"] == True) & (df["finish"] == "stop")].reset_index(drop=True)
    return pd.DataFrame({
        "id": [f"darragh_{i:05d}" for i in range(len(df))],
        "prompt_id": "darragh_" + df["prompt_id"].astype(str),
        "text": df["text"].astype(str),
        "model": df["model"],
        "source_dataset": "darragh1walsh",
    })


def main():
    official = pd.read_csv(data_dir() / "train.csv")
    out = out_dir()

    for name, prep_fn in [("prvsiyan", prep_prvsiyan), ("darragh1walsh", prep_darragh)]:
        df = prep_fn()
        before = len(df)
        sim = check_leakage(df["text"].tolist(), official["text"].tolist())
        df["max_sim_official_train"] = sim
        n_flagged = (df["max_sim_official_train"] > LEAKAGE_SIM_THRESHOLD).sum()
        df = df[df["max_sim_official_train"] <= LEAKAGE_SIM_THRESHOLD].reset_index(drop=True)
        print(f"{name}: {before} rows -> dropped {n_flagged} too-similar-to-official-train -> {len(df)} final")
        print(df["model"].value_counts())
        out_path = out / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"saved -> {out_path}\n")


if __name__ == "__main__":
    main()
