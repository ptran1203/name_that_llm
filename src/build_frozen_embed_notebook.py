"""Builds notebooks/frozen_embeddings.ipynb cell-by-cell via nbformat.

NOTE: the live notebook has since been hand-edited beyond what this script generates (e.g. the
external-data cells were patched in directly). This script is kept as a rough source of truth for
a from-scratch rebuild, but isn't guaranteed byte-identical to the current notebooks/frozen_embeddings.ipynb.

Approach: extract mean-pooled embeddings from several FROZEN pretrained models (no fine-tuning,
no backward pass, no Trainer/optimizer -- pure forward-pass inference, so it's far lighter than
anything we've done with Trainer), concatenate with SVD-compressed TF-IDF + stylometric features,
and fit a classical ML head (LogisticRegression / LightGBM / XGBoost) via GroupKFold CV. Optionally
blends in a large external pooled dataset (train-side only, down-weighted) for extra signal.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Frozen embeddings + TF-IDF -> classical ML

No fine-tuning anywhere in this notebook: every pretrained model is used strictly in
`eval()` mode under `torch.no_grad()` for a single forward pass per passage (mean-pooled
last hidden state). No `Trainer`, no optimizer, no backward pass, no `GradScaler` -- this
sidesteps every issue we've hit trying to fine-tune (DDP eval-gather bugs, fp16 GradScaler
crashes, transformers-version incompatibilities). It's also much cheaper: embeddings only
need to be extracted once for the whole dataset (frozen models can't leak train labels into
the embedding, so there's no need to refit per fold -- unlike TF-IDF, which still needs a
fresh fit per fold to avoid vocabulary leakage).""")

code("""import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.class_weight import compute_sample_weight
from transformers import AutoModel, AutoTokenizer

import sys
sys.path.insert(0, '../src')
from paths import data_dir
from stylometric import stylometric_matrix
from train import build_vectorizers, fit_transform

pd.set_option('display.max_columns', None)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEVICE)""")

md("""## 1. Load data""")

code("""df = pd.read_csv(data_dir() / 'train.csv')
labels_sorted = sorted(df['model'].unique())
label2id = {l: i for i, l in enumerate(labels_sorted)}
df['label'] = df['model'].map(label2id)
print(df.shape)
df.head(2)""")

md("""## 2. Frozen embedding extraction (forward pass only, computed once for the whole dataset)

Six models for diversity: DeBERTa-v3-base + RoBERTa-base (MLM encoders, different corpora),
ELECTRA-base-discriminator (different pretraining objective -- replaced-token detection, not MLM),
Qwen2.5-0.5B (decoder-only, causal -- structurally different from the rest), and two sentence-embedding
models (MiniLM-L6-v2, e5-small-v2) trained via contrastive objectives rather than plain MLM.""")

code("""MODEL_NAMES = [
    'microsoft/deberta-v3-base',
    'roberta-base',
    'sentence-transformers/all-MiniLM-L6-v2',
    'google/electra-base-discriminator',
    'Qwen/Qwen2.5-0.5B',
    'intfloat/e5-small-v2',
]
MAX_LENGTH = 384
BATCH_SIZE = 16  # inference-only -- no gradients/optimizer state, so this is far lighter than training


@torch.no_grad()
def extract_embeddings(model_name, texts, max_length=MAX_LENGTH, batch_size=BATCH_SIZE):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:  # some decoder-only models (e.g. Qwen) have no pad token by default
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=max_length, padding=True, return_tensors='pt').to(DEVICE)
        hidden = model(**enc).last_hidden_state
        mask = enc['attention_mask'].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(pooled.float().cpu().numpy())
    del model
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(out, axis=0)""")

code("""texts = df['text'].tolist()
embeddings = {}
for name in MODEL_NAMES:
    print(f'extracting: {name}')
    embeddings[name] = extract_embeddings(name, texts)
    print(f'  -> shape {embeddings[name].shape}')

emb_concat = np.concatenate([embeddings[n] for n in MODEL_NAMES], axis=1).astype(np.float32)
print('concatenated embedding shape:', emb_concat.shape)""")

md("""## 3. Stylometric features (stateless, also computed once)""")

code("""stylo = stylometric_matrix(df['text']).astype(np.float32)
print('stylometric shape:', stylo.shape)""")

md("""## 3b. External data (optional extra signal)

Three pooled sources, all normalized to [id, text, model, source_dataset]:
- `ntl_open_pool.parquet` (226k rows, pooled from many public Kaggle datasets) -- filtered to drop
  rows flagged near-duplicate to official train prompts, regenerated, or at risk of Gemini-3.7 API
  routing mislabeling, then capped per class (prioritized by `n_sources`, i.e. how many independent
  public datasets agree on this row).
- `prvsiyan.parquet` (5050 rows, all 18 labels) -- cleaned via `src/prepare_external_data.py`:
  source's own funnel + n-gram-overlap check, plus our own TF-IDF cosine-similarity check against
  official train.csv.
- `darragh1walsh.parquet` (2584 rows, 12 of 18 labels -- missing GLM/Kimi/Qwen/claude-fable-5/
  gemini-3.7-flash) -- same cleaning: source's `passed_filter` + `finish=='stop'` (drop token-limit-
  truncated rows), plus our own leakage check.

**External rows only ever enter the train side of each CV fold, never val** -- val stays pure
official data so CV remains an honest estimate of real generalization.""")

code("""EXTERNAL_CAP_PER_CLASS = 1000  # applied to ntl_open_pool only -- the other two sources are small enough to use whole
EXTERNAL_WEIGHT = 0.25  # down-weight relative to official rows when fitting

# ntl_open_pool: filter leakage/quality flags, then cap per class (prioritized by n_sources)
pool = pd.read_parquet('../data/external/ntl_open_pool.parquet')
pool_clean = pool[~pool['flag_near_train_prompt'] & ~pool['flag_train_prompt_regen'] & ~pool['flag_gemini37_routing_risk']].copy()
pool_clean = pool_clean.sort_values('n_sources', ascending=False)
pool_sample = pool_clean.groupby('model', group_keys=False).head(EXTERNAL_CAP_PER_CLASS)
pool_sample = pool_sample[['id', 'text', 'model']].copy()
pool_sample['source_dataset'] = 'ntl_open_pool'

# prvsiyan + darragh1walsh: already cleaned by src/prepare_external_data.py
prvsiyan = pd.read_parquet('../data/external/prvsiyan.parquet')[['id', 'text', 'model', 'source_dataset']]
darragh = pd.read_parquet('../data/external/darragh1walsh.parquet')[['id', 'text', 'model', 'source_dataset']]

ext_sample = pd.concat([pool_sample, prvsiyan, darragh], ignore_index=True)
ext_sample['label'] = ext_sample['model'].map(label2id)
print('external sample shape:', ext_sample.shape)
print()
print('by source:')
print(ext_sample['source_dataset'].value_counts())
print()
print('by model (combined):')
print(ext_sample['model'].value_counts())""")

code("""ext_texts = ext_sample['text'].tolist()
ext_embeddings = {}
for name in MODEL_NAMES:
    print(f'extracting (external): {name}')
    ext_embeddings[name] = extract_embeddings(name, ext_texts)
    print(f'  -> shape {ext_embeddings[name].shape}')

ext_emb_concat = np.concatenate([ext_embeddings[n] for n in MODEL_NAMES], axis=1).astype(np.float32)
ext_stylo = stylometric_matrix(ext_sample['text']).astype(np.float32)
print('external emb/stylo shapes:', ext_emb_concat.shape, ext_stylo.shape)""")

md("""## 4. GroupKFold CV: [frozen embeddings | SVD-TFIDF (fit per fold) | stylometric] -> classical ML

TF-IDF/SVD still needs a fresh fit per fold (unlike the frozen embeddings) since it's fit
directly on these texts' vocabulary and would leak fold-specific words otherwise.""")

code("""SVD_DIM = 150
N_SPLITS = 5


def run_cv(clf_factory, name, use_external=False):
    gkf = GroupKFold(n_splits=N_SPLITS)
    train_scores, val_scores = [], []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, df['label'], groups=df['prompt_id'])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        tfidf_fit_texts = pd.concat([train_df['text'], ext_sample['text']], ignore_index=True) if use_external else train_df['text']
        word_vec, char_vec = build_vectorizers()
        X_tfidf_fit = fit_transform(tfidf_fit_texts, word_vec, char_vec, fit=True)
        X_tfidf_val = fit_transform(val_df['text'], word_vec, char_vec, fit=False)
        svd = TruncatedSVD(n_components=SVD_DIM, random_state=fold)
        svd_fit = svd.fit_transform(X_tfidf_fit)
        svd_val = svd.transform(X_tfidf_val)

        if use_external:
            n_off = len(train_df)
            X_train = np.concatenate([
                np.concatenate([emb_concat[train_idx], svd_fit[:n_off], stylo[train_idx]], axis=1),
                np.concatenate([ext_emb_concat, svd_fit[n_off:], ext_stylo], axis=1),
            ], axis=0).astype(np.float32)
            y_train = np.concatenate([train_df['label'].values, ext_sample['label'].values])
            is_ext = np.concatenate([np.zeros(n_off, dtype=bool), np.ones(len(ext_sample), dtype=bool)])
            sw = compute_sample_weight('balanced', y_train)
            sw[is_ext] *= EXTERNAL_WEIGHT
        else:
            X_train = np.concatenate([emb_concat[train_idx], svd_fit, stylo[train_idx]], axis=1).astype(np.float32)
            y_train = train_df['label'].values
            sw = compute_sample_weight('balanced', y_train)

        X_val = np.concatenate([emb_concat[val_idx], svd_val, stylo[val_idx]], axis=1).astype(np.float32)

        clf = clf_factory()
        clf.fit(X_train, y_train, sample_weight=sw)
        train_score = balanced_accuracy_score(y_train, clf.predict(X_train))
        val_score = balanced_accuracy_score(val_df['label'], clf.predict(X_val))
        train_scores.append(train_score)
        val_scores.append(val_score)
        print(f'  fold {fold}: train={train_score:.4f}  val={val_score:.4f}  gap={train_score - val_score:.4f}')

    print(f'{name}: train={np.mean(train_scores):.4f}  val={np.mean(val_scores):.4f} +/- {np.std(val_scores):.4f}')
    return val_scores""")

code("""lr_scores = run_cv(
    lambda: LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1),
    'LogisticRegression (official only)', use_external=False,
)""")

code("""lr_ext_scores = run_cv(
    lambda: LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1),
    'LogisticRegression (+external)', use_external=True,
)""")

code("""import xgboost as xgb

xgb_scores = run_cv(
    lambda: xgb.XGBClassifier(tree_method='hist', device='cuda'),
    'XGBoost (official only)', use_external=False,
)""")

md("""## 5. Summary

Compare against the other approaches tried so far:
- TF-IDF + LogisticRegression baseline: 0.3086 CV
- DeBERTa-v3-base full fine-tune (plain classifier head): ~0.30 CV
- DeBERTa-v3-base, `unfreeze_last_n=2` partial fine-tune: 0.3216 CV
- Fine-tuned DeBERTa embedding + SVD-TFIDF + stylometric -> LightGBM: 0.3482 CV (best so far)

This notebook's numbers tell us whether *frozen* (unfine-tuned) multi-model embeddings can
get close to the fine-tuned fusion result without any of the fine-tuning risk/cost -- and
if so, adding more/bigger frozen models here is a much safer way to keep improving than
more fine-tuning experiments.""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {
        "display_name": "Python (name_llm)",
        "language": "python",
        "name": "py_namellm",
    }
}

with open("notebooks/frozen_embeddings.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/frozen_embeddings.ipynb with", len(cells), "cells")
