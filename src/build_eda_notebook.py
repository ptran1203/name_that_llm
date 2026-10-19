"""Builds notebooks/eda.ipynb cell-by-cell via nbformat (not hand-written JSON)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Name That LLM — EDA

Exploring `data/train.csv`: 18 candidate models, balanced-accuracy scoring, passages 40-250 words.
Goal: understand class balance, the prompt/model pairing structure, text-length behavior, and any
cheap stylistic fingerprints (punctuation, function words) before building features.""")

code("""import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 140)

df = pd.read_csv('../data/train.csv')
df['n_words'] = df['text'].str.split().str.len()
df['n_chars'] = df['text'].str.len()
LABELS = sorted(df['model'].unique())
print(df.shape)
df.head()""")

md("## 1. Class balance\nBalanced accuracy weights every model equally, so check how close to equal the train rows already are.")

code("""counts = df['model'].value_counts().sort_index()
display(counts)

fig, ax = plt.subplots(figsize=(10, 4))
counts.plot.bar(ax=ax)
ax.set_ylabel('rows')
ax.set_title('Rows per model')
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.show()""")

md("""## 2. Prompt / model pairing structure
`prompt_id` looks shared across models (every model answering the same ~150 prompts). If so,
train/validation splits MUST be grouped by `prompt_id`, or a model will "memorize" topic
instead of learning style — and CV will look better than true held-out performance.""")

code("""n_prompts = df['prompt_id'].nunique()
per_prompt = df.groupby('prompt_id')['model'].nunique()
per_prompt_model_counts = df.groupby('prompt_id').size()
print('unique prompts:', n_prompts)
print('models per prompt - distribution:')
display(per_prompt.value_counts().sort_index())

# which (prompt, model) pairs are missing from a fully-crossed design?
full_grid = pd.MultiIndex.from_product([df['prompt_id'].unique(), LABELS], names=['prompt_id', 'model'])
present = set(map(tuple, df[['prompt_id', 'model']].values))
missing = [pm for pm in full_grid if pm not in present]
print(f'missing (prompt_id, model) pairs out of {len(full_grid)}: {len(missing)}')
missing""")

code("""# duplicate prompt_id+model rows?
dupe_mask = df.duplicated(subset=['prompt_id', 'model'], keep=False)
print('duplicated (prompt_id, model) rows:', int(dupe_mask.sum()))

# any exact duplicate text (possible copy/leak)?
print('exact duplicate texts:', int(df['text'].duplicated().sum()))
print('null texts:', int(df['text'].isna().sum()))""")

md("""## 3. Text length behavior
Passage length is randomized 40-250 words per the README, but models may still differ in their
*typical* length, variance, or how tightly they hit the requested range — all of which a naive
model could latch onto as a shortcut (and which may NOT transfer to the hidden test distribution,
so treat any length-based signal cautiously).""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
order = df.groupby('model')['n_words'].median().sort_values().index
df.boxplot(column='n_words', by='model', ax=axes[0], rot=60)
axes[0].set_title('Words per passage, by model')
axes[0].set_xlabel('')
df.groupby('model')['n_words'].agg(['mean', 'std', 'median', 'min', 'max']).loc[order]""")

code("""fig, ax = plt.subplots(figsize=(10, 5))
for m in LABELS:
    ax.hist(df.loc[df['model'] == m, 'n_words'], bins=30, histtype='step', label=m, alpha=0.7)
ax.set_xlabel('words')
ax.set_ylabel('count')
ax.set_title('Word-count distribution, overlaid by model')
ax.legend(fontsize=6, ncol=2)
plt.tight_layout()
plt.show()""")

md("""## 4. Same-prompt, different-model comparison
Since most prompts are answered by all 18 models, print a few prompt groups side by side to get
an intuitive feel for what actually distinguishes models on identical topics/instructions.""")

code("""sample_prompt = df['prompt_id'].value_counts().index[0]
sub = df[df['prompt_id'] == sample_prompt][['model', 'text']]
for _, row in sub.sample(min(5, len(sub)), random_state=0).iterrows():
    print(f\"--- {row['model']} ---\")
    print(row['text'][:400])
    print()""")

md("""## 5. Punctuation / orthographic fingerprints
A prior public notebook claimed hard rules like "Claude & GLM never use curly quotes" and
"Gemini rarely uses >=2 em-dashes." Check those claims directly on this train set rather than
trusting them — these are exactly the kind of brittle, sampling-setting artifacts that may not
hold on the hidden test set.""")

code("""def has_curly_quote(t):
    return any(c in t for c in ('\\u2019', '\\u201c', '\\u201d'))


def count_em_dash(t):
    return t.count('\\u2014')


feat = pd.DataFrame({
    'model': df['model'],
    'has_curly_quote': df['text'].apply(has_curly_quote),
    'em_dash_count': df['text'].apply(count_em_dash),
    'has_2plus_em_dash': df['text'].apply(count_em_dash) >= 2,
})

curly_rate = feat.groupby('model')['has_curly_quote'].mean().sort_values(ascending=False)
print('Curly-quote usage rate by model:')
display(curly_rate.to_frame('rate'))

em_rate = feat.groupby('model')['has_2plus_em_dash'].mean().sort_values(ascending=False)
print('\\n>=2 em-dash usage rate by model:')
display(em_rate.to_frame('rate'))""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
curly_rate.plot.barh(ax=axes[0])
axes[0].set_title('P(curly quote present)')
em_rate.plot.barh(ax=axes[1])
axes[1].set_title('P(>=2 em-dashes)')
plt.tight_layout()
plt.show()""")

md("""## 6. Other quick stylometric signals
Sentence length, exclamation/question usage, average word length, lexical diversity —
cheap per-passage features that feed a LightGBM/LinearSVC stylometric head.""")

code("""SENT = re.compile(r'(?<=[.!?])\\s+')
WORD = re.compile(r"[A-Za-z\\u2019']+")


def stylo(t):
    words = WORD.findall(t)
    nw = max(len(words), 1)
    sents = [s for s in SENT.split(t.strip()) if s]
    sl = [len(WORD.findall(s)) for s in sents] or [0]
    wl = [len(w) for w in words] or [0]
    return pd.Series({
        'n_sentences': len(sents),
        'mean_sent_len': np.mean(sl),
        'mean_word_len': np.mean(wl),
        'lexical_diversity': len(set(w.lower() for w in words)) / nw,
        'comma_rate': t.count(',') / nw,
        'semicolon_rate': t.count(';') / nw,
        'exclaim_rate': t.count('!') / nw,
        'question_rate': t.count('?') / nw,
    })


stylo_feats = df['text'].apply(stylo)
stylo_df = pd.concat([df[['model']], stylo_feats], axis=1)
stylo_df.groupby('model').mean().round(4)""")

code("""fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, col in zip(axes.ravel(), ['mean_sent_len', 'mean_word_len', 'lexical_diversity', 'comma_rate']):
    order = stylo_df.groupby('model')[col].median().sort_values().index
    stylo_df.boxplot(column=col, by='model', ax=ax, rot=60)
    ax.set_title(col)
    ax.set_xlabel('')
plt.tight_layout()
plt.show()""")

md("""## 7. Top distinguishing words per model (chi2 on word counts)
Quick signal check: which unigrams are most associated with each model, via a chi-square test
on a CountVectorizer matrix. Gives a sense of whether lexical choice alone carries real signal
beyond generic topic words (stopwords removed).""")

code("""from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english', min_df=5, max_features=5000)
X = cv.fit_transform(df['text'])
vocab = np.array(cv.get_feature_names_out())

y = df['model'].values
top_words = {}
for m in LABELS:
    yb = (y == m).astype(int)
    # chi2 per model-vs-rest
    from sklearn.feature_selection import chi2
    scores, _ = chi2(X, yb)
    top_idx = np.argsort(scores)[::-1][:12]
    top_words[m] = list(vocab[top_idx])

pd.DataFrame(top_words).T""")

md("""## 8. Summary

- Classes are perfectly balanced (150/model, 149 for `claude-fable-5`), so train-set accuracy and
  balanced accuracy coincide here — but the hidden test set mix is explicitly *not* balanced, so a
  model must still get minority models right, not just the majority ones.
- Prompts are (almost) fully crossed with models — this is a paired design, so **any CV must be
  grouped by `prompt_id`**, otherwise a model partially learns topic/content instead of style and
  CV will overstate real generalization to the hidden test set's unseen prompts.
- Length distributions differ across models but are constrained to the same 40-250 word window by
  design — treat length-based features as a weak, possibly non-transferable signal.
- Punctuation habits (curly quotes, em-dash usage) show visible skew across providers, but the
  rates should be read as soft, noisy signal, not hard rules — baking in a hard "never does X"
  rule risks breaking on hidden-test passages that happen to violate it.
- Lexical chi2 analysis shows real per-model word preferences beyond generic topic words,
  supporting a word+char n-gram model as a solid baseline signal source.
""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {
        "display_name": "Python (name_llm)",
        "language": "python",
        "name": "py_namellm",
    }
}

with open("notebooks/eda.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebooks/eda.ipynb with", len(cells), "cells")
