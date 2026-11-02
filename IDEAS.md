# Ideas backlog

Not yet implemented / evaluated. Notes on feasibility and risk for later.

## 1. Data augmentation by combining same-model passages (mixup-style)

Synthesize extra same-class training examples by combining two passages from the *same*
model (so the label stays valid), to cheaply grow the effective dataset beyond 150 rows/class.

- **Feature-space mixup** (safer): interpolate two same-class examples' TF-IDF vectors or
  transformer pooled embeddings, train on the blend. Well-studied for text (manifold mixup);
  doesn't touch raw text, so no risk of creating unnatural sentences.
- **Text-level splicing** (riskier): concatenate/interleave sentences from two same-model
  passages into a new synthetic one. Risk: sentence-boundary and length-based stylometric
  features (mean_sent_len, n_sentences, etc.) get distorted by the splice point in ways a real
  single response never would — could teach the model a splice artifact instead of real style.
- Validate with the existing GroupKFold-by-prompt_id CV before trusting any gain — augmentation
  that only helps train-loss but not held-out balanced accuracy isn't worth the complexity.

> **TRIED — negative, see TRIED.md.** Feature-space mixup on TF-IDF tested at two multipliers,
> both below the plain baseline. Text-level splicing never implemented.

## 2. Stack a fine-tuned encoder (RoBERTa/DeBERTa) with TF-IDF for ensembling

(Note: RoBERTa/DeBERTa are encoder-only, not decoder — worth keeping straight since decoder-only
vs encoder-only was already a discussion point here.)

Two ways to combine them:
- **Model-level blend** (what the downloaded reference notebook does): train each model
  separately, fit a small meta-learner (logistic regression / L-BFGS weights) on out-of-fold
  predictions from both. Safer with this little data — less to overfit, each model keeps its
  own inductive bias.
- **Feature-level fusion**: concatenate the encoder's pooled embedding with the TF-IDF vector
  into one classifier's input. Riskier — the sparse high-dim TF-IDF space and dense embedding
  space have very different scales, and a single classifier on a tiny dataset (2699 rows) is
  more prone to overfitting the concatenated space than two independent models blended after.
- Recommendation if/when we get here: start with the model-level blend, it's what's already
  validated by prior public solutions on this exact problem.

> **Feature-level fusion variant TRIED — succeeded.** `train_fusion_lgbm.py`: fine-tuned
> DeBERTa embedding + SVD-compressed TF-IDF + stylometric -> LightGBM. 0.3482 CV / 0.353 LB,
> our best result overall. The "riskier" option here actually won — key difference from the
> plain concatenation that failed later (see idea 1's TRIED note and TRIED.md) is the TF-IDF
> side was SVD-compressed first, not fed in raw/sparse. Model-level blend still untried.

## 3. Few-shot / in-context classification at inference time

Prompt an LLM at inference with a handful of labeled exemplar passages per candidate model,
then ask it to classify the new passage — effectively retrieval-augmented in-context learning
instead of a trained classifier head.

- Interesting because it's a genuinely different signal source: a big pretrained LLM may have
  implicit "meta-knowledge" of how GPT/Claude/Gemini/etc. write, picked up from the web
  discourse about these models during its own pretraining — something no stylometric feature
  or fine-tuned encoder has access to.
- Costly: ~4000 test passages x enough exemplars across 18 classes = long prompts, slow/expensive
  per-row inference compared to a trained classifier's single forward pass.
- Uncertain reliability: LLMs are not reliably good at *explicit* stylometric comparison tasks
  unless they have strong priors about these specific models; needs an actual CV check (grouped
  by prompt_id, same as everything else) before trusting it over the cheaper methods.
- Best framed as one more ensemble member to test, not a replacement for the feature-based
  models.


## 4. Mixture of experts — TRIED, see TRIED.md

~~1. first classify the provider~~
~~2. each provider have they own models~~

Implemented as a soft family-gate (base classifier + per-family re-ranker), not a hard
provider-then-model cascade. No CV improvement observed — see TRIED.md for the result and a
theory on why (the gates likely inherit the same p>>n overfitting as the base classifier).

## 5. Similarity search

- **k-NN as a classifier**: embed passages (TF-IDF or a transformer's pooled embedding) and
  classify by majority vote among nearest training neighbors, instead of / blended with a
  parametric classifier. Simple; embedding-space locality might capture style once topic is
  controlled for.
- **Exploit the paired-prompt structure via retrieval** — ~~priority candidate~~ **downgraded,
  see below**: train is ~18 responses to each of 150 shared prompts (confirmed in
  notebooks/eda.ipynb). The idea was: if hidden test has the same structure, similarity search
  could detect which test rows share an unseen prompt, then treat prediction as a **constrained
  assignment problem** (Hungarian algorithm) since each model answers a given prompt at most
  once. README confirms this assumption is false for test: "the number of test passages per
  model is not equal, and the mix of models differs between the public and private
  leaderboard" -- contradicts the clean "every model answers every prompt exactly once"
  structure this depended on. The hard-constraint version is very likely not viable; the
  near-duplicate-detection safety net (last bullet below) is unaffected by this and still fine.
- **Free safety net**: similarity search between train and the hidden test set for
  near-duplicates (quote reuse, accidental leakage). Cheap, zero downside, unlikely to hit
  but free correct labels if it ever does.

## 6. Few-shot / metric-learning: pairwise same-model verification

Instead of direct 18-way softmax classification, reframe as "do these two passages have the
same author?" (binary verification), then classify a new passage by comparing it against a
small stored reference set per model (e.g. 5 reference passages/model = a 90-passage
"database") and predicting whichever model's references it matches best. This is standard
few-shot metric learning (prototypical-networks style), not a new idea invented here, but
untried on this problem so far.

- **Why it could help with so little data**: pairwise framing turns ~150 examples/class into
  a combinatorially larger training set — same-model pairs plus (critically) hard
  different-model pairs. Comparable to how verification/metric-learning tasks are used in
  face recognition and authorship-verification literature specifically because per-class
  example counts are too small for plain softmax classifiers to learn cleanly.
- **Must sample hard negatives deliberately**: random different-class pairs are mostly
  "easy" (different provider families, obviously different). The pairs that actually matter
  are the ones matching our confusion-matrix findings — claude-vs-glm, and sibling models
  within a family (gemini-3.6/3.7/3.8-flash, gpt-5.6-luna/sol/terra, claude-opus-4-6/4-8).
  A pair-sampler that oversamples exactly those hard negatives would directly target the
  weakest point of every model we've tried so far.
- **Inference-time reference set matters**: could store raw reference passages, or learned
  per-model prototype embeddings (centroid of several references) — prototypes are cheaper
  at inference (one comparison per model instead of k) and less noisy than any single
  reference passage.
- **Engineering cost is real**: pair generation, a similarity/distance network (could reuse
  existing frozen or fine-tuned embeddings as the base, only need to learn a distance/
  similarity function on top), and inference-time nearest-prototype matching — more moving
  parts than a direct classifier. CV also needs care: reference passages must come from the
  train side of each fold only, same leakage discipline as the external-data work.
- Best treated as a parallel ensemble member / experiment once the simpler stacking and
  external-data wins are banked, not a replacement for them.

> **Step 1 TRIED — negative, see TRIED.md.** Zero-training cosine-similarity-to-reference-set
> baseline scored 0.17-0.20 across variants, well below trained classifiers (~0.39-0.41). Per
> the staged plan, this was the gate for Step 2 (learned pairwise verifier) — result says raw
> embedding similarity doesn't carry enough signal on its own, so Step 2 was not attempted.

## 7. NBSVM (Naive Bayes-weighted TF-IDF)

Scale each TF-IDF feature by its Naive Bayes log-count ratio before feeding it to Logistic
Regression, instead of using raw/plain TF-IDF weights as `train.py` does now. Wang & Manning
2012 ("Baselines and Bigrams"), popularized via Jeremy Howard's "NB-SVM strong linear baseline"
Kaggle kernel for the Jigsaw Toxic Comment competition. Cheap: one extra transform step before
`.fit()`, no architecture change. See COMPETITION_RESEARCH.md for sources.

## 8. Corpus-specific BPE tokenizer (trained from scratch) for TF-IDF

We confirmed reusing a *pretrained* transformer's subword tokenizer (DeBERTa-v3) for TF-IDF
gives +0.03 balanced accuracy (0.3382 vs 0.3086, see TRIED.md / experiments). The Kaggle
LLM-Detect-AI-Generated-Text winners did something related but distinct: train a BPE tokenizer
*from scratch* on the competition's own corpus (train+test combined, unsupervised — no leakage
risk since no labels involved), then TF-IDF over those corpus-specific subword tokens. Different
lever from what we just tried — captures vocabulary specific to *our* 18 models' outputs rather
than whatever DeBERTa happened to learn from its own pretraining data. Cheap to test
(HuggingFace `tokenizers` library).

> **TRIED (fold 0 only) — negative, see TRIED.md.** Train.csv-only: 0.3296. Train.csv + external
> data: 0.3222 (worse, counter to the hypothesis that more data helps the from-scratch vocab).
> Neither beat reusing DeBERTa's pretrained subword tokenizer (0.3352 on the same fold). Single-
> fold, not exhaustively tuned (one vocab_size) — directionally discouraging, not fully confirmed.

## 9. Punctuation n-grams (order-preserving) as a dedicated feature channel

Our stylometric features count punctuation *rates* (comma_rate, semicolon_rate, etc.) — loses
sequence/order information. PAN authorship-attribution research found punctuation n-grams (char
n-grams computed over just the extracted punctuation stream, preserving order) add signal beyond
rate-based features and beyond general char n-grams (where punctuation gets drowned out in a 3-5
char window dominated by letters). Cheap: regex-extract the punctuation-only string per passage,
run a small char-n-gram TfidfVectorizer over it, concatenate into the existing fusion. Directly
relevant to the claude<->glm confusion given punctuation is already our strongest known signal
there (semicolon_rate, curly quotes).

## 10. Pseudo-label external unlabeled data (not the competition test set)

Already deprioritized pseudo-labeling our own ~4000-row test set directly (risk of reinforcing
existing blind spots on hard classes at ~0.32-0.35 balanced accuracy). Jigsaw Multilingual's
winners did something more specific: pseudo-label a *separate* unlabeled corpus with a trained
model, then add high-confidence pseudo-labels as extra weighted training data — same discipline
already applied to external data (train-fold-only, down-weighted), just generating labels
ourselves instead of relying on a third party. Could apply to unlabeled/low-confidence rows in
`ntl_open_pool.parquet` or any other unlabeled LLM-text source.