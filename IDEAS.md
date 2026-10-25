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


## 4 Mixture of experts
1. first classify the provider
2. each provider have they own models

## 5. Similarity search

- **k-NN as a classifier**: embed passages (TF-IDF or a transformer's pooled embedding) and
  classify by majority vote among nearest training neighbors, instead of / blended with a
  parametric classifier. Simple; embedding-space locality might capture style once topic is
  controlled for.
- **Exploit the paired-prompt structure via retrieval** (priority candidate): train is ~18
  responses to each of 150 shared prompts (confirmed in notebooks/eda.ipynb). If the hidden
  test set has the same structure, similarity search could detect which test rows share an
  unseen prompt by content similarity, then treat prediction as a **constrained assignment
  problem** — since each model answers a given prompt at most once, no two passages in the
  same detected cluster should get the same predicted label. Enforce with the Hungarian
  algorithm over the cluster's prediction probabilities. Stronger than the downloaded
  reference notebook's soft "topic clustering" trick since it's an actual hard constraint
  from a confirmed property of the dataset, not a heuristic. Needs checking test.csv for the
  same pairing structure before betting on it.
- **Free safety net**: similarity search between train and the hidden test set for
  near-duplicates (quote reuse, accidental leakage). Cheap, zero downside, unlikely to hit
  but free correct labels if it ever does.