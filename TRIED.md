# Tried, didn't help

Negative/neutral results, kept so we don't re-attempt them later without a genuinely new angle.
(See IDEAS.md for the untried backlog, and experiments/ + `compare_experiments.py` for the raw
numbers behind the ones that *did* work.)

## Mixup augmentation (feature-space, on TF-IDF)
`experiment_mixup.py`. Interpolating two same-class TF-IDF vectors at multiplier 1.0 and 2.0:
0.3049 and 0.3012 CV, both *below* the plain baseline (0.3086). Cause: TF-IDF is topic-driven,
so blending two same-model passages on different topics makes an unrealistic synthetic point
rather than real stylistic variation. Text-level splicing (never implemented) would likely be
worse for the same reason, plus it directly distorts the sentence-length stylometric features.

## Full fine-tuning of DeBERTa-v3 (all layers trainable)
`train_transformer.py` default mode. Overfits badly on ~2160 rows/fold -- train_loss drops below
1 within a few epochs while eval_loss stays near the random-guess floor (~2.6-2.9). CV ~0.30,
no better than the plain TF-IDF baseline despite being "deep learning." Fixed by
`--unfreeze_last_n`, which freezes everything but the last N encoder layers + head: `n=2` on
deberta-v3-base reached 0.3216 CV, 3x lower fold-to-fold variance than full fine-tune.

## Boosted trees (XGBoost / LightGBM) on dense embedding features, no regularization tuning
`frozen_embeddings.ipynb`. Default `XGBClassifier()` on the concatenated frozen embeddings:
train=1.0000 (perfect memorization) every fold, val~0.39-0.41. Severe overfitting from zero
depth/regularization constraints. Consistent with the TF-IDF-era finding that linear models
beat trees here -- small n (~120 rows/class), moderate/high p, and embeddings that are plausibly
already close to linearly separable (the premise behind "linear probing" as a technique) all
favor a regularized linear model over trees. The one place boosting *did* work (fine-tuned
DeBERTa embedding + SVD-TFIDF + stylometric -> LightGBM, 0.3482 CV, our best result) only worked
because of heavy hand-tuned regularization (`num_leaves=15, min_child_samples=10, subsample=0.8,
colsample_bytree=0.5, reg_lambda=1.0`) specifically to fight this same failure mode -- a
naive/default boosting config is not a fair comparison and reliably loses.

## Concatenating all frozen embeddings + SVD-TFIDF + stylometric into one LogisticRegression
`frozen_embeddings.ipynb`, concat version of `run_cv`. Going from 3 to 5-6 embedding models
pushed the feature count to ~3000-4200 dims against ~2160 train rows/fold (p > n). Even
LogisticRegression (`C=1.0`) hit train=1.0000, val~0.39-0.41, gap~0.59-0.61 -- the "linear
models don't overfit as easily" assumption breaks down once dimensionality outgrows the row
count by this much.
- **Adding external data to the same concat approach helps a lot**: with ~7.6k external rows
  blended in (train side only, down-weighted), gap dropped to ~0.35-0.37 and val rose to
  ~0.41-0.44 (2/5 folds, run interrupted before completing -- re-run for the full mean before
  fully trusting this).
- **Stacking fixes it even more directly**: per-source base LR -> small meta-LR on ~251-dim
  OOF probabilities (instead of ~3000+ raw concatenated dims) dropped the gap to ~0.10-0.13,
  val ~0.42 (2/5 folds, also interrupted -- re-run for the full mean).

## Mixture of experts (family-specific soft gate on top of the base classifier)
`frozen_embeddings.ipynb`, `run_cv_moe`. Base 18-way LogisticRegression + per-family
(Claude/GPT/Gemini/GLM) gate classifiers, re-ranking only within the family the base model
already predicted (gate_lambda=0.5). Confirmed negative: fold 0 delta -0.0093, fold 1 delta
-0.0019 (run interrupted after 2 folds, but the pattern is consistent -- small, real underperformance,
not noise). Plausible reason: the gate classifiers are trained on the same overfitting-prone
feature space as the base model (same p>>n problem, fewer rows per gate since restricted to one
family), so they likely inherit the overfitting rather than fixing the confusion they were meant
to address.

## Few-shot Step 1: zero-training cosine-similarity-to-reference-set classifier
`frozen_embeddings.ipynb`, `run_cv_prototype`. Pure nearest-prototype/neighbor matching on
frozen embeddings, no learned classifier at all. All variants well below trained classifiers:
- All-refs centroid: 0.2049 +/- 0.0167
- 5-shot centroid: 0.1441 +/- 0.0162
- All-refs max-similarity: 0.1733 +/- 0.0164

(For reference, random guessing on 18 classes is ~0.056, and the trained LR baseline is
~0.39-0.41 -- so there's *some* signal, just much weaker than a fitted classifier gets from the
same embeddings.) More references beats both max-sim and literal 5-shot, consistent with "more
reference data helps," but even the best variant isn't remotely competitive. Per the staged plan
in IDEAS.md #6, Step 1 was meant to gate whether Step 2 (a learned pairwise verifier with
deliberate hard-negative sampling) is worth building -- this result says **no**, raw embedding
similarity doesn't carry enough same/different-model signal on its own. Step 2 not attempted.

## Label smoothing (0.1) in train_transformer.py, on Kaggle

`train_transformer.py`'s custom loss. `--label_smoothing 0.1` + `--unfreeze_last_n 1` (fp32,
single-GPU, confirmed via `CUDA_VISIBLE_DEVICES=0`, no external data or `family_aux_weight`
needed to reproduce) hit a NaN gradient explosion by epoch 2 on Kaggle: `grad_norm: nan`,
`loss: 0`, then `eval_loss: nan` / `eval_balanced_accuracy: 0.05556` (== 1/18 -- every prediction
had already collapsed to one constant class, i.e. the weights were NaN before eval even ran).
Same config with `--label_smoothing 0` (everything else identical) trained clean. Same command
reportedly fine on the user's local machine too, so this looks like an environment-specific
(Kaggle fp32, this transformers/torch version, or the `unfreeze_last_n=1`-sized trainable set
specifically) interaction, not a universal label-smoothing bug -- but since label smoothing is
normally a *stabilizer*, not a destabilizer, the mechanism is genuinely unclear and wasn't root-
caused (ruled out along the way: DDP -- reproduces single-GPU too; the family-aux loss's
clamp+log formulation -- reproduces with `family_aux_weight=0`; external data -- reproduces
without it). Default changed to `--label_smoothing 0` (off); re-enable only with a `--max_folds 1`
quick-check on the actual target environment first.

## Multi-GPU (DataParallel / DDP) training on Kaggle, in general

Three separate, never-root-caused instability bugs hit on this exact script on Kaggle whenever
more than one GPU is visible -- none reproduce on single GPU with the same config:
1. **fp16 + GradScaler crash**: `ValueError: Attempting to unscale FP16 gradients`, both naive
   `DataParallel` and proper `accelerate launch --multi_gpu` DDP. Worked around by defaulting
   `--precision` to fp32.
2. **Eval-gather misalignment**: a confirmed-good config (fine single-GPU) scored 0.0056 -- below
   the random floor -- under DDP. Likely a label/prediction misalignment during distributed-eval
   gather.
3. **NaN gradient explosion independent of label smoothing**: `--label_smoothing 0` +
   `--unfreeze_last_n 1` (fp32) trains clean on single GPU (confirmed via `CUDA_VISIBLE_DEVICES=0`)
   but still NaNs by epoch 2 on two GPUs, config otherwise identical. So this is a second,
   independent multi-GPU NaN trigger on top of #1 (the label-smoothing entry above is also
   independent of GPU count -- it reproduces on single GPU too).

No live CUDA debugging access to root-cause any of these properly. Pragmatic policy: **always
train this script single-GPU on Kaggle** (`CUDA_VISIBLE_DEVICES=0`) -- a runtime warning in
`train_transformer.py` now fires if `torch.cuda.device_count() > 1`. Partial fine-tuning
(`--unfreeze_last_n`) is fast enough single-GPU that the multi-GPU speedup isn't worth chasing
these bugs further right now.
