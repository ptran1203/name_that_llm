# Environment

Local machine specs for this workspace root (`D:\Master\hk3\msc`). Sub-projects can assume this environment unless they state otherwise.

- OS: Windows 10 Pro, Build 19041 (x64)
- CPU: Intel Core i5-12600K (10 cores / 16 logical processors)
- RAM: 16 GB (16,158 MB)
- GPU: NVIDIA GeForce RTX 3060, 12 GB VRAM (driver 536.23, supports CUDA 12.2); integrated Intel UHD Graphics 770 also present
- Python: 3.10.6 (`python` on PATH; `python3` not aliased)
- PyTorch: 2.6.0+cu124 installed, `torch.cuda.is_available() == True` — GPU-accelerated training/inference works out of the box
- CUDA toolkit (`nvcc`, the compiler): not installed — irrelevant for pip-installed PyTorch, which bundles its own CUDA runtime; only matters if compiling custom CUDA kernels from source
- Shell: PowerShell (primary)

# Kaggle submission paths

Kaggle datasets/models this project's inference notebooks attach, by Kaggle username `ptran1203`
unless noted. Local repo has no access to any of these paths — they only resolve inside a Kaggle
notebook session.

- **Training notebooks have internet access — don't use the packaged code dataset for them,
  `git clone` the repo directly instead.** Only INFERENCE/submission notebooks need code
  delivered via a Kaggle input dataset (competition submissions typically run with internet
  disabled). So: `train_qwen_ensemble_artifacts.py`, `train_fusion_lgbm.py`, `train_qwen_lora.py`
  and any other artifact-producing/training script → clone `origin/main` fresh each run, always
  latest code, no manual sync step needed at all. `predict.py`, `predict_qwen_ensemble_artifacts.py`,
  `submit_frozen_qwen_tfidf_cluster.py`, `submit_multi_qwen_ensemble.py` and anything else run as
  the FINAL submission → needs the packaged dataset below, kept in sync via `update_code.bat`/
  the `sync-kaggle` skill, since internet won't be available to clone from at submission time.
- **Inference code dataset**: `/kaggle/input/datasets/ptran1203/name-that-llm-inference-code/`
  — a packaged copy of (a subset of) `src/`, re-uploaded manually after local changes. Currently
  holds `predict.py` + its direct dependencies (`prompt_cluster_assign.py`, `stylometric.py`,
  `train_fusion_lgbm.py`, `train.py`, `paths.py`, `exp_log.py`). Any new script run on Kaggle
  needs ITS sibling-module dependencies added here too, not just the entry script.
- **Weights dataset**: `/kaggle/input/datasets/ptran1203/name-that-llm-code-n-weights/` — a
  packaged `experiments/<run_id>/` directory (DeBERTa fold checkpoints + `fold*_fusion.joblib`
  + tokenizer), passed as `predict.py --run_dir`. Only needed by the `train_fusion_lgbm.py`/
  `predict.py` pipeline — the newer frozen-embedding pipelines
  (`submit_frozen_qwen_tfidf_cluster.py`, `submit_multi_qwen_ensemble.py`) don't use this at all;
  they extract frozen embeddings fresh and fit LR/SVC/LGBM from scratch each run.
- **Code sync script**: `scripts\update_code.bat ["commit message"]` — copies a HARDCODED file
  list from `src/` into `kaggle_code_package/`, commits the repo, then runs
  `kaggle datasets version -p kaggle_code_package` to push a new version of the
  `name-that-llm-inference-code` dataset above. Use the `sync-kaggle` skill (`.claude/skills/`)
  for this rather than redoing the steps manually — it already knows the dependency closures per
  entry script. The file list goes stale whenever a new script is added; check it's current
  before assuming a script is live on Kaggle.
- **2-notebook training/inference split**: `train_qwen_ensemble_artifacts.py` (does all the
  TRAIN-side work -- embedding extraction, OOF, calibration, blend-weight fitting, final
  classifier fitting -- then saves a bundle to `--artifacts_dir`: `config.json`,
  `class_labels.json`, `embedding_clf_<name>.joblib` per classifier, `tfidf_word_vec.joblib`/
  `tfidf_char_vec.joblib`/`tfidf_clf.joblib`, `temperatures.json`, `blend_weights.json`) +
  `predict_qwen_ensemble_artifacts.py` (loads that bundle, extracts embeddings for TEST ONLY,
  predicts, applies the saved calibration/blend/Hungarian, writes submission -- fast, never
  touches train.csv). Run the first as a "training notebook" that outputs a new Kaggle dataset
  (the artifacts dir), attach that dataset to a separate "inference notebook" that runs the
  second. Pseudo-labeling was discussed as a possible future addition to the inference script
  (concatenate high-confidence test predictions into train, rerun training) -- NOT implemented.
- **2-GPU extraction**: `submit_multi_qwen_ensemble.py --extract_only` caches embeddings for a
  model subset then exits (no classifier fitting) -- run it twice with different `--model_names`
  subsets and `CUDA_VISIBLE_DEVICES`, then one final full run hits the cache for everything.
  No `Trainer`/DataParallel/DDP anywhere in this script (plain forward-pass extraction), so none
  of TRIED.md's multi-GPU `Trainer` bugs apply -- safe to use 2 GPUs here unlike training.
  **Must use real `subprocess.Popen` in a Python cell to actually run the two in parallel** --
  Jupyter/Kaggle notebooks reject `!cmd &` background-job syntax outright
  (`OSError: Background processes not supported.`), it's an IPython restriction, not a Kaggle
  bug. `train_qwen_ensemble_artifacts.py` doesn't have `--extract_only` yet (only
  `submit_multi_qwen_ensemble.py` does) -- add it there too if this 2-notebook split also needs
  2-GPU extraction.
- **`bitsandbytes` is NOT preinstalled on Kaggle's base image** (or too old for 4-bit) -- run
  `!pip install -U "bitsandbytes>=0.46.1"` as its own cell before anything using
  `--load_in_4bit_for`.
- **Current submission command** (fusion_lgbm + cluster-assign pipeline):
  ```
  !python /kaggle/input/datasets/ptran1203/name-that-llm-inference-code/predict.py \
      --run_dir /kaggle/input/datasets/ptran1203/name-that-llm-code-n-weights --cluster_assign \
      --out submission.parquet
  ```
  Add `--cluster_resolve hungarian` to this (code already supports it) — Hungarian beat greedy
  in every test run so far (train OOF: fusion_lgbm+greedy ~0.353 flat, +hungarian ~0.386).
- **Qwen3 models** (Kaggle Models input, not a Datasets path):
  - `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1` — base decoder LM, fits a
    12GB-class GPU in fp16/bf16.
  - `/kaggle/input/models/qwen-lm/qwen-3/transformers/14b/1` — base decoder LM, bigger; needs
    4-bit.
  - Hit a CUDA OOM using `--load_in_4bit_for 14b` alone -- the 8B models' fp16 weights alone
    (~16GB) are ALSO too big for a single 16GB Kaggle GPU before any activation memory. Use
    `--load_in_4bit_for 8b,14b` (substring match catches `8b-base` and `qwen-3-embedding/8b`
    too) to quantize every model >=8B, leaving only the 4B embedding model in fp16.
  - `/kaggle/input/models/qwen-lm/qwen-3-embedding/transformers/{4b,8b}/1` — retrieval/semantic-
    similarity embedding models, NOT base decoders. Flagged as likely bad for this task: the
    same category as bge/e5/mpnet, already tested frozen and found catastrophic (0.13-0.17,
    TRIED.md) because contrastive-similarity training is backwards for authorship attribution
    (wants topic invariance; we want topic invariance stripped OUT). Don't use for embeddings
    without re-deriving why first.
- **Best validated pipeline so far** (train OOF, nested-CV blend + Hungarian): frozen
  Qwen2.5-3B-layer-7 embedding + combined TF-IDF, temperature-calibrated, weighted-blended,
  then Hungarian cluster-assignment = **0.5309** — see `submit_frozen_qwen_tfidf_cluster.py`.
  Caveat: the cluster-assignment gain is validated on train (confirmed ~150 real shared-prompt
  groups of ~18 rows exist there); whether the REAL hidden test set has the same structure is
  still unconfirmed (local `data/test.csv` is a 12-row dummy, all exact train duplicates).
