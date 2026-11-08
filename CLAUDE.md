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
  `name-that-llm-inference-code` dataset above. The hardcoded list (inside the .bat) is
  currently: `paths.py stylometric.py train.py train_transformer.py train_fusion_lgbm.py
  exp_log.py predict.py prepare_external_data.py prompt_cluster_assign.py`.
  **This list does NOT include any of this session's new scripts** (`ensemble_utils.py`,
  `experiment_decoder_embedding.py`, `experiment_multi_embed_concat.py`, `recon_checks.py`,
  `experiment_hard_cluster_specialists.py`, `submit_frozen_qwen_tfidf_cluster.py`,
  `submit_multi_qwen_ensemble.py`, `contrastive_losses.py`, `train_qwen_lora.py`) — running the
  script as-is will NOT sync the new pipelines. Add the needed ones to that `for %%F in (...)`
  line before running it for anything beyond the existing fusion_lgbm/predict.py path.
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
  - `/kaggle/input/models/qwen-lm/qwen-3/transformers/14b/1` — base decoder LM, bigger; use
    `--load_in_4bit_for 14b` with `submit_multi_qwen_ensemble.py`.
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
