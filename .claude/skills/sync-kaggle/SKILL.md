---
name: sync-kaggle
description: Commit changed src/ code and push a new version of the Kaggle inference-code dataset for this project (name-that-llm). Use whenever the user asks to commit+sync, "update Kaggle", or push code changes to the Kaggle dataset.
---

# Sync code to Kaggle

This project ships inference code to Kaggle as a packaged dataset
(`ptran1203/name-that-llm-inference-code`), via `scripts\update_code.bat`. That script copies a
HARDCODED file list from `src/` into `kaggle_code_package/`, commits the repo locally, then runs
`kaggle datasets version` to push a new dataset version. It does NOT push to the GitHub remote —
that's a separate step if wanted.

## Steps

1. `git status` / `git diff` to see what actually changed.
2. Open `scripts\update_code.bat` and check the `for %%F in (...)` file list against what
   changed. This goes stale often — any new script (or a script whose new imports pull in a
   module not yet listed) needs adding here. Current known entry points and their dependency
   closures:
   - `predict.py` → `paths.py stylometric.py train.py train_transformer.py
     train_fusion_lgbm.py exp_log.py prepare_external_data.py prompt_cluster_assign.py`
   - `submit_frozen_qwen_tfidf_cluster.py` → adds `ensemble_utils.py
     experiment_decoder_embedding.py`
   - `submit_multi_qwen_ensemble.py` → adds `experiment_multi_embed_concat.py` (also pulls in
     the two above)
   If unsure whether a module is needed, `grep -n "^from \|^import " <script>.py` and check
   which hits resolve to a sibling file in `src/` rather than a pip package.
3. Add any missing filenames into the `for %%F in (...)` line, and add the entry script itself
   if it's new.
4. Stage and commit (`git add src requirements.txt`, plus `scripts/update_code.bat` if edited)
   with a concise message, ending with the attribution lines given in the conversation's
   system-reminder.
5. Run the sync: from the repo root, `cmd /c scripts\update_code.bat "commit message"` (same
   message is reused for both the git commit and the Kaggle dataset version note). Check the
   exit code — it prints `Kaggle upload failed.` and exits non-zero on failure.
6. Report back which files actually got packaged (the script's own "=== Staging code files ==="
   output lists them) and whether the Kaggle upload succeeded.

## Notes

- Needs the `kaggle` CLI authenticated (`kaggle.json` already configured) — if it fails on an
  auth/403 error, say so plainly rather than retrying the same command.
- This only updates the CODE dataset, never the WEIGHTS dataset
  (`name-that-llm-code-n-weights`) — that's a separate, manually-packaged trained-model
  artifact, out of scope for this skill.
- `update_code.bat` only commits locally; if the user also wants `origin` updated, run
  `git push` as an explicit separate step after confirming with them (pushing to a shared
  remote is a user-confirm action, not implied by "sync to Kaggle").
