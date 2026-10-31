@echo off
REM Syncs src/*.py (predict.py's dependency closure) into kaggle_code_package/,
REM commits the repo, and pushes a new version of the Kaggle code dataset.
REM Usage: scripts\update_code.bat ["commit/version message"]
REM First-time dataset creation is a one-off (already done): kaggle datasets create -p kaggle_code_package -q
setlocal

cd /d "%~dp0\.."

set "MSG=%~1"
if "%MSG%"=="" set "MSG=sync inference code"

if not exist "kaggle_code_package" mkdir "kaggle_code_package"
if not exist "kaggle_code_package\dataset-metadata.json" (
    echo {> "kaggle_code_package\dataset-metadata.json"
    echo   "title": "name-that-llm-inference-code",>> "kaggle_code_package\dataset-metadata.json"
    echo   "id": "ptran1203/name-that-llm-inference-code",>> "kaggle_code_package\dataset-metadata.json"
    echo   "licenses": [{"name": "CC0-1.0"}]>> "kaggle_code_package\dataset-metadata.json"
    echo }>> "kaggle_code_package\dataset-metadata.json"
)

echo === Staging code files ===
for %%F in (paths.py stylometric.py train.py train_transformer.py train_fusion_lgbm.py exp_log.py predict.py prepare_external_data.py) do (
    copy /Y "src\%%F" "kaggle_code_package\%%F" >nul
    if errorlevel 1 (
        echo Failed to copy src\%%F
        exit /b 1
    )
)
copy /Y "requirements.txt" "kaggle_code_package\requirements.txt" >nul

echo === Committing to git ===
git add src requirements.txt
git commit -m "%MSG%" -m "" -m "Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
if errorlevel 1 echo (nothing to commit, or commit skipped - continuing)

echo === Uploading new Kaggle dataset version ===
kaggle datasets version -p kaggle_code_package -m "%MSG%"
if errorlevel 1 (
    echo Kaggle upload failed.
    exit /b 1
)

echo Done.
endlocal
