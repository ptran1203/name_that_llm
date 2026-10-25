@echo off
REM Repackages the most recent fusion_lgbm run (experiments/*fusion_lgbm*) and pushes a new
REM version of the Kaggle weights dataset. Full repackage + upload each time (no incremental
REM diffing), so this can take a while for the ~4GB weights bundle.
REM Usage: scripts\update_weights.bat ["version message"]
REM First-time dataset creation is a one-off (already done): kaggle datasets create -p kaggle_package --dir-mode tar
setlocal

cd /d "%~dp0\.."

set "MSG=%~1"
if "%MSG%"=="" set "MSG=sync weights"

echo === Packaging latest fusion_lgbm run ===
python src\package_for_kaggle.py --run_dir latest --out_dir kaggle_package --slug name-that-llm-code-n-weights --title "name-that-llm-code-n-weights"
if errorlevel 1 (
    echo Packaging failed, aborting.
    exit /b 1
)

echo === Uploading new Kaggle dataset version (this uploads several GB, be patient) ===
kaggle datasets version -p kaggle_package -m "%MSG%" --dir-mode tar --delete-old-versions
if errorlevel 1 (
    echo Kaggle upload failed.
    exit /b 1
)

echo Done.
endlocal
