@echo off
REM Packages data\external\{prvsiyan,darragh1walsh}.parquet and pushes a new version of the
REM Kaggle external-data dataset, so train_transformer.py --external_paths can resolve them on
REM Kaggle via /kaggle/input/name-that-llm-external-data/ (see paths.external_data_dir()).
REM ntl_open_pool.parquet is NOT included -- it needs its own flag-filtering/capping first (see
REM the frozen_embeddings notebook), not something the generic --external_paths loader does.
REM Usage: scripts\update_external_data.bat ["version message"]
REM First-time dataset creation (one-off, not done by this script):
REM   kaggle datasets create -p kaggle_external_package --dir-mode skip
setlocal

cd /d "%~dp0\.."

set "MSG=%~1"
if "%MSG%"=="" set "MSG=sync external data"

echo === Packaging external data ===
if exist kaggle_external_package rmdir /s /q kaggle_external_package
mkdir kaggle_external_package
copy /y data\external\prvsiyan.parquet kaggle_external_package\ >nul
copy /y data\external\darragh1walsh.parquet kaggle_external_package\ >nul

python -c "import json; json.dump({'title': 'name-that-llm-external-data', 'id': 'ptran1203/name-that-llm-external-data', 'licenses': [{'name': 'CC0-1.0'}]}, open('kaggle_external_package/dataset-metadata.json', 'w'), indent=2)"

echo === Uploading new Kaggle dataset version ===
kaggle datasets version -p kaggle_external_package -m "%MSG%" --dir-mode skip --delete-old-versions
if errorlevel 1 (
    echo Kaggle upload failed. If this is the first upload, run instead:
    echo   kaggle datasets create -p kaggle_external_package --dir-mode skip
    exit /b 1
)

echo Done.
endlocal
