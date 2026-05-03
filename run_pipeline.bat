@echo off
:: Switch console to UTF-8 to prevent UnicodeEncodeError
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
:: ============================================================
:: Astro-Anomaly-Detector — Full Pipeline Runner (Windows)
:: ============================================================
::
:: This script runs the complete pipeline in order:
::   Step 1: Verify the dataset can be found
::   Step 2: Train the Convolutional Autoencoder
::   Step 3: Detect anomalies and score all tiles
::   Step 4: Generate visualisation dashboard
::
:: Activate your virtual environment before running:
::   .venv\Scripts\activate
:: then:
::   run_pipeline.bat
:: ============================================================

echo.
echo  ============================================================
echo   ASTRO ANOMALY DETECTOR — FULL PIPELINE
echo  ============================================================
echo.

:: ─── Step 1: Sanity check ───────────────────────────────────
echo [Step 1/4] Checking dataset ...
python -c "import os; d='../Hybrid-Astro-Deblender/data/processed/dataset_full/train'; files=[f for f in os.listdir(d) if f.endswith('.npy')] if os.path.exists(d) else []; print(f'  Found {len(files)} training tiles')"

:: ─── Step 2: Train ──────────────────────────────────────────
echo.
echo [Step 2/4] Training Autoencoder ...
python src/train_autoencoder.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Training failed. Exiting.
    exit /b 1
)

:: ─── Step 3: Detect ─────────────────────────────────────────
echo.
echo [Step 3/4] Detecting Anomalies ...
python src/detect_anomalies.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Anomaly detection failed. Exiting.
    exit /b 1
)

:: ─── Step 4: Visualise ──────────────────────────────────────
echo.
echo [Step 4/4] Generating Visualisation Dashboard ...
python src/visualize_anomalies.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Visualisation failed. Exiting.
    exit /b 1
)

echo.
echo  ============================================================
echo   PIPELINE COMPLETE!
echo   Results are in outputs/reports/  and outputs/anomaly_maps/
echo  ============================================================
echo.
pause
