@echo off
:: ============================================================
:: Astro-Anomaly-Detector — First-Time Setup Script
:: ============================================================
:: Run this ONCE before running run_pipeline.bat
:: It will ask where your Hybrid-Astro-Deblender folder is
:: and update configs/config.yaml automatically.
:: ============================================================

echo.
echo  ============================================================
echo   ASTRO ANOMALY DETECTOR — First-Time Setup
echo  ============================================================
echo.
echo  This script will configure the data path for your machine.
echo.

:: ─── Step 1: Ask for the dataset root path ──────────────────
echo  Enter the FULL path to your Hybrid-Astro-Deblender folder.
echo  Example:  E:\COLLAGE\ml\research\Hybrid-Astro-Deblender
echo.
set /p DATA_ROOT="  Path: "

:: ─── Step 2: Validate the path exists ───────────────────────
if not exist "%DATA_ROOT%" (
    echo.
    echo  [ERROR] That path does not exist:
    echo    %DATA_ROOT%
    echo.
    echo  Please check the path and run setup.bat again.
    pause
    exit /b 1
)

:: ─── Step 3: Check for expected subdirectory ────────────────
if not exist "%DATA_ROOT%\data\processed\dataset_full\train" (
    echo.
    echo  [WARNING] Could not find the expected tile directory:
    echo    %DATA_ROOT%\data\processed\dataset_full\train
    echo.
    echo  Make sure the Hybrid-Astro-Deblender pipeline has been
    echo  run first to generate the processed tiles.
    echo.
    pause
    exit /b 1
)

:: ─── Step 4: Write the path into config.yaml ────────────────
:: Replace backslashes with forward slashes for YAML compatibility
set "DATA_ROOT_YAML=%DATA_ROOT:\=/%"

python -c "
import re, sys
path = sys.argv[1]
with open('configs/config.yaml', 'r') as f:
    content = f.read()
content = re.sub(r'data_root:\s*\".*?\"', 'data_root: \"' + path + '\"', content)
with open('configs/config.yaml', 'w') as f:
    f.write(content)
print('  config.yaml updated successfully.')
" "%DATA_ROOT_YAML%"

if %ERRORLEVEL% NEQ 0 (
    echo  [ERROR] Failed to update config.yaml
    pause
    exit /b 1
)

:: ─── Step 5: Verify Python + PyTorch ───────────────────────
echo.
echo  Checking Python environment ...
python -c "import torch; print('  PyTorch version:', torch.__version__); print('  CUDA available :', torch.cuda.is_available())"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [WARNING] PyTorch check failed. See SETUP.md for fix instructions.
)

:: ─── Done ────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   Setup complete! Data root configured to:
echo   %DATA_ROOT%
echo.
echo   Next step: run  run_pipeline.bat  to train the model.
echo  ============================================================
echo.
pause
