# Astro-Anomaly-Detector — Setup Guide

## Quick Start (3 Steps)

```
1. setup.bat          ← Run this FIRST (configures data path)
2. install.bat        ← Run this to install Python packages
3. run_pipeline.bat   ← Run the full training + detection pipeline
```

---

## Step 1 — Configure Your Data Path

The project needs to know where your **Hybrid-Astro-Deblender** dataset folder is.
Run the setup script and paste the full path when asked:

```bat
setup.bat
```

**When prompted, enter the full path. Examples:**

| Your OS | Example Path |
|---|---|
| Windows | `E:\COLLAGE\ml\research\Hybrid-Astro-Deblender` |
| Windows | `C:\Users\YourName\projects\Hybrid-Astro-Deblender` |

> Or manually edit `configs/config.yaml` and set:
> ```yaml
> data_root: "E:/COLLAGE/ml/research/Hybrid-Astro-Deblender"
> ```
> Use **forward slashes** `/` even on Windows.

---

## Step 2 — Install Python Dependencies

### Option A: pip (recommended if you have a venv)

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib tqdm pyyaml
```

> **CUDA 12.1** version (`cu121`) matches most RTX 30xx / 40xx cards.
> For older GPUs (CUDA 11.8): replace `cu121` with `cu118`.
> For CPU only: replace the whole URL part with just `pip install torch`.

### Option B: conda

```bat
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install numpy pandas matplotlib tqdm pyyaml
```

---

## Fixing the PyTorch DLL Error (WinError 126)

**Error message looks like:**
```
OSError: [WinError 126] The specified module could not be found
fbgemm.dll  /  caffe2_nvrtc.dll  /  libomp140.x86_64.dll
```

**This is a missing Visual C++ Redistributable. Fix it with these steps:**

### Fix 1 — Install Visual C++ Redistributable (most common fix)

Download and install from Microsoft:
- **VC++ 2015-2022 x64**: https://aka.ms/vs/17/release/vc_redist.x64.exe

Then **restart your terminal** and try again.

### Fix 2 — Reinstall PyTorch cleanly

```bat
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Fix 3 — If using Anaconda (PATH issue)

Open **Anaconda Prompt** (not regular CMD) and run:
```bat
conda activate your_env_name
python -c "import torch; print(torch.__version__)"
```

If that works, always run the project from Anaconda Prompt, not regular CMD.

### Fix 4 — Add Conda to PATH manually

```bat
set PATH=C:\ProgramData\Anaconda3\Library\bin;%PATH%
set PATH=C:\ProgramData\Anaconda3\Library\mingw-w64\bin;%PATH%
```
Replace `C:\ProgramData\Anaconda3` with your actual Anaconda installation path.

---

## Step 3 — Run the Pipeline

Once setup and install are done:

```bat
run_pipeline.bat
```

This will automatically run all 4 steps:
1. Dataset check
2. Train the autoencoder (60 epochs, ~15-20 hours with RTX GPU)
3. Detect anomalies on validation set
4. Generate visualisation dashboard

**Results will be saved to:**
```
outputs/
  checkpoints/    ← saved model weights
  reports/        ← graphs, CSV scores, summary
  anomaly_maps/   ← per-tile anomaly visualisations
```

---

## Requirements

| Software | Minimum Version |
|---|---|
| Python | 3.9+ |
| PyTorch | 2.0+ |
| CUDA (optional) | 11.8 or 12.1 |
| RAM | 8 GB |
| Disk space | 5 GB (for tiles + outputs) |

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `data_root: UNCONFIGURED` | Setup not run | Run `setup.bat` |
| `[PATH ERROR] train_dir not found` | Wrong data_root path | Re-run `setup.bat` or edit `configs/config.yaml` |
| `WinError 126 fbgemm.dll` | Missing VC++ runtime | Install VC++ Redistributable (see above) |
| `ModuleNotFoundError: torch` | PyTorch not installed | Run `pip install torch ...` |
| `CUDA out of memory` | GPU VRAM too low | Reduce `batch_size` in `configs/config.yaml` to 16 or 32 |
| `UnicodeEncodeError` | Windows cmd encoding | Use PowerShell or Windows Terminal instead of CMD |
