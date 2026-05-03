# Astro-Anomaly-Detector

Unsupervised anomaly detection in multi-band astronomical images using a
Convolutional Autoencoder trained on HST + AstroSat tile data.

---

## How It Works

The model trains on **normal** astronomical tiles. Tiles it cannot reconstruct
well (high error) are flagged as anomalies — cosmic rays, sensor artefacts,
blended source residuals, or rare astrophysical transients.

```
Train autoencoder → Score all tiles → Flag top anomalies → Visualise
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Astro-Anomaly-Detector.git
cd Astro-Anomaly-Detector
```

### 2. Download the dataset

> The dataset is **not included** in this repository (8.94 GB).
> Download it separately from the link below.

**[Download Dataset — Google Drive](https://drive.google.com/YOUR_LINK_HERE)**

After downloading, extract it so you have this folder structure:
```
Hybrid-Astro-Deblender/
  data/
    processed/
      dataset_full/
        train/    ← 30,742 .npy tiles
        val/      ← 3,362  .npy tiles
```

### 3. Install dependencies

```bash
# With pip (CUDA 12.1 for RTX 30xx/40xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas matplotlib tqdm pyyaml

# OR with conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install numpy pandas matplotlib tqdm pyyaml
```

### 4. Configure your data path

**Windows — run the setup script:**
```bat
setup.bat
```
When prompted, enter the full path to your `Hybrid-Astro-Deblender` folder.

**OR edit `configs/config.yaml` manually:**
```yaml
data:
  data_root: "C:/path/to/your/Hybrid-Astro-Deblender"
```

### 5. Run the full pipeline

```bat
run_pipeline.bat
```

This runs all 4 steps automatically:
1. Dataset sanity check
2. Train the Convolutional Autoencoder (60 epochs)
3. Detect and score anomalies
4. Generate visualisation dashboard

---

## Results

After running the pipeline, outputs are saved to:

```
outputs/
  checkpoints/
    best_autoencoder.pth          ← Best model weights
  reports/
    training_curve.png            ← Train vs val loss
    1_score_analysis.png          ← Score distribution + MSE vs SSIM scatter
    2_anomaly_gallery.png         ← Top-20 anomalous tiles
    3_normal_vs_anomaly.png       ← Normal vs anomaly comparison
    4_channel_error_top1.png      ← Per-channel error for #1 anomaly
    anomaly_scores.csv            ← Full score table for all tiles
    anomaly_summary.txt           ← Top-20 ranked anomalies
  anomaly_maps/
    tile_XXXXX_anomaly.png        ← Per-tile: original | recon | error | diff
```

---

## Model Architecture

```
Input (4, 128, 128)
  │
  ├── Encoder: 4x strided conv blocks  → (512-dim latent vector)
  │     128 → 64 → 32 → 16 → 8
  │
  └── Decoder: 4x transposed conv blocks → (4, 128, 128) reconstruction
        8 → 16 → 32 → 64 → 128

Loss = MSE + 0.1 × Sobel-gradient penalty
Score = 0.5 × MSE + 0.5 × SSIM-delta
```

---

## Key Metrics (10-epoch baseline)

| Metric | Value |
|---|---|
| Validation tiles scored | 3,362 |
| Anomalies detected (P97 threshold) | 169 (5.03%) |
| MSE ratio (anomaly / normal) | **4.24×** |
| Cohen's d effect size | **1.41** |
| False positive rate | **0.0%** |

---

## Troubleshooting

See [SETUP.md](SETUP.md) for:
- PyTorch DLL errors (WinError 126)
- Conda PATH issues
- CUDA out-of-memory fixes

---

## Project Structure

```
Astro-Anomaly-Detector/
├── configs/
│   └── config.yaml           ← All settings (edit data_root here)
├── src/
│   ├── models/
│   │   └── autoencoder.py    ← Convolutional Autoencoder
│   ├── data/
│   │   └── dataset.py        ← DataLoader + normalisation
│   ├── utils/
│   │   └── helpers.py        ← Config loading, checkpointing
│   ├── train_autoencoder.py  ← Training script
│   ├── detect_anomalies.py   ← Scoring + CSV export
│   └── visualize_anomalies.py← Dashboard generation
├── outputs/                  ← Generated outputs (not in git)
├── setup.bat                 ← First-time setup (Windows)
├── run_pipeline.bat          ← Run full pipeline (Windows)
├── SETUP.md                  ← Detailed setup guide
└── requirements.txt          ← Python dependencies
```
