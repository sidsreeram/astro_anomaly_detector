import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Set matplotlib style to look like the LaTeX compiled paper plots
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize": 14,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
})

def calculate_z_scores(series):
    return (series - series.mean()) / series.std()

def main():
    # Load the real dataset scores
    reports_dir = os.path.join("outputs", "reports")
    cae_path = os.path.join(reports_dir, "anomaly_scores.csv")
    vae_path = os.path.join(reports_dir, "anomaly_scores_vae.csv")
    
    if not os.path.exists(cae_path) or not os.path.exists(vae_path):
        print("CSV files not found!")
        return
        
    df_cae = pd.read_csv(cae_path)
    df_vae = pd.read_csv(vae_path)
    
    # Merge on filename
    df_merged = pd.merge(df_cae, df_vae, on="Filename", suffixes=("_cae", "_vae"))
    
    # We will use Anomaly_Score as the primary score
    # To make them comparable on the same axis for raw plot (if scales differ), we just plot them raw
    raw_cae = df_merged["Anomaly_Score_cae"].values
    raw_vae = df_merged["Anomaly_Score_vae"].values
    
    # Let's see if we should use MSE instead since VAE and CAE MSE are somewhat closer
    # Actually, for the raw plots, let's just use the anomaly scores as they are.
    
    # Calculate sigmas (Z-scores)
    sigma_cae = calculate_z_scores(raw_cae)
    sigma_vae = calculate_z_scores(raw_vae)
    sigma_total = (sigma_cae + sigma_vae) / 2.0
    
    # --- PLOT 1: Replication of samp.jpeg (Raw Scores) ---
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Histograms
    bins_raw = np.linspace(min(raw_cae.min(), raw_vae.min()), max(raw_cae.max(), raw_vae.max()), 60)
    
    ax1.hist(raw_vae, bins=bins_raw, histtype='step', color='mediumblue', linestyle=':', label=r'VAE score ($s_{\mathrm{VAE}}$)', linewidth=1.5)
    ax1.hist(raw_cae, bins=bins_raw, histtype='step', color='forestgreen', linestyle='-.', label=r'CAE score ($s_{\mathrm{CAE}}$)', linewidth=1.5)
    
    ax1.set_xlabel('score (raw)')
    ax1.set_ylabel('number')
    ax1.legend(loc='upper right')
    
    # Right: Scatter
    # Sort by total score to draw high anomalies on top
    sort_idx = np.argsort(sigma_total)
    
    sc = ax2.scatter(raw_vae[sort_idx], raw_cae[sort_idx], c=sigma_total[sort_idx], 
                     cmap='plasma_r', s=2, alpha=0.5, rasterized=True)
    
    # Diagonal line
    min_val = min(raw_vae.min(), raw_cae.min())
    max_val = max(raw_vae.max(), raw_cae.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=0.5, alpha=0.5)
    
    ax2.set_xlabel(r'$s_{\mathrm{VAE}}$, VAE score (raw)')
    ax2.set_ylabel(r'$s_{\mathrm{CAE}}$, CAE score (raw)')
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label(r'$s_{\mathrm{Total}}$ ($\sigma$)', rotation=270, labelpad=15)
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(reports_dir, "cmp_paper_fig1_raw.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    
    # --- PLOT 2: Replication of sample.jpeg (Sigma Scores) ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Histograms
    bins_sigma = np.linspace(-3, 6, 60)
    
    ax3.hist(sigma_total, bins=bins_sigma, histtype='step', color='purple', linestyle='-', label=r'total score ($s_{\mathrm{Total}}$)', linewidth=1.5)
    ax3.hist(sigma_vae, bins=bins_sigma, histtype='step', color='mediumblue', linestyle=':', label=r'VAE score ($s_{\mathrm{VAE}}$)', linewidth=1.5)
    ax3.hist(sigma_cae, bins=bins_sigma, histtype='step', color='firebrick', linestyle='--', label=r'CAE score ($s_{\mathrm{CAE}}$)', linewidth=1.5)
    
    ax3.set_xlabel(r'score ($\sigma$)')
    ax3.set_ylabel('number')
    ax3.legend(loc='upper right')
    ax3.set_xlim(-3, 6)
    
    # Right: Scatter
    sc2 = ax4.scatter(sigma_vae[sort_idx], sigma_cae[sort_idx], c=sigma_total[sort_idx], 
                      cmap='plasma_r', s=2, alpha=0.5, rasterized=True)
    
    ax4.plot([-5, 15], [-5, 15], 'k-', linewidth=0.5, alpha=0.5)
    ax4.set_xlim([-5, 15])
    ax4.set_ylim([-5, 15])
    
    ax4.set_xlabel(r'$s_{\mathrm{VAE}}$, VAE score ($\sigma$)')
    ax4.set_ylabel(r'$s_{\mathrm{CAE}}$, CAE score ($\sigma$)')
    
    # Colorbar
    cbar2 = plt.colorbar(sc2, ax=ax4)
    cbar2.set_label(r'$s_{\mathrm{Total}}$ ($\sigma$)', rotation=270, labelpad=15)
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(reports_dir, "cmp_paper_fig2_sigma.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("Recreated plots saved to outputs/reports/!")

if __name__ == "__main__":
    main()
