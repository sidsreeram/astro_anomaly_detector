import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")

def generate_comparison_plots():
    print("Generating Autoencoder vs VAE comparison plots...")
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    csv_ae = os.path.join(REPORT_DIR, "anomaly_scores.csv")
    csv_vae = os.path.join(REPORT_DIR, "anomaly_scores_vae.csv")
    
    if not os.path.exists(csv_ae) or not os.path.exists(csv_vae):
        print(f"[ERROR] Missing CSV files. Please run both detection scripts first.")
        sys.exit(1)
        
    df_ae = pd.read_csv(csv_ae)
    df_vae = pd.read_csv(csv_vae)
    
    # Merge on filename
    df_merged = pd.merge(df_ae, df_vae, on="Filename", suffixes=('_AE', '_VAE'))
    
    # Extract data
    ae_scores = df_merged['Anomaly_Score_AE'].values
    vae_scores = df_merged['Anomaly_Score_VAE'].values
    ae_mse = df_merged['MSE_AE'].values
    vae_mse = df_merged['MSE_VAE'].values
    
    # Set dark theme
    plt.style.use('dark_background')
    colors = ['#00ffcc', '#ff0055']
    
    # 1. Distribution Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(ae_scores, bins=50, ax=ax1, color=colors[0], alpha=0.6, kde=True, label='Autoencoder')
    ax1.axvline(np.percentile(ae_scores, 95), color='white', linestyle='--', label='95th Pct')
    ax1.set_title("Standard Autoencoder - Score Distribution", fontsize=14)
    ax1.set_xlabel("Anomaly Score (Combined)")
    ax1.set_ylabel("Count")
    ax1.legend()
    
    sns.histplot(vae_scores, bins=50, ax=ax2, color=colors[1], alpha=0.6, kde=True, label='VAE')
    ax2.axvline(np.percentile(vae_scores, 95), color='white', linestyle='--', label='95th Pct')
    ax2.set_title("Variational Autoencoder - Score Distribution", fontsize=14)
    ax2.set_xlabel("Anomaly Score (Combined)")
    ax2.set_ylabel("Count")
    ax2.legend()
    
    out_path1 = os.path.join(REPORT_DIR, "cmp_1_score_distributions.png")
    plt.tight_layout()
    plt.savefig(out_path1, dpi=150)
    plt.close()
    
    # 2. MSE Comparison Scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    # Calculate difference magnitude and use logical masks
    diff = np.abs(vae_mse - ae_mse)
    
    # Create scatter with sizes based on difference to highlight interesting points
    sc = ax.scatter(ae_mse, vae_mse, c=diff, cmap='inferno', alpha=0.7, s=20 + diff*1000)
    
    # Add a diagonal line y=x representing identical performance
    max_val = max(ae_mse.max(), vae_mse.max())
    ax.plot([0, max_val], [0, max_val], 'w--', alpha=0.5, label='y=x (Equal MSE)')
    
    ax.set_title("Mean Squared Error (MSE) Comparison: AE vs VAE", fontsize=15)
    ax.set_xlabel("Standard Autoencoder MSE", fontsize=12)
    ax.set_ylabel("Variational Autoencoder MSE", fontsize=12)
    plt.colorbar(sc, label='Absolute MSE Difference')
    ax.legend(loc='lower right')
    
    out_path2 = os.path.join(REPORT_DIR, "cmp_2_mse_scatter.png")
    plt.tight_layout()
    plt.savefig(out_path2, dpi=150)
    plt.close()
    
    # 3. Bar chart showing top 20 anomaly agreement
    ae_top20 = set(df_merged.sort_values('Anomaly_Score_AE', ascending=False).head(20)['Filename'])
    vae_top20 = set(df_merged.sort_values('Anomaly_Score_VAE', ascending=False).head(20)['Filename'])
    
    overlap = len(ae_top20.intersection(vae_top20))
    only_ae = len(ae_top20 - vae_top20)
    only_vae = len(vae_top20 - ae_top20)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(['In Both Top 20', 'Only in AE Top 20', 'Only in VAE Top 20'], 
                  [overlap, only_ae, only_vae], 
                  color=['#aaaaaa', colors[0], colors[1]])
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title("Top 20 Anomalies Agreement", fontsize=15)
    ax.set_ylabel("Number of Tiles")
    ax.set_ylim(0, 25)
    
    out_path3 = os.path.join(REPORT_DIR, "cmp_3_top20_agreement.png")
    plt.tight_layout()
    plt.savefig(out_path3, dpi=150)
    plt.close()
    
    # Generate text summary report
    rep_path = os.path.join(REPORT_DIR, "comparison_summary.txt")
    with open(rep_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL COMPARISON SUMMARY: AUTOENCODER vs VAE\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total tiles evaluated: {len(df_merged)}\n\n")
        
        f.write("1. STATISTICS\n")
        f.write(f"  Standard Autoencoder:\n")
        f.write(f"    Mean MSE:  {ae_mse.mean():.6f}\n")
        f.write(f"    Max MSE:   {ae_mse.max():.6f}\n")
        f.write(f"    Score Std: {ae_scores.std():.6f}\n\n")
        
        f.write(f"  Variational Autoencoder (VAE):\n")
        f.write(f"    Mean MSE:  {vae_mse.mean():.6f}\n")
        f.write(f"    Max MSE:   {vae_mse.max():.6f}\n")
        f.write(f"    Score Std: {vae_scores.std():.6f}\n\n")
        
        f.write("2. TOP 10 ANOMALIES (By Filename)\n")
        ae_t10 = df_merged.sort_values('Anomaly_Score_AE', ascending=False).head(10)['Filename'].tolist()
        vae_t10 = df_merged.sort_values('Anomaly_Score_VAE', ascending=False).head(10)['Filename'].tolist()
        
        f.write(f"  {'Rank':<5} | {'Autoencoder':<20} | {'VAE':<20}\n")
        f.write("  " + "-"*50 + "\n")
        for i in range(10):
            f.write(f"  {i+1:<5} | {ae_t10[i]:<20} | {vae_t10[i]:<20}\n")
            
        f.write(f"\n3. INTERPRETATION\n")
        f.write(f"  * Overlap in Top 20: {overlap}/20 ({overlap/20*100:.1f}%)\n")
        if vae_mse.max() > ae_mse.max():
             f.write(f"  * VAE causes higher maximum reconstruction error ({(vae_mse.max()/ae_mse.max()):.1f}x higher) on anomalous tiles.\n")
             f.write(f"    This suggests VAE is MORE strict about the 'normal' manifold, failing harder on anomalies.\n")
        
    print(f"Generated comparison plots in {REPORT_DIR}")
    print(f"  -> {out_path1}")
    print(f"  -> {out_path2}")
    print(f"  -> {out_path3}")
    print(f"  -> {rep_path}")

if __name__ == "__main__":
    generate_comparison_plots()
