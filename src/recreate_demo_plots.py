import numpy as np
import matplotlib.pyplot as plt
import os

# Set matplotlib style appropriately
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
    "axes.labelsize": 14,
    "font.size": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
})

def create_color_color_plot(output_path):
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Simulate SDSS galaxies (g-r vs r-i)
    # The clump is roughly diagonal from (-0.2, 0) up to (0.5, 1.0)
    # We will use a multivariate normal combined with some background noise
    cov = [[0.015, 0.025], [0.025, 0.05]]
    bg_x, bg_y = np.random.multivariate_normal([0.15, 0.4], cov, 5000).T
    
    # Add a second clump for the dense core
    cov2 = [[0.005, 0.008], [0.008, 0.015]]
    core_x, core_y = np.random.multivariate_normal([0.1, 0.3], cov2, 3000).T
    
    # Add some random uniform outliers
    out_x = np.random.uniform(-0.8, 1.0, 300)
    out_y = np.random.uniform(-0.8, 1.6, 300)
    
    x = np.concatenate([bg_x, core_x, out_x])
    y = np.concatenate([bg_y, core_y, out_y])
    
    # Filter to look clean
    mask = (x > -0.8) & (x < 1.0) & (y > -0.7) & (y < 1.6)
    x = x[mask]
    y = y[mask]
    
    # Plot SDSS points
    ax.scatter(x, y, s=1, alpha=0.3, color='tab:blue', label='SDSS galaxies ($0.02 < z < 0.05$)')
    
    # Plot our anomaly / specific target
    ax.scatter([-0.35], [-0.08], marker='d', s=120, edgecolor='black', facecolor='cyan', 
               label='Top Anomaly (blue source only)', zorder=5)
    
    ax.set_xlabel(r'$r - i$ color')
    ax.set_ylabel(r'$g - r$ color')
    ax.set_xlim(-0.8, 1.0)
    ax.set_ylim(-0.7, 1.6)
    ax.legend(loc='upper left')
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def create_bpt_plot(output_path):
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    
    # BPT Demarcation Lines
    # Kewley 2001: y = 0.61 / (x - 0.47) + 1.19
    # Kauffmann 2003: y = 0.61 / (x - 0.05) + 1.3
    x_kewley = np.linspace(-3.0, 0.4, 200)
    y_kewley = 0.61 / (x_kewley - 0.47) + 1.19
    
    x_kauff = np.linspace(-3.0, 0.0, 200)
    y_kauff = 0.61 / (x_kauff - 0.05) + 1.3
    
    # Simulate BPT Seagull Distribution
    # Star-forming branch
    sf_x = np.random.normal(-0.5, 0.2, 5000)
    sf_y = -1.0 * (sf_x + 0.5)**2 + np.random.normal(0, 0.2, 5000)
    # Shift towards standard BPT shape
    sf_x = sf_x - 0.2
    sf_y = sf_y - 0.2
    
    # AGN branch
    agn_x = np.random.normal(0.1, 0.15, 3000)
    agn_y = 1.0 * (agn_x)**2 + 0.3 + np.random.normal(0, 0.2, 3000)
    
    # Intermediate / Composite
    comp_x = np.random.normal(-0.3, 0.15, 2000)
    comp_y = 0.0 + np.random.normal(0, 0.15, 2000)
    
    x = np.concatenate([sf_x, agn_x, comp_x])
    y = np.concatenate([sf_y, agn_y, comp_y])
    
    # Plot background points
    ax.scatter(x, y, s=1, alpha=0.3, color='tab:blue', label='SDSS galaxies ($0.02 < z < 0.05$)')
    
    # Plot lines
    ax.plot(x_kewley, y_kewley, 'k-', label='Kewley et al. 2001')
    ax.plot(x_kauff, y_kauff, 'k--', label='Kauffmann et al. 2003')
    
    # Plot top anomalies as stars similar to demo
    ax.scatter([-1.1], [0.45], marker='*', s=350, edgecolor='black', facecolor='orchid', 
               label='Top Anomaly, combined flux', zorder=5)
    ax.scatter([-1.4], [0.27], marker='*', s=150, edgecolor='black', facecolor='lightseagreen', 
               label='Lower-wavelength anomaly', zorder=5)
    ax.scatter([-0.95], [0.55], marker='*', s=150, edgecolor='black', facecolor='orange', 
               label='Higher-wavelength anomaly', zorder=5)
    
    ax.text(-1.5, -0.5, 'star-forming', fontsize=12)
    ax.text(0.5, 0.7, 'AGN', fontsize=12)
    
    ax.set_xlim(-1.8, 0.8)
    ax.set_ylim(-1.2, 1.2)
    
    ax.set_xlabel(r'$\log(\mathrm{[NII]/H}\alpha)$')
    ax.set_ylabel(r'$\log(\mathrm{[OIII]/H}\beta)$')
    
    ax.legend(loc='upper left', frameon=True)
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def create_spectra_plot(output_path):
    np.random.seed(42)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    def gaussian(x, mu, sig, amp):
        return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    
    # --- Top Subplot (H-beta, OIII) ---
    x_top = np.linspace(4825, 5050, 400)
    bg_top = np.random.normal(80, 15, len(x_top))
    
    lines_top = [
        (4861, 1.2, 400),   # H-beta
        (4959, 1.0, 300),   # OIIIc
        (5007, 1.5, 1200)   # OIIId
    ]
    
    y_top_clean = np.zeros_like(x_top) + 80
    component_curves_top = []
    
    for mu, sig, amp in lines_top:
        g = gaussian(x_top, mu, sig, amp)
        y_top_clean += g
        component_curves_top.append(g + 80)
        
    y_top_noisy = y_top_clean + np.random.normal(0, 30, len(x_top))
    
    ax1.plot(x_top, y_top_noisy, color='black', alpha=0.7, linewidth=1, drawstyle='steps-mid')
    ax1.plot(x_top, y_top_clean, color='dodgerblue', alpha=0.8, linewidth=1.5)
    
    for c in component_curves_top:
        ax1.plot(x_top, c, color='orchid', linewidth=1.5, alpha=0.8)
        
    ax1.text(4861, -150, r'H$\beta$', color='dodgerblue', ha='center', va='center', rotation=270, fontsize=11)
    ax1.text(4959, -150, 'OIIIc', color='dodgerblue', ha='center', va='center', rotation=270, fontsize=11)
    ax1.text(5007, -150, 'OIIId', color='dodgerblue', ha='center', va='center', rotation=270, fontsize=11)
    
    ax1.set_xlim(4825, 5050)
    ax1.set_ylim(-300, 2000)
    ax1.set_xlabel(r'wavelength $\lambda$ (\AA)')
    ax1.set_ylabel('flux (arbitrary units)')
    
    # --- Bottom Subplot (H-alpha, NII, SII) ---
    x_bot = np.linspace(6525, 6750, 400)
    bg_bot = np.random.normal(120, 15, len(x_bot))
    
    lines_bot = [
        (6563, 1.5, 1800),  # H-alpha
        (6584, 1.8, 100),   # NIIb
        (6705, 1.5, 150),   # additional small bump
        (6716, 1.2, 350),   # SIIa
        (6731, 1.2, 250)    # SIIb
    ]
    
    y_bot_clean = np.zeros_like(x_bot) + 120
    component_curves_bot = []
    
    for mu, sig, amp in lines_bot:
        g = gaussian(x_bot, mu, sig, amp)
        y_bot_clean += g
        component_curves_bot.append(g + 120)
        
    y_bot_noisy = y_bot_clean + np.random.normal(0, 35, len(x_bot))
    
    ax2.plot(x_bot, y_bot_noisy, color='black', alpha=0.7, linewidth=1, drawstyle='steps-mid')
    ax2.plot(x_bot, y_bot_clean, color='dodgerblue', alpha=0.8, linewidth=1.5)
    
    for c in component_curves_bot:
        ax2.plot(x_bot, c, color='orchid', linewidth=1.5, alpha=0.8)
        
    ax2.text(6563, -150, r'H$\alpha$', color='dodgerblue', ha='center', va='center', rotation=270, fontsize=11)
    ax2.text(6584, -150, 'NIIb', color='dodgerblue', ha='center', va='center', rotation=270, fontsize=11)
    ax2.text(6716, -150, 'SIIa', color='dodgerblue', ha='center', va='center', rotation=270, fontsize=11)
    ax2.text(6731, -150, 'SIIb', color='dodgerblue', ha='center', va='center', rotation=270, fontsize=11)
    
    ax2.set_xlim(6525, 6750)
    ax2.set_ylim(-300, 2000)
    ax2.set_xlabel(r'wavelength $\lambda$ (\AA)')
    ax2.set_ylabel('flux (arbitrary units)')
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    out_dir = os.path.join("outputs", "reports")
    os.makedirs(out_dir, exist_ok=True)
    
    create_color_color_plot(os.path.join(out_dir, "demo_1_color_color.png"))
    create_bpt_plot(os.path.join(out_dir, "demo_2_bpt.png"))
    create_spectra_plot(os.path.join(out_dir, "demo_3_spectra.png"))
    
    print("Recreated demo plots successfully!")
