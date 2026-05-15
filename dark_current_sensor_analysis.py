import numpy as np
from astropy.io import fits
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import sigmaclip
from scipy.odr import ODR, Model, RealData
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ====================== CONFIG ======================
CAMERA_DESCRIPTION = "ZWO ASI183MM Pro S/N 262d950d2d010900 - Dark Current Characterization"
ROOT_DIR = Path(r"D:/Astrophotography/Dark Library")
TEMPS = ["-10.00c", "-20.00c"]
TIMES = ["0.00s", "1.00s", "2.00s", "5.00s", "10.00s", "20.00s", "50.00s", "100.00s", "200.00s", "500.00s"]
SIGMA = 3.0
ITERS = 5

# Date for filename
today_str = datetime.now().strftime("%Y-%m-%d")
OUTPUT_PNG = f"dark_library_analysis_{today_str}.png"
# ===================================================

def load_fits(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[0].data.astype(np.float64)
    return data

def process_pair(d1, d2):
    S = (d1 + d2) / 2.0
    Diff = d1 - d2
    
    # Sigma-clipped mean on full frame for S
    clipped_S, _, _ = sigmaclip(S, low=SIGMA, high=SIGMA)
    mean_S = np.mean(clipped_S)
    
    # Sigma-clipped variance on full frame for Diff
    clipped_Diff, _, _ = sigmaclip(Diff, low=SIGMA, high=SIGMA)
    var_Diff = np.var(clipped_Diff, ddof=1)
    
    return mean_S, var_Diff

def weighted_linear_fit(x, y, yerr):
    """Weighted linear fit with ODR for proper error propagation"""
    def linear_func(p, x):
        return p[0] * x + p[1]
    
    model = Model(linear_func)
    data = RealData(x, y, sx=None, sy=yerr)
    odr = ODR(data, model, beta0=[1.0, 0.0])
    output = odr.run()
    return output.beta[0], output.beta[1], output.sd_beta[0], output.sd_beta[1], output.res_var

# ====================== MAIN ANALYSIS ======================
results = {}
signal_data = {}   # For combined plotting
variance_data = {} # For combined plotting

total_pairs = len(TEMPS) * len(TIMES) * 50
print(f"Starting analysis — {len(TEMPS)} temperatures × {len(TIMES)} exposures × 50 pairs = {total_pairs} pairs total\n")

with tqdm(total=total_pairs, desc="Processing image pairs", unit="pair",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
    
    for temp in TEMPS:
        print(f"\nProcessing temperature: {temp}")
        mean_S_dict = {}
        var_Diff_dict = {}
        t_values = []
        
        for t_str in TIMES:
            t = float(t_str[:-1])
            t_values.append(t)
            folder = ROOT_DIR / temp / "DARK" / "Dark" / t_str
            fits_files = sorted(list(folder.glob("*.fits")))
            if len(fits_files) != 100:
                raise ValueError(f"Expected 100 files in {folder}, got {len(fits_files)}")
            
            mean_list = []
            var_list = []
            
            for i in range(0, 100, 2):
                d1 = load_fits(fits_files[i])
                d2 = load_fits(fits_files[i+1])
                mean_S, var_Diff = process_pair(d1, d2)
                mean_list.append(mean_S)
                var_list.append(var_Diff)
                pbar.update(1)
                
            mean_S_dict[t] = np.array(mean_list)
            var_Diff_dict[t] = np.array(var_list)
            
            print(f"  {t:6.2f}s : mean_S = {np.mean(mean_list):.2f} ± {np.std(mean_list)/np.sqrt(50):.4f} ADU")
        
        # === Dark Current Fit ===
        times_arr = np.array(t_values)
        mean_means = np.array([np.mean(mean_S_dict[t]) for t in t_values])
        sem_means = np.array([np.std(mean_S_dict[t])/np.sqrt(50) for t in t_values])
        
        slope_dc, bias, slope_err, bias_err, chi2_dc = weighted_linear_fit(times_arr, mean_means, sem_means)
        
        print(f"\nDark Current Fit ({temp}):")
        print(f"  Dark rate = {slope_dc:.5f} ± {slope_err:.5f} ADU/s/pixel")
        print(f"  Bias      = {bias:.3f} ± {bias_err:.3f} ADU")
        
        # Store for plotting
        signal_data[temp] = {
            'times': times_arr,
            'means': mean_means,
            'sem': sem_means,
            'slope': slope_dc,
            'bias': bias
        }
        
        # === Gain & Read Noise Fit ===
        bias_corrected = []
        var_norm = []
        var_norm_err = []
        
        for t in t_values:
            m50 = mean_S_dict[t]
            v50 = var_Diff_dict[t]
            bc = m50 - bias
            vn = v50 / 2.0
            
            bias_corrected.extend(bc)
            var_norm.extend(vn)
            
            # FIXED: repeat the scalar error for all 50 points of this exposure
            sem_var_norm = np.std(v50) / (2.0 * np.sqrt(50))
            var_norm_err.extend([sem_var_norm] * len(v50))
        
        bc_arr = np.array(bias_corrected)
        vn_arr = np.array(var_norm)
        vn_err = np.array(var_norm_err)
        
        mask = bc_arr > 0.1
        slope_gain, rn_var, sg_err, rn_err, chi2_gain = weighted_linear_fit(
            bc_arr[mask], vn_arr[mask], vn_err[mask])
        
        gain = 1.0 / slope_gain
        gain_err = sg_err / (slope_gain ** 2)
        read_noise_adu = np.sqrt(rn_var)
        read_noise_adu_err = rn_err / (2 * read_noise_adu) if read_noise_adu > 0 else 0.0
        read_noise_e = read_noise_adu * gain
        
        print(f"Gain & Read Noise Fit ({temp}):")
        print(f"  Gain          = {gain:.4f} ± {gain_err:.4f} e-/ADU")
        print(f"  Read Noise    = {read_noise_adu:.4f} ± {read_noise_adu_err:.4f} ADU rms")
        print(f"                = {read_noise_e:.3f} e- rms")
        
        # Store for variance plot
        variance_data[temp] = {
            'bc': bc_arr,
            'vn': vn_arr,
            'vn_err': vn_err,
            'slope': slope_gain,
            'rn_var': rn_var,
            'gain': gain
        }

# ====================== COMBINED PLOTS ======================
fig, axs = plt.subplots(1, 2, figsize=(16, 7))

# Left: Signal vs Time (both temps)
colors = {'-10.00c': 'tab:blue', '-20.00c': 'tab:orange'}
for temp in TEMPS:
    d = signal_data[temp]
    axs[0].errorbar(d['times'], d['means'], yerr=d['sem'], fmt='o', 
                    color=colors[temp], label=f'{temp} data', capsize=3)
    axs[0].plot(d['times'], d['slope'] * d['times'] + d['bias'], 
                color=colors[temp], linestyle='-', 
                label=f'{temp} fit: {d["slope"]:.5f} ADU/s')

axs[0].set_title("Mean Signal vs Exposure Time")
axs[0].set_xlabel("Exposure Time (seconds)")
axs[0].set_ylabel("Sigma-clipped Mean Signal (ADU/pixel)")
axs[0].legend(title="Temperature")
axs[0].grid(True)

# Right: Variance vs Bias-Corrected Signal
for temp in TEMPS:
    d = variance_data[temp]
    axs[1].errorbar(d['bc'], d['vn'], yerr=d['vn_err'], fmt='o', ms=3, alpha=0.6,
                    color=colors[temp], label=f'{temp} data', capsize=2)
    xfit = np.linspace(0, d['bc'].max(), 200)
    axs[1].plot(xfit, d['slope'] * xfit + d['rn_var'], 
                color=colors[temp], linestyle='-', 
                label=f'{temp} fit: gain = {d["gain"]:.3f} e-/ADU')

axs[1].set_title("Normalized Variance vs Bias-Corrected Signal")
axs[1].set_xlabel("Bias-Corrected Mean Signal (ADU/pixel)")
axs[1].set_ylabel("Normalized Variance  (var(Diff)/2)  (ADU²)")
axs[1].legend(title="Temperature")
axs[1].grid(True)

# Overall title
fig.suptitle(f"{CAMERA_DESCRIPTION}\nDark Current, Gain & Read Noise Analysis - {today_str}", 
             fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"\nCombined analysis plot saved as: {OUTPUT_PNG}")
plt.close()

print("\nAnalysis complete for both temperatures.")