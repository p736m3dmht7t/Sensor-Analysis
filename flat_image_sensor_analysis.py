###
# flat_image_sensor_analysis.py
# John Phillips (Improved)
# john.d.phillips@comcast.net
# 2026-05-16
###

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
CAMERA_DESCRIPTION = "ZWO ASI183MM Pro S/N 262d950d2d010900 -20C - Flat Image Gain Characterization"
FLAT_DIR = Path(r"D:\Astrophotography\2026-05-19\FLAT\V")
SIGMA_LOW = 3.0          # Pixel-level low-side clip 
SIGMA_HIGH = 5.0         # Outlier rejection threshold (residuals)
SIGMA_SLOPE = 2.5        # Slope outlier rejection threshold
SATURATION_ADU = 65504   # Absolute native sensor saturation ceiling

# Date for filename
today_str = datetime.now().strftime("%Y-%m-%d")
OUTPUT_PNG = f"flat_sensor_gain_analysis_{today_str}.png"
# ===================================================

def load_fits(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[0].data.astype(np.float64)
    return data

def process_pair(d1, d2):
    # Drop individual pixels from frame math if either hits absolute saturation
    sat_mask = (d1 < SATURATION_ADU) & (d2 < SATURATION_ADU)
    d1v = d1[sat_mask]
    d2v = d2[sat_mask]

    if len(d1v) == 0:
        return np.nan, np.nan

    S = (d1v + d2v) / 2.0
    Diff = d1v - d2v

    # Asymmetric sigma-clipped mean on full frame for S
    clipped_S, _, _ = sigmaclip(S, low=SIGMA_LOW, high=SIGMA_HIGH)
    mean_S = np.mean(clipped_S)

    # Symmetric sigma-clipped variance on full frame for Diff
    clipped_Diff, _, _ = sigmaclip(Diff, low=SIGMA_HIGH, high=SIGMA_HIGH)
    var_Diff = np.var(clipped_Diff, ddof=1)

    return mean_S, var_Diff

def linear_func(p, x):
    return p[0] * x + p[1]

def linear_fit(x, y):
    """Linear fit with ODR"""
    model = Model(linear_func)
    data = RealData(x, y)
    odr = ODR(data, model, beta0=[1.0, 0.0])
    output = odr.run()
    return output

# ====================== MAIN ANALYSIS ======================
print(f"Scanning directory: {FLAT_DIR}")
fits_files = sorted(list(FLAT_DIR.glob("*.fits")))
num_files = len(fits_files)
num_pairs = num_files // 2

if num_pairs == 0:
    raise ValueError(f"No FITS file pairs found in {FLAT_DIR}")

print(f"Found {num_files} files. Processing as {num_pairs} pairs.\n")

means = []
vars_norm = []

with tqdm(total=num_pairs, desc="Processing flat pairs", unit="pair",
          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
    
    for i in range(0, num_pairs * 2, 2):
        d1 = load_fits(fits_files[i])
        d2 = load_fits(fits_files[i+1])
        
        mean_S, var_Diff = process_pair(d1, d2)
        
        if not np.isnan(mean_S):
            means.append(mean_S)
            vars_norm.append(var_Diff / 2.0)
        
        pbar.update(1)

means_arr = np.array(means)
vars_norm_arr = np.array(vars_norm)

# === Iterative Residual Outlier Elimination ===
print("\nInitiating iterative residual outlier clipping...")
valid_mask = np.ones(len(means_arr), dtype=bool)

iteration = 0
while True:
    iteration += 1
    # Fit only using current valid points
    x_fit = means_arr[valid_mask]
    y_fit = vars_norm_arr[valid_mask]
    
    if len(x_fit) < 30:
        print("Warning: Dropped too many points to reliably fit!")
        break
        
    output = linear_fit(x_fit, y_fit)
    slope_gain, intercept = output.beta[0], output.beta[1]
    sg_err, int_err = output.sd_beta[0], output.sd_beta[1]
    
    # Calculate residuals for ALL points based on current model fit
    # Using vertical/y-residuals for thresholding standard deviation
    all_residuals = vars_norm_arr - linear_func([slope_gain, intercept], means_arr)
    valid_residuals = all_residuals[valid_mask]
    
    # Standard deviation of the valid residuals
    std_residuals = np.std(valid_residuals)
    threshold = SIGMA_SLOPE * std_residuals
    
    # Check absolute residuals of currently valid points
    abs_valid_residuals = np.abs(valid_residuals)
    max_res_idx_in_valid = np.argmax(abs_valid_residuals)
    max_res_val = abs_valid_residuals[max_res_idx_in_valid]
    
    if max_res_val > threshold:
        # Find the absolute index in the original array to drop it
        actual_indices = np.where(valid_mask)[0]
        drop_target = actual_indices[max_res_idx_in_valid]
        valid_mask[drop_target] = False
        print(f"  Iteration {iteration:02d}: Dropped point at X={means_arr[drop_target]:.1f}, Residual={valid_residuals[max_res_idx_in_valid]:.1f} (> {threshold:.1f})")
    else:
        print(f"  Convergence achieved at iteration {iteration}. All remaining residuals are < {SIGMA_SLOPE} sigma.")
        break

gain = 1.0 / slope_gain
gain_err = sg_err / (slope_gain ** 2)
num_dropped = np.sum(~valid_mask)

print("\nFinal Gain Fit Results:")
print(f"  Slope = {slope_gain:.5f} ± {sg_err:.5f}")
print(f"  Gain  = {gain:.4f} ± {gain_err:.4f} e-/ADU")
print(f"  Dropped Points: {num_dropped} out of {len(means_arr)}")

# ====================== PLOTTING ======================
fig, ax = plt.subplots(figsize=(10, 8))

# Scatter plot: color-code included vs excluded points
ax.scatter(means_arr[valid_mask], vars_norm_arr[valid_mask], 
           alpha=0.7, s=20, color='tab:blue', label='Included pairs')
if num_dropped > 0:
    ax.scatter(means_arr[~valid_mask], vars_norm_arr[~valid_mask], 
               alpha=0.3, s=20, color='tab:gray', marker='x', label=f'Excluded Outliers ({num_dropped})')

# Line of best fit
xfit = np.linspace(0, means_arr.max() * 1.05, 200)
yfit = slope_gain * xfit + intercept
ax.plot(xfit, yfit, color='tab:red', linestyle='-', 
        label=f'Fit: gain = {gain:.4f} e-/ADU')

ax.set_title("Photon Transfer Curve: Variance vs Mean Signal (Outlier Adjusted)")
ax.set_xlabel("Mean Signal (ADU/pixel) - Uncorrected")
ax.set_ylabel("Normalized Variance (var(Diff)/2) (ADU²)")
ax.legend()
ax.grid(True)

# Overall title
fig.suptitle(f"{CAMERA_DESCRIPTION}\nGain Analysis - {today_str}",
             fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0.15, 1, 0.95])

# ====================== ON-PLOT STATS FOOTER ======================
txt = (
    f"  Total Image Pairs Processed : {len(means_arr)}\n"
    f"  Pairs Used in Fit / Dropped : {np.sum(valid_mask)} / {num_dropped}\n"
    f"  Slope (1/Gain)              : {slope_gain:.5f} ± {sg_err:.5f}\n"
    f"  Calculated Gain             : {gain:.5f} ± {gain_err:.5f} e-/ADU\n"
    f"  *Note: Outliers pruned iteratively using a {SIGMA_SLOPE}-sigma residual threshold."
)

fig.text(
    0.5, 0.02, txt,
    family='monospace', fontsize=11,
    verticalalignment='bottom', horizontalalignment='center',
    color='black',
    bbox=dict(boxstyle='round,pad=0.8',
              facecolor='#f8f8f8',
              edgecolor='tab:blue',
              linewidth=1.2),
)

plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"\nAnalysis plot saved as: {OUTPUT_PNG}")
plt.close()

print("\nAnalysis complete.")