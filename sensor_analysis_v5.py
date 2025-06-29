import os
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg') # This line must come BEFORE importing pyplot
import matplotlib.pyplot as plt
import csv
from scipy.stats import linregress

# --- Configuration ---
# IMPORTANT: Set this to the directory containing your FITS images.
INPUT_DIR = r"D:\Astrophotography\FLATS_For_Sensor_Analysis"

# Heuristics for automatically detecting the 'jump' point in the signal-exposure curve.
JUMP_DETECTION_STD_MULTIPLIER = 5           # How many std deviations away from initial fit for a new region
JUMP_DETECTION_CONSECUTIVE_POINTS = 25      # How many consecutive points must deviate to confirm a jump
JUMP_DETECTION_MIN_EXP = 0.9                # Smallest exposure time to consider for jump detection (to avoid noise at 0 exp time)

# Heuristic for detecting the saturation point by finding where data deviates from the linear fit.
SATURATION_DETECTION_STD_MULTIPLIER = 5      # How many std deviations below the fit a point must be to be considered saturated.
SATURATION_DETECTION_CONSECUTIVE_POINTS = 10 # How many consecutive points must deviate to confirm saturation.

# Keywords to identify filter types in FITS header (case-insensitive check will be used)
DARK_FILTER_KEYWORDS = ['DARK', 'BIAS'] # Common keywords for dark/bias frames
LIGHT_FILTER_KEYWORDS = ['B', 'V', 'R', 'I', 'NONE'] # Common keywords for light/flat frames

# --- Functions ---

def load_fits_and_extract_info(image_path):
    """
    Loads a FITS image, converts its data to float32, and extracts relevant header info.
    """
    try:
        with fits.open(image_path) as hdul:
            img_data = hdul[0].data.astype(np.float32)
            header = hdul[0].header
            exposure_time = header.get('EXPOSURE', header.get('EXPTIME', None))
            filter_name = str(header.get('FILTER', 'N/A')).strip()
            instrume = str(header.get('INSTRUME', 'N/A')).strip()
            set_temp = header.get('SET-TEMP', 'N/A')
            gain_setting = header.get('GAIN', 'N/A')
            offset_setting = header.get('OFFSET', 'N/A')
        return img_data, exposure_time, filter_name, instrume, set_temp, gain_setting, offset_setting
    except Exception as e:
        print(f"Error loading FITS image {image_path}: {e}")
        return None, None, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'

def calculate_subframe_metrics(img_data1, img_data2):
    """
    Calculates raw signal and noise for 9 subframes of an image pair.
    """
    if img_data1 is None or img_data2 is None: return [], []
    height, width = img_data1.shape
    sub_h, sub_w = height // 3, width // 3
    raw_signals, noises = [], []
    for i in range(3):
        for j in range(3):
            h_start, w_start = i * sub_h, j * sub_w
            h_end, w_end = (i + 1) * sub_h if i < 2 else height, (j + 1) * sub_w if j < 2 else width
            sub_img1, sub_img2 = img_data1[h_start:h_end, w_start:w_end], img_data2[h_start:h_end, w_start:w_end]
            raw_signal = np.mean((sub_img1 + sub_img2) / 2.0)
            noise = np.std(sub_img2 - sub_img1, ddof=1) / np.sqrt(2.0)
            noises.append(1e-6 if np.isnan(noise) or noise == 0 else noise)
            raw_signals.append(raw_signal)
    return raw_signals, noises

def collect_all_data(input_dir):
    """
    Recursively finds and processes all FITS pairs in a directory.
    """
    all_fits_files = sorted([os.path.join(root, file) for root, _, files in os.walk(input_dir) for file in files if file.lower().endswith(('.fits', '.fit'))])
    if len(all_fits_files) < 2:
        print(f"Error: Need at least 2 FITS files for pairing. Found {len(all_fits_files)}.")
        return [], {}
    if len(all_fits_files) % 2 != 0:
        print(f"Warning: Odd number of FITS files ({len(all_fits_files)}). The last file will be ignored.")
    
    all_data, metadata = [], {}
    for i in range(0, len(all_fits_files) - 1, 2):
        img1_path, img2_path = all_fits_files[i], all_fits_files[i+1]
        
        # RESTORED: Print progress for each pair being processed
        pair_count = i // 2
        print(f"Processing pair: {pair_count:04d} {os.path.basename(img1_path)} and {os.path.basename(img2_path)}")

        img_data1, exp1, filt1, inst1, temp1, gain1, off1 = load_fits_and_extract_info(img1_path)
        img_data2, exp2, _, _, _, _, _ = load_fits_and_extract_info(img2_path)
        
        if not metadata and inst1 != 'N/A':
            metadata = {'instrume': inst1, 'set_temp': temp1, 'gain_setting': gain1, 'offset_setting': off1}
            
        if img_data1 is not None and img_data2 is not None and img_data1.shape == img_data2.shape and exp1 is not None and abs(exp1 - exp2) < 1e-6:
            signals, noises = calculate_subframe_metrics(img_data1, img_data2)
            all_data.append({'raw_signal_subframes': signals, 'noise_subframes': noises, 'exposure_time': exp1, 'filter': filt1})
    print(f"\nSuccessfully processed {len(all_data)} image pairs.")
    return all_data, metadata

def calculate_linregress_stats(x, y):
    """
    Calculates slope, intercept, their standard errors, and the overall model fit error.
    """
    if len(x) < 2: return np.nan, np.nan, np.nan, np.nan, np.nan
    slope, intercept, _, _, stderr_slope = linregress(x, y)
    residuals = y - (slope * x + intercept)
    if len(x) > 2:
        std_err_estimate = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
        sum_x_sq_dev = np.sum((x - np.mean(x))**2)
        stderr_intercept = std_err_estimate * np.sqrt(np.sum(x**2) / (len(x) * sum_x_sq_dev)) if sum_x_sq_dev > 0 else np.nan
    else:
        std_err_estimate, stderr_intercept = np.nan, np.nan
    return slope, intercept, stderr_slope, stderr_intercept, std_err_estimate

def analyze_and_plot_all(all_data, metadata):
    """
    Master function to perform all analysis and generate plots and files.
    """
    # --- Step 1: Basic Data Prep & Bias Calculation ---
    light_data = [d for d in all_data if d['filter'].upper() in LIGHT_FILTER_KEYWORDS and d.get('exposure_time') is not None]
    dark_data = [d for d in all_data if d['filter'].upper() in DARK_FILTER_KEYWORDS and d.get('exposure_time') is not None]
    
    if not light_data:
        print("Error: No valid LIGHT frames found for analysis.")
        return

    # Calculate subframe biases
    min_dark_exp = min((d['exposure_time'] for d in dark_data), default=0)
    min_exp_darks = [d for d in dark_data if d['exposure_time'] == min_dark_exp]
    subframe_biases = [np.mean([d['raw_signal_subframes'][i] for d in min_exp_darks]) if min_exp_darks else 0 for i in range(9)]
    
    # Prepare averaged data for signal-exposure curve
    avg_raw_signals = np.array([np.mean(d['raw_signal_subframes']) for d in light_data])
    exposure_times = np.array([d['exposure_time'] for d in light_data])
    sort_indices = np.argsort(exposure_times)
    avg_raw_signals, exposure_times = avg_raw_signals[sort_indices], exposure_times[sort_indices]

    # --- Step 2: Find Jump Point from Signal-Exposure Curve ---
    fit1_mask = exposure_times < JUMP_DETECTION_MIN_EXP
    s1, i1, s_err1, i_err1, fit_err1 = calculate_linregress_stats(exposure_times[fit1_mask], avg_raw_signals[fit1_mask])
    
    jump_idx = -1
    if not np.isnan(s1):
        residuals = avg_raw_signals - (s1 * exposure_times + i1)
        std_resid1 = np.std(residuals[fit1_mask])
        consecutive_dev, start_check_idx = 0, np.searchsorted(exposure_times, JUMP_DETECTION_MIN_EXP)
        for i in range(start_check_idx, len(exposure_times)):
            if residuals[i] > JUMP_DETECTION_STD_MULTIPLIER * std_resid1:
                consecutive_dev += 1
                if consecutive_dev >= JUMP_DETECTION_CONSECUTIVE_POINTS:
                    jump_idx = i - JUMP_DETECTION_CONSECUTIVE_POINTS + 1
                    break
            else: consecutive_dev = 0
    
    # --- Step 3: Prepare Signal-Noise Data & Find Saturation ---
    eff_signals_all = np.array([s - b for d in light_data for s, b in zip(d['raw_signal_subframes'], subframe_biases)])
    noises_all = np.array([n for d in light_data for n in d['noise_subframes']])
    exp_times_all = np.array([d['exposure_time'] for d in light_data for _ in range(9)])
    
    sort_indices_sn = np.argsort(exp_times_all)
    eff_signals_all, noises_all = eff_signals_all[sort_indices_sn], noises_all[sort_indices_sn]
    
    read_noise_noises = [n for d in min_exp_darks for n in d['noise_subframes'] if n > 0]
    read_noise_mean = np.mean(read_noise_noises) if read_noise_noises else 0
    
    # Find saturation from S-N curve using a log-log deviation method
    saturation_idx = -1
    if jump_idx != -1:
        sn_jump_idx = jump_idx * 9  # Corresponding index in the subframe array
        
        # Define a mask for a clean linear fit in log-log space, avoiding the very end which might be saturated.
        # We'll fit up to a certain percentile of the signal in the second region to get a robust baseline.
        fit_region_mask = (np.arange(len(eff_signals_all)) >= sn_jump_idx)
        
        # Only proceed if there are points in the second region
        if np.any(fit_region_mask):
            signals_in_region = eff_signals_all[fit_region_mask & (eff_signals_all > 0)]
            
            if len(signals_in_region) > SATURATION_DETECTION_CONSECUTIVE_POINTS:
                # Use a high percentile to define the upper bound for the linear fit
                signal_upper_bound_for_fit = np.percentile(signals_in_region, 85)
                
                fit_mask = fit_region_mask & \
                           (eff_signals_all < signal_upper_bound_for_fit) & \
                           (eff_signals_all > 0) & \
                           (noises_all > 0)

                # Ensure there are enough points to perform a meaningful fit
                if np.sum(fit_mask) > 10: 
                    log_signals_fit = np.log10(eff_signals_all[fit_mask])
                    log_noises_fit = np.log10(noises_all[fit_mask])
                    
                    s_log, i_log, _, _, fit_err_log = calculate_linregress_stats(log_signals_fit, log_noises_fit)

                    if not np.isnan(s_log) and fit_err_log > 0:
                        consecutive_saturated = 0
                        # Start checking for saturation from the beginning of region 2
                        for i in range(sn_jump_idx, len(eff_signals_all)):
                            if eff_signals_all[i] > 0 and noises_all[i] > 0:
                                predicted_log_noise = s_log * np.log10(eff_signals_all[i]) + i_log
                                residual_log = np.log10(noises_all[i]) - predicted_log_noise
                                
                                if residual_log < -SATURATION_DETECTION_STD_MULTIPLIER * fit_err_log:
                                    consecutive_saturated += 1
                                    if consecutive_saturated >= SATURATION_DETECTION_CONSECUTIVE_POINTS:
                                        # Found the start of the saturation region
                                        sat_start_idx = i - SATURATION_DETECTION_CONSECUTIVE_POINTS + 1
                                        saturation_idx = sat_start_idx // 9  # Convert subframe index to avg_raw_signals index
                                        break
                                else:
                                    consecutive_saturated = 0 # Reset counter if a point is back in line
                            else:
                                 consecutive_saturated = 0 # Reset if data is invalid
    if saturation_idx == -1: saturation_idx = len(avg_raw_signals) # No saturation found, use all points

    # --- Step 4: Finalize Fits with Known Boundaries ---
    fit1_mask = np.arange(len(avg_raw_signals)) < jump_idx if jump_idx != -1 else exposure_times < JUMP_DETECTION_MIN_EXP
    s1, i1, s_err1, i_err1, fit_err1 = calculate_linregress_stats(exposure_times[fit1_mask], avg_raw_signals[fit1_mask])

    s2, i2, s_err2, i_err2, fit_err2 = (np.nan,)*5
    if jump_idx != -1:
        fit2_mask = (np.arange(len(avg_raw_signals)) >= jump_idx) & (np.arange(len(avg_raw_signals)) < saturation_idx)
        s2, i2, s_err2, i_err2, fit_err2 = calculate_linregress_stats(exposure_times[fit2_mask], avg_raw_signals[fit2_mask])

    # --- Step 5: Gain Calculations (with Read Noise Correction) ---
    def calculate_gain(mask, read_noise_sq):
        if np.sum(mask) < 2: return (np.nan,)*4
        # Correct for read noise before fitting
        noise_sq_corrected = noises_all[mask]**2 - read_noise_sq
        positive_noise_mask = noise_sq_corrected > 0
        if np.sum(positive_noise_mask) < 2: return (np.nan,)*4
        
        signal_subset = eff_signals_all[mask][positive_noise_mask]
        noise_sq_subset = noise_sq_corrected[positive_noise_mask]

        slope_ns2, _, _, _, _ = linregress(signal_subset, noise_sq_subset)
        log_slope, _, _, _, stderr_log_slope = linregress(np.log10(signal_subset), np.log10(np.sqrt(noise_sq_subset)))
        
        gain = 1.0 / slope_ns2 if slope_ns2 > 0 else np.nan
        return gain, log_slope, stderr_log_slope, (np.min(eff_signals_all[mask]), np.max(eff_signals_all[mask]))

    sn_jump_idx = jump_idx * 9 if jump_idx != -1 else -1
    sn_sat_idx = saturation_idx * 9
    
    gain1_mask = (np.arange(len(eff_signals_all)) < sn_jump_idx) & (eff_signals_all > read_noise_mean * 3) if sn_jump_idx != -1 else (eff_signals_all < np.percentile(eff_signals_all, 50)) & (eff_signals_all > read_noise_mean * 3)
    gain2_mask = (np.arange(len(eff_signals_all)) >= sn_jump_idx) & (np.arange(len(eff_signals_all)) < sn_sat_idx) if sn_jump_idx != -1 else np.zeros_like(eff_signals_all, dtype=bool)
    
    read_noise_sq = read_noise_mean**2
    gain1, log_s1, log_s_err1, range1 = calculate_gain(gain1_mask, read_noise_sq)
    gain2, log_s2, log_s_err2, range2 = calculate_gain(gain2_mask, read_noise_sq)
    
    # --- Step 6: Generate Filenames ---
    gain_str = metadata.get('gain_setting', 'GNA')
    off_str = metadata.get('offset_setting', 'ONA')
    base_filename = f"sensor_analysis_G{gain_str}_O{off_str}"
    
    # --- Step 7: Print Final Console Summary ---
    print("\n--- Light Curve Analysis (Region 1) ---")
    print(f"Fitted model: Y = ({s1:.2f} +/- {s_err1:.2f}) * ExposureTime + ({i1:.2f} +/- {i_err1:.2f})")
    print(f"Model fit: +/-{fit_err1:.2f} ADU")
    if jump_idx != -1:
        print("\n--- Light Curve Analysis (Region 2) ---")
        print(f"Fitted model: Y = ({s2:.2f} +/- {s_err2:.2f}) * ExposureTime + ({i2:.2f} +/- {i_err2:.2f})")
        print(f"Model fit: +/-{fit_err2:.2f} ADU")
    print(f"\nSaturation detected at index {saturation_idx}, Exposure {exposure_times[saturation_idx]:.2f}s, ADU {avg_raw_signals[saturation_idx]:.0f}")

    print(f"\n--- Read Noise Calculation ---\nRead Noise: {read_noise_mean:.4f} ADU\n")
    
    if not np.isnan(gain1):
        print(f"--- Photon Noise Analysis (Gain 1) ---")
        print(f"Fit Range (Effective Signal): {range1[0]:.0f} - {range1[1]:.0f} ADU")
        print(f"Gain 1: {gain1:.3f} e-/ADU ({1/gain1:.4f} ADU/e-)")
        print(f"Log-Log Slope: {log_s1:.4f} +/- {log_s_err1:.4f}\n")

    if not np.isnan(gain2):
        print(f"--- Photon Noise Analysis (Gain 2) ---")
        print(f"Fit Range (Effective Signal): {range2[0]:.0f} - {range2[1]:.0f} ADU")
        print(f"Gain 2: {gain2:.3f} e-/ADU ({1/gain2:.4f} ADU/e-)")
        print(f"Log-Log Slope: {log_s2:.4f} +/- {log_s_err2:.4f}\n")
        
    # --- Step 8: Plotting ---
    # Plot 1: Signal vs. Exposure
    plt.figure(figsize=(12, 7))
    unsat_mask = np.arange(len(avg_raw_signals)) < saturation_idx
    sat_mask = ~unsat_mask
    plt.plot(exposure_times[unsat_mask], avg_raw_signals[unsat_mask], 'o', c='tab:blue', markersize=5, label='Unsaturated')
    plt.plot(exposure_times[sat_mask], avg_raw_signals[sat_mask], 'x', c='tab:red', markersize=5, label='Saturated')
    
    if not np.isnan(s1):
        x_fit1 = exposure_times[fit1_mask]
        plt.plot(x_fit1, s1 * x_fit1 + i1, 'r--', label=f'Fit 1: Y = {s1:.2f}X + {i1:.2f}')
    if not np.isnan(s2):
        x_fit2 = exposure_times[(np.arange(len(avg_raw_signals)) >= jump_idx) & unsat_mask]
        plt.plot(x_fit2, s2 * x_fit2 + i2, 'b--', label=f'Fit 2: Y = {s2:.2f}X + {i2:.2f}')
        boundary_exp = (exposure_times[jump_idx - 1] + exposure_times[jump_idx]) / 2.0
        plt.axvline(boundary_exp, color='purple', ls=':', label=f'Region Boundary ({boundary_exp:.2f}s)')
        
    plt.axhline(avg_raw_signals[saturation_idx], color='gray', ls=':', label=f'Saturation Threshold ({avg_raw_signals[saturation_idx]:.0f} ADU)')
    plt.title(f"Raw Signal vs. Exposure Time for {metadata.get('instrume', 'N/A')}, {metadata.get('set_temp', 'N/A')}C, Gain {gain_str}, Offset {off_str}", fontsize=14)
    plt.xlabel('Exposure Time (s)', fontsize=12)
    plt.ylabel('Average ADU (Raw)', fontsize=12)
    plt.grid(True, which="both", ls="-", color='0.8')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{base_filename}_signal_exposure.png", dpi=300)
    plt.close()
    print(f"Signal vs Exposure plot saved to {base_filename}_signal_exposure.png")

    # Plot 2: Signal vs. Noise
    plt.figure(figsize=(12, 8))
    sn_sat_idx_all = saturation_idx * 9
    unsat_mask_sn = np.arange(len(eff_signals_all)) < sn_sat_idx_all
    pos_mask = (eff_signals_all > 0) & (noises_all > 0)
    
    plt.loglog(eff_signals_all[unsat_mask_sn & pos_mask], noises_all[unsat_mask_sn & pos_mask], 'o', c='tab:blue', markersize=4, alpha=0.7, label='Unsaturated')
    plt.loglog(eff_signals_all[~unsat_mask_sn & pos_mask], noises_all[~unsat_mask_sn & pos_mask], 'x', c='tab:red', markersize=4, label='Saturated')

    legend_handles = plt.gca().get_legend_handles_labels()[0]
    legend_labels = ['Unsaturated', 'Saturated']

    if read_noise_mean > 0:
        h = plt.axhline(read_noise_mean, color='red', ls='--', lw=2, label=f'Read Noise: {read_noise_mean:.2f} ADU')
        legend_handles.append(h)
        legend_labels.append(f'Read Noise: {read_noise_mean:.2f} ADU')
        
    if not np.isnan(gain1):
        legend_labels.append(f'Gain 1: {gain1:.4f} e-/ADU (Slope: {log_s1:.4f})')
        legend_handles.append(plt.Line2D([0], [0], color='green', lw=2.5, label='Gain 1')) # Dummy handle
    if not np.isnan(gain2):
        legend_labels.append(f'Gain 2: {gain2:.4f} e-/ADU (Slope: {log_s2:.4f})')
        legend_handles.append(plt.Line2D([0], [0], color='forestgreen', ls='--', lw=2.5, label='Gain 2')) # Dummy handle

    if jump_idx != -1:
        boundary_signal = (eff_signals_all[sn_jump_idx - 1] + eff_signals_all[sn_jump_idx]) / 2.0
        plt.axvline(boundary_signal, color='darkviolet', ls=':', lw=2, label=f'Jump ({exposure_times[jump_idx]:.2f}s)')
        legend_labels.append(f'Response Jump ({exposure_times[jump_idx]:.2f}s)')
        legend_handles.append(plt.Line2D([0],[0], color='darkviolet', ls=':', lw=2))

    plt.title(f"CMOS Sensor Noise Characterization for {metadata.get('instrume', 'N/A')}, {metadata.get('set_temp', 'N/A')}C, Gain {gain_str}, Offset {off_str}", fontsize=14)
    plt.xlabel('Effective Signal (ADU)', fontsize=12)
    plt.ylabel('Noise (ADU)', fontsize=12)
    plt.grid(True, which="both", ls="-", color='0.8')
    plt.legend(handles=legend_handles, labels=legend_labels, fontsize=11)
    plt.xlim(left=max(1, np.min(eff_signals_all[pos_mask])*0.8))
    plt.tight_layout()
    plt.savefig(f"{base_filename}_signal_noise.png", dpi=300)
    plt.close()
    print(f"Signal vs Noise plot saved to {base_filename}_signal_noise.png")
    
    # --- Step 9: Save CSV ---
    with open(f"{base_filename}_data.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Effective_Signal', 'Noise', 'Raw_Signal', 'Exposure_Time', 'Subframe_Index', 'Saturated'])
        for i, (eff_s, n, exp_t) in enumerate(zip(eff_signals_all, noises_all, exp_times_all)):
            raw_s = eff_s + subframe_biases[i % 9]
            is_saturated = i >= sn_sat_idx_all
            writer.writerow([eff_s, n, raw_s, exp_t, i % 9, is_saturated])
    print(f"Data saved to {base_filename}_data.csv")


# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        exit()

    all_collected_data, metadata = collect_all_data(INPUT_DIR)

    if all_collected_data:
        analyze_and_plot_all(all_collected_data, metadata)
    else:
        print("\nNo valid data collected. Please check the input directory and FITS files.\n")