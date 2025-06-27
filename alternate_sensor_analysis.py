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
INPUT_DIR = r'D:\Astrophotography\2025-06-10\FLAT\B' 
OUTPUT_CSV_FILE = 'signal_noise_data.csv'
OUTPUT_PLOT_FILE_SN_STANDARD = 'signal_noise_plot_standard.png' # Renamed for clarity
OUTPUT_PLOT_FILE_SN_ADJUSTED = 'signal_noise_plot_adjusted_bias.png' # New alternative plot
OUTPUT_PLOT_FILE_SE = 'signal_exposure_plot.png'

# Define *BIAS-SUBTRACTED* signal ranges (in ADU) for fitting different noise regions on the S/N plot.
# These values are now split into two regions based on the expected sensor behavior change.
# Adjust these by examining the generated plots and console output.

# Region 1: Low-to-Mid Signal (before the detected 'jump' in gain/response)
READ_NOISE_SIGNAL_MAX_EFFECTIVE = 30        # Max bias-subtracted signal for estimating read noise
PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG1 = 50 # Min effective signal for Photon Noise Fit 1
PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG1 = 9000 # Max effective signal for Photon Noise Fit 1 (before jump)
FPN_SIGNAL_MIN_EFFECTIVE_REG1 = 9000        # Min effective signal for FPN Fit 1 (less likely to be pure FPN in this region)
FPN_SIGNAL_MAX_EFFECTIVE_REG1 = 9500        # Max effective signal for FPN Fit 1

# Region 2: Mid-to-High Signal (after the detected 'jump' in gain/response, before saturation)
PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG2 = 15000 # Min effective signal for Photon Noise Fit 2 (after jump)
PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG2 = 40000 # Max effective signal for Photon Noise Fit 2
FPN_SIGNAL_MIN_EFFECTIVE_REG2 = 50000       # Min effective signal for FPN Fit 2
FPN_SIGNAL_MAX_EFFECTIVE_REG2 = 61000       # Max effective signal for FPN Fit 2

# Initial maximum exposure time for the first linear fit (bias calculation).
# This is kept as a constant to define the very first, often cleanest, linear region.
INITIAL_LINEAR_EXPOSURE_MAX_SECONDS = 1.0 

# Heuristics for automatically detecting the 'jump' point in the signal-exposure curve.
JUMP_DETECTION_STD_MULTIPLIER = 5           # How many std deviations away from initial fit for a new region
JUMP_DETECTION_CONSECUTIVE_POINTS = 3       # How many consecutive points must deviate to confirm a jump
JUMP_DETECTION_MIN_EXP = 0.1                # Smallest exposure time to consider for jump detection (to avoid noise at 0 exp time)

# --- Functions ---

def load_and_convert_fits(image_path):
    """
    Loads a FITS image and converts its data to float32, also retrieves exposure time.
    """
    try:
        with fits.open(image_path) as hdul:
            img_data = hdul[0].data.astype(np.float32)
            
            # Try to get exposure time from header
            exposure_time = None
            if 'EXPOSURE' in hdul[0].header:
                exposure_time = hdul[0].header['EXPOSURE']
            elif 'EXPTIME' in hdul[0].header:
                exposure_time = hdul[0].header['EXPTIME']
            else:
                print(f"Warning: No 'EXPOSURE' or 'EXPTIME' found in header for {image_path}. Skipping exposure time for this image.")
                
        return img_data, exposure_time
    except Exception as e:
        print(f"Error loading FITS image {image_path}: {e}")
        return None, None

def calculate_subframe_metrics(img_data1, img_data2):
    """
    Divides images into 9 subframes and calculates signal and noise for each.
    Returns raw signal and noise, as well as exposure time (if available).
    """
    if img_data1 is None or img_data2 is None:
        return []

    height, width = img_data1.shape
    
    subframe_height = height // 3
    subframe_width = width // 3

    results = []

    for i in range(3): # Row index for subframes
        for j in range(3): # Column index for subframes
            h_start = i * subframe_height
            h_end = (i + 1) * subframe_height if i < 2 else height 
            w_start = j * subframe_width
            w_end = (j + 1) * subframe_width if j < 2 else width 

            sub_img1 = img_data1[h_start:h_end, w_start:w_end]
            sub_img2 = img_data2[h_start:h_end, w_start:w_end]

            # Raw signal (average ADU before bias subtraction)
            raw_signal = np.mean((sub_img1 + sub_img2) / 2.0)

            diff_img = sub_img2 - sub_img1
            noise = np.std(diff_img, ddof=1) / np.sqrt(2.0)
            
            # Prevent issues from constant subframes; will be filtered for plotting
            if np.isnan(noise) or noise == 0: 
                noise = 1e-6 

            results.append({'raw_signal': raw_signal, 'noise': noise})
    return results

def collect_all_data(input_dir):
    """
    Recursively iterates through the directory, finds FITS pairs,
    and collects all signal/noise data from their subframes, including exposure time.
    """
    all_fits_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.fits', '.fit')):
                all_fits_files.append(os.path.join(root, file))

    all_fits_files.sort()

    if len(all_fits_files) < 2:
        print(f"Error: Found {len(all_fits_files)} FITS files. Need at least 2 files for pairing.")
        return []
    if len(all_fits_files) % 2 != 0:
        print(f"Warning: Odd number of FITS files ({len(all_fits_files)}) found in '{input_dir}'. "
              "The last file will be ignored. Ensure all images have a pair.")

    all_data = []
    processed_pairs_count = 0
    
    for i in range(0, len(all_fits_files) - 1, 2):
        img_path1 = all_fits_files[i]
        img_path2 = all_fits_files[i+1]

        print(f"Processing pair: {os.path.basename(img_path1)} and {os.path.basename(img_path2)}")

        img_data1, exp_time1 = load_and_convert_fits(img_path1)
        img_data2, exp_time2 = load_and_convert_fits(img_path2)

        if img_data1 is not None and img_data2 is not None:
            if img_data1.shape != img_data2.shape:
                print(f"Warning: Image dimensions mismatch for pair {os.path.basename(img_path1)} and {os.path.basename(img_path2)}. Skipping pair.")
                continue
            
            # Ensure exposure times are consistent for the pair and not None
            if exp_time1 is None or exp_time2 is None or abs(exp_time1 - exp_time2) > 1e-6:
                print(f"Warning: Inconsistent or missing exposure times for pair {os.path.basename(img_path1)} and {os.path.basename(img_path2)}. Using average if present, otherwise skipping exposure time for this pair.")
                if exp_time1 is not None and exp_time2 is not None:
                    avg_exp_time = (exp_time1 + exp_time2) / 2.0
                else:
                    avg_exp_time = None # Cannot use exposure time for this pair
            else:
                avg_exp_time = exp_time1 # They should be the same
            
            subframe_results = calculate_subframe_metrics(img_data1, img_data2)
            
            for res in subframe_results:
                res['exposure_time'] = avg_exp_time # Add exposure time to each subframe result
            all_data.extend(subframe_results)
            processed_pairs_count += 1
        else:
            print(f"Skipping pair due to loading error: {os.path.basename(img_path1)}, {os.path.basename(img_path2)}")

    print(f"\nSuccessfully processed {processed_pairs_count} image pairs.")
    return all_data

def analyze_signal_exposure_curve(data, output_plot_file):
    """
    Analyzes the signal vs exposure time curve, performing two linear fits:
    1. For the initial region to calculate sensor bias.
    2. For a second linear region (e.g., higher gain) by automatically detecting the jump.
    Also dynamically detects saturation.
    Plots the raw signal vs. exposure time with the fits.
    Returns the calculated sensor bias, saturation threshold, effective signal at jump point, and second_intercept.
    """
    # Filter data with valid exposure times
    filtered_data = [d for d in data if d['exposure_time'] is not None]
    if not filtered_data:
        print("Error: No valid data points with exposure time found for signal-exposure analysis.")
        return 0.0, 0.0, np.nan, np.nan

    # Sort data by exposure time for proper analysis
    filtered_data.sort(key=lambda x: x['exposure_time'])
    all_raw_signals_sorted = np.array([d['raw_signal'] for d in filtered_data])
    all_exposure_times_sorted = np.array([d['exposure_time'] for d in filtered_data])

    # --- Dynamic Saturation Threshold Detection ---
    # Use a high percentile of observed signals to estimate the saturation level
    # Then apply a small safety margin for where the linear fit should end
    dynamic_saturation_level = np.percentile(all_raw_signals_sorted, 99.5) # Captures the plateau
    fit_saturation_threshold_adu = dynamic_saturation_level * 0.98 # Safety margin for linear fit

    # --- Fit 1: Initial Linear Region (for Bias Calculation) ---
    # Corrected line: removed the extra 'd'
    valid_data_for_bias_fit = [d for d in filtered_data if 
                                d['exposure_time'] <= INITIAL_LINEAR_EXPOSURE_MAX_SECONDS and
                                d['raw_signal'] < fit_saturation_threshold_adu]

    sensor_bias = 0.0
    initial_light_response_rate = 0.0
    initial_r_squared = 0.0
    initial_fit_successful = False

    if len(valid_data_for_bias_fit) >= 2:
        raw_signals_bias_fit = np.array([d['raw_signal'] for d in valid_data_for_bias_fit])
        exposure_times_bias_fit = np.array([d['exposure_time'] for d in valid_data_for_bias_fit])
        slope_bias_fit, intercept_bias_fit, r_value_bias_fit, _, _ = linregress(exposure_times_bias_fit, raw_signals_bias_fit)
        sensor_bias = intercept_bias_fit
        initial_light_response_rate = slope_bias_fit
        initial_r_squared = r_value_bias_fit**2
        initial_fit_successful = True

        print(f"\n--- Sensor Bias Calculation (from data <= {INITIAL_LINEAR_EXPOSURE_MAX_SECONDS}s) ---")
        print(f"Fitted linear model (Initial Region): Signal = {initial_light_response_rate:.2f} * ExposureTime + {sensor_bias:.2f}")
        print(f"Estimated Sensor Bias (ADU offset): {sensor_bias:.2f} ADU")
        print(f"Estimated Light Current Rate (Initial Region): {initial_light_response_rate:.2f} ADU/s")
        print(f"R-squared for fit: {initial_r_squared:.4f}")
        print(f"--------------------------------------------------")
    else:
        print("Warning: Not enough valid data points for initial bias calculation. Returning 0 bias for further calculations.")

    # --- Fit 2: Second Linear Region (Dynamic Detection of Jump) ---
    second_light_response_rate = np.nan
    second_intercept = np.nan
    second_linear_min_exp = np.nan
    second_linear_max_exp = np.nan
    effective_signal_at_jump_point = np.nan
    second_fit_successful = False

    if initial_fit_successful and initial_r_squared > 0.9: # Only proceed if initial fit was reasonably good
        # Calculate residuals against the *extrapolation* of the initial fit for all non-saturated data
        extrapolated_initial_signals = initial_light_response_rate * all_exposure_times_sorted + sensor_bias
        residuals = all_raw_signals_sorted - extrapolated_initial_signals

        # Calculate std dev of residuals in the initial fit region to set a threshold for deviation
        initial_residuals_in_fit_region = residuals[all_exposure_times_sorted <= INITIAL_LINEAR_EXPOSURE_MAX_SECONDS]
        # Avoid std dev issues if very few points or perfectly linear
        std_dev_initial_residuals = np.std(initial_residuals_in_fit_region) 
        if std_dev_initial_residuals < 1e-6: # Set a floor to avoid division by zero or too small a threshold
            std_dev_initial_residuals = 1.0 # Arbitrary small but non-zero value for thresholding

        # Find the jump point: first exposure where residuals consistently exceed a threshold
        consecutive_deviations = 0
        jump_idx = -1
        # Start checking for jump *after* the initial linear fit region
        start_check_idx = np.where(all_exposure_times_sorted > INITIAL_LINEAR_EXPOSURE_MAX_SECONDS)[0]
        if len(start_check_idx) > 0:
            start_check_idx = start_check_idx[0]
        else:
            start_check_idx = len(all_exposure_times_sorted) # No points beyond initial fit

        for i in range(start_check_idx, len(all_exposure_times_sorted)):
            if all_exposure_times_sorted[i] < JUMP_DETECTION_MIN_EXP:
                continue 

            if all_raw_signals_sorted[i] >= fit_saturation_threshold_adu:
                break # Stop if we hit saturation

            # Check for significant positive deviation (the 'jump')
            if residuals[i] > JUMP_DETECTION_STD_MULTIPLIER * std_dev_initial_residuals:
                consecutive_deviations += 1
                if consecutive_deviations >= JUMP_DETECTION_CONSECUTIVE_POINTS:
                    jump_idx = i - JUMP_DETECTION_CONSECUTIVE_POINTS + 1
                    break
            else:
                consecutive_deviations = 0 

        if jump_idx != -1:
            # second_linear_min_exp is the exposure time at the start of the detected jump
            second_linear_min_exp = all_exposure_times_sorted[jump_idx]
            # effective_signal_at_jump_point is the effective signal at that exact point
            # based on the *initially calculated bias* (for the standard S/N plot X-axis)
            effective_signal_at_jump_point = all_raw_signals_sorted[jump_idx] - sensor_bias

            # Determine the end of the second linear region (before saturation fully hits)
            valid_exposure_times_below_saturation = all_exposure_times_sorted[all_raw_signals_sorted < fit_saturation_threshold_adu]
            if len(valid_exposure_times_below_saturation) > 0:
                second_linear_max_exp = np.max(valid_exposure_times_below_saturation) * 0.99
                # Ensure max_exp is greater than min_exp, and has enough points for a fit
                if second_linear_max_exp <= second_linear_min_exp + 0.1: 
                    second_linear_max_exp = np.max(all_exposure_times_sorted[all_raw_signals_sorted < dynamic_saturation_level]) * 0.95 
                    print("Warning: second_linear_max_exp adjusted for robust second fit range.")
            else:
                second_linear_max_exp = np.max(all_exposure_times_sorted) # Fallback if no non-saturated points found

            # Ensure the second fit range has at least 2 points
            valid_data_for_second_fit = [d for d in filtered_data if 
                                          d['exposure_time'] >= second_linear_min_exp and
                                          d['exposure_time'] <= second_linear_max_exp and
                                          d['raw_signal'] < fit_saturation_threshold_adu]

            if len(valid_data_for_second_fit) >= 2:
                raw_signals_second_fit = np.array([d['raw_signal'] for d in valid_data_for_second_fit])
                exposure_times_second_fit = np.array([d['exposure_time'] for d in valid_data_for_second_fit])
                slope_second_fit, intercept_second_fit, r_value_second_fit, _, _ = linregress(exposure_times_second_fit, raw_signals_second_fit)
                second_light_response_rate = slope_second_fit
                second_intercept = intercept_second_fit # This is the "new bias" for the second region's extrapolation
                second_fit_successful = True

                print(f"\n--- Second Linear Region Fit (Detected between {second_linear_min_exp:.2f}s and {second_linear_max_exp:.2f}s) ---")
                print(f"Fitted linear model (Second Region): Signal = {second_light_response_rate:.2f} * ExposureTime + {second_intercept:.2f}")
                print(f"Estimated Light Current Rate (Second Region): {second_light_response_rate:.2f} ADU/s")
                print(f"R-squared for fit: {r_value_second_fit**2:.4f}")
                print(f"------------------------------------------------------------------------------------")
            else:
                print("Warning: Not enough valid data points for second linear fit after jump detection. Skipping second fit.")
        else:
            print("Info: No significant jump detected in the signal-exposure curve after initial fit. Only one linear region assumed.")
            effective_signal_at_jump_point = np.nan # Ensure it's NaN if no jump found

    # --- Plot Signal vs Exposure Time ---
    plt.figure(figsize=(10, 6))
    
    # Plot all raw data points first
    plt.plot(all_exposure_times_sorted, all_raw_signals_sorted, 'o', markersize=4, alpha=0.6, label='All Raw Data Points')

    # Plot the first fitted line ONLY over its region
    if initial_fit_successful:
        fit_x1 = np.linspace(0, INITIAL_LINEAR_EXPOSURE_MAX_SECONDS, 100)
        plt.plot(fit_x1, initial_light_response_rate * fit_x1 + sensor_bias, 'r--', 
                 label=f'Linear Fit (0-{INITIAL_LINEAR_EXPOSURE_MAX_SECONDS}s): Y = {initial_light_response_rate:.2f}X + {sensor_bias:.2f}')
    
    # Plot the second fitted line ONLY over its region
    if second_fit_successful:
        fit_x2 = np.linspace(second_linear_min_exp, second_linear_max_exp, 100)
        plt.plot(fit_x2, second_light_response_rate * fit_x2 + second_intercept, 'b--', 
                 label=f'Linear Fit ({second_linear_min_exp:.1f}-{second_linear_max_exp:.1f}s): Y = {second_light_response_rate:.2f}X + {second_intercept:.2f}')

    # Add a horizontal line for the dynamically detected saturation threshold
    plt.axhline(fit_saturation_threshold_adu, color='gray', linestyle=':', 
                label=f'Auto Saturation Threshold for Fit ({fit_saturation_threshold_adu:.0f} ADU)')
    
    # Add vertical lines for the exposure time cutoffs/jump points
    if initial_fit_successful:
        plt.axvline(INITIAL_LINEAR_EXPOSURE_MAX_SECONDS, color='darkorange', linestyle=':', label=f'1st Fit End ({INITIAL_LINEAR_EXPOSURE_MAX_SECONDS}s)')
    if not np.isnan(second_linear_min_exp):
        plt.axvline(second_linear_min_exp, color='purple', linestyle=':', label=f'2nd Fit Start ({second_linear_min_exp:.1f}s)')

    plt.xlabel('Exposure Time (s)', fontsize=12)
    plt.ylabel('Average ADU (Raw)', fontsize=12)
    plt.title('Raw Signal vs. Exposure Time for Bias & Linear Region Analysis', fontsize=14)
    plt.grid(True, which="both", ls="-", color='0.8')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_plot_file, dpi=300)
    plt.close()
    print(f"Signal vs Exposure Time plot saved to {output_plot_file}")

    return sensor_bias, fit_saturation_threshold_adu, effective_signal_at_jump_point, second_intercept

def save_to_csv(data, filename, bias):
    """
    Saves the collected signal (bias-subtracted) and noise data to a CSV file.
    """
    if not data:
        print("No data to save to CSV.")
        return

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Effective_Signal', 'Noise', 'Raw_Signal', 'Exposure_Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            effective_signal = row['raw_signal'] - bias # Always subtract the single fundamental bias for CSV
            writer.writerow({
                'Effective_Signal': effective_signal, 
                'Noise': row['noise'],
                'Raw_Signal': row['raw_signal'],
                'Exposure_Time': row['exposure_time']
            })
    print(f"Data saved to {filename}")

def create_loglog_plots(data, filename_standard, filename_adjusted, sensor_bias, dynamic_saturation_adu, effective_signal_at_jump_point, second_intercept):
    """
    Creates TWO log-log plots of bias-subtracted Signal vs. Noise and identifies four key regions.
    One standard plot, and one experimental plot with adjusted bias for the second region.
    """
    # Prepare data common to both plots
    all_raw_signals = np.array([d['raw_signal'] for d in data])
    all_exposure_times = np.array([d['exposure_time'] for d in data])
    noises_all = np.array([d['noise'] for d in data])

    # --- Plot 1: Standard PTC (Single Bias Subtraction) ---
    effective_signals_standard = all_raw_signals - sensor_bias
    
    positive_mask_for_plot_standard = (effective_signals_standard > 0) & (noises_all > 0)
    signals_for_plot_standard = effective_signals_standard[positive_mask_for_plot_standard]
    noises_for_plot_standard = noises_all[positive_mask_for_plot_standard]

    plt.figure(figsize=(12, 8))
    plt.loglog(signals_for_plot_standard, noises_for_plot_standard, 'o', markersize=4, alpha=0.6, label='Data Points')

    plt.xlim(1e-1, 1e5) 
    plt.ylim(1e1, 1e3)  
    plt.grid(True, which="both", ls="-", color='0.8') 

    # 1. Read Noise Region (Standard Plot)
    read_noise_mask = (effective_signals_standard >= 1e-6) & (effective_signals_standard <= READ_NOISE_SIGNAL_MAX_EFFECTIVE) & (noises_all > 0)
    if np.sum(read_noise_mask) > 0:
        read_noise_subset_noises = noises_all[read_noise_mask]
        read_noise_adu = np.mean(read_noise_subset_noises)
        plt.axhline(read_noise_adu, color='red', linestyle='--', linewidth=2, label=f'Read Noise (Estimate at low signal): {read_noise_adu:.2f} ADU')
        plt.text(plt.xlim()[0] * 1.5, read_noise_adu * 1.1, f'Read Noise: {read_noise_adu:.2f} ADU', color='red', verticalalignment='bottom', fontsize=10)
    else:
        print(f"Warning: Not enough data points in Read Noise region (Effective Signal between 0 and {READ_NOISE_SIGNAL_MAX_EFFECTIVE} ADU). Adjust `READ_NOISE_SIGNAL_MAX_EFFECTIVE` or check data.")

    # 2. Photon Noise Region 1 (Standard Plot)
    photon_noise_mask_reg1 = (effective_signals_standard >= PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG1) & \
                             (effective_signals_standard <= PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG1) & \
                             (noises_all > 0)
    if np.sum(photon_noise_mask_reg1) > 1:
        signal_subset_reg1 = effective_signals_standard[photon_noise_mask_reg1]
        noise_squared_subset_reg1 = noises_all[photon_noise_mask_reg1]**2
        slope_ns2_reg1, intercept_ns2_reg1, _, _, _ = linregress(signal_subset_reg1, noise_squared_subset_reg1)
        if slope_ns2_reg1 > 0:
            gain_electrons_per_ADU_1 = 1.0 / slope_ns2_reg1 
            fit_signals_reg1 = np.linspace(PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG1, PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG1, 100)
            fitted_noise_photon_reg1 = np.sqrt(slope_ns2_reg1 * fit_signals_reg1 + intercept_ns2_reg1)
            fitted_noise_photon_reg1[fitted_noise_photon_reg1 <= 0] = np.nan 
            plt.loglog(fit_signals_reg1, fitted_noise_photon_reg1, 'green', linestyle='-', linewidth=2, label=f'Photon Noise Fit 1 (Slope 1/2)')
            gain_x_intercept_1 = gain_electrons_per_ADU_1 
            plt.axvline(gain_x_intercept_1, color='green', linestyle=':', alpha=0.7, label=f'Gain 1 ({gain_x_intercept_1:.3f} e-/ADU)')
            plt.text(gain_x_intercept_1 * 1.05, plt.ylim()[1] * 0.7, f'Gain 1: {gain_electrons_per_ADU_1:.3f} e-/ADU', color='green', rotation=90, verticalalignment='top', fontsize=10)
        else: print("Warning: Photon noise fit REGION 1 resulted in non-positive slope.")
    else: print(f"Warning: Not enough data points in Photon Noise REGION 1 (Effective Signal between {PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG1} and {PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG1} ADU).")

    # 3. Photon Noise Region 2 (Standard Plot)
    photon_noise_mask_reg2 = (effective_signals_standard >= PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG2) & \
                             (effective_signals_standard <= PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG2) & \
                             (noises_all > 0)
    if np.sum(photon_noise_mask_reg2) > 1:
        signal_subset_reg2 = effective_signals_standard[photon_noise_mask_reg2]
        noise_squared_subset_reg2 = noises_all[photon_noise_mask_reg2]**2
        slope_ns2_reg2, intercept_ns2_reg2, _, _, _ = linregress(signal_subset_reg2, noise_squared_subset_reg2)
        if slope_ns2_reg2 > 0:
            gain_electrons_per_ADU_2 = 1.0 / slope_ns2_reg2 
            fit_signals_reg2 = np.linspace(PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG2, PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG2, 100)
            fitted_noise_photon_reg2 = np.sqrt(slope_ns2_reg2 * fit_signals_reg2 + intercept_ns2_reg2)
            fitted_noise_photon_reg2[fitted_noise_photon_reg2 <= 0] = np.nan 
            plt.loglog(fit_signals_reg2, fitted_noise_photon_reg2, 'forestgreen', linestyle='--', linewidth=2, label=f'Photon Noise Fit 2 (Slope 1/2)')
            gain_x_intercept_2 = gain_electrons_per_ADU_2
            plt.axvline(gain_x_intercept_2, color='forestgreen', linestyle=':', alpha=0.7, label=f'Gain 2 ({gain_x_intercept_2:.3f} e-/ADU)')
            plt.text(gain_x_intercept_2 * 1.05, plt.ylim()[1] * 0.7, f'Gain 2: {gain_electrons_per_ADU_2:.3f} e-/ADU', color='forestgreen', rotation=90, verticalalignment='top', fontsize=10)
        else: print("Warning: Photon noise fit REGION 2 resulted in non-positive slope.")
    else: print(f"Warning: Not enough data points in Photon Noise REGION 2 (Effective Signal between {PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG2} and {PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG2} ADU).")

    # 4. Fixed Pattern Noise Region 1 (Standard Plot)
    fpn_mask_reg1 = (effective_signals_standard >= FPN_SIGNAL_MIN_EFFECTIVE_REG1) & (effective_signals_standard <= FPN_SIGNAL_MAX_EFFECTIVE_REG1) & (noises_all > 0)
    if np.sum(fpn_mask_reg1) > 1:
        fpn_signal_subset_log_reg1 = np.log10(effective_signals_standard[fpn_mask_reg1])
        fpn_noise_subset_log_reg1 = np.log10(noises_all[fpn_mask_reg1])
        slope_fpn_reg1, intercept_fpn_reg1, _, _, _ = linregress(fpn_signal_subset_log_reg1, fpn_noise_subset_log_reg1)
        fit_signals_fpn_reg1 = np.linspace(FPN_SIGNAL_MIN_EFFECTIVE_REG1, FPN_SIGNAL_MAX_EFFECTIVE_REG1, 100)
        fitted_noise_fpn_reg1 = 10**(slope_fpn_reg1 * np.log10(fit_signals_fpn_reg1) + intercept_fpn_reg1)
        fitted_noise_fpn_reg1[fitted_noise_fpn_reg1 <= 0] = np.nan 
        plt.loglog(fit_signals_fpn_reg1, fitted_noise_fpn_reg1, 'blue', linestyle='-', linewidth=2, label=f'Fixed Pattern Noise Fit 1 (Slope {slope_fpn_reg1:.2f})')
        if slope_fpn_reg1 != 0 and not np.isinf(slope_fpn_reg1) and not np.isnan(slope_fpn_reg1):
            fpn_x_intercept_log_reg1 = -intercept_fpn_reg1 / slope_fpn_reg1
            fpn_x_intercept_adu_reg1 = 10**fpn_x_intercept_log_reg1
            if 100 < fpn_x_intercept_adu_reg1 < 100000: 
                plt.axvline(fpn_x_intercept_adu_reg1, color='blue', linestyle=':', alpha=0.7, label=f'FPN X-Int 1 ({fpn_x_intercept_adu_reg1:.0f} ADU)')
                plt.text(fpn_x_intercept_adu_reg1 * 1.02, plt.ylim()[0] * 1.5, f'FPN 1: {fpn_x_intercept_adu_reg1:.0f} ADU\n(slope {slope_fpn_reg1:.2f})', color='blue', rotation=90, verticalalignment='bottom', fontsize=10)
            else: print(f"Warning: Calculated FPN X-intercept REGION 1 {fpn_x_intercept_adu_reg1:.0f} ADU is outside typical plot range.")
        else: print("Warning: FPN slope REGION 1 is zero or invalid.")
    else: print(f"Warning: Not enough data points in FPN REGION 1 (Effective Signal between {FPN_SIGNAL_MIN_EFFECTIVE_REG1} and {FPN_SIGNAL_MAX_EFFECTIVE_REG1} ADU).")

    # 5. Fixed Pattern Noise Region 2 (Standard Plot)
    fpn_mask_reg2 = (effective_signals_standard >= FPN_SIGNAL_MIN_EFFECTIVE_REG2) & \
                    (effective_signals_standard <= FPN_SIGNAL_MAX_EFFECTIVE_REG2) & \
                    (noises_all > 0)
    if np.sum(fpn_mask_reg2) > 1:
        fpn_signal_subset_log_reg2 = np.log10(effective_signals_standard[fpn_mask_reg2])
        fpn_noise_subset_log_reg2 = np.log10(noises_all[fpn_mask_reg2])
        slope_fpn_reg2, intercept_fpn_reg2, _, _, _ = linregress(fpn_signal_subset_log_reg2, fpn_noise_subset_log_reg2)
        fit_signals_fpn_reg2 = np.linspace(FPN_SIGNAL_MIN_EFFECTIVE_REG2, FPN_SIGNAL_MAX_EFFECTIVE_REG2, 100)
        fitted_noise_fpn_reg2 = 10**(slope_fpn_reg2 * np.log10(fit_signals_fpn_reg2) + intercept_fpn_reg2)
        fitted_noise_fpn_reg2[fitted_noise_fpn_reg2 <= 0] = np.nan 
        plt.loglog(fit_signals_fpn_reg2, fitted_noise_fpn_reg2, 'steelblue', linestyle='--', linewidth=2, label=f'Fixed Pattern Noise Fit 2 (Slope {slope_fpn_reg2:.2f})')
        if slope_fpn_reg2 != 0 and not np.isinf(slope_fpn_reg2) and not np.isnan(slope_fpn_reg2):
            fpn_x_intercept_log_reg2 = -intercept_fpn_reg2 / slope_fpn_reg2
            fpn_x_intercept_adu_reg2 = 10**fpn_x_intercept_log_reg2
            if 100 < fpn_x_intercept_adu_reg2 < 100000: 
                plt.axvline(fpn_x_intercept_adu_reg2, color='steelblue', linestyle=':', alpha=0.7, label=f'FPN X-Int 2 ({fpn_x_intercept_adu_reg2:.0f} ADU)')
                plt.text(fpn_x_intercept_adu_reg2 * 1.02, plt.ylim()[0] * 1.5, f'FPN 2: {fpn_x_intercept_adu_reg2:.0f} ADU\n(slope {slope_fpn_reg2:.2f})', color='steelblue', rotation=90, verticalalignment='bottom', fontsize=10)
            else: print(f"Warning: Calculated FPN X-intercept REGION 2 {fpn_x_intercept_adu_reg2:.0f} ADU is outside typical plot range.")
        else: print("Warning: FPN slope REGION 2 is zero or invalid.")
    else: print(f"Warning: Not enough data points in FPN REGION 2 (Effective Signal between {FPN_SIGNAL_MIN_EFFECTIVE_REG2} and {FPN_SIGNAL_MAX_EFFECTIVE_REG2} ADU).")

    # 6. Jump Point and Saturation (Standard Plot)
    if not np.isnan(effective_signal_at_jump_point) and effective_signal_at_jump_point > 0:
        plt.axvline(effective_signal_at_jump_point, color='darkviolet', linestyle=':', linewidth=2, label=f'Sensor Response Jump ({effective_signal_at_jump_point:.0f} ADU)')
        plt.text(effective_signal_at_jump_point * 1.05, plt.ylim()[1] * 0.5, 'Response Jump', color='darkviolet', rotation=90, verticalalignment='center', fontsize=10)
    saturation_effective_adu = dynamic_saturation_adu - sensor_bias
    if saturation_effective_adu > 0:
        plt.axvline(saturation_effective_adu, color='purple', linestyle=':', linewidth=2, label='Approx. Saturation Region (Effective Signal)')
        plt.text(saturation_effective_adu * 1.05, plt.ylim()[1] * 0.8, 'Saturation', color='purple', rotation=90, verticalalignment='top', fontsize=10)
    else: print(f"Warning: Calculated effective saturation ADU ({saturation_effective_adu:.2f}) is not positive. Saturation line not plotted.")

    plt.xlabel('Effective Signal (ADU)', fontsize=12)
    plt.ylabel('Noise (ADU)', fontsize=12)
    plt.title('CMOS Sensor Noise Characterization (Log-Log Plot - Standard Bias)', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(filename_standard, dpi=300)
    plt.close()
    print(f"Standard Signal vs Noise plot saved to {filename_standard}")

    # --- Plot 2: Adjusted Bias Plot (Hypothesis Test) ---
    effective_signals_adjusted = np.array([d['raw_signal'] - sensor_bias if d['exposure_time'] <= INITIAL_LINEAR_EXPOSURE_MAX_SECONDS 
                                           else d['raw_signal'] - second_intercept 
                                           for d in data])
    
    # Filter out non-positive values for log plot
    positive_mask_for_plot_adjusted = (effective_signals_adjusted > 0) & (noises_all > 0)
    signals_for_plot_adjusted = effective_signals_adjusted[positive_mask_for_plot_adjusted]
    noises_for_plot_adjusted = noises_all[positive_mask_for_plot_adjusted]

    plt.figure(figsize=(12, 8))
    plt.loglog(signals_for_plot_adjusted, noises_for_plot_adjusted, 'o', markersize=4, alpha=0.6, label='Data Points')

    plt.xlim(1e-1, 1e5) 
    plt.ylim(1e1, 1e3)  
    plt.grid(True, which="both", ls="-", color='0.8') 

    # Plot fundamental read noise for context
    if np.sum(read_noise_mask) > 0:
        read_noise_adu = np.mean(noises_all[read_noise_mask]) # Use noises_all for read noise calculation
        plt.axhline(read_noise_adu, color='red', linestyle='--', linewidth=2, label=f'Read Noise (Fundamental): {read_noise_adu:.2f} ADU')
        plt.text(plt.xlim()[0] * 1.5, read_noise_adu * 1.1, f'Read Noise: {read_noise_adu:.2f} ADU', color='red', verticalalignment='bottom', fontsize=10)

    # Plot the jump point (using the original bias to calculate its position on the X-axis for consistency)
    if not np.isnan(effective_signal_at_jump_point) and effective_signal_at_jump_point > 0:
        plt.axvline(effective_signal_at_jump_point, color='darkviolet', linestyle=':', linewidth=2, label=f'Original Jump Point ({effective_signal_at_jump_point:.0f} ADU)')
        plt.text(effective_signal_at_jump_point * 1.05, plt.ylim()[1] * 0.5, 'Original Jump Point', color='darkviolet', rotation=90, verticalalignment='center', fontsize=10)

    # Plot saturation (using the original bias to calculate its position on the X-axis for consistency)
    if saturation_effective_adu > 0:
        plt.axvline(saturation_effective_adu, color='purple', linestyle=':', linewidth=2, label='Approx. Saturation (Original Bias)')
        plt.text(saturation_effective_adu * 1.05, plt.ylim()[1] * 0.8, 'Saturation', color='purple', rotation=90, verticalalignment='top', fontsize=10)


    plt.xlabel('Effective Signal (ADU) - Bias Adjusted by Region', fontsize=12)
    plt.ylabel('Noise (ADU)', fontsize=12)
    plt.title('CMOS Sensor Noise Characterization (Log-Log Plot - Adjusted Bias by Region)', fontsize=14)
    plt.legend(fontsize=10)
    plt.text(0.5, 0.95, "Note: This plot uses a piecewise bias subtraction for visualization.\n"
             "The X-axis does not consistently represent photoelectrons throughout.", 
             transform=plt.gca().transAxes, fontsize=9, color='gray', ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="gray", lw=1, alpha=0.6))
    plt.tight_layout()
    plt.savefig(filename_adjusted, dpi=300)
    plt.close()
    print(f"Adjusted Bias Signal vs Noise plot saved to {filename_adjusted}")

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        print("Please create this directory and place your FITS image pairs inside.")
        print("Example: `mkdir fits_images_directory` and then copy your `.fits` files into it.")
        exit()

    all_raw_data = collect_all_data(INPUT_DIR)

    if all_raw_data:
        # Step 1: Analyze signal vs exposure curve to get bias, dynamic saturation, jump point, and second_intercept
        sensor_bias_value, dynamic_saturation_adu, effective_signal_at_jump_point, second_intercept_value = analyze_signal_exposure_curve(all_raw_data, OUTPUT_PLOT_FILE_SE)
        
        # Step 2: Save the data to CSV (always using the initial fundamental bias for consistency in CSV)
        save_to_csv(all_raw_data, OUTPUT_CSV_FILE, sensor_bias_value)
        
        # Step 3: Create the log-log plots
        create_loglog_plots(all_raw_data, 
                            OUTPUT_PLOT_FILE_SN_STANDARD, 
                            OUTPUT_PLOT_FILE_SN_ADJUSTED, 
                            sensor_bias_value, 
                            dynamic_saturation_adu, 
                            effective_signal_at_jump_point, 
                            second_intercept_value)
    else:
        print("No valid data collected. Please check the input directory and FITS file formats.")