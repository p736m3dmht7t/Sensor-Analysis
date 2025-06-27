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
# Using the path provided by the user
INPUT_DIR = r'D:\Astrophotography\2025-06-10\FLAT\B' 
OUTPUT_CSV_FILE = 'signal_noise_data.csv'
OUTPUT_PLOT_FILE_SN = 'signal_noise_plot.png'
OUTPUT_PLOT_FILE_SE = 'signal_exposure_plot.png'

# Define *BIAS-SUBTRACTED* signal ranges (in ADU) for fitting different noise regions on the S/N plot.
# These values are approximate and may need tuning based on your specific sensor and data.
# They are now relative to an estimated zero signal level (after bias subtraction).
READ_NOISE_SIGNAL_MAX_EFFECTIVE = 8         # Max bias-subtracted signal for estimating read noise
PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE = 400     # Min bias-subtracted signal for fitting photon noise
PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE = 40000   # Max bias-subtracted signal for fitting photon noise
FPN_SIGNAL_MIN_EFFECTIVE = 50000            # Min bias-subtracted signal for fitting fixed pattern noise
FPN_SIGNAL_MAX_EFFECTIVE = 61000            # Max bias-subtracted signal for fitting fixed pattern noise

# Initial maximum exposure time for the first linear fit (bias calculation).
# This is kept as a constant to define the very first, often cleanest, linear region.
INITIAL_LINEAR_EXPOSURE_MAX_SECONDS = 1.0 

# Heuristic for detecting jump point: how many std deviations away from initial fit for a new region
JUMP_DETECTION_STD_MULTIPLIER = 5 
# How many consecutive points must deviate to confirm a jump
JUMP_DETECTION_CONSECUTIVE_POINTS = 3 
# Smallest exposure time to consider for jump detection (to avoid noise at 0 exp time)
JUMP_DETECTION_MIN_EXP = 0.1 

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
    Returns the calculated sensor bias and saturation threshold.
    """
    # Filter data with valid exposure times
    filtered_data = [d for d in data if d['exposure_time'] is not None]
    if not filtered_data:
        print("Error: No valid data points with exposure time found for signal-exposure analysis.")
        return 0.0, 0.0

    # Sort data by exposure time for proper analysis
    filtered_data.sort(key=lambda x: x['exposure_time'])
    all_raw_signals = np.array([d['raw_signal'] for d in filtered_data])
    all_exposure_times = np.array([d['exposure_time'] for d in filtered_data])

    # --- Dynamic Saturation Threshold Detection ---
    # Use a high percentile of observed signals to estimate the saturation level
    # Then apply a small safety margin for where the linear fit should end
    dynamic_saturation_level = np.percentile(all_raw_signals, 99.5) # Captures the plateau
    fit_saturation_threshold_adu = dynamic_saturation_level * 0.98 # Safety margin for linear fit

    # --- Fit 1: Initial Linear Region (for Bias Calculation) ---
    valid_data_for_bias_fit = [d for d in filtered_data if 
                                d['exposure_time'] <= INITIAL_LINEAR_EXPOSURE_MAX_SECONDS and
                                d['raw_signal'] < fit_saturation_threshold_adu]

    sensor_bias = 0.0
    initial_light_response_rate = 0.0
    initial_r_squared = 0.0

    if len(valid_data_for_bias_fit) >= 2:
        raw_signals_bias_fit = np.array([d['raw_signal'] for d in valid_data_for_bias_fit])
        exposure_times_bias_fit = np.array([d['exposure_time'] for d in valid_data_for_bias_fit])
        slope_bias_fit, intercept_bias_fit, r_value_bias_fit, _, _ = linregress(exposure_times_bias_fit, raw_signals_bias_fit)
        sensor_bias = intercept_bias_fit
        initial_light_response_rate = slope_bias_fit
        initial_r_squared = r_value_bias_fit**2

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

    if initial_r_squared > 0.9: # Only proceed if initial fit was reasonably good
        # Calculate residuals against the *extrapolation* of the initial fit for all non-saturated data
        extrapolated_initial_signals = initial_light_response_rate * all_exposure_times + sensor_bias
        residuals = all_raw_signals - extrapolated_initial_signals

        # Calculate std dev of residuals in the initial fit region to set a threshold for deviation
        initial_residuals_in_fit_region = residuals[all_exposure_times <= INITIAL_LINEAR_EXPOSURE_MAX_SECONDS]
        # Avoid std dev issues if very few points or perfectly linear
        std_dev_initial_residuals = np.std(initial_residuals_in_fit_region) 
        if std_dev_initial_residuals < 1e-6: # Set a floor to avoid division by zero or too small a threshold
            std_dev_initial_residuals = 1.0 # Arbitrary small but non-zero value for thresholding

        # Find the jump point: first exposure where residuals consistently exceed a threshold
        consecutive_deviations = 0
        jump_idx = -1
        for i in range(len(all_exposure_times)):
            if all_exposure_times[i] <= INITIAL_LINEAR_EXPOSURE_MAX_SECONDS:
                continue # Skip points already in the initial fit region

            if all_exposure_times[i] < JUMP_DETECTION_MIN_EXP:
                continue # Skip very early points if jump detection should start later

            # Consider only points below saturation threshold for jump detection
            if all_raw_signals[i] >= fit_saturation_threshold_adu:
                break 

            if residuals[i] > JUMP_DETECTION_STD_MULTIPLIER * std_dev_initial_residuals:
                consecutive_deviations += 1
                if consecutive_deviations >= JUMP_DETECTION_CONSECUTIVE_POINTS:
                    # Found consistent deviation, set jump_idx to the start of this consistency
                    jump_idx = i - JUMP_DETECTION_CONSECUTIVE_POINTS + 1
                    break
            else:
                consecutive_deviations = 0 # Reset if deviation is not consistent

        if jump_idx != -1:
            second_linear_min_exp = all_exposure_times[jump_idx]
            
            # Determine the end of the second linear region (before saturation fully hits)
            # Find the largest exposure time where raw signal is still below the fit_saturation_threshold_adu
            valid_exposure_times_below_saturation = all_exposure_times[all_raw_signals < fit_saturation_threshold_adu]
            if len(valid_exposure_times_below_saturation) > 0:
                # Take the max exposure time in the non-saturated region, apply a small buffer backward
                second_linear_max_exp = np.max(valid_exposure_times_below_saturation) * 0.99 
                # Ensure max_exp is greater than min_exp
                if second_linear_max_exp <= second_linear_min_exp:
                    second_linear_max_exp = np.max(all_exposure_times) # Fallback to max if calculation is bad
                    print("Warning: second_linear_max_exp adjusted due to calculation issue.")
            else:
                second_linear_max_exp = np.max(all_exposure_times) # Fallback

            # Filter data for the second fit using dynamically determined ranges
            valid_data_for_second_fit = [d for d in filtered_data if 
                                          d['exposure_time'] >= second_linear_min_exp and
                                          d['exposure_time'] <= second_linear_max_exp and
                                          d['raw_signal'] < fit_saturation_threshold_adu]

            if len(valid_data_for_second_fit) >= 2:
                raw_signals_second_fit = np.array([d['raw_signal'] for d in valid_data_for_second_fit])
                exposure_times_second_fit = np.array([d['exposure_time'] for d in valid_data_for_second_fit])
                slope_second_fit, intercept_second_fit, r_value_second_fit, _, _ = linregress(exposure_times_second_fit, raw_signals_second_fit)
                second_light_response_rate = slope_second_fit
                second_intercept = intercept_second_fit

                print(f"\n--- Second Linear Region Fit (Detected between {second_linear_min_exp:.2f}s and {second_linear_max_exp:.2f}s) ---")
                print(f"Fitted linear model (Second Region): Signal = {second_light_response_rate:.2f} * ExposureTime + {second_intercept:.2f}")
                print(f"Estimated Light Current Rate (Second Region): {second_light_response_rate:.2f} ADU/s")
                print(f"R-squared for fit: {r_value_second_fit**2:.4f}")
                print(f"------------------------------------------------------------------------------------")
            else:
                print("Warning: Not enough valid data points for second linear fit after jump detection. Skipping second fit.")
        else:
            print("Info: No significant jump detected in the signal-exposure curve after initial fit.")
            second_linear_min_exp = np.nan # Ensure it's NaN if no jump found
            second_linear_max_exp = np.nan # Ensure it's NaN if no jump found

    # --- Plot Signal vs Exposure Time ---
    plt.figure(figsize=(10, 6))
    
    # Plot all raw data points first
    plt.plot(all_exposure_times, all_raw_signals, 'o', markersize=4, alpha=0.6, label='All Raw Data Points')

    # Plot the first fitted line ONLY over its region
    if initial_r_squared > 0.0:
        fit_x1 = np.linspace(0, INITIAL_LINEAR_EXPOSURE_MAX_SECONDS, 100)
        plt.plot(fit_x1, initial_light_response_rate * fit_x1 + sensor_bias, 'r--', 
                 label=f'Linear Fit (0-{INITIAL_LINEAR_EXPOSURE_MAX_SECONDS}s): Y = {initial_light_response_rate:.2f}X + {sensor_bias:.2f}')
    
    # Plot the second fitted line ONLY over its region
    if not np.isnan(second_light_response_rate) and not np.isnan(second_linear_min_exp) and not np.isnan(second_linear_max_exp):
        fit_x2 = np.linspace(second_linear_min_exp, second_linear_max_exp, 100)
        plt.plot(fit_x2, second_light_response_rate * fit_x2 + second_intercept, 'b--', 
                 label=f'Linear Fit ({second_linear_min_exp:.1f}-{second_linear_max_exp:.1f}s): Y = {second_light_response_rate:.2f}X + {second_intercept:.2f}')

    # Add a horizontal line for the dynamically detected saturation threshold
    plt.axhline(fit_saturation_threshold_adu, color='gray', linestyle=':', 
                label=f'Auto Saturation Threshold for Fit ({fit_saturation_threshold_adu:.0f} ADU)')
    
    # Add vertical lines for the exposure time cutoffs/jump points
    plt.axvline(INITIAL_LINEAR_EXPOSURE_MAX_SECONDS, color='darkorange', linestyle=':', label=f'1st Fit End ({INITIAL_LINEAR_EXPOSURE_MAX_SECONDS}s)')
    if not np.isnan(second_linear_min_exp):
        plt.axvline(second_linear_min_exp, color='purple', linestyle=':', label=f'2nd Fit Start ({second_linear_min_exp:.1f}s)')
        # Optionally add a line for the end of the second fit region if it's distinct
        # plt.axvline(second_linear_max_exp, color='cyan', linestyle=':', label=f'2nd Fit End ({second_linear_max_exp:.1f}s)')


    plt.xlabel('Exposure Time (s)', fontsize=12)
    plt.ylabel('Average ADU (Raw)', fontsize=12)
    plt.title('Raw Signal vs. Exposure Time for Bias & Linear Region Analysis', fontsize=14)
    plt.grid(True, which="both", ls="-", color='0.8')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_plot_file, dpi=300)
    plt.close()
    print(f"Signal vs Exposure Time plot saved to {output_plot_file}")

    return sensor_bias, fit_saturation_threshold_adu # Return saturation threshold for S/N plot

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
            effective_signal = row['raw_signal'] - bias
            # Keep original negative/zero effective signals; filtering for plot will handle log
            writer.writerow({
                'Effective_Signal': effective_signal, 
                'Noise': row['noise'],
                'Raw_Signal': row['raw_signal'],
                'Exposure_Time': row['exposure_time']
            })
    print(f"Data saved to {filename}")

def create_loglog_plot(data, filename, sensor_bias, saturation_effective_adu):
    """
    Creates a log-log plot of bias-subtracted Signal vs. Noise and identifies four key regions.
    """
    # Apply bias subtraction to all signal values
    effective_signals_all = np.array([d['raw_signal'] - sensor_bias for d in data])
    noises_all = np.array([d['noise'] for d in data])

    # Filter out any data points where effective_signal is <= 0 or noise is <= 0 for log plot
    positive_mask_for_plot = (effective_signals_all > 0) & (noises_all > 0)
    signals_for_plot = effective_signals_all[positive_mask_for_plot]
    noises_for_plot = noises_all[positive_mask_for_plot]

    plt.figure(figsize=(12, 8))
    plt.loglog(signals_for_plot, noises_for_plot, 'o', markersize=4, alpha=0.6, label='Data Points')

    # Set specific x and y-axis limits and ensure minor grid lines on both axes
    plt.xlim(1e-1, 1e5) # From 0.1 ADU to 100,000 ADU (6 log cycles)
    plt.ylim(1e1, 1e3)  # From 10 ADU to 1,000 ADU
    plt.grid(True, which="both", ls="-", color='0.8') # Ensure minor grid lines for both axes

    # --- Region Analysis and Fitting ---

    # 1. Read Noise Region (Slope ~0)
    # Use effective_signals_all for mask to include all data for fitting
    read_noise_mask = (effective_signals_all >= 1e-6) & (effective_signals_all <= READ_NOISE_SIGNAL_MAX_EFFECTIVE) & (noises_all > 0)
    if np.sum(read_noise_mask) > 0:
        read_noise_subset_noises = noises_all[read_noise_mask]
        read_noise_adu = np.mean(read_noise_subset_noises)
        
        plt.axhline(read_noise_adu, color='red', linestyle='--', linewidth=2, label=f'Read Noise (Estimate at low signal): {read_noise_adu:.2f} ADU')
        # Adjust text position slightly for better visibility
        # Using fixed offset from xlim for annotation
        plt.text(plt.xlim()[0] * 1.5, read_noise_adu * 1.1, 
                 f'Read Noise: {read_noise_adu:.2f} ADU', 
                 color='red', verticalalignment='bottom', fontsize=10)
    else:
        print(f"Warning: Not enough data points in Read Noise region (Effective Signal between 0 and {READ_NOISE_SIGNAL_MAX_EFFECTIVE} ADU). Adjust `READ_NOISE_SIGNAL_MAX_EFFECTIVE` or check data.")


    # 2. Photon Noise Region (Slope 1/2) - Gain calculation
    photon_noise_mask = (effective_signals_all >= PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE) & \
                        (effective_signals_all <= PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE) & \
                        (noises_all > 0)
    
    if np.sum(photon_noise_mask) > 1:
        signal_subset = effective_signals_all[photon_noise_mask]
        noise_squared_subset = noises_all[photon_noise_mask]**2

        slope_ns2, intercept_ns2, r_value, p_value, std_err = linregress(signal_subset, noise_squared_subset)

        if slope_ns2 > 0:
            # Renamed for clarity: Gain in electrons per ADU (e-/ADU)
            gain_electrons_per_ADU = 1.0 / slope_ns2 
            
            # Plot the fitted line within its relevant fitting range
            fit_signals = np.linspace(PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE, PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE, 100)
            # Reconstruct noise from the Noise^2 fit: Noise = sqrt(slope*Signal + intercept)
            fitted_noise_photon = np.sqrt(slope_ns2 * fit_signals + intercept_ns2)
            fitted_noise_photon[fitted_noise_photon <= 0] = np.nan 
            plt.loglog(fit_signals, fitted_noise_photon, 'green', linestyle='--', linewidth=2, label=f'Photon Noise Fit (Slope 1/2)')
            
            # --- Annotate Gain as X-intercept (where Noise = 1 ADU) ---
            # The X-intercept of the slope 1/2 line is the Gain_e_per_ADU
            gain_x_intercept = gain_electrons_per_ADU 
            
            # Plot vertical line for gain
            plt.axvline(gain_x_intercept, color='green', linestyle=':', alpha=0.7, label=f'Gain ({gain_x_intercept:.3f} e-/ADU)') # Label changed, 3 decimal places
            
            # Annotate gain value
            # Place annotation above the line, shifted slightly right, with 3 decimal places
            plt.text(gain_x_intercept * 1.05, plt.ylim()[1] * 0.7, 
                     f'Gain: {gain_electrons_per_ADU:.3f} e-/ADU', # Text changed to 3 decimal places and correct unit
                     color='green', rotation=90, verticalalignment='top', fontsize=10)

        else:
            print("Warning: Photon noise fit resulted in non-positive slope. Cannot calculate gain.")
    else:
        print(f"Warning: Not enough data points in Photon Noise region (Effective Signal between {PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE} and {PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE} ADU). Adjust ranges or check data.")


    # 3. Fixed Pattern Noise Region (Slope 1)
    fpn_mask = (effective_signals_all >= FPN_SIGNAL_MIN_EFFECTIVE) & (effective_signals_all <= FPN_SIGNAL_MAX_EFFECTIVE) & (noises_all > 0)
    if np.sum(fpn_mask) > 1:
        fpn_signal_subset_log = np.log10(effective_signals_all[fpn_mask])
        fpn_noise_subset_log = np.log10(noises_all[fpn_mask])

        slope_fpn, intercept_fpn, r_value, p_value, std_err = linregress(fpn_signal_subset_log, fpn_noise_subset_log)

        # Plot the fitted line within its relevant fitting range
        fit_signals_fpn = np.linspace(FPN_SIGNAL_MIN_EFFECTIVE, FPN_SIGNAL_MAX_EFFECTIVE, 100)
        fitted_noise_fpn = 10**(slope_fpn * np.log10(fit_signals_fpn) + intercept_fpn)
        fitted_noise_fpn[fitted_noise_fpn <= 0] = np.nan 
        plt.loglog(fit_signals_fpn, fitted_noise_fpn, 'blue', linestyle='--', linewidth=2, label=f'Fixed Pattern Noise Fit (Slope {slope_fpn:.2f})')

        # Calculate x-intercept for FPN (where Noise = 1 ADU)
        if slope_fpn != 0 and not np.isinf(slope_fpn) and not np.isnan(slope_fpn):
            fpn_x_intercept_log = -intercept_fpn / slope_fpn
            fpn_x_intercept_adu = 10**fpn_x_intercept_log
            if 100 < fpn_x_intercept_adu < 100000: 
                plt.axvline(fpn_x_intercept_adu, color='blue', linestyle=':', alpha=0.7, label=f'FPN X-Intercept ({fpn_x_intercept_adu:.0f} ADU)')
                plt.text(fpn_x_intercept_adu * 1.02, plt.ylim()[0] * 1.5, 
                         f'FPN X-Intercept: {fpn_x_intercept_adu:.0f} ADU\n(based on slope {slope_fpn:.2f})', 
                         color='blue', rotation=90, verticalalignment='bottom', fontsize=10)
            else:
                 print(f"Warning: Calculated FPN X-intercept {fpn_x_intercept_adu:.0f} ADU is outside typical plot range, not annotating.")
        else:
            print("Warning: FPN slope is zero or invalid. Cannot calculate or plot x-intercept as described.")
    else:
        print(f"Warning: Not enough data points in FPN region (Effective Signal between {FPN_SIGNAL_MIN_EFFECTIVE} and {FPN_SIGNAL_MAX_EFFECTIVE} ADU). Adjust ranges or check data.")


    # 4. Saturation Region
    # Saturation threshold is for RAW ADU, so subtract bias for effective signal axis placement
    # This comes directly from the analysis function now
    if saturation_effective_adu > 0:
        plt.axvline(saturation_effective_adu, color='purple', linestyle=':', linewidth=2, label='Approx. Saturation Region (Effective Signal)')
        plt.text(saturation_effective_adu * 1.05, plt.ylim()[1] * 0.8, 
                 'Saturation', color='purple', rotation=90, verticalalignment='top', fontsize=10)
    else:
        print(f"Warning: Calculated effective saturation ADU ({saturation_effective_adu:.2f}) is not positive. Saturation line not plotted.")

    plt.xlabel('Effective Signal (ADU)', fontsize=12)
    plt.ylabel('Noise (ADU)', fontsize=12)
    plt.title('CMOS Sensor Noise Characterization (Log-Log Plot)', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Signal vs Noise plot saved to {filename}")

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        print("Please create this directory and place your FITS image pairs inside.")
        print("Example: `mkdir fits_images_directory` and then copy your `.fits` files into it.")
        exit()

    all_raw_data = collect_all_data(INPUT_DIR)

    if all_raw_data:
        # Step 1: Analyze signal vs exposure curve to get bias and dynamic saturation
        sensor_bias_value, dynamic_saturation_adu = analyze_signal_exposure_curve(all_raw_data, OUTPUT_PLOT_FILE_SE)
        
        # Step 2: Save the data to CSV (now including effective signal)
        save_to_csv(all_raw_data, OUTPUT_CSV_FILE, sensor_bias_value)
        
        # Step 3: Create the log-log plot using bias-subtracted data and dynamic saturation
        create_loglog_plot(all_raw_data, OUTPUT_PLOT_FILE_SN, sensor_bias_value, dynamic_saturation_adu - sensor_bias_value)
    else:
        print("No valid data collected. Please check the input directory and FITS file formats.")