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
INPUT_DIR = r'D:\Astrophotography\2025-06-15\FLAT'
OUTPUT_CSV_FILE = 'signal_noise_data.csv'
OUTPUT_PLOT_FILE_SN = 'signal_noise_plot.png'
OUTPUT_PLOT_FILE_SE = 'signal_exposure_plot.png'

# Define *BIAS-SUBTRACTED* signal ranges (in ADU) for fitting different noise regions on the S/N plot.
# These values are now split into two regions based on the expected sensor behavior change.
# Adjust these by examining the generated plots and console output.

# Region 1: Low-to-Mid Signal (before the detected 'jump' in gain/response)
PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG1 = 500   # Min effective signal for Photon Noise Fit 1
PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG1 = 10000 # Max effective signal for Photon Noise Fit 1 (before jump)
FPN_SIGNAL_MIN_EFFECTIVE_REG1 = 100000          # Min effective signal for FPN Fit 1
FPN_SIGNAL_MAX_EFFECTIVE_REG1 = 110000          # Max effective signal for FPN Fit 1

# Region 2: Mid-to-High Signal (after the detected 'jump' in gain/response, before saturation)
PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG2 = 10000 # Min effective signal for Photon Noise Fit 2 (after jump)
PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG2 = 50000 # Max effective signal for Photon Noise Fit 2
FPN_SIGNAL_MIN_EFFECTIVE_REG2 = 100000          # Min effective signal for FPN Fit 2
FPN_SIGNAL_MAX_EFFECTIVE_REG2 = 110000          # Max effective signal for FPN Fit 2

# Initial maximum exposure time for the first linear fit in signal-exposure curve (for light frames).
INITIAL_LINEAR_EXPOSURE_MAX_SECONDS = 1.0

# Heuristics for automatically detecting the 'jump' point in the signal-exposure curve.
JUMP_DETECTION_STD_MULTIPLIER = 5           # How many std deviations away from initial fit for a new region
JUMP_DETECTION_CONSECUTIVE_POINTS = 3       # How many consecutive points must deviate to confirm a jump
JUMP_DETECTION_MIN_EXP = 0.1                # Smallest exposure time to consider for jump detection (to avoid noise at 0 exp time)

# Keywords to identify filter types in FITS header (case-insensitive check will be used)
DARK_FILTER_KEYWORDS = ['DARK'] # Common keywords for dark/bias frames
LIGHT_FILTER_KEYWORDS = ['B', 'V', 'R', 'I', 'NONE'] # Common keywords for light/flat frames (changed 'None' to 'NONE' as it's often uppercase in headers)

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
            filter_name = str(header.get('FILTER', 'N/A')).strip() # Ensure string and strip whitespace
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
    Divides images into 9 subframes and calculates raw signal and noise for each.
    Returns lists of raw_signals and noises for all 9 subframes.
    """
    if img_data1 is None or img_data2 is None:
        return [], []

    height, width = img_data1.shape

    subframe_height = height // 3
    subframe_width = width // 3

    raw_signals_subframes = []
    noises_subframes = []

    for i in range(3): # Row index for subframes
        for j in range(3): # Column index for subframes
            h_start = i * subframe_height
            h_end = (i + 1) * subframe_height if i < 2 else height
            w_start = j * subframe_width
            w_end = (j + 1) * subframe_width if j < 2 else width

            sub_img1 = img_data1[h_start:h_end, w_start:w_end]
            sub_img2 = img_data2[h_start:h_end, w_start:w_end]

            raw_signal = np.mean((sub_img1 + sub_img2) / 2.0)
            diff_img = sub_img2 - sub_img1
            noise = np.std(diff_img, ddof=1) / np.sqrt(2.0)

            if np.isnan(noise) or noise == 0:
                noise = 1e-6 # Avoid log(0) or division by zero issues later

            raw_signals_subframes.append(raw_signal)
            noises_subframes.append(noise)
    return raw_signals_subframes, noises_subframes

def collect_all_data(input_dir):
    """
    Recursively iterates through the directory, finds FITS pairs,
    and collects all raw signal, noise, and header metadata for each subframe.
    """
    all_fits_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.fits', '.fit')):
                all_fits_files.append(os.path.join(root, file))

    all_fits_files.sort()

    if len(all_fits_files) < 2:
        print(f"Error: Found {len(all_fits_files)} FITS files. Need at least 2 files for pairing.")
        return [], 'N/A', 'N/A', 'N/A', 'N/A'
    if len(all_fits_files) % 2 != 0:
        print(f"Warning: Odd number of FITS files ({len(all_fits_files)}) found in '{input_dir}'. "
              "The last file will be ignored. Ensure all images have a pair.")

    all_data = []
    processed_pairs_count = 0

    # Store first header metadata found for plot titles (assuming consistent across all images)
    global_instrume, global_set_temp, global_gain_setting, global_offset_setting = 'N/A', 'N/A', 'N/A', 'N/A'
    metadata_found = False

    for i in range(0, len(all_fits_files) - 1, 2):
        img_path1 = all_fits_files[i]
        img_path2 = all_fits_files[i+1]

        print(f"Processing pair: {os.path.basename(img_path1)} and {os.path.basename(img_path2)}")

        img_data1, exp_time1, filter_name1, instrume1, set_temp1, gain_setting1, offset_setting1 = load_fits_and_extract_info(img_path1)
        img_data2, exp_time2, filter_name2, instrume2, set_temp2, gain_setting2, offset_setting2 = load_fits_and_extract_info(img_path2)

        # Store metadata from the first valid image loaded
        if not metadata_found and instrume1 != 'N/A':
            global_instrume = instrume1
            global_set_temp = set_temp1
            global_gain_setting = gain_setting1
            global_offset_setting = offset_setting1
            metadata_found = True

        if img_data1 is not None and img_data2 is not None:
            if img_data1.shape != img_data2.shape:
                print(f"Warning: Image dimensions mismatch for pair {os.path.basename(img_path1)} and {os.path.basename(img_path2)}. Skipping pair.")
                continue
            if exp_time1 is None or exp_time2 is None or abs(exp_time1 - exp_time2) > 1e-6:
                print(f"Warning: Inconsistent or missing exposure times for pair {os.path.basename(img_path1)} and {os.path.basename(img_path2)}. Using average if present, otherwise skipping exposure time for this pair.")
                avg_exp_time = (exp_time1 + exp_time2) / 2.0 if exp_time1 is not None and exp_time2 is not None else None
            else:
                avg_exp_time = exp_time1 # They should be the same

            # Ensure filters are consistent for the pair
            if filter_name1.upper() != filter_name2.upper(): # Case-insensitive comparison
                print(f"Warning: Filter mismatch for pair {os.path.basename(img_path1)} ({filter_name1}) and {os.path.basename(img_path2)} ({filter_name2}). Using filter from first image.")

            raw_signals_subframes, noises_subframes = calculate_subframe_metrics(img_data1, img_data2)

            all_data.append({
                'raw_signal_subframes': raw_signals_subframes,
                'noise_subframes': noises_subframes,
                'exposure_time': avg_exp_time,
                'filter': filter_name1, # Use filter from first image in pair
            })
            processed_pairs_count += 1
        else:
            print(f"Skipping pair due to loading error: {os.path.basename(img_path1)}, {os.path.basename(img_path2)}")

    print(f"\nSuccessfully processed {processed_pairs_count} image pairs.")
    return all_data, global_instrume, global_set_temp, global_gain_setting, global_offset_setting

def calculate_subframe_biases(all_collected_data):
    """
    Calculates the bias for each of the 9 subframes using minimum exposure DARK images.
    Returns lists of 9 bias values (mean) and 9 bias std dev values.
    """
    dark_images_data = [d for d in all_collected_data if d['filter'].upper() in DARK_FILTER_KEYWORDS and d['exposure_time'] is not None]

    if not dark_images_data:
        print(f"Warning: No DARK images found with filter keywords {DARK_FILTER_KEYWORDS} for subframe bias calculation. All subframe biases will be set to 0.")
        return [0.0] * 9, [0.0] * 9

    min_dark_exp_time = min(d['exposure_time'] for d in dark_images_data)
    min_exp_dark_images = [d for d in dark_images_data if d['exposure_time'] == min_dark_exp_time]

    if not min_exp_dark_images: # Should not happen if dark_images_data is not empty
        print("Warning: Could not find minimum exposure DARK images. All subframe biases will be set to 0.")
        return [0.0] * 9, [0.0] * 9

    subframe_raw_signals_per_idx = [[] for _ in range(9)]
    for dark_entry in min_exp_dark_images:
        if len(dark_entry['raw_signal_subframes']) == 9:
            for j in range(9):
                subframe_raw_signals_per_idx[j].append(dark_entry['raw_signal_subframes'][j])
        else:
            print(f"Warning: Unexpected number of subframes ({len(dark_entry['raw_signal_subframes'])}) in a dark image entry. Skipping.")

    subframe_biases = []
    subframe_biases_std = []

    print(f"\n--- Subframe Bias Calculation (from DARK images at {min_dark_exp_time}s) ---")
    for i in range(9):
        if subframe_raw_signals_per_idx[i]:
            bias_mean = np.mean(subframe_raw_signals_per_idx[i])
            bias_std = np.std(subframe_raw_signals_per_idx[i], ddof=1) # Sample standard deviation
            subframe_biases.append(bias_mean)
            subframe_biases_std.append(bias_std)
            print(f"  Subframe {i}: {bias_mean:.3f} +/- {bias_std:.3f} ADU Bias")
        else:
            subframe_biases.append(0.0)
            subframe_biases_std.append(0.0)
            print(f"  Subframe {i}: No data for bias calculation.")
    print(f"--------------------------------------------------\n")

    return subframe_biases, subframe_biases_std

def calculate_linregress_stats(x, y):
    """
    Calculates slope, intercept, and their standard errors using linregress.
    """
    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan # Not enough points for fit

    slope, intercept, r_value, p_value, stderr_slope = linregress(x, y)

    # Calculate residuals and standard error of the estimate (residual standard deviation)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    if len(x) > 2: # Need at least 3 points for n-2 degrees of freedom
        s_y_x = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
        # Calculate standard error of the intercept
        # Formula: s_y_x * sqrt( (sum(x^2)) / (N * sum((x - mean(x))^2)) )
        sum_x_squared = np.sum(x**2)
        sum_x_minus_mean_x_squared = np.sum((x - np.mean(x))**2)
        if sum_x_minus_mean_x_squared == 0: # Avoid division by zero if all x are same
            stderr_intercept = np.nan
        else:
            stderr_intercept = s_y_x * np.sqrt(sum_x_squared / (len(x) * sum_x_minus_mean_x_squared))
    else: # For exactly 2 points, stderr is not well-defined or very large
        stderr_intercept = np.nan

    return slope, intercept, stderr_slope, stderr_intercept

def analyze_signal_exposure_curve(all_collected_data, subframe_biases, plot_metadata, output_plot_file):
    """
    Analyzes the signal vs exposure time curve for LIGHT images.
    Performs two linear fits:
    1. For the initial region to analyze light response.
    2. For a second linear region by automatically detecting the jump.
    Uses the mean of subframe biases for calculating average effective signal for this plot.
    Returns dynamic saturation threshold, effective signal at jump point.
    """
    # Initialize return values (ensuring they are always defined)
    dynamic_saturation_adu = 0.0
    raw_signal_at_jump_point = np.nan

    # Filter for LIGHT images and average signal per exposure time across all subframes
    light_images_data = [d for d in all_collected_data if d['filter'].upper() in LIGHT_FILTER_KEYWORDS and d['exposure_time'] is not None]

    if not light_images_data:
        print(f"Error: No LIGHT images found with filter keywords {LIGHT_FILTER_KEYWORDS} for signal-exposure analysis.")
        return dynamic_saturation_adu, raw_signal_at_jump_point

    # Calculate average raw signal for each light image pair (mean across its 9 subframes)
    avg_raw_signals = np.array([np.mean(d['raw_signal_subframes']) for d in light_images_data])
    exposure_times_for_plot = np.array([d['exposure_time'] for d in light_images_data])

    # Sort data by exposure time for proper analysis
    sort_indices = np.argsort(exposure_times_for_plot)
    all_raw_signals_sorted = avg_raw_signals[sort_indices]
    all_exposure_times_sorted = exposure_times_for_plot[sort_indices]

    # --- Dynamic Saturation Threshold Detection ---
    dynamic_saturation_level = np.percentile(all_raw_signals_sorted, 99.5)
    fit_saturation_threshold_adu = dynamic_saturation_level * 0.98
    dynamic_saturation_adu = fit_saturation_threshold_adu # Assign the actual value for return

    # --- Fit 1: Initial Linear Region (from LIGHT frames) ---
    mask_initial_fit = (all_exposure_times_sorted <= INITIAL_LINEAR_EXPOSURE_MAX_SECONDS) & \
                       (all_raw_signals_sorted < fit_saturation_threshold_adu)

    # Use the mean of the calculated subframe biases for the intercept of this plot's line
    mean_subframe_bias = np.mean(subframe_biases) if len(subframe_biases) > 0 else 0.0

    initial_light_response_rate = 0.0
    initial_fit_successful = False

    if np.sum(mask_initial_fit) >= 2:
        exposure_times_initial_fit = all_exposure_times_sorted[mask_initial_fit]
        raw_signals_initial_fit = all_raw_signals_sorted[mask_initial_fit]

        slope_initial_fit, intercept_raw_initial_fit, stderr_slope_initial_fit, stderr_intercept_raw_initial_fit = \
            calculate_linregress_stats(exposure_times_initial_fit, raw_signals_initial_fit)

        initial_light_response_rate = slope_initial_fit
        initial_fit_successful = True

        # Calculate bias-subtracted intercept and its error (slope error is unaffected by constant subtraction)
        bias_subtracted_intercept_initial_fit = intercept_raw_initial_fit - mean_subframe_bias
        stderr_bias_subtracted_intercept_initial_fit = stderr_intercept_raw_initial_fit # Std error of a constant subtracted doesn't change

        print(f"--- Light Curve Analysis (Initial Region <= {INITIAL_LINEAR_EXPOSURE_MAX_SECONDS}s) ---")
        print(f"Fitted linear model of the bias-subtracted signal (via linregress): "
              f"Y = ({initial_light_response_rate:.2f} +/- {stderr_slope_initial_fit:.2f}) * ExposureTime + "
              f"({bias_subtracted_intercept_initial_fit:.2f} +/- {stderr_bias_subtracted_intercept_initial_fit:.2f})")
        print(f"--------------------------------------------------\n")
    else:
        print("Warning: Not enough unique light curve points for initial linear fit. Skipping analysis of this region.\n")

    # --- Fit 2: Second Linear Region (Dynamic Detection of Jump) ---
    second_light_response_rate = np.nan
    second_intercept_fit = np.nan # This is the intercept from the fit of the second region
    second_linear_min_exp = np.nan
    second_linear_max_exp = np.nan
    second_fit_successful = False

    if initial_fit_successful:
        # Calculate residuals against the *extrapolation* of the initial fit using its OWN intercept
        extrapolated_initial_signals = initial_light_response_rate * all_exposure_times_sorted + intercept_raw_initial_fit
        residuals = all_raw_signals_sorted - extrapolated_initial_signals # Corrected: Use all_raw_signals_sorted

        initial_residuals_in_fit_region = residuals[mask_initial_fit]
        std_dev_initial_residuals = np.std(initial_residuals_in_fit_region)
        if std_dev_initial_residuals < 1e-6:
            std_dev_initial_residuals = 1.0 # Prevent division by zero if residuals are flat

        consecutive_deviations = 0
        jump_idx = -1
        start_check_idx = np.where(all_exposure_times_sorted > INITIAL_LINEAR_EXPOSURE_MAX_SECONDS)[0]
        if len(start_check_idx) > 0:
            start_check_idx = start_check_idx[0]
        else:
            start_check_idx = len(all_exposure_times_sorted) # No points beyond initial fit

        for i in range(start_check_idx, len(all_exposure_times_sorted)):
            if all_exposure_times_sorted[i] < JUMP_DETECTION_MIN_EXP:
                continue

            if all_raw_signals_sorted[i] >= fit_saturation_threshold_adu:
                break # Stop checking once we hit saturation

            if residuals[i] > JUMP_DETECTION_STD_MULTIPLIER * std_dev_initial_residuals:
                consecutive_deviations += 1
                if consecutive_deviations >= JUMP_DETECTION_CONSECUTIVE_POINTS:
                    jump_idx = i - JUMP_DETECTION_CONSECUTIVE_POINTS + 1
                    break
            else:
                consecutive_deviations = 0 # Reset if deviation not consecutive

        if jump_idx != -1:
            second_linear_min_exp = all_exposure_times_sorted[jump_idx]
            # Raw signal at jump point from data
            raw_signal_at_jump_point = all_raw_signals_sorted[jump_idx]

            # Determine the end of the second linear region (before saturation fully hits)
            mask_below_saturation = (all_exposure_times_sorted >= second_linear_min_exp) & \
                                    (all_raw_signals_sorted < fit_saturation_threshold_adu)

            valid_exposure_times_below_saturation = all_exposure_times_sorted[mask_below_saturation]
            if len(valid_exposure_times_below_saturation) > 0:
                second_linear_max_exp = np.max(valid_exposure_times_below_saturation) * 0.99
                if second_linear_max_exp <= second_linear_min_exp + 0.1: # Ensure reasonable range
                    second_linear_max_exp = np.max(all_exposure_times_sorted[all_raw_signals_sorted < dynamic_saturation_level]) * 0.95
                    print("Warning: second_linear_max_exp adjusted for robust second fit range.")
            else:
                second_linear_max_exp = np.max(all_exposure_times_sorted) # Fallback to max exposure

            mask_second_fit = (all_exposure_times_sorted >= second_linear_min_exp) & \
                              (all_exposure_times_sorted <= second_linear_max_exp) & \
                              (all_raw_signals_sorted < fit_saturation_threshold_adu)

            if np.sum(mask_second_fit) >= 2:
                exposure_times_second_fit = all_exposure_times_sorted[mask_second_fit]
                raw_signals_second_fit = all_raw_signals_sorted[mask_second_fit]

                slope_second_fit, intercept_second_fit, stderr_slope_second_fit, stderr_intercept_second_fit = \
                    calculate_linregress_stats(exposure_times_second_fit, raw_signals_second_fit)

                second_light_response_rate = slope_second_fit
                second_fit_successful = True

                # Calculate bias-subtracted intercept and its error (slope error is unaffected by constant subtraction)
                bias_subtracted_intercept_second_fit = intercept_second_fit - mean_subframe_bias
                stderr_bias_subtracted_intercept_second_fit = stderr_intercept_second_fit # Std error of a constant subtracted doesn't change

                print(f"--- Second Linear Region Fit (LIGHT Images, Detected between {second_linear_min_exp:.2f}s and {second_linear_max_exp:.2f}s) ---")
                print(f"Fitted linear model of the bias-subtracted signal (via linregress): "
                      f"Y = ({second_light_response_rate:.2f} +/- {stderr_slope_second_fit:.2f}) * ExposureTime + "
                      f"({bias_subtracted_intercept_second_fit:.2f} +/- {stderr_bias_subtracted_intercept_second_fit:.2f})")
                print(f"------------------------------------------------------------------------------------\n")
            else:
                print("Warning: Not enough valid data points for second linear fit after jump detection. Skipping second fit.\n")
        else:
            print("Info: No significant jump detected in the light curve. Only one linear region assumed for light response.\n")
            raw_signal_at_jump_point = np.nan # Ensure raw_signal_at_jump_point is NaN if no jump found

    # --- Plot Signal vs Exposure Time ---
    plt.figure(figsize=(10, 6))

    # Enhanced title
    plot_title_str = f"Raw Signal vs. Exposure Time for {plot_metadata['instrume']}, {plot_metadata['set_temp']}C, Gain {plot_metadata['gain_setting']}, Offset {plot_metadata['offset_setting']}"
    plt.title(plot_title_str, fontsize=14)

    plt.plot(all_exposure_times_sorted, all_raw_signals_sorted, 'o', markersize=4, alpha=0.6, label='All Raw Data Points')

    if initial_fit_successful:
        # Plot the first fitted line, intercepting at the mean subframe bias
        fit_x1 = np.linspace(0, INITIAL_LINEAR_EXPOSURE_MAX_SECONDS, 100)
        plt.plot(fit_x1, initial_light_response_rate * fit_x1 + mean_subframe_bias, 'r--',
                 label=f'Linear Fit (0-{INITIAL_LINEAR_EXPOSURE_MAX_SECONDS}s): Y = {initial_light_response_rate:.2f}X + {mean_subframe_bias:.2f}')

    if second_fit_successful:
        # Plot the second fitted line, with its own calculated intercept
        fit_x2 = np.linspace(second_linear_min_exp, second_linear_max_exp, 100)
        plt.plot(fit_x2, second_light_response_rate * fit_x2 + second_intercept_fit, 'b--',
                 label=f'Linear Fit ({second_linear_min_exp:.1f}-{second_linear_max_exp:.1f}s): Y = {second_light_response_rate:.2f}X + {second_intercept_fit:.2f}')

    plt.axhline(fit_saturation_threshold_adu, color='gray', linestyle=':',
                label=f'Auto Saturation Threshold for Fit ({fit_saturation_threshold_adu:.0f} ADU)')

    if initial_fit_successful:
        plt.axvline(INITIAL_LINEAR_EXPOSURE_MAX_SECONDS, color='darkorange', linestyle=':', label=f'1st Fit End ({INITIAL_LINEAR_EXPOSURE_MAX_SECONDS}s)')
    if not np.isnan(second_linear_min_exp):
        plt.axvline(second_linear_min_exp, color='purple', linestyle=':', label=f'2nd Fit Start ({second_linear_min_exp:.1f}s)')

    plt.xlabel('Exposure Time (s)', fontsize=12)
    plt.ylabel('Average ADU (Raw)', fontsize=12)
    plt.grid(True, which="both", ls="-", color='0.8')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_plot_file, dpi=300)
    plt.close()
    print(f"Signal vs Exposure Time plot saved to {output_plot_file}\n")

    return dynamic_saturation_adu, raw_signal_at_jump_point

def save_to_csv(all_collected_data, filename, subframe_biases):
    """
    Saves the collected signal (bias-subtracted) and noise data to a CSV file.
    Only includes LIGHT images.
    """
    # Filter only LIGHT image data for CSV output
    light_images_data_for_csv = [d for d in all_collected_data if d['filter'].upper() in LIGHT_FILTER_KEYWORDS and d['exposure_time'] is not None]

    if not light_images_data_for_csv:
        print("No LIGHT image data to save to CSV.\n")
        return

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Effective_Signal', 'Noise', 'Raw_Signal', 'Exposure_Time', 'Subframe_Index']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in light_images_data_for_csv:
            if len(entry['raw_signal_subframes']) == 9 and len(entry['noise_subframes']) == 9:
                for j in range(9):
                    effective_signal = entry['raw_signal_subframes'][j] - subframe_biases[j]
                    # Data is written to CSV regardless of positivity, filtering for plot happens later
                    writer.writerow({
                        'Effective_Signal': effective_signal,
                        'Noise': entry['noise_subframes'][j],
                        'Raw_Signal': entry['raw_signal_subframes'][j],
                        'Exposure_Time': entry['exposure_time'],
                        'Subframe_Index': j
                    })
            else:
                print(f"Warning: Skipping CSV entry for a LIGHT image due to inconsistent subframe data: {entry['exposure_time']}s")
    print(f"Data saved to {filename}\n")

def create_loglog_plots(all_collected_data, filename_sn, subframe_biases, subframe_biases_std, dynamic_saturation_adu, raw_signal_at_jump_point, plot_metadata):
    """
    Creates a log-log plot of bias-subtracted Signal vs. Noise and identifies four key regions.
    """
    # Prepare data from LIGHT images for the log-log plot
    light_images_data_for_plot = [d for d in all_collected_data if d['filter'].upper() in LIGHT_FILTER_KEYWORDS and d['exposure_time'] is not None]

    effective_signals_all = []
    noises_all = []

    for entry in light_images_data_for_plot:
        if len(entry['raw_signal_subframes']) == 9 and len(entry['noise_subframes']) == 9:
            for j in range(9):
                effective_signals_all.append(entry['raw_signal_subframes'][j] - subframe_biases[j])
                noises_all.append(entry['noise_subframes'][j])

    effective_signals_all = np.array(effective_signals_all)
    noises_all = np.array(noises_all)

    # Calculate effective signal at jump point using the mean subframe bias for plot annotation consistency
    mean_subframe_bias = np.mean(subframe_biases) if len(subframe_biases) > 0 else 0.0
    effective_signal_at_jump_point_for_plot = raw_signal_at_jump_point - mean_subframe_bias if not np.isnan(raw_signal_at_jump_point) else np.nan

    # Filter out any data points where effective_signal is <= 0 or noise is <= 0 for log plot
    positive_mask_for_plot = (effective_signals_all > 0) & (noises_all > 0)
    signals_for_plot = effective_signals_all[positive_mask_for_plot]
    noises_for_plot = noises_all[positive_mask_for_plot]

    plt.figure(figsize=(12, 8))
    plt.loglog(signals_for_plot, noises_for_plot, 'o', markersize=4, alpha=0.6, label='Data Points')

    plt.xlim(1e-1, 1e5)
    plt.ylim(1e1, 1e3)
    plt.grid(True, which="both", ls="-", color='0.8')

    # --- Plot Titles ---
    plot_title_str = f"CMOS Sensor Noise Characterization for {plot_metadata['instrume']}, {plot_metadata['set_temp']}C, Gain {plot_metadata['gain_setting']}, Offset {plot_metadata['offset_setting']}"
    plt.title(plot_title_str, fontsize=14)

    # --- Region Analysis and Fitting ---

    read_noise_adu_mean = np.nan
    read_noise_adu_std = np.nan

    dark_images_data = [d for d in all_collected_data if d['filter'].upper() in DARK_FILTER_KEYWORDS and d['exposure_time'] is not None]
    min_dark_exp_time = min(d['exposure_time'] for d in dark_images_data) if dark_images_data else None
    min_exp_dark_entries = [d for d in dark_images_data if d['exposure_time'] == min_dark_exp_time] if min_dark_exp_time is not None else []

    all_dark_noises = []
    for entry in min_exp_dark_entries:
        all_dark_noises.extend([n for n in entry['noise_subframes'] if n > 0]) # Ensure positive for log plot

    if all_dark_noises:
        read_noise_adu_mean = np.mean(all_dark_noises)
        read_noise_adu_std = np.std(all_dark_noises, ddof=1) # Sample standard deviation
        read_noise_label = f'Read Noise (Estimate at low signal): {read_noise_adu_mean:.2f} +/- {read_noise_adu_std:.2f} ADU'
        plt.axhline(read_noise_adu_mean, color='red', linestyle='--', linewidth=2, label=read_noise_label)
        plt.text(plt.xlim()[0] * 1.5, read_noise_adu_mean * 1.1,
                 f'Read Noise: {read_noise_adu_mean:.2f} +/- {read_noise_adu_std:.2f} ADU',
                 color='red', verticalalignment='bottom', fontsize=10)
        print(f"--- Read Noise Calculation ---")
        print(f"Read Noise (from min-exposure darks): {read_noise_adu_mean:.4f} +/- {read_noise_adu_std:.4f} ADU")
        print(f"----------------------------\n")
    else:
        print("Warning: No valid noise data from minimum exposure darks for Read Noise calculation. Read Noise not plotted.\n")


    # 2. Photon Noise Region 1 (Slope 1/2) - Gain 1 calculation ---
    photon_noise_mask_reg1 = (effective_signals_all >= PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG1) & \
                             (effective_signals_all <= PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG1) & \
                             (noises_all > 0)

    if np.sum(photon_noise_mask_reg1) > 1:
        signal_subset_reg1 = effective_signals_all[photon_noise_mask_reg1]
        noise_subset_reg1 = noises_all[photon_noise_mask_reg1]
        noise_squared_subset_reg1 = noise_subset_reg1**2

        slope_ns2_reg1, intercept_ns2_reg1, _, _, _ = linregress(signal_subset_reg1, noise_squared_subset_reg1)

        log_signal_subset_reg1 = np.log10(signal_subset_reg1)
        log_noise_subset_reg1 = np.log10(noise_subset_reg1)
        slope_loglog_reg1, _, _, _, stderr_loglog_reg1 = linregress(log_signal_subset_reg1, log_noise_subset_reg1)


        if slope_ns2_reg1 > 0:
            gain_electrons_per_ADU_1 = 1.0 / slope_ns2_reg1

            fit_signals_reg1 = np.linspace(PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG1, PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG1, 100)
            fitted_noise_photon_reg1 = np.sqrt(slope_ns2_reg1 * fit_signals_reg1 + intercept_ns2_reg1)
            fitted_noise_photon_reg1[fitted_noise_photon_reg1 <= 0] = np.nan
            plt.loglog(fit_signals_reg1, fitted_noise_photon_reg1, 'green', linestyle='-', linewidth=2,
                       label=f'Photon Noise Fit 1 (Log-Log Slope: {slope_loglog_reg1:.4f} +/- {stderr_loglog_reg1:.4f})')

            plt.text(np.mean(fit_signals_reg1), np.mean(fitted_noise_photon_reg1) * 0.5,
                     f'Gain 1: {gain_electrons_per_ADU_1:.3f} e-/ADU',
                     color='green', horizontalalignment='center', verticalalignment='top', fontsize=10)
            print(f"--- Photon Noise Analysis Region 1 (Gain 1) ---")
            print(f"Gain 1 (from Noise^2 vs Signal): {gain_electrons_per_ADU_1:.3f} e-/ADU")
            print(f"Log-Log Slope (Noise vs Signal): {slope_loglog_reg1:.4f} +/- {stderr_loglog_reg1:.4f}")
            print(f"----------------------------------------------\n")
        else:
            print("Warning: Photon noise fit REGION 1 resulted in non-positive slope for Noise^2. Cannot calculate gain.\n")
    else:
        print(f"Warning: Not enough data points in Photon Noise REGION 1 (Effective Signal between {PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG1} and {PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG1} ADU). Adjust ranges or check data.\n")

    # 3. Photon Noise Region 2 (Slope 1/2) - Gain 2 calculation ---
    photon_noise_mask_reg2 = (effective_signals_all >= PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG2) & \
                             (effective_signals_all <= PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG2) & \
                             (noises_all > 0)

    if np.sum(photon_noise_mask_reg2) > 1:
        signal_subset_reg2 = effective_signals_all[photon_noise_mask_reg2]
        noise_subset_reg2 = noises_all[photon_noise_mask_reg2]
        noise_squared_subset_reg2 = noise_subset_reg2**2

        slope_ns2_reg2, intercept_ns2_reg2, _, _, _ = linregress(signal_subset_reg2, noise_squared_subset_reg2)

        log_signal_subset_reg2 = np.log10(signal_subset_reg2)
        log_noise_subset_reg2 = np.log10(noise_subset_reg2)
        slope_loglog_reg2, _, _, _, stderr_loglog_reg2 = linregress(log_signal_subset_reg2, log_noise_subset_reg2)

        if slope_ns2_reg2 > 0:
            gain_electrons_per_ADU_2 = 1.0 / slope_ns2_reg2

            fit_signals_reg2 = np.linspace(PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG2, PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG2, 100)
            fitted_noise_photon_reg2 = np.sqrt(slope_ns2_reg2 * fit_signals_reg2 + intercept_ns2_reg2)
            fitted_noise_photon_reg2[fitted_noise_photon_reg2 <= 0] = np.nan
            plt.loglog(fit_signals_reg2, fitted_noise_photon_reg2, 'forestgreen', linestyle='--', linewidth=2,
                       label=f'Photon Noise Fit 2 (Log-Log Slope: {slope_loglog_reg2:.4f} +/- {stderr_loglog_reg2:.4f})')

            plt.text(np.mean(fit_signals_reg2), np.mean(fitted_noise_photon_reg2) * 0.5,
                     f'Gain 2: {gain_electrons_per_ADU_2:.3f} e-/ADU',
                     color='forestgreen', horizontalalignment='center', verticalalignment='top', fontsize=10)
            print(f"--- Photon Noise Analysis Region 2 (Gain 2) ---")
            print(f"Gain 2 (from Noise^2 vs Signal): {gain_electrons_per_ADU_2:.3f} e-/ADU")
            print(f"Log-Log Slope (Noise vs Signal): {slope_loglog_reg2:.4f} +/- {stderr_loglog_reg2:.4f}")
            print(f"----------------------------------------------\n")
        else:
            print("Warning: Photon noise fit REGION 2 resulted in non-positive slope for Noise^2. Cannot calculate gain.\n")
    else:
        print(f"Warning: Not enough data points in Photon Noise REGION 2 (Effective Signal between {PHOTON_NOISE_SIGNAL_MIN_EFFECTIVE_REG2} and {PHOTON_NOISE_SIGNAL_MAX_EFFECTIVE_REG2} ADU). Adjust ranges or check data.\n")


    # 4. Fixed Pattern Noise Region 1 (Slope 1) ---
    fpn_mask_reg1 = (effective_signals_all >= FPN_SIGNAL_MIN_EFFECTIVE_REG1) & \
                    (effective_signals_all <= FPN_SIGNAL_MAX_EFFECTIVE_REG1) & \
                    (noises_all > 0)
    if np.sum(fpn_mask_reg1) > 1:
        valid_mask = (effective_signals_all[fpn_mask_reg1] > 0) & (noises_all[fpn_mask_reg1] > 0)
        fpn_signal_subset_log_reg1 = np.log10(effective_signals_all[fpn_mask_reg1][valid_mask])
        fpn_noise_subset_log_reg1 = np.log10(noises_all[fpn_mask_reg1][valid_mask])

        slope_fpn_reg1, intercept_fpn_reg1, r_value_reg1, _, stderr_fpn_reg1 = linregress(fpn_signal_subset_log_reg1, fpn_noise_subset_log_reg1)

        fit_signals_fpn_reg1 = np.linspace(FPN_SIGNAL_MIN_EFFECTIVE_REG1, FPN_SIGNAL_MAX_EFFECTIVE_REG1, 100)
        fitted_noise_fpn_reg1 = 10**(slope_fpn_reg1 * np.log10(fit_signals_fpn_reg1) + intercept_fpn_reg1)
        fitted_noise_fpn_reg1[fitted_noise_fpn_reg1 <= 0] = np.nan
        plt.loglog(fit_signals_fpn_reg1, fitted_noise_fpn_reg1, 'blue', linestyle='-', linewidth=2,
                   label=f'Fixed Pattern Noise Fit 1 (Log-Log Slope: {slope_fpn_reg1:.4f} +/- {stderr_fpn_reg1:.4f})')
        print(f"--- Fixed Pattern Noise Analysis Region 1 ---")
        print(f"Log-Log Slope (Noise vs Signal): {slope_fpn_reg1:.4f} +/- {stderr_fpn_reg1:.4f}")
        print(f"---------------------------------------------\n")
    else:
        print(f"Warning: Not enough data points in FPN REGION 1 (Effective Signal between {FPN_SIGNAL_MIN_EFFECTIVE_REG1} and {FPN_SIGNAL_MAX_EFFECTIVE_REG1} ADU). Adjust ranges or check data.\n")


    # 5. Fixed Pattern Noise Region 2 (Slope 1) ---
    fpn_mask_reg2 = (effective_signals_all >= FPN_SIGNAL_MIN_EFFECTIVE_REG2) & \
                    (effective_signals_all <= FPN_SIGNAL_MAX_EFFECTIVE_REG2) & \
                    (noises_all > 0)
    if np.sum(fpn_mask_reg2) > 1:
        valid_mask = (effective_signals_all[fpn_mask_reg2] > 0) & (noises_all[fpn_mask_reg2] > 0)
        fpn_signal_subset_log_reg2 = np.log10(effective_signals_all[fpn_mask_reg2][valid_mask])
        fpn_noise_subset_log_reg2 = np.log10(noises_all[fpn_mask_reg2][valid_mask])

        slope_fpn_reg2, intercept_fpn_reg2, r_value_reg2, _, stderr_fpn_reg2 = linregress(fpn_signal_subset_log_reg2, fpn_noise_subset_log_reg2)

        fit_signals_fpn_reg2 = np.linspace(FPN_SIGNAL_MIN_EFFECTIVE_REG2, FPN_SIGNAL_MAX_EFFECTIVE_REG2, 100)
        fitted_noise_fpn_reg2 = 10**(slope_fpn_reg2 * np.log10(fit_signals_fpn_reg2) + intercept_fpn_reg2)
        fitted_noise_fpn_reg2[fitted_noise_fpn_reg2 <= 0] = np.nan
        plt.loglog(fit_signals_fpn_reg2, fitted_noise_fpn_reg2, 'steelblue', linestyle='--', linewidth=2,
                   label=f'Fixed Pattern Noise Fit 2 (Log-Log Slope: {slope_fpn_reg2:.4f} +/- {stderr_fpn_reg2:.4f})')
        print(f"--- Fixed Pattern Noise Analysis Region 2 ---")
        print(f"Log-Log Slope (Noise vs Signal): {slope_fpn_reg2:.4f} +/- {stderr_fpn_reg2:.4f}")
        print(f"---------------------------------------------\n")
    else:
        print(f"Warning: Not enough data points in FPN REGION 2 (Effective Signal between {FPN_SIGNAL_MIN_EFFECTIVE_REG2} and {FPN_SIGNAL_MAX_EFFECTIVE_REG2} ADU). Adjust ranges or check data.\n")


    # 6. Jump Point and Saturation (Standard Plot)
    if not np.isnan(effective_signal_at_jump_point_for_plot) and effective_signal_at_jump_point_for_plot > 0:
        # Corrected variable name in the f-string:
        plt.axvline(effective_signal_at_jump_point_for_plot, color='darkviolet', linestyle=':', linewidth=2, label=f'Sensor Response Jump ({effective_signal_at_jump_point_for_plot:.0f} ADU)')
        plt.text(effective_signal_at_jump_point_for_plot * 1.05, plt.ylim()[1] * 0.5,
                'Response Jump', color='darkviolet', rotation=90, verticalalignment='center', fontsize=10)

    # Saturation threshold is from the light curve analysis's dynamic detection
    saturation_effective_adu = dynamic_saturation_adu - mean_subframe_bias
    if saturation_effective_adu > 0:
        plt.axvline(saturation_effective_adu, color='purple', linestyle=':', linewidth=2, label='Approx. Saturation Region (Effective Signal)')
        plt.text(saturation_effective_adu * 1.05, plt.ylim()[1] * 0.8,
                 'Saturation', color='purple', rotation=90, verticalalignment='top', fontsize=10)
    else:
        print(f"Warning: Calculated effective saturation ADU ({saturation_effective_adu:.2f}) is not positive. Saturation line not plotted.\n")

    plt.xlabel('Effective Signal (ADU)', fontsize=12)
    plt.ylabel('Noise (ADU)', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(filename_sn, dpi=300)
    plt.close()
    print(f"Signal vs Noise plot saved to {filename_sn}\n")


# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        print("Please create this directory and place your FITS image pairs inside.")
        print("Example: `mkdir fits_images_directory` and then copy your `.fits` files into it.")
        exit()

    # Step 1: Collect all raw data and primary header info
    all_collected_data_points, instrume, set_temp, gain_setting, offset_setting = collect_all_data(INPUT_DIR)
    plot_metadata = {
        'instrume': instrume,
        'set_temp': set_temp,
        'gain_setting': gain_setting,
        'offset_setting': offset_setting
    }

    if all_collected_data_points:
        # Step 2: Calculate subframe biases from dark frames
        subframe_biases_list, subframe_biases_std_list = calculate_subframe_biases(all_collected_data_points)

        # Step 3: Analyze signal vs exposure curve for light frames
        # This returns saturation ADU and raw signal at jump point (from light curve analysis)
        dynamic_saturation_adu, raw_signal_at_jump_point = analyze_signal_exposure_curve(all_collected_data_points, subframe_biases_list, plot_metadata, OUTPUT_PLOT_FILE_SE)

        # Step 4: Save the data to CSV (effective signal calculated using subframe biases)
        save_to_csv(all_collected_data_points, OUTPUT_CSV_FILE, subframe_biases_list)

        # Step 5: Create the log-log plot (effective signal calculated using subframe biases)
        create_loglog_plots(all_collected_data_points,
                            OUTPUT_PLOT_FILE_SN,
                            subframe_biases_list,
                            subframe_biases_std_list,
                            dynamic_saturation_adu,
                            raw_signal_at_jump_point,
                            plot_metadata)
    else:
        print("\nNo valid data collected. Please check the input directory and FITS file formats and ensure dark frames are present.\n")