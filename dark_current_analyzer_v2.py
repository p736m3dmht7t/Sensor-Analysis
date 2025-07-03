###
# dark_frame_characterizer.py 2025-07-03
# John Phillips, john.d.phillips@comcast.net
###

# dark_frame_characterizer.py: Performs a detailed dark frame analysis using a time-based regression model.
# Sections:
# - imports_and_configuration
# - error_propagation_utilities
# - data_collection_and_processing
# - core_analysis_engine
# - plotting_and_reporting
# - main_execution_logic

# --- imports_and_configuration ---
# Handles all necessary library imports and user-configurable settings.

import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress
from datetime import datetime
import warnings

# --- User Configuration ---
# IMPORTANT: Set this to the directory containing your dark FITS image pairs.
INPUT_DIR = r"D:\Astrophotography\2025-07-02\DARK\Dark"
# Directory to save the output plots and report.
OUTPUT_DIR = r"."
# Exposure time threshold in seconds to separate the two ADC readout modes.
ADC_THRESHOLD_S = 1.0

# --- Plotting Style Configuration ---
FIG_SIZE = (12, 8)
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 10
GRID_COLORS = plt.cm.viridis(np.linspace(0, 1, 9))
GRID_MARKERS = ['o', 's', 'v', '^', '<', '>', 'D', 'p', '*']

# --- error_propagation_utilities ---
# Contains helper functions for calculating the standard error of derived quantities.

def propagate_division_error(val_a, val_b, se_a, se_b):
    # Calculates the standard error for C = A / B.
    if val_b == 0 or val_a == 0: return np.nan
    rel_se_a = se_a / val_a
    rel_se_b = se_b / val_b
    val_c = val_a / val_b
    se_c = abs(val_c) * np.sqrt(rel_se_a**2 + rel_se_b**2)
    return se_c

def propagate_multiplication_error(val_a, val_b, se_a, se_b):
    # Calculates the standard error for C = A * B.
    if val_a == 0 or val_b == 0: return np.nan
    rel_se_a = se_a / val_a
    rel_se_b = se_b / val_b
    val_c = val_a * val_b
    se_c = abs(val_c) * np.sqrt(rel_se_a**2 + rel_se_b**2)
    return se_c

def propagate_sqrt_error(val_a, se_a):
    # Calculates the standard error for C = sqrt(A).
    if val_a <= 0: return np.nan
    val_c = np.sqrt(val_a)
    se_c = 0.5 * se_a / val_c if val_c != 0 else np.nan
    return se_c

# --- data_collection_and_processing ---
# Functions for finding, loading, and processing the FITS image pairs.

def load_fits_info(image_path):
    # Loads a FITS image and extracts relevant header information.
    try:
        with fits.open(image_path) as hdul:
            header = hdul[0].header
            img_data = hdul[0].data.astype(np.float32)
            exposure = header.get('EXPOSURE', header.get('EXPTIME', 0.0))
            instrume = str(header.get('INSTRUME', 'N/A')).strip()
            set_temp = header.get('SET-TEMP', 'N/A')
            gain = header.get('GAIN', 'N/A')
            offset = header.get('OFFSET', 'N/A')
        return img_data, exposure, instrume, set_temp, gain, offset
    except Exception as e:
        print(f"Error reading FITS file {image_path}: {e}")
        return None, 0.0, 'N/A', 'N/A', 'N/A', 'N/A'

def collect_and_process_pairs(input_dir):
    # Finds all FITS files, pairs them, and calculates statistics for each 3x3 subframe.
    print("\n--- Starting Data Collection ---")
    all_files = sorted([os.path.join(root, f) for root, _, files in os.walk(input_dir) for f in files if f.lower().endswith(('.fits', '.fit'))])
    if len(all_files) < 2:
        print(f"Error: Need at least 2 FITS files for analysis. Found {len(all_files)}.")
        return pd.DataFrame(), {}
    if len(all_files) % 2 != 0:
        print(f"Warning: Found an odd number of FITS files ({len(all_files)}). The last file will be ignored.")

    all_data = []
    metadata = {}
    
    for i in range(0, len(all_files) - 1, 2):
        path1, path2 = all_files[i], all_files[i+1]
        print(f"Processing pair {i//2 + 1:03d}: ({os.path.basename(path1)}, {os.path.basename(path2)})")

        img1, exp1, inst, temp, gain, offset = load_fits_info(path1)
        img2, exp2, _, _, _, _ = load_fits_info(path2)

        if not metadata and inst != 'N/A':
            metadata = {'instrume': inst, 'set_temp': temp, 'gain_setting': gain, 'offset_setting': offset}

        if img1 is None or img2 is None or img1.shape != img2.shape or abs(exp1 - exp2) > 1e-5:
            print("  -> Skipping pair due to mismatch or loading error.")
            continue

        sum_img = img1 + img2
        diff_img = img1 - img2
        height, width = img1.shape
        sub_h, sub_w = height // 3, width // 3
        
        for r in range(3):
            for c in range(3):
                h_start, w_start = r * sub_h, c * sub_w
                h_end, w_end = h_start + sub_h, w_start + sub_w
                
                sub_sum = sum_img[h_start:h_end, w_start:w_end]
                sub_diff = diff_img[h_start:h_end, w_start:w_end]
                
                mean_sum_div_2 = np.mean(sub_sum) / 2.0
                var_sum = np.var(sub_sum, ddof=1)
                var_diff = np.var(sub_diff, ddof=1)
                
                all_data.append({
                    'subframe_index': r * 3 + c,
                    'exposure_time': exp1,
                    'mean_sum_div_2': mean_sum_div_2,
                    'var_sum': var_sum,
                    'var_diff': var_diff
                })

    if not all_data:
        print("Error: No valid image pairs were processed.")
        return pd.DataFrame(), {}
        
    df = pd.DataFrame(all_data)
    df['exposure_time_sq'] = df['exposure_time']**2
    # This is the spatial variance component from the sum image, which models FPN
    df['var_fpn_unscaled'] = df['var_sum'] - df['var_diff']
    df.loc[df['var_fpn_unscaled'] < 0, 'var_fpn_unscaled'] = 0 # Variance cannot be negative

    print(f"\nSuccessfully processed {len(df) // 9} pairs into a DataFrame with {len(df)} data points.")
    return df, metadata

# --- core_analysis_engine ---
# The primary function for performing the regressions and deriving physical parameters.

def analyze_data_region(df_region):
    # Performs time-based regressions on a specific dataset (one subframe, one ADC mode).
    results = {key: (np.nan, np.nan) for key in [
        'fixed_bias_adu', 'dark_current_adu_s', 'gain_e_adu', 'read_noise_adu',
        'read_noise_e', 'dark_current_e_s', 'dsnu_e_s',
        'var_diff_intercept', 'var_diff_slope' # Store fit parameters for plotting
    ]}
    
    if len(df_region) < 3:
        print("  -> Not enough data points (< 3) for regression. Skipping.")
        return results

    # --- Fit 1: Mean Signal vs. Time ---
    fit_mean = linregress(df_region['exposure_time'], df_region['mean_sum_div_2'])
    c_mean, m_mean = fit_mean.intercept, fit_mean.slope
    se_c_mean, se_m_mean = fit_mean.intercept_stderr, fit_mean.stderr
    results['fixed_bias_adu'] = (c_mean, se_c_mean)
    results['dark_current_adu_s'] = (m_mean, se_m_mean)

    # --- Fit 2: Variance(Difference) vs. Time ---
    fit_var = linregress(df_region['exposure_time'], df_region['var_diff'])
    c_var, m_var = fit_var.intercept, fit_var.slope
    se_c_var, se_m_var = fit_var.intercept_stderr, fit_var.stderr
    results['var_diff_intercept'] = (c_var, se_c_var)
    results['var_diff_slope'] = (m_var, se_m_var)
    
    # --- Fit 3: Variance(FPN) vs. Time^2 ---
    fit_fpn = linregress(df_region['exposure_time_sq'], df_region['var_fpn_unscaled'])
    m_fpn, se_m_fpn = fit_fpn.slope, fit_fpn.stderr
    
    # --- Derive Physical Parameters and Propagate Errors ---
    # Gain (K = 2 * m_mean / m_var)
    if m_var > 0:
        gain = (2 * m_mean) / m_var
        se_gain = propagate_division_error(2 * m_mean, m_var, 2 * se_m_mean, se_m_var)
        results['gain_e_adu'] = (gain, se_gain)
    else:
        gain, se_gain = np.nan, np.nan
        
    # Read Noise (ADU and e-)
    if c_var > 0:
        rn_adu_sq = c_var / 2.0
        se_rn_adu_sq = se_c_var / 2.0
        rn_adu = np.sqrt(rn_adu_sq)
        se_rn_adu = propagate_sqrt_error(rn_adu_sq, se_rn_adu_sq)
        results['read_noise_adu'] = (rn_adu, se_rn_adu)
        if not np.isnan(gain):
            rn_e = rn_adu * gain
            se_rn_e = propagate_multiplication_error(rn_adu, gain, se_rn_adu, se_gain)
            results['read_noise_e'] = (rn_e, se_rn_e)
            
    # Dark Current (e-/s)
    if not np.isnan(gain):
        dc_e_s = m_mean * gain
        se_dc_e_s = propagate_multiplication_error(m_mean, gain, se_m_mean, se_gain)
        results['dark_current_e_s'] = (dc_e_s, se_dc_e_s)
        
    # DSNU (e-/s)
    if m_fpn > 0 and not np.isnan(gain):
        # m_fpn = 4 * (sigma_DSNU / K)^2  => sigma_DSNU = K/2 * sqrt(m_fpn)
        term_to_sqrt = m_fpn
        se_term_to_sqrt = se_m_fpn
        
        sqrt_val = np.sqrt(term_to_sqrt)
        se_sqrt_val = propagate_sqrt_error(term_to_sqrt, se_term_to_sqrt)

        dsnu_e_s = (gain / 2.0) * sqrt_val
        se_dsnu_e_s = propagate_multiplication_error(gain/2.0, sqrt_val, se_gain/2.0, se_sqrt_val)
        results['dsnu_e_s'] = (dsnu_e_s, se_dsnu_e_s)

    return results

# --- plotting_and_reporting ---
# Functions to visualize the data and results, and generate the final report.

def generate_all_plots(df, all_results, metadata, output_dir, base_filename):
    # Orchestrates the creation of all plots for both ADC modes.
    print("\n--- Generating Plots ---")
    
    # Amp Glow Map
    _generate_amp_glow_map(df, metadata, output_dir, base_filename)

    # Plot sets for each mode
    df_mode1 = df[df['exposure_time'] < ADC_THRESHOLD_S]
    df_mode2 = df[df['exposure_time'] >= ADC_THRESHOLD_S]
    
    if not df_mode1.empty:
        _generate_plot_set(df_mode1, all_results, 1, metadata, output_dir, base_filename)
    else:
        print("Skipping plots for Mode 1 (<1s): No data.")

    if not df_mode2.empty:
        _generate_plot_set(df_mode2, all_results, 2, metadata, output_dir, base_filename)
    else:
        print("Skipping plots for Mode 2 (>=1s): No data.")

def _generate_amp_glow_map(df, metadata, output_dir, base_filename):
    # Generates a representative amp glow map from the longest exposure.
    if df.empty: return
    longest_exp_time = df['exposure_time'].max()
    longest_exp_pair_mean = df[df['exposure_time'] == longest_exp_time]['mean_sum_div_2'].mean()
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    # Placeholder for a real image, using a title to convey the info
    ax.text(0.5, 0.5, "Amp Glow Map Placeholder\n(Full image loading omitted for speed;\nrefer to a FITS viewer for actual image)",
            ha='center', va='center', fontsize=18, color='gray')
    height, width = 3856, 5496 # Typical ASI183 dimensions
    for i in range(1, 3):
        ax.axhline(i * height / 3, color='r', linestyle=':'); ax.axvline(i * width / 3, color='r', linestyle=':')
    for i in range(3):
        for j in range(3):
            ax.text(j * width / 3 + width / 6, i * height / 3 + height / 6, str(i * 3 + j),
                    fontsize=20, color='red', ha='center', va='center', weight='bold')
    
    title = (f"Longest Exposure: {longest_exp_time:.2f}s - Avg. Signal: {longest_exp_pair_mean:.1f} ADU\n"
             f"{metadata.get('instrume', '')} (G:{metadata.get('gain_setting', 'NA')}, O:{metadata.get('offset_setting', 'NA')}, T:{metadata.get('set_temp', 'NA')}C)")
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, f"{base_filename}_amp_glow_map.png")
    plt.savefig(plot_path, dpi=150); plt.close(fig)
    print(f"Amp Glow map placeholder saved to: {plot_path}")

def _generate_plot_set(df_region, all_results, mode_num, metadata, output_dir, base_filename):
    # Helper to generate a standardized set of 3 plots for a given data region.
    mode_suffix = f"_mode{mode_num}"
    mode_title = f"Mode {mode_num} ({'<' if mode_num==1 else '>='}{ADC_THRESHOLD_S}s)"
    cam_info = f"{metadata.get('instrume', 'N/A')} (G:{metadata.get('gain_setting', 'NA')}, O:{metadata.get('offset_setting', 'NA')}, T:{metadata.get('set_temp', 'NA')}C)"
    
    # Plot 1: Mean Signal vs. Time
    fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
    for i in range(9):
        df_sub = df_region[df_region['subframe_index'] == i]
        if df_sub.empty: continue
        ax1.plot(df_sub['exposure_time'], df_sub['mean_sum_div_2'], marker=GRID_MARKERS[i], ls='none', color=GRID_COLORS[i], label=f'Grid {i}')
        # Plot regression line
        res = all_results[i][f'mode{mode_num}']
        bias, _ = res['fixed_bias_adu']
        dc_adu, _ = res['dark_current_adu_s']
        if not np.isnan(bias):
            t_fit = np.array([0, df_region['exposure_time'].max()]) if not df_region.empty else np.array([0,1])
            ax1.plot(t_fit, bias + dc_adu * t_fit, color=GRID_COLORS[i], ls='--')
    ax1.set_title(f"Mean Signal vs. Exposure Time - {mode_title}\n{cam_info}", fontsize=TITLE_FONTSIZE)
    ax1.set_xlabel("Exposure Time (s)", fontsize=LABEL_FONTSIZE); ax1.set_ylabel("Mean Signal (ADU)", fontsize=LABEL_FONTSIZE)
    ax1.grid(True, which="both", ls=":"); ax1.legend(ncol=3); fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{base_filename}{mode_suffix}_1_mean_vs_time.png"), dpi=150); plt.close(fig1)

    # Plot 2: Variance(Difference) vs. Time
    fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
    for i in range(9):
        df_sub = df_region[df_region['subframe_index'] == i]
        if df_sub.empty: continue
        ax2.plot(df_sub['exposure_time'], df_sub['var_diff'], marker=GRID_MARKERS[i], ls='none', color=GRID_COLORS[i], label=f'Grid {i}')
        # Plot regression line using the actual fit parameters
        res = all_results[i][f'mode{mode_num}']
        intercept, _ = res['var_diff_intercept']
        slope, _ = res['var_diff_slope']
        if not np.isnan(intercept):
            t_fit = np.array([0, df_region['exposure_time'].max()]) if not df_region.empty else np.array([0,1])
            ax2.plot(t_fit, intercept + slope * t_fit, color=GRID_COLORS[i], ls='--')
    ax2.set_title(f"Variance(Difference) vs. Exposure Time - {mode_title}\n{cam_info}", fontsize=TITLE_FONTSIZE)
    ax2.set_xlabel("Exposure Time (s)", fontsize=LABEL_FONTSIZE); ax2.set_ylabel("Variance of Difference Image (ADU²)", fontsize=LABEL_FONTSIZE)
    ax2.grid(True, which="both", ls=":"); ax2.legend(ncol=3); fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{base_filename}{mode_suffix}_2_vardiff_vs_time.png"), dpi=150); plt.close(fig2)

    # Plot 3: Variance(FPN) vs. Time^2
    fig3, ax3 = plt.subplots(figsize=FIG_SIZE)
    for i in range(9):
        df_sub = df_region[df_region['subframe_index'] == i]
        if df_sub.empty: continue
        ax3.plot(df_sub['exposure_time_sq'], df_sub['var_fpn_unscaled'], marker=GRID_MARKERS[i], ls='none', color=GRID_COLORS[i], label=f'Grid {i}')
    ax3.set_title(f"Fixed Pattern Noise Variance vs. Time² - {mode_title}\n{cam_info}", fontsize=TITLE_FONTSIZE)
    ax3.set_xlabel("Exposure Time Squared (s²)", fontsize=LABEL_FONTSIZE); ax3.set_ylabel("FPN Component of Variance (ADU²)", fontsize=LABEL_FONTSIZE)
    ax3.grid(True, which="both", ls=":"); ax3.legend(ncol=3); fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{base_filename}{mode_suffix}_3_varfpn_vs_timesq.png"), dpi=150); plt.close(fig3)
    print(f"Generated plot set for {mode_title}")

def generate_markdown_report(all_results, metadata, output_dir, base_filename):
    # Generates a detailed Markdown report with all results and embedded plots.
    report_path = os.path.join(output_dir, f"{base_filename}_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# Dark Frame Characterization Report\n\n")
        f.write(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Camera:** `{metadata.get('instrume', 'N/A')}`\n")
        f.write(f"- **Settings:** Temp=`{metadata.get('set_temp', 'N/A')}C`, Gain=`{metadata.get('gain_setting', 'N/A')}`, Offset=`{metadata.get('offset_setting', 'N/A')}`\n")
        f.write(f"- **Analysis Method:** Time-based linear regression with full error propagation.\n\n")

        f.write(f"## Amp Glow Map\n")
        f.write(f"This map indicates the general location of analysis grids. The title provides info from the longest exposure pair.\n\n")
        f.write(f"![Amp Glow Map]({os.path.basename(base_filename)}_amp_glow_map.png)\n\n")

        for mode_num in [1, 2]:
            mode_title = f"Mode {mode_num} (`{'<' if mode_num==1 else '>='}{ADC_THRESHOLD_S}s`)"
            f.write(f"---\n\n## Results for {mode_title}\n\n")
            
            # Check if any data exists for this mode
            has_data = any(not np.isnan(res[f'mode{mode_num}']['gain_e_adu'][0]) for res in all_results.values())
            if not has_data:
                f.write("No data available for this readout mode.\n\n")
                continue

            f.write("| Grid | Fixed Bias<br>(ADU) | Read Noise<br>(e-) | Gain<br>(e-/ADU) | Dark Current<br>(e-/pixel/s) | DSNU<br>(e-/pixel/s) |\n")
            f.write("|:----:|:---:|:---:|:---:|:---:|:---:|\n")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for i in range(9):
                    res = all_results[i][f'mode{mode_num}']
                    f.write(f"| **{i}** | `{res['fixed_bias_adu'][0]:.2f} ± {res['fixed_bias_adu'][1]:.2f}` "
                            f"| `{res['read_noise_e'][0]:.2f} ± {res['read_noise_e'][1]:.2f}` "
                            f"| `{res['gain_e_adu'][0]:.3f} ± {res['gain_e_adu'][1]:.3f}` "
                            f"| `{res['dark_current_e_s'][0]:.4f} ± {res['dark_current_e_s'][1]:.4f}` "
                            f"| `{res['dsnu_e_s'][0]:.4f} ± {res['dsnu_e_s'][1]:.4f}` |\n")
            
            f.write(f"\n### Diagnostic Plots for {mode_title}\n\n")
            f.write(f"![Mean vs Time]({os.path.basename(base_filename)}_mode{mode_num}_1_mean_vs_time.png)\n")
            f.write(f"![Var(Diff) vs Time]({os.path.basename(base_filename)}_mode{mode_num}_2_vardiff_vs_time.png)\n")
            f.write(f"![Var(FPN) vs Time^2]({os.path.basename(base_filename)}_mode{mode_num}_3_varfpn_vs_timesq.png)\n\n")
    
    print(f"\nMarkdown report saved to: {report_path}")

# --- main_execution_logic ---
# The main block that orchestrates the entire analysis from start to finish.

def orchestrate_analysis(input_dir, output_dir):
    # Main function to run the entire analysis pipeline.
    os.makedirs(output_dir, exist_ok=True)
    
    master_df, metadata = collect_and_process_pairs(input_dir)
    if master_df.empty:
        print("\nAnalysis halted: No data was collected.")
        return

    base_filename = f"dark_analysis_{metadata.get('instrume', 'UnknownCam').replace(' ', '_')}"
    
    print("\n--- Starting Core Analysis ---")
    all_results = {}
    for i in range(9):
        print(f"Analyzing Grid {i}...")
        df_sub = master_df[master_df['subframe_index'] == i].copy()
        
        df_mode1 = df_sub[df_sub['exposure_time'] < ADC_THRESHOLD_S]
        df_mode2 = df_sub[df_sub['exposure_time'] >= ADC_THRESHOLD_S]
        
        results_mode1 = analyze_data_region(df_mode1)
        results_mode2 = analyze_data_region(df_mode2)
        
        all_results[i] = {'mode1': results_mode1, 'mode2': results_mode2}

    print("\n--- Analysis Complete. Generating Outputs. ---")
    generate_all_plots(master_df, all_results, metadata, output_dir, base_filename)
    generate_markdown_report(all_results, metadata, output_dir, base_filename)
    
    csv_path = os.path.join(output_dir, f"{base_filename}_raw_metrics.csv")
    master_df.to_csv(csv_path, index=False)
    print(f"\nRaw metrics data saved to: {csv_path}")

if __name__ == "__main__":
    if not os.path.isdir(INPUT_DIR):
        print(f"\nError: Input directory '{INPUT_DIR}' not found. Please check the 'INPUT_DIR' variable.")
    else:
        orchestrate_analysis(INPUT_DIR, OUTPUT_DIR)
        print("\n--- Script Finished ---\n")