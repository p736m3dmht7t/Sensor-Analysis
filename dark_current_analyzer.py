# dark_current_analyzer_v7.py

import os
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')  # This line must come BEFORE importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import linregress
from datetime import datetime
import warnings

# --- Configuration ---
# IMPORTANT: Set this to the directory containing your dark/bias FITS images.
INPUT_DIR = r"D:\Astrophotography\2025-07-02\DARK\Dark"
# Directory to save the output plots and report.
OUTPUT_DIR = r"."
# Exposure time threshold to separate the two ADC modes.
ADC_THRESHOLD_S = 1.0

# --- Plotting Style Configuration ---
FIG_SIZE = (11, 8.5) # Landscape 8.5x11 aspect ratio
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 10

# --- Functions ---

def load_fits_and_extract_info(image_path, load_data=True):
    """Loads a FITS image and/or header, and extracts relevant info."""
    try:
        with fits.open(image_path) as hdul:
            header = hdul[0].header
            img_data = hdul[0].data.astype(np.float32) if load_data else None
            exposure = header.get('EXPOSURE', header.get('EXPTIME', 0.0))
            instrume = str(header.get('INSTRUME', 'N/A')).strip()
            set_temp = header.get('SET-TEMP', 'N/A')
            gain = header.get('GAIN', 'N/A')
            offset = header.get('OFFSET', 'N/A')
        return img_data, header, exposure, instrume, set_temp, gain, offset
    except Exception as e:
        print(f"Error processing FITS file {image_path}: {e}"); return None, None, 0.0, 'N/A', 'N/A', 'N/A', 'N/A'

def generate_amp_glow_map(input_dir, output_dir, base_filename):
    """Finds the longest exposure by ONLY reading headers, then loads and plots that single image."""
    print("\nGenerating Amp Glow Map...")
    all_files = [os.path.join(root, f) for root, _, files in os.walk(input_dir) for f in files if f.lower().endswith(('.fits', '.fit'))]
    if not all_files: print("Warning: Could not generate Amp Glow map, no FITS files found."); return

    longest_exp_path = max(all_files, key=lambda p: load_fits_and_extract_info(p, load_data=False)[2])
    if not longest_exp_path: print("Warning: Could not find a valid FITS file for Amp Glow map."); return

    img_data, _, exp, instr, temp, gain, offset = load_fits_and_extract_info(longest_exp_path, load_data=True)
    if img_data is None: return
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    vmin, vmax = np.percentile(img_data, 1), np.percentile(img_data, 99.5)
    im = ax.imshow(img_data, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    
    height, width = img_data.shape
    for i in range(1, 3):
        ax.axhline(i * height / 3, color='r', linestyle=':'); ax.axvline(i * width / 3, color='r', linestyle=':')
    for i in range(3):
        for j in range(3):
            ax.text(j * width / 3 + width / 6, i * height / 3 + height / 6, str(i * 3 + j),
                    fontsize=20, color='red', ha='center', va='center', weight='bold')

    title = f"{instr} ({exp:.2f}s, {temp}C, G:{gain}, O:{offset})"
    ax.set_title(title, fontsize=TITLE_FONTSIZE); ax.set_xlabel('X Pixel', fontsize=LABEL_FONTSIZE); ax.set_ylabel('Y Pixel', fontsize=LABEL_FONTSIZE)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1); fig.colorbar(im, cax=cax, label='ADU')
    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"{base_filename}_amp_glow_map.png")
    plt.savefig(plot_path, dpi=150); plt.close(fig)
    print(f"Amp Glow map saved to: {plot_path}")

def collect_all_data(input_dir):
    """Recursively finds and processes all FITS pairs, returning a pandas DataFrame."""
    all_fits_files = sorted([os.path.join(root, f) for root, _, files in os.walk(input_dir) for f in files if f.lower().endswith(('.fits', '.fit'))])
    if len(all_fits_files) < 2: print(f"Error: Need at least 2 FITS files. Found {len(all_fits_files)}."); return pd.DataFrame(), {}
    if len(all_fits_files) % 2 != 0: print(f"Warning: Odd number of FITS files ({len(all_fits_files)}). Last file ignored.")
    
    all_data, metadata = [], {}
    for i in range(0, len(all_fits_files) - 1, 2):
        img1_path, img2_path = all_fits_files[i], all_fits_files[i+1]
        print(f"Processing pair: {i//2:04d} ({os.path.basename(img1_path)}, {os.path.basename(img2_path)})")
        img_data1, _, exp1, inst1, temp1, gain1, off1 = load_fits_and_extract_info(img1_path)
        img_data2, _, exp2, _, _, _, _ = load_fits_and_extract_info(img2_path)
        
        if not metadata and inst1 != 'N/A': metadata = {'instrume': inst1, 'set_temp': temp1, 'gain_setting': gain1, 'offset_setting': off1}
        if img_data1 is not None and img_data2 is not None and img_data1.shape == img_data2.shape and exp1 is not None and abs(exp1 - exp2) < 1e-6:
            all_data.extend(calculate_dark_noise_metrics(img_data1, img_data2, exp1))
    print(f"\nSuccessfully processed {len(all_data) // 9} image pairs into {len(all_data)} subframe data points.")
    return pd.DataFrame(all_data), metadata

def calculate_dark_noise_metrics(img_data1, img_data2, exposure_time):
    # This function is unchanged from the previous version.
    height, width = img_data1.shape
    sub_h, sub_w = height // 3, width // 3
    subframe_data = []
    diff_img = img_data2 - img_data1
    for i in range(3):
        for j in range(3):
            h_start, w_start, h_end, w_end = i * sub_h, j * sub_w, (i + 1) * sub_h, (j + 1) * sub_w
            sub_img1, sub_img2 = img_data1[h_start:h_end, w_start:w_end], img_data2[h_start:h_end, w_start:w_end]
            sub_diff = diff_img[h_start:h_end, w_start:w_end]
            raw_signal = np.mean((sub_img1 + sub_img2) / 2.0)
            noise_total = np.std(sub_img1, ddof=1)
            noise_temporal = np.std(sub_diff, ddof=1) / np.sqrt(2.0)
            subframe_data.append({'subframe_index': i * 3 + j, 'exposure_time': exposure_time, 'raw_signal': raw_signal,
                                  'noise_total': 1e-9 if np.isnan(noise_total) or noise_total == 0 else noise_total,
                                  'noise_temporal': 1e-9 if np.isnan(noise_temporal) or noise_temporal == 0 else noise_temporal})
    return subframe_data

def _perform_single_analysis(df_region):
    # This function is unchanged from the previous version.
    results = {k: np.nan for k in ['gain', 'dark_current_adu_per_sec', 'fpn_k']}
    if df_region.shape[0] < 5: return results
    fit_df = df_region[df_region['exposure_time'] > df_region['exposure_time'].min()] if len(df_region['exposure_time'].unique()) > 1 else df_region
    if fit_df.shape[0] > 2: results['dark_current_adu_per_sec'] = linregress(fit_df['exposure_time'], fit_df['effective_signal']).slope
    shot_fit_mask = (df_region['effective_signal'] > 0) & (df_region['shot_noise_var'] > 0)
    if shot_fit_mask.sum() > 5:
        gain_fit = linregress(df_region.loc[shot_fit_mask, 'effective_signal'], df_region.loc[shot_fit_mask, 'shot_noise_var'])
        results['gain'] = 1.0 / gain_fit.slope if gain_fit.slope > 0 else np.nan
    fpn_fit_mask = (df_region['effective_signal'] > 0) & (df_region['fpn_var'] > 0)
    if fpn_fit_mask.sum() > 5:
        df_region['effective_signal_sq'] = df_region['effective_signal']**2
        fpn_fit = linregress(df_region.loc[fpn_fit_mask, 'effective_signal_sq'], df_region.loc[fpn_fit_mask, 'fpn_var'])
        results['fpn_k'] = np.sqrt(fpn_fit.slope) if fpn_fit.slope > 0 else np.nan
    return results

def analyze_and_present(df, metadata, output_dir):
    """Orchestrates the entire subframe-by-subframe, dual-region analysis."""
    if df.empty: print("Error: DataFrame is empty."); return

    base_filename = f"dark_analysis_{metadata.get('instrume', 'UnknownCam').replace(' ', '_')}_G{metadata.get('gain_setting', 'NA')}_O{metadata.get('offset_setting', 'NA')}"
    generate_amp_glow_map(INPUT_DIR, output_dir, base_filename)
    
    df = df.sort_values(by='exposure_time').reset_index(drop=True)
    bias_frames = df[df['exposure_time'] == df['exposure_time'].min()]
    global_bias = bias_frames.groupby('subframe_index')['raw_signal'].mean()
    global_rn = bias_frames.groupby('subframe_index')['noise_temporal'].mean()
    
    for i in range(9):
        bias_adu, read_noise_adu = global_bias.get(i, 0), global_rn.get(i, 0)
        sub_idx = df['subframe_index'] == i
        df.loc[sub_idx, 'effective_signal'] = df.loc[sub_idx, 'raw_signal'] - bias_adu
        df.loc[sub_idx, 'shot_noise_var'] = df.loc[sub_idx, 'noise_temporal']**2 - read_noise_adu**2
        df.loc[sub_idx, 'fpn_var'] = df.loc[sub_idx, 'noise_total']**2 - df.loc[sub_idx, 'noise_temporal']**2
    
    for col in ['shot_noise_var', 'fpn_var']: df.loc[df[col] < 0, col] = 0
    df['fpn'] = np.sqrt(df['fpn_var'])

    all_results = {}
    for i in range(9):
        print(f"\n--- Analyzing Subframe {i} ---")
        df_sub = df[df['subframe_index'] == i]
        df_r1, df_r2 = df_sub[df_sub['exposure_time'] < ADC_THRESHOLD_S].copy(), df_sub[df_sub['exposure_time'] >= ADC_THRESHOLD_S].copy()
        print(f"Region 1 (<{ADC_THRESHOLD_S}s): {len(df_r1)} data points; Region 2 (>=...): {len(df_r2)} data points")
        all_results[i] = {'region1': _perform_single_analysis(df_r1), 'region2': _perform_single_analysis(df_r2)}

    generate_all_plots(df, all_results, global_rn, metadata, base_filename, output_dir)
    print_console_summary(all_results, global_bias, global_rn, metadata)
    generate_markdown_report(all_results, global_bias, global_rn, metadata, base_filename, output_dir)
    
    csv_path = os.path.join(output_dir, f"{base_filename}_full_data.csv"); df.to_csv(csv_path, index=False)
    print(f"\nFull data saved to: {csv_path}")

def _generate_plot_set(df_region, region_suffix, region_title, avg_k, global_rn, metadata, base_filename, output_dir):
    """Helper to generate a set of 3 plots for a specific data region."""
    if df_region.empty: return
    
    colors = plt.cm.viridis(np.linspace(0, 1, 9)); markers = ['o', 's', 'v', '^', '<', '>', 'D', 'p', '*']
    cam_info = f"{metadata.get('instrume', 'N/A')} (G:{metadata.get('gain_setting', 'NA')}, O:{metadata.get('offset_setting', 'NA')}, T:{metadata.get('set_temp', 'NA')}C)"
    full_title = f"{region_title}\n{cam_info}"

    # Plot 1: Temporal Noise vs Signal
    plt.figure(figsize=FIG_SIZE); ax1 = plt.gca()
    for i in range(9):
        df_sub = df_region[df_region['subframe_index'] == i]
        ax1.loglog(df_sub['effective_signal'], df_sub['noise_temporal'], marker=markers[i], ls='none', color=colors[i], markersize=5, alpha=0.7, label=f'SubF {i}')
    ax1.axhline(global_rn.mean(), color='r', ls='--', lw=2, label=f'Avg. RN ({global_rn.mean():.2f} ADU)')
    ax1.set_title(f'Temporal Noise vs. Signal - {full_title}', fontsize=TITLE_FONTSIZE); ax1.set_xlabel('Effective Signal (ADU)', fontsize=LABEL_FONTSIZE); ax1.set_ylabel('Temporal Noise (ADU)', fontsize=LABEL_FONTSIZE)
    ax1.grid(True, which="both", ls=":", color='0.7'); ax1.legend(ncol=2, fontsize=LEGEND_FONTSIZE)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{base_filename}{region_suffix}_1_temporal_noise.png"), dpi=150); plt.close()
    
    # Plot 2: Effective Signal vs Exposure
    plt.figure(figsize=FIG_SIZE); ax2 = plt.gca()
    for i in range(9):
        df_sub = df_region[df_region['subframe_index'] == i]
        ax2.plot(df_sub['exposure_time'], df_sub['effective_signal'], marker=markers[i], ls='none', color=colors[i], markersize=5, alpha=0.7, label=f'SubF {i}')
    ax2.set_title(f'Effective Signal vs. Exposure Time - {full_title}', fontsize=TITLE_FONTSIZE); ax2.set_xlabel('Exposure Time (s)', fontsize=LABEL_FONTSIZE); ax2.set_ylabel('Effective Signal (ADU)', fontsize=LABEL_FONTSIZE)
    ax2.grid(True, which="both", ls=":", color='0.7'); ax2.legend(ncol=2, fontsize=LEGEND_FONTSIZE)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{base_filename}{region_suffix}_2_signal_vs_exposure.png"), dpi=150); plt.close()

    # Plot 3: FPN vs Signal
    plt.figure(figsize=FIG_SIZE); ax3 = plt.gca()
    for i in range(9):
        df_sub = df_region[(df_region['subframe_index'] == i) & (df_region['effective_signal'] > 0) & (df_region['fpn'] > 0)]
        ax3.loglog(df_sub['effective_signal'], df_sub['fpn'], marker=markers[i], ls='none', color=colors[i], markersize=5, alpha=0.7, label=f'SubF {i}')
    if not df_region.empty and df_region.effective_signal.max() > 0:
        sig_range = np.logspace(np.log10(max(1, df_region.effective_signal.min())), np.log10(max(10, df_region.effective_signal.max())), 100)
        if not np.isnan(avg_k): ax3.plot(sig_range, sig_range * avg_k, 'k--', label=f'Avg. FPN (k={avg_k*100:.3f}%)')
    ax3.set_title(f'Fixed Pattern Noise vs. Signal - {full_title}', fontsize=TITLE_FONTSIZE); ax3.set_xlabel('Effective Signal (ADU)', fontsize=LABEL_FONTSIZE); ax3.set_ylabel('Fixed Pattern Noise (ADU)', fontsize=LABEL_FONTSIZE)
    ax3.grid(True, which="both", ls=":", color='0.7'); ax3.legend(ncol=2, fontsize=LEGEND_FONTSIZE)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"{base_filename}{region_suffix}_3_fpn_vs_signal.png"), dpi=150); plt.close()

def generate_all_plots(df, all_results, global_rn, metadata, base_filename, output_dir):
    """Generates two full sets of plots, one for each ADC region."""
    print("\nGenerating summary plots for each ADC region...")
    
    # Data for Region 1 (< 1.0s)
    df_lt_1s = df[df['exposure_time'] < ADC_THRESHOLD_S]
    k_vals_lt_1s = [res['region1'].get('fpn_k', np.nan) for res in all_results.values()]
    avg_k_lt_1s = np.nanmean(k_vals_lt_1s)
    _generate_plot_set(df_lt_1s, "_lt_1s", "Region < 1.0s", avg_k_lt_1s, global_rn, metadata, base_filename, output_dir)
    
    # Data for Region 2 (>= 1.0s)
    df_ge_1s = df[df['exposure_time'] >= ADC_THRESHOLD_S]
    k_vals_ge_1s = [res['region2'].get('fpn_k', np.nan) for res in all_results.values()]
    avg_k_ge_1s = np.nanmean(k_vals_ge_1s)
    _generate_plot_set(df_ge_1s, "_ge_1s", "Region >= 1.0s", avg_k_ge_1s, global_rn, metadata, base_filename, output_dir)
    
    print("All plot sets saved.")

def print_console_summary(results, bias, read_noise, metadata):
    # This function is unchanged from the previous version.
    print("\n" + "="*80); print(" " * 20 + "Dark Current and Noise Analysis Summary"); print("="*80)
    print(f"Camera: {metadata.get('instrume', 'N/A')}, Temp: {metadata.get('set_temp', 'N/A')}C, Gain: {metadata.get('gain_setting', 'N/A')}, Offset: {metadata.get('offset_setting', 'N/A')}")
    print("-" * 80)
    print(f"{'SubF':>4} | {'Bias':>7} {'RN':>6} | {'Gain (<1s)':>10} {'DC/s (<1s)':>10} | {'Gain (≥1s)':>10} {'DC/s (≥1s)':>10} | {'FPN k (≥1s)':>12}")
    print(f"{'':>4} | {'(ADU)':>7} {'(ADU)':>6} | {'(e-/ADU)':>10} {'(e-/s)':>10} | {'(e-/ADU)':>10} {'(e-/s)':>10} | {'(%)':>12}")
    print("-" * 80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(9):
            r1, r2 = results.get(i, {}).get('region1', {}), results.get(i, {}).get('region2', {})
            dc1_e, dc2_e = r1.get('dark_current_adu_per_sec', np.nan) * r1.get('gain', np.nan), r2.get('dark_current_adu_per_sec', np.nan) * r2.get('gain', np.nan)
            fpn_k_pct = r2.get('fpn_k', np.nan) * 100
            print(f"{i:4} | {bias.get(i, 0):7.1f} {read_noise.get(i, 0):6.2f} | {r1.get('gain', np.nan):10.3f} {dc1_e:10.3f} | {r2.get('gain', np.nan):10.3f} {dc2_e:10.3f} | {fpn_k_pct:11.4f}%")
    print("="*80)

def generate_markdown_report(results, bias, read_noise, metadata, base_filename, output_dir):
    """Generates a detailed Markdown report file with all results and plots."""
    report_path = os.path.join(output_dir, f"{base_filename}_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header and Summary Table (unchanged)
        header = f"""# Dark Current and Noise Analysis Report
- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Camera:** `{metadata.get('instrume', 'N/A')}`
- **Settings:** Temp=`{metadata.get('set_temp', 'N/A')}C`, Gain=`{metadata.get('gain_setting', 'N/A')}`, Offset=`{metadata.get('offset_setting', 'N/A')}`
- **Analysis Regions:** Region 1 (`<{ADC_THRESHOLD_S}s`), Region 2 (`≥{ADC_THRESHOLD_S}s`)
## Summary Table
"""
        table_header = f"| SubF | Bias<br>(ADU) | Read Noise<br>(ADU) | Gain<br>(e-/ADU)<br><{ADC_THRESHOLD_S}s | Dark Current<br>(e-/s)<br><{ADC_THRESHOLD_S}s | Gain<br>(e-/ADU)<br>≥{ADC_THRESHOLD_S}s | Dark Current<br>(e-/s)<br>≥{ADC_THRESHOLD_S}s | FPN Coeff.<br>(k, %)<br>≥{ADC_THRESHOLD_S}s |\n"
        table_align = "|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
        table_rows = ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(9):
                r1, r2 = results.get(i, {}).get('region1', {}), results.get(i, {}).get('region2', {})
                dc1_e, dc2_e = r1.get('dark_current_adu_per_sec', np.nan) * r1.get('gain', np.nan), r2.get('dark_current_adu_per_sec', np.nan) * r2.get('gain', np.nan)
                fpn_k_pct = r2.get('fpn_k', np.nan) * 100
                table_rows += f"| **{i}** | {bias.get(i, 0):.1f} | {read_noise.get(i, 0):.2f} | {r1.get('gain', np.nan):.3f} | {dc1_e:.3f} | {r2.get('gain', np.nan):.3f} | {dc2_e:.3f} | {fpn_k_pct:.4f}% |\n"
        
        # Updated plots section to include both sets
        plots_section = f"""## Amp Glow Map
![Amp Glow Map]({os.path.basename(base_filename)}_amp_glow_map.png)

---

## Analysis Plots for Region < 1.0s
This set of plots shows the sensor characteristics for very short exposures, governed by one ADC mode.

![Temporal Noise < 1.0s]({os.path.basename(base_filename)}_lt_1s_1_temporal_noise.png)
![Signal vs Exposure < 1.0s]({os.path.basename(base_filename)}_lt_1s_2_signal_vs_exposure.png)
![FPN vs Signal < 1.0s]({os.path.basename(base_filename)}_lt_1s_3_fpn_vs_signal.png)

---

## Analysis Plots for Region ≥ 1.0s
This set of plots shows the sensor characteristics for longer exposures, governed by the second ADC mode.

![Temporal Noise >= 1.0s]({os.path.basename(base_filename)}_ge_1s_1_temporal_noise.png)
![Signal vs Exposure >= 1.0s]({os.path.basename(base_filename)}_ge_1s_2_signal_vs_exposure.png)
![FPN vs Signal >= 1.0s]({os.path.basename(base_filename)}_ge_1s_3_fpn_vs_signal.png)
"""
        f.write(header + table_header + table_align + table_rows + plots_section)
    print(f"\nMarkdown report saved to: {report_path}")

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.isdir(INPUT_DIR): print(f"Error: Input directory '{INPUT_DIR}' not found."); exit()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_df, metadata = collect_all_data(INPUT_DIR)
    if not data_df.empty: analyze_and_present(data_df, metadata, OUTPUT_DIR)
    else: print("\nNo valid data collected. Please check the input directory and FITS files.\n")