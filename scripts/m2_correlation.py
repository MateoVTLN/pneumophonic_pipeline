"""
M2 — Exploratory Correlation Analysis
======================================

Three levels of analysis on the paired HDF5 corpus:

1. GLOBAL: per-segment summary statistics correlated across subjects
2. TIME-RESOLVED: sliding-window cross-correlation within recordings
3. EVENT-ALIGNED: above-FRC vs below-FRC comparison (sustained tasks only)

Outputs:
    data_target/<batch>/m2_correlation/
    ├── global_summary.csv           — summary stats per subject × task
    ├── global_correlation_matrix.pdf — heatmap of cross-feature correlations
    ├── global_scatter_plots.pdf     — key scatter plots (energy↔volume, f0↔flow)
    ├── time_resolved/               — per-subject cross-correlation plots
    └── m2_report.txt                — text summary of findings


"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pneumophonic_analysis.paired_features import PairedFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_TARGET  = PROJECT_ROOT / "data_target"

BATCHES = ["healthy_subjects", "pathological_subjects"]

# ---- Batch selection ----
def select_batch():
    print("\nAvailable batches:")
    for idx, name in enumerate(BATCHES):
        print(f"  [{idx}] {name}")
    while True:
        sel = input("Select batch by number: ")
        if sel.isdigit() and 0 <= int(sel) < len(BATCHES):
            return BATCHES[int(sel)]
        print("Invalid selection. Try again.")

# ---- Task categories ----
SUSTAINED_TASKS = {'a', 'a_2', 'a_3', 'a_7', 'r'}
SPEECH_TASKS = {'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'testo'}
VOWEL_TASKS = {'a', 'e', 'i', 'o', 'u'}

# =====================================================================
# LEVEL 1: GLOBAL (per-segment) correlations
# =====================================================================

def compute_segment_summary(df, meta):
    """
    Compute summary statistics for one paired segment.
    Returns a dict with one value per feature.
    """
    voiced = df[df['voiced'] == 1.0]
    n_voiced = len(voiced)
    n_total = len(df)

    summary = {
        'subject_id': meta.get('subject_id', ''),
        'task': meta.get('task_name', ''),
        'duration_sec': meta.get('audio_duration_sec', n_total * 0.015),
        'n_frames': n_total,
        'n_voiced': n_voiced,
        'voiced_ratio': n_voiced / n_total if n_total > 0 else 0,
    }

    # --- Audio features (voiced frames only) ---
    if n_voiced > 10:
        summary['f0_mean'] = np.nanmean(voiced['f0'])
        summary['f0_std'] = np.nanstd(voiced['f0'])
        summary['f0_range'] = np.nanmax(voiced['f0']) - np.nanmin(voiced['f0'])
        summary['energy_mean'] = voiced['energy'].mean()
        summary['energy_std'] = voiced['energy'].std()
        summary['spectral_centroid_mean'] = voiced['spectral_centroid'].mean()

        # MFCCs — mean of first 5
        for i in range(min(5, sum(1 for c in voiced.columns if c.startswith('mfcc_')))):
            summary[f'mfcc_{i}_mean'] = voiced[f'mfcc_{i}'].mean()
    else:
        summary['f0_mean'] = np.nan
        summary['f0_std'] = np.nan
        summary['f0_range'] = np.nan
        summary['energy_mean'] = np.nan
        summary['energy_std'] = np.nan
        summary['spectral_centroid_mean'] = np.nan

    # --- OEP features (all frames) ---
    summary['vcw_mean'] = df['vcw'].mean()
    summary['delta_vcw_range'] = df['delta_vcw'].max() - df['delta_vcw'].min()
    summary['flow_cw_mean'] = df['flow_cw'].mean()
    summary['flow_cw_std'] = df['flow_cw'].std()
    summary['flow_rc_mean'] = df['flow_rc'].mean()
    summary['flow_ab_mean'] = df['flow_ab'].mean()
    summary['pct_rc_mean'] = df['pct_rc'].mean()
    summary['pct_rc_std'] = df['pct_rc'].std()
    summary['pct_ab_mean'] = df['pct_ab'].mean()

    # --- Frame-level correlations (within this segment, voiced only) ---
    if n_voiced > 20:
        # Energy ↔ delta_vcw
        r, p = stats.pearsonr(voiced['energy'], voiced['delta_vcw'])
        summary['corr_energy_deltavcw'] = r
        summary['pval_energy_deltavcw'] = p

        # F0 ↔ flow_cw (drop NaN F0 frames)
        f0_valid = voiced.dropna(subset=['f0'])
        if len(f0_valid) > 20:
            r, p = stats.pearsonr(f0_valid['f0'], f0_valid['flow_cw'])
            summary['corr_f0_flowcw'] = r
            summary['pval_f0_flowcw'] = p
        else:
            summary['corr_f0_flowcw'] = np.nan
            summary['pval_f0_flowcw'] = np.nan

        # Energy ↔ flow_cw
        r, p = stats.pearsonr(voiced['energy'], voiced['flow_cw'])
        summary['corr_energy_flowcw'] = r
        summary['pval_energy_flowcw'] = p
    else:
        summary['corr_energy_deltavcw'] = np.nan
        summary['pval_energy_deltavcw'] = np.nan
        summary['corr_f0_flowcw'] = np.nan
        summary['pval_f0_flowcw'] = np.nan
        summary['corr_energy_flowcw'] = np.nan
        summary['pval_energy_flowcw'] = np.nan

    return summary


def run_global_analysis(paired_dir, output_dir):
    """Level 1: compute summary stats and cross-subject correlations."""
    print("\n" + "="*60)
    print("LEVEL 1: Global per-segment correlations")
    print("="*60)

    h5_files = sorted(paired_dir.glob("*.h5"))
    print(f"  Loading {len(h5_files)} paired datasets...")

    summaries = []
    for h5 in h5_files:
        try:
            df, meta = PairedFeatureExtractor.load_hdf5(h5)
            summary = compute_segment_summary(df, meta)
            summaries.append(summary)
        except Exception as e:
            logger.warning(f"  Failed to load {h5.name}: {e}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "global_summary.csv", index=False)
    print(f"  Summary: {len(summary_df)} segments from "
          f"{summary_df['subject_id'].nunique()} subjects")

    # ---- Correlation heatmap (audio ↔ OEP features) ----
    audio_cols = ['f0_mean', 'f0_std', 'energy_mean', 'energy_std', 'spectral_centroid_mean']
    oep_cols = ['delta_vcw_range', 'flow_cw_mean', 'flow_cw_std', 'pct_rc_mean', 'pct_rc_std']
    all_corr_cols = audio_cols + oep_cols

    available = [c for c in all_corr_cols if c in summary_df.columns]
    corr_data = summary_df[available].dropna()

    if len(corr_data) > 10:
        corr_matrix = corr_data.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, ax=ax,
            linewidths=0.5
        )
        ax.set_title('Global Correlation: Audio ↔ OEP Features\n(per-segment summaries across subjects)')
        plt.tight_layout()
        fig.savefig(output_dir / "global_correlation_matrix.pdf", bbox_inches='tight')
        plt.close(fig)
        print("  📄 Saved: global_correlation_matrix.pdf")

    # ---- Correlation heatmap by task type ----
    for task_group_name, task_set in [("sustained", SUSTAINED_TASKS), ("speech", SPEECH_TASKS)]:
        subset = summary_df[summary_df['task'].isin(task_set)]
        sub_corr = subset[available].dropna()
        if len(sub_corr) > 10:
            corr_m = sub_corr.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                corr_m, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                linewidths=0.5
            )
            ax.set_title(f'Global Correlation — {task_group_name.upper()} tasks only')
            plt.tight_layout()
            fig.savefig(output_dir / f"global_correlation_{task_group_name}.pdf", bbox_inches='tight')
            plt.close(fig)
            print(f"  📄 Saved: global_correlation_{task_group_name}.pdf")

    # ---- Key scatter plots ----
    scatter_pairs = [
        ('delta_vcw_range', 'energy_mean', 'Volume Excursion vs Mean Energy'),
        ('flow_cw_mean', 'f0_mean', 'Mean Flow vs Mean F0'),
        ('flow_cw_std', 'energy_std', 'Flow Variability vs Energy Variability'),
        ('pct_rc_mean', 'f0_mean', 'Mean %RC vs Mean F0'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (x_col, y_col, title) in enumerate(scatter_pairs):
        ax = axes[idx]
        if x_col in summary_df.columns and y_col in summary_df.columns:
            valid = summary_df[[x_col, y_col, 'task']].dropna()
            # Color by task type
            colors = []
            for t in valid['task']:
                if t in SUSTAINED_TASKS:
                    colors.append('steelblue')
                elif t in SPEECH_TASKS:
                    colors.append('coral')
                else:
                    colors.append('gray')
            ax.scatter(valid[x_col], valid[y_col], c=colors, alpha=0.6, s=30)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(title)

            # Add regression line
            r, p = stats.pearsonr(valid[x_col], valid[y_col])
            ax.annotate(f'r={r:.3f}, p={p:.3e}', xy=(0.05, 0.95),
                       xycoords='axes fraction', fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', label='Sustained', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='coral', label='Speech', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='Vowels', markersize=8),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(output_dir / "global_scatter_plots.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  📄 Saved: global_scatter_plots.pdf")

    # ---- Within-segment correlation distributions ----
    corr_cols_within = ['corr_energy_deltavcw', 'corr_f0_flowcw', 'corr_energy_flowcw']
    available_within = [c for c in corr_cols_within if c in summary_df.columns]

    if available_within:
        fig, axes = plt.subplots(1, len(available_within), figsize=(6 * len(available_within), 5))
        if len(available_within) == 1:
            axes = [axes]

        for ax, col in zip(axes, available_within):
            data = summary_df[col].dropna()
            ax.hist(data, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax.axvline(data.median(), color='red', linestyle='--', linewidth=2,
                      label=f'median={data.median():.3f}')
            ax.set_xlabel(f'Pearson r')
            ax.set_ylabel('Count')
            ax.set_title(col.replace('corr_', '').replace('_', ' ↔ '))
            ax.legend()
            ax.set_xlim(-1, 1)

        plt.suptitle('Distribution of Within-Segment Correlations Across All Subjects × Tasks')
        plt.tight_layout()
        fig.savefig(output_dir / "within_segment_correlation_distributions.pdf", bbox_inches='tight')
        plt.close(fig)
        print("  📄 Saved: within_segment_correlation_distributions.pdf")

    return summary_df


# =====================================================================
# LEVEL 2: TIME-RESOLVED cross-correlations
# =====================================================================

def compute_sliding_correlation(x, y, window_frames=33):
    """
    Sliding-window Pearson correlation.
    window_frames=33 ≈ 0.5s at ~66 fps.
    """
    n = len(x)
    if n < window_frames:
        return np.array([]), np.array([])

    half_w = window_frames // 2
    r_values = np.full(n, np.nan)

    for i in range(half_w, n - half_w):
        x_win = x[i - half_w:i + half_w + 1]
        y_win = y[i - half_w:i + half_w + 1]
        valid = ~(np.isnan(x_win) | np.isnan(y_win))
        if np.sum(valid) > 10:
            r, _ = stats.pearsonr(x_win[valid], y_win[valid])
            r_values[i] = r

    return r_values


def run_time_resolved(paired_dir, output_dir, max_subjects=10):
    """
    Level 2: sliding-window cross-correlation within recordings.
    Runs on sustained tasks only (clearest signal).
    """
    print("\n" + "="*60)
    print("LEVEL 2: Time-resolved cross-correlations")
    print("="*60)

    tr_dir = output_dir / "time_resolved"
    tr_dir.mkdir(exist_ok=True)

    h5_files = sorted(paired_dir.glob("*.h5"))
    processed = 0

    # Collect all sliding correlations for aggregate statistics
    all_energy_vcw_corrs = []

    for h5 in h5_files:
        try:
            df, meta = PairedFeatureExtractor.load_hdf5(h5)
            task = meta.get('task_name', '')
            sid = meta.get('subject_id', '')

            # Only sustained tasks for time-resolved analysis
            if task not in SUSTAINED_TASKS:
                continue
            if len(df) < 100:  # need at least ~1.5s
                continue

            # Sliding correlation: energy ↔ delta_vcw
            r_energy_vcw = compute_sliding_correlation(
                df['energy'].values, df['delta_vcw'].values, window_frames=33
            )

            all_energy_vcw_corrs.append({
                'subject': sid, 'task': task,
                'r_values': r_energy_vcw,
                'time': df['time'].values
            })

            # Plot for first N subjects
            if processed < max_subjects:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

                # Top: signals
                ax1.plot(df['time'], df['energy'], color='steelblue', alpha=0.7, label='Energy')
                ax1b = ax1.twinx()
                ax1b.plot(df['time'], df['delta_vcw'], color='coral', alpha=0.7, label='ΔVcw')
                ax1.set_ylabel('Energy', color='steelblue')
                ax1b.set_ylabel('ΔVcw (L)', color='coral')
                ax1.set_title(f'{sid} — {task}: Time-Resolved Correlation')
                ax1.legend(loc='upper left')
                ax1b.legend(loc='upper right')

                # Bottom: sliding correlation
                ax2.plot(df['time'], r_energy_vcw, color='purple', linewidth=1.5)
                ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
                ax2.fill_between(df['time'], r_energy_vcw, 0,
                               where=~np.isnan(r_energy_vcw),
                               alpha=0.3, color='purple')
                ax2.set_ylabel('Pearson r (0.5s window)')
                ax2.set_xlabel('Time (s)')
                ax2.set_ylim(-1, 1)
                ax2.set_title('Sliding Energy ↔ ΔVcw Correlation')

                plt.tight_layout()
                fig.savefig(tr_dir / f"{sid}_{task}_time_resolved.pdf", bbox_inches='tight')
                plt.close(fig)

            processed += 1

        except Exception as e:
            logger.warning(f"  Time-resolved failed for {h5.name}: {e}")

    print(f"  Processed {processed} sustained segments")
    print(f"  📄 Saved {min(processed, max_subjects)} individual plots in time_resolved/")

    # Aggregate: mean sliding correlation across all subjects
    if all_energy_vcw_corrs:
        # Compute distribution of mean correlations
        mean_rs = [np.nanmean(entry['r_values']) for entry in all_energy_vcw_corrs]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(mean_rs, bins=15, edgecolor='black', alpha=0.7, color='purple')
        ax.axvline(np.median(mean_rs), color='red', linestyle='--',
                   label=f'median={np.median(mean_rs):.3f}')
        ax.set_xlabel('Mean sliding r (Energy ↔ ΔVcw)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Time-Resolved Energy–Volume Coupling\n(sustained tasks, 0.5s window)')
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_dir / "time_resolved_aggregate.pdf", bbox_inches='tight')
        plt.close(fig)
        print("  📄 Saved: time_resolved_aggregate.pdf")

    return all_energy_vcw_corrs


# =====================================================================
# LEVEL 3: EVENT-ALIGNED (FRC crossing)
# =====================================================================

def run_frc_analysis(paired_dir, output_dir):
    """
    Level 3: compare audio features above vs below FRC.
    Only for sustained tasks (a_2, a_3, a_7) where volume
    crosses FRC during a single expiration.
    """
    print("\n" + "="*60)
    print("LEVEL 3: Event-aligned FRC analysis")
    print("="*60)

    frc_tasks = {'a_2', 'a_3', 'a_7'}
    h5_files = sorted(paired_dir.glob("*.h5"))

    frc_results = []

    for h5 in h5_files:
        try:
            df, meta = PairedFeatureExtractor.load_hdf5(h5)
            task = meta.get('task_name', '')
            sid = meta.get('subject_id', '')

            if task not in frc_tasks:
                continue

            # Find FRC crossing: volume rises during inspiration, then descends
            # during phonation. FRC ≈ the volume at segment start (pre-inspiration level).
            # Strategy: find the peak volume (end of inspiration), then find where
            # delta_vcw crosses zero AFTER that peak (volume returns to starting level).
            dcw = df['delta_vcw'].values

            # Find the volume peak (end of inspiration / start of expiration)
            peak_idx = np.argmax(dcw)

            # After the peak, find where delta_vcw crosses zero (descending)
            post_peak = dcw[peak_idx:]
            descending_crossings = np.where(
                (post_peak[:-1] > 0) & (post_peak[1:] <= 0)
            )[0]

            if len(descending_crossings) > 0:
                cross_idx = peak_idx + descending_crossings[0]
            else:
                # No zero crossing after peak — volume never returns to start
                # Use the midpoint of the descent as a fallback
                cross_idx = peak_idx + len(post_peak) // 2
                if cross_idx >= len(df) - 20:
                    continue  # not enough data after crossing

            # Split into above and below FRC
            above = df.iloc[:cross_idx]
            below = df.iloc[cross_idx:]

            if len(above) < 20 or len(below) < 20:
                continue

            above_voiced = above[above['voiced'] == 1.0]
            below_voiced = below[below['voiced'] == 1.0]

            if len(above_voiced) < 10 or len(below_voiced) < 10:
                continue

            result = {
                'subject_id': sid,
                'task': task,
                'frc_cross_time': df['time'].iloc[cross_idx],
                'duration_above': len(above) * 0.015,
                'duration_below': len(below) * 0.015,

                # Audio above FRC
                'f0_above': np.nanmean(above_voiced['f0']),
                'energy_above': above_voiced['energy'].mean(),
                'spectral_centroid_above': above_voiced['spectral_centroid'].mean(),

                # Audio below FRC
                'f0_below': np.nanmean(below_voiced['f0']),
                'energy_below': below_voiced['energy'].mean(),
                'spectral_centroid_below': below_voiced['spectral_centroid'].mean(),

                # OEP above vs below
                'flow_cw_above': above['flow_cw'].mean(),
                'flow_cw_below': below['flow_cw'].mean(),
                'pct_rc_above': above['pct_rc'].mean(),
                'pct_rc_below': below['pct_rc'].mean(),
            }

            # Deltas (below - above)
            result['f0_shift'] = result['f0_below'] - result['f0_above']
            result['energy_shift'] = result['energy_below'] - result['energy_above']
            result['pct_rc_shift'] = result['pct_rc_below'] - result['pct_rc_above']
            result['flow_shift'] = result['flow_cw_below'] - result['flow_cw_above']

            frc_results.append(result)

        except Exception as e:
            logger.warning(f"  FRC analysis failed for {h5.name}: {e}")

    if not frc_results:
        print("  No valid FRC segments found.")
        return pd.DataFrame()

    frc_df = pd.DataFrame(frc_results)
    frc_df.to_csv(output_dir / "frc_analysis.csv", index=False)
    print(f"  Analyzed {len(frc_df)} segments from {frc_df['subject_id'].nunique()} subjects")

    # ---- Plot: above vs below FRC comparison ----
    shift_cols = [
        ('f0_shift', 'F0 shift (Hz)', 'Below−Above FRC'),
        ('energy_shift', 'Energy shift', 'Below−Above FRC'),
        ('pct_rc_shift', '%RC shift', 'Below−Above FRC'),
        ('flow_shift', 'Flow shift (L/s)', 'Below−Above FRC'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (col, ylabel, xlabel) in enumerate(shift_cols):
        ax = axes[idx]
        data = frc_df[col].dropna()
        ax.hist(data, bins=15, edgecolor='black', alpha=0.7, color='teal')
        ax.axvline(0, color='gray', linestyle='--')
        ax.axvline(data.median(), color='red', linestyle='--',
                   label=f'median={data.median():.4f}')
        ax.set_xlabel(ylabel)
        ax.set_ylabel('Count')
        ax.set_title(f'{xlabel}: {ylabel}')
        ax.legend(fontsize=8)

        # Wilcoxon test: is the shift significantly different from 0?
        if len(data) > 5:
            try:
                stat, p = stats.wilcoxon(data)
                ax.annotate(f'Wilcoxon p={p:.3e}', xy=(0.05, 0.85),
                           xycoords='axes fraction', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except Exception:
                pass

    plt.suptitle('Effect of FRC Crossing on Audio and Respiratory Features', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "frc_shifts.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  📄 Saved: frc_shifts.pdf")

    # ---- Paired scatter: above vs below ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (above_col, below_col, label) in zip(axes, [
        ('f0_above', 'f0_below', 'F0 (Hz)'),
        ('energy_above', 'energy_below', 'Energy'),
        ('pct_rc_above', 'pct_rc_below', '%RC'),
    ]):
        valid = frc_df[[above_col, below_col]].dropna()
        ax.scatter(valid[above_col], valid[below_col], alpha=0.6, s=30, color='teal')
        lims = [
            min(valid[above_col].min(), valid[below_col].min()),
            max(valid[above_col].max(), valid[below_col].max()),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='identity')
        ax.set_xlabel(f'{label} — Above FRC')
        ax.set_ylabel(f'{label} — Below FRC')
        ax.set_title(f'{label}: Above vs Below FRC')
        ax.legend()

    plt.suptitle('Paired Comparison: Above vs Below FRC', fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "frc_paired_scatter.pdf", bbox_inches='tight')
    plt.close(fig)
    print("  📄 Saved: frc_paired_scatter.pdf")

    return frc_df


# =====================================================================
# TEXT REPORT
# =====================================================================

def write_report(summary_df, frc_df, output_dir):
    """Write a text summary of M2 findings."""
    report_path = output_dir / "m2_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("M2 — EXPLORATORY CORRELATION ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total segments analyzed: {len(summary_df)}\n")
        f.write(f"Unique subjects: {summary_df['subject_id'].nunique()}\n")
        f.write(f"Tasks: {sorted(summary_df['task'].unique())}\n\n")

        # Within-segment correlations
        f.write("WITHIN-SEGMENT CORRELATIONS (per recording)\n")
        f.write("-" * 40 + "\n")
        for col, label in [
            ('corr_energy_deltavcw', 'Energy ↔ ΔVcw'),
            ('corr_f0_flowcw', 'F0 ↔ Flow CW'),
            ('corr_energy_flowcw', 'Energy ↔ Flow CW'),
        ]:
            if col in summary_df.columns:
                data = summary_df[col].dropna()
                f.write(f"  {label}:\n")
                f.write(f"    median r = {data.median():.3f}\n")
                f.write(f"    mean r   = {data.mean():.3f} ± {data.std():.3f}\n")
                f.write(f"    range    = [{data.min():.3f}, {data.max():.3f}]\n")
                f.write(f"    n        = {len(data)}\n\n")

        # FRC analysis
        if frc_df is not None and len(frc_df) > 0:
            f.write("\nFRC CROSSING ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Segments with FRC crossing: {len(frc_df)}\n\n")
            for col, label in [
                ('f0_shift', 'F0 shift (Below−Above)'),
                ('energy_shift', 'Energy shift'),
                ('pct_rc_shift', '%RC shift'),
            ]:
                data = frc_df[col].dropna()
                f.write(f"  {label}:\n")
                f.write(f"    median = {data.median():.4f}\n")
                f.write(f"    mean   = {data.mean():.4f} ± {data.std():.4f}\n")
                if len(data) > 5:
                    try:
                        stat, p = stats.wilcoxon(data)
                        f.write(f"    Wilcoxon p = {p:.3e}\n")
                    except Exception:
                        pass
                f.write("\n")

    print(f"\n  📄 Saved: m2_report.txt")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    batch_name = select_batch()
    paired_dir = DATA_TARGET / batch_name / "paired"
    output_dir = DATA_TARGET / batch_name / "m2_correlation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Level 1: Global
    summary_df = run_global_analysis(paired_dir, output_dir)

    # Level 2: Time-resolved
    time_corrs = run_time_resolved(paired_dir, output_dir, max_subjects=10)

    # Level 3: FRC
    frc_df = run_frc_analysis(paired_dir, output_dir)

    # Report
    write_report(summary_df, frc_df, output_dir)

    print("\n" + "="*60)
    print("M2 COMPLETE")
    print(f"All outputs in: {output_dir.relative_to(PROJECT_ROOT)}")
    print("="*60)