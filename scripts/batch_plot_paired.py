"""
Batch plot: generate PDF figures for all .h5 files in a batch.
Saves to data_target/<batch>/figures/paired/<subject_id>/
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for batch processing
import matplotlib.pyplot as plt
from pneumophonic_analysis.paired_features import PairedFeatureExtractor

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_TARGET  = PROJECT_ROOT / "data_target"

BATCHES = ["healthy_subjects", "pathological_subjects"]

# ---- Batch selection ----
print("\nAvailable batches:")
for idx, name in enumerate(BATCHES):
    print(f"  [{idx}] {name}")
while True:
    sel = input("Select batch by number: ")
    if sel.isdigit() and 0 <= int(sel) < len(BATCHES):
        batch_name = BATCHES[int(sel)]
        break
    print("Invalid selection. Try again.")

paired_dir = DATA_TARGET / batch_name / "paired"
h5_files = sorted(paired_dir.glob("*.h5"))

if not h5_files:
    print(f"\n⚠️  No .h5 files found in {paired_dir}")
    exit(1)

print(f"\nFound {len(h5_files)} paired datasets. Generating figures...\n")

# ---- Process each file ----
for h5_path in h5_files:
    try:
        df, meta = PairedFeatureExtractor.load_hdf5(h5_path)
        sid = meta['subject_id']
        task = meta['task_name']

        fig_dir = DATA_TARGET / batch_name / "figures" / "paired" / str(sid)
        fig_dir.mkdir(parents=True, exist_ok=True)

        voiced = df[df['voiced'] == 1.0]

        # ---- Plot 1: Energy vs Volume ----
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(df['time'], df['energy'], color='steelblue', label='Audio Energy')
        ax1.set_ylabel('Energy', color='steelblue')
        ax2 = ax1.twinx()
        ax2.plot(df['time'], df['delta_vcw'], color='coral', label='ΔVcw')
        ax2.set_ylabel('ΔVcw (L)', color='coral')
        ax1.set_xlabel('Time (s)')
        ax1.set_title(f"{sid} — {task}: Energy vs Volume")
        fig.legend(loc='upper right')
        plt.tight_layout()
        fig.savefig(fig_dir / f"{sid}_{task}_energy_vs_volume.pdf", bbox_inches='tight')
        plt.close(fig)

        # ---- Plot 2: F0 vs Flow (voiced only) ----
        if len(voiced) > 0:
            fig, ax1 = plt.subplots(figsize=(14, 5))
            ax1.plot(voiced['time'], voiced['f0'], color='purple', alpha=0.7, label='F0 (Hz)')
            ax1.set_ylabel('F0 (Hz)', color='purple')
            ax2 = ax1.twinx()
            ax2.plot(voiced['time'], voiced['flow_cw'], color='green', alpha=0.7, label='Flow CW (L/s)')
            ax2.set_ylabel('Flow CW (L/s)', color='green')
            ax1.set_xlabel('Time (s)')
            ax1.set_title(f"{sid} — {task}: F0 vs Expiratory Flow (voiced only)")
            fig.legend(loc='upper right')
            plt.tight_layout()
            fig.savefig(fig_dir / f"{sid}_{task}_f0_vs_flow.pdf", bbox_inches='tight')
            plt.close(fig)

        # ---- Plot 3: Compartmental strategy ----
        fig, ax1 = plt.subplots(figsize=(14, 5))
        ax1.plot(df['time'], df['pct_rc'], color='royalblue', label='%RC (rib cage)')
        ax1.plot(df['time'], df['pct_ab'], color='darkorange', label='%AB (abdomen)')
        ax1.set_ylabel('Compartmental contribution')
        ax1.set_xlabel('Time (s)')
        ax1.set_title(f"{sid} — {task}: Compartmental Strategy")
        ax1.legend()
        plt.tight_layout()
        fig.savefig(fig_dir / f"{sid}_{task}_compartmental.pdf", bbox_inches='tight')
        plt.close(fig)

        print(f"  ✅ {sid}/{task} — 3 figures saved")

    except Exception as e:
        print(f"  ❌ {h5_path.name} — {e}")

print(f"\n✅ Done. Figures in: data_target/{batch_name}/figures/paired/")