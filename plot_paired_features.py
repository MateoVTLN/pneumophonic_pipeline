"""
Explore and plot paired HDF5 data produced by test_paired.py.
Saves figures as PDF in data_target/<batch>/figures/paired/
"""
from pathlib import Path
import matplotlib.pyplot as plt
from pneumophonic_analysis.paired_features import PairedFeatureExtractor

# ---- Paths ----
PROJECT_ROOT = Path(__file__).parent
DATA_TARGET  = PROJECT_ROOT / "data_target"

BATCHES = ["healthy_subjects", "pathological_subjects"]

# ---- Selection helpers ----
def select_from_list(items, label):
    print(f"\nAvailable {label}:")
    for idx, item in enumerate(items):
        print(f"  [{idx}] {item}")
    while True:
        sel = input(f"Select {label} by number: ")
        if sel.isdigit() and 0 <= int(sel) < len(items):
            return int(sel)
        print("Invalid selection. Try again.")

def select_batch():
    idx = select_from_list(BATCHES, "batches")
    return BATCHES[idx]

def select_h5(batch_name):
    paired_dir = DATA_TARGET / batch_name / "paired"
    if not paired_dir.exists():
        print(f"\n⚠️  No paired/ folder found in data_target/{batch_name}/")
        print("   Run test_paired.py first to generate HDF5 files.")
        exit(1)

    h5_files = sorted(paired_dir.glob("*.h5"))
    if not h5_files:
        print(f"\n⚠️  No .h5 files found in {paired_dir}")
        print("   Run test_paired.py first to generate HDF5 files.")
        exit(1)

    idx = select_from_list([f.name for f in h5_files], f"paired datasets in {batch_name}")
    return h5_files[idx]

# ---- Selection ----
batch_name = select_batch()
h5_path = select_h5(batch_name)

# ---- Load ----
df, meta = PairedFeatureExtractor.load_hdf5(h5_path)

subject_id = meta['subject_id']
task_name = meta['task_name']

print(f"\n✅ Loaded: {h5_path.name}")
print(f"   Subject: {subject_id}, Task: {task_name}")
print(f"   Matrix: {df.shape[0]} frames × {df.shape[1]} features")

# ---- Output folder ----
fig_dir = DATA_TARGET / batch_name / "figures" / "paired" / subject_id
fig_dir.mkdir(parents=True, exist_ok=True)

def save_fig(fig, name):
    """Save figure as PDF and print path."""
    path = fig_dir / f"{subject_id}_{task_name}_{name}.pdf"
    fig.savefig(path, bbox_inches='tight')
    print(f"  📄 Saved: {path.relative_to(PROJECT_ROOT)}")
    return path

# ---- Plot 1: Audio Energy vs Chest Wall Volume ----
fig1, ax1 = plt.subplots(figsize=(14, 5))

ax1.plot(df['time'], df['energy'], color='steelblue', label='Audio Energy')
ax1.set_ylabel('Energy', color='steelblue')

ax2 = ax1.twinx()
ax2.plot(df['time'], df['delta_vcw'], color='coral', label='ΔVcw')
ax2.set_ylabel('ΔVcw (L)', color='coral')

ax1.set_xlabel('Time (s)')
ax1.set_title(f"{subject_id} — {task_name}: Energy vs Volume")
fig1.legend(loc='upper right')
plt.tight_layout()
save_fig(fig1, "energy_vs_volume")

# ---- Plot 2: F0 vs Expiratory Flow (voiced only) ----
voiced = df[df['voiced'] == 1.0].copy()

if len(voiced) > 0:
    fig2, ax1 = plt.subplots(figsize=(14, 5))

    ax1.plot(voiced['time'], voiced['f0'], color='purple', alpha=0.7, label='F0 (Hz)')
    ax1.set_ylabel('F0 (Hz)', color='purple')

    ax2 = ax1.twinx()
    ax2.plot(voiced['time'], voiced['flow_cw'], color='green', alpha=0.7, label='Flow CW (L/s)')
    ax2.set_ylabel('Flow CW (L/s)', color='green')

    ax1.set_xlabel('Time (s)')
    ax1.set_title(f"{subject_id} — {task_name}: F0 vs Expiratory Flow (voiced only)")
    fig2.legend(loc='upper right')
    plt.tight_layout()
    save_fig(fig2, "f0_vs_flow")
else:
    print("\n⚠️  No voiced frames found — skipping F0 vs Flow plot.")

# ---- Plot 3: Compartmental contributions over time ----
fig3, ax1 = plt.subplots(figsize=(14, 5))

ax1.plot(df['time'], df['pct_rc'], color='royalblue', label='%RC (rib cage)')
ax1.plot(df['time'], df['pct_ab'], color='darkorange', label='%AB (abdomen)')
ax1.set_ylabel('Compartmental contribution')
ax1.set_xlabel('Time (s)')
ax1.set_title(f"{subject_id} — {task_name}: Compartmental Strategy")
ax1.legend()
plt.tight_layout()
save_fig(fig3, "compartmental")

# ---- Correlation summary (voiced frames) ----
if len(voiced) > 10:
    cols = ['f0', 'energy', 'spectral_centroid', 'flow_cw', 'vcw', 'pct_rc', 'delta_vcw']
    available = [c for c in cols if c in voiced.columns]
    corr = voiced[available].corr()
    print(f"\n📊 Correlation matrix (voiced frames, n={len(voiced)}):\n")
    print(corr.round(3))

print(f"\n✅ All figures saved in: {fig_dir.relative_to(PROJECT_ROOT)}")
plt.show()
