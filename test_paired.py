"""
Quick test script for paired feature extraction.
Reads phonation timings from the subject's Excel file to trim
the extraction to the actual vocal task window.
"""
import logging
from pathlib import Path
import pandas as pd
from pneumophonic_analysis import create_config
from pneumophonic_analysis.paired_features import PairedFeatureExtractor

logging.basicConfig(level=logging.INFO)

# ---- Paths ----
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT    = PROJECT_ROOT / "data_root"
DATA_TARGET  = PROJECT_ROOT / "data_target"

BATCHES = ["healthy_subjects", "pathological_subjects"]

# ---- Mapping: Excel task label → audio filename in renders/ ----
TASK_TO_FILE = {
    'a':     'a.wav',
    'e':     'e.wav',
    'i':     'i.wav',
    'o':     'o.wav',
    'u':     'u.wav',
    'a_2':   'phonema_a_2.wav',
    'a_3':   'phonema_a_3.wav',
    'a_7':   'phonema_a_7_2.wav',
    'r':     'r.wav',
    'f_1':   'phrase_1.wav',
    'f_2':   'phrase_2.wav',
    'f_3':   'phrase_3.wav',
    'f_4':   'phrase_4.wav',
    'f_5':   'phrase_5.wav',
    'testo': 'testo.wav',
}

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

def select_subject(batch_name):
    source_dir = DATA_ROOT / batch_name
    subjects = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir() and (d / "renders").exists()
    ])
    names = [s.name for s in subjects]
    idx = select_from_list(names, f"subjects in {batch_name}")
    return subjects[idx]

def load_timing(subject_folder, subject_id):
    """Load the Timing sheet from the subject's Excel file."""
    excel_path = subject_folder / f"{subject_id}_audio.xlsx"
    if not excel_path.exists():
        print(f"\n⚠️  No Excel timing file found: {excel_path}")
        return None
    df = pd.read_excel(excel_path, sheet_name="Timing")
    # Normalize the task label column
    label_col = df.columns[0]
    df[label_col] = df[label_col].astype(str).str.strip()
    return df

def select_task(timing_df):
    """Let user pick a task from the available timings."""
    if timing_df is None:
        # No timing file — fall back to manual entry
        task = input("\nEnter task name (e.g. a, r, testo): ").strip()
        audio_file = TASK_TO_FILE.get(task, f"{task}.wav")
        start = float(input("Enter start time (s): "))
        stop = float(input("Enter stop time (s): "))
        return task, audio_file, start, stop

    label_col = timing_df.columns[0]
    tasks = timing_df[label_col].tolist()
    idx = select_from_list(tasks, "tasks")

    row = timing_df.iloc[idx]
    task_label = str(row[label_col]).strip()
    start = float(row['start'])
    stop = float(row['stop'])

    audio_file = TASK_TO_FILE.get(task_label, f"{task_label}.wav")

    print(f"\n  Task: {task_label}")
    print(f"  Audio file: {audio_file}")
    print(f"  Phonation window: {start:.2f}s → {stop:.2f}s")

    return task_label, audio_file, start, stop

# ---- Run ----
batch_name = select_batch()
SUBJECT_FOLDER = select_subject(batch_name)

subject_id = SUBJECT_FOLDER.name.split('_')[1] if '_' in SUBJECT_FOLDER.name else SUBJECT_FOLDER.name

# Load timings
timing_df = load_timing(SUBJECT_FOLDER, subject_id)

# Select task
task_label, audio_file, start_sec, stop_sec = select_task(timing_df)

# OEP CSV path
OEP_CSV = f"csv/{subject_id}_Vocali.csv"
TAKE_NUMBER = 1

# ---- Config ----
config = create_config(
    data_root=DATA_ROOT / batch_name,
    output_root=DATA_TARGET / batch_name
)

# ---- Extract (trimmed to phonation window) ----
extractor = PairedFeatureExtractor(config)

paired = extractor.extract(
    subject_folder=SUBJECT_FOLDER,
    task_name=task_label,
    audio_filename=audio_file,
    oep_csv_path=OEP_CSV,
    take_number=TAKE_NUMBER,
    audio_start_sec=start_sec,
    audio_end_sec=stop_sec,
)

df = paired.dataframe
print(f"\n✅ Aligned Matrix: {df.shape[0]} frames × {df.shape[1]} features")
print(f"   Duration: {stop_sec - start_sec:.2f}s (trimmed to phonation)")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nPreview (first 5 rows):\n{df.head()}")

# Quick verification: Vrc + Vab ≈ Vcw ?
err = (df['vrc'] + df['vab'] - df['vcw']).abs().mean()
print(f"\n🔍 Average Error |Vrc + Vab - Vcw| = {err:.6f} L")
if err < 0.01:
    print("   → Compartmental Mapping OK ✓")
else:
    print("   → ⚠️ Check the column mapping")

# Save HDF5 into data_target/<batch>/paired/
output_dir = DATA_TARGET / batch_name / "paired"
h5_path = PairedFeatureExtractor.save_hdf5(
    paired,
    output_dir / f"{paired.subject_id}_{task_label}.h5"
)
print(f"\n💾 Saved: {h5_path}")