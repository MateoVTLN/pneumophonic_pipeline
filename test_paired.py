"""
Quick test script for paired feature extraction.
Reads phonation timings from the subject's Excel file and automatically
selects the correct OEP CSV and take number for each task.
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

# ---- Mapping: task label → (audio filename, OEP csv suffix, take number) ----
# OEP csv will be resolved as: csv/{SubjectID}_{suffix}.csv
# Tasks sharing the same OEP session use different take numbers.
TASK_MAP = {
    # Sustained vowels — all in vocali.csv, one take per vowel
    'a':     ('a.wav',              'Vocali',       1),
    'e':     ('e.wav',              'Vocali',       2),
    'i':     ('i.wav',              'Vocali',       3),
    'o':     ('o.wav',              'Vocali',       4),
    'u':     ('u.wav',              'Vocali',       5),

    # Sustained phonation tasks — each has its own OEP CSV
    'a_2':   ('phonema_a_2.wav',    'phonema_a_2',  1),
    'a_3':   ('phonema_a_3.wav',    'phonema_a_3',  1),
    'a_7':   ('phonema_a_7_2.wav',  'phonema_a_7',  1),

    # Trill
    'r':     ('r.wav',              'r',            1),

    # Phrases — all in frasi.csv, one take per phrase
    'f_1':   ('phrase_1.wav',       'frasi',        1),
    'f_2':   ('phrase_2.wav',       'frasi',        2),
    'f_3':   ('phrase_3.wav',       'frasi',        3),
    'f_4':   ('phrase_4.wav',       'frasi',        4),
    'f_5':   ('phrase_5.wav',       'frasi',        5),

    # Text reading
    'testo': ('testo.wav',          'testo',        1),
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
    label_col = df.columns[0]
    df[label_col] = df[label_col].astype(str).str.strip()
    return df

def select_task(timing_df, subject_id, subject_folder):
    """Let user pick a task, resolve audio file, OEP CSV, take number, and falling edge."""
    if timing_df is None:
        task = input("\nEnter task name (e.g. a, r, testo): ").strip()
        if task not in TASK_MAP:
            print(f"⚠️  Unknown task '{task}'. Known tasks: {list(TASK_MAP.keys())}")
            exit(1)
        audio_file, csv_suffix, take_number = TASK_MAP[task]
        start = float(input("Enter start time (s): "))
        stop = float(input("Enter stop time (s): "))
        falling_edge = float(input("Enter falling edge (OEP time, s): "))
        oep_csv = f"csv/{subject_id}_{csv_suffix}.csv"
        return task, audio_file, oep_csv, take_number, start, stop, falling_edge

    label_col = timing_df.columns[0]
    tasks = timing_df[label_col].tolist()
    idx = select_from_list(tasks, "tasks")

    row = timing_df.iloc[idx]
    task_label = str(row[label_col]).strip()
    start = float(row['start'])
    stop = float(row['stop'])
    falling_edge = float(row['falling edge'])

    if task_label not in TASK_MAP:
        print(f"⚠️  Task '{task_label}' not in TASK_MAP. Add it manually.")
        exit(1)

    audio_file, csv_suffix, take_number = TASK_MAP[task_label]
    oep_csv = f"csv/{subject_id}_{csv_suffix}.csv"

    # Verify the OEP CSV exists
    oep_path = subject_folder / oep_csv
    if not oep_path.exists():
        print(f"\n⚠️  OEP file not found: {oep_csv}")
        print(f"   Available CSVs:")
        csv_dir = subject_folder / "csv"
        if csv_dir.exists():
            for f in sorted(csv_dir.glob("*.csv")):
                print(f"     {f.name}")
        exit(1)

    print(f"\n  Task:          {task_label}")
    print(f"  Audio file:    {audio_file}")
    print(f"  OEP CSV:       {oep_csv}")
    print(f"  Falling edge:  {falling_edge:.2f}s (OEP time)")
    print(f"  Phonation:     {start:.2f}s → {stop:.2f}s")

    return task_label, audio_file, oep_csv, take_number, start, stop, falling_edge

# ---- Run ----
batch_name = select_batch()
SUBJECT_FOLDER = select_subject(batch_name)

subject_id = SUBJECT_FOLDER.name.split('_')[1] if '_' in SUBJECT_FOLDER.name else SUBJECT_FOLDER.name

# Load timings
timing_df = load_timing(SUBJECT_FOLDER, subject_id)

# Select task (now returns OEP CSV, take number, and falling edge)
task_label, audio_file, oep_csv, take_number, start_sec, stop_sec, falling_edge = select_task(
    timing_df, subject_id, SUBJECT_FOLDER
)

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
    oep_csv_path=oep_csv,
    take_number=take_number,
    audio_start_sec=start_sec,
    audio_end_sec=stop_sec,
    oep_falling_edge_sec=falling_edge,
)

df = paired.dataframe
print(f"\n✅ Aligned Matrix: {df.shape[0]} frames × {df.shape[1]} features")
print(f"   Duration: {stop_sec - start_sec:.2f}s (trimmed to phonation)")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nPreview (first 5 rows):\n{df.head()}")

# Quick Verif: Vrc + Vab ≈ Vcw ?
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