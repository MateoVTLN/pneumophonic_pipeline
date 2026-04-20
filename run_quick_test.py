from pneumophonic_analysis import run_pipeline
import logging
logging.basicConfig(level=logging.DEBUG)
from pathlib import Path

_repo_root = Path(__file__).parent

# Batch selection
print("\n Which subjects to analyse?")
print("  1. healthy subjects")
print("  2. pathological subjects")
print("  3. singers")
print("  4. non-singers")

while True:
    choice = input("Enter 1, 2, 3, or 4: ").strip()
    if choice == "1":
        batch = "healthy_subjects"
        break
    elif choice == "2":
        batch = "pathological_subjects"
        break
    elif choice == "3":
        batch = "singer_subjects"
        break
    elif choice == "4":
        batch = "notsinger_subjects"
        break
    print("Please enter 1, 2, 3, or 4.")

data_roots = _repo_root / "data_root" / batch
data_target = _repo_root / "data_target" / batch

# Subject exclusion (if dataset too noisy or simply to exclude)
available = sorted([d.name for d in data_roots.glob("*_*") if d.is_dir()])
subjects_to_run = None

if available:
    print(f"\nSubjects found in {batch}:")
    for s in available:
        print(f"  - {s}")
    print("\nEnter subject names to EXCLUDE (comma-separated), or press Enter to include all:")
    raw = input("> ").strip()
    if raw:
        excluded = {s.strip() for s in raw.split(",")}
        unknown = excluded - set(available)
        if unknown:
            print(f"Warning: these subjects were not found and will be ignored: {unknown}")
        subjects_to_run = [s for s in available if s not in excluded]
        print(f"\nRunning with {len(subjects_to_run)} subject(s): {subjects_to_run}")
    else:
        print(f"\nRunning with all {len(available)} subject(s).")
else:
    print(f"\nNo subjects found in {data_roots}. Proceeding anyway.")

results = run_pipeline(
    data_root=data_roots,
    output_root=data_target,
    subjects=subjects_to_run,
    tasks=['vowel', 'trill', 'glide']
)
print(f"Analyzed {results.n_subjects} subjects")
