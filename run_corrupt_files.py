from pathlib import Path

_repo_root = Path(__file__).parent
data_roots = [
    _repo_root / "data_root" / "healthy_subjects",
    _repo_root / "data_root" / "pathological_subjects",
]

for root in data_roots:
    if not root.exists():
        continue
    print(f"\n{'='*50}")
    print(f"📁 {root.name}")
    print('='*50)
    
    for subject in sorted(root.glob("*_*")):
        if not subject.is_dir():
            continue
        renders = subject / "renders"
        if renders.exists():
            wav_files = list(renders.glob("*.wav"))
            valid = [f for f in wav_files if f.stat().st_size > 0]
            empty = len(wav_files) - len(valid)
            status = "| ✓ |" if valid else "| ✗ |"
            print(f"{status} {subject.name}: {len(valid)} valid, {empty} empty")
        else:
            print(f"| ✗ | {subject.name}: no renders/")