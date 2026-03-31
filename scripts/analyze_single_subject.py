#!/usr/bin/env python3
"""
Example: Analysis of a single subject
======================================

This script demonstrates how to use the pipeline to analyze
data from a single subject.

Usage:
    python analyze_single_subject.py /path/to/subject_folder

Example:
    python analyze_single_subject.py /data/20260218_GaBa
"""

import sys
import argparse
from pathlib import Path

# Add the package to the path if run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumophonic_analysis import (
    PneumophonicPipeline,
    create_config,
    DataLoader,
    AudioProcessor,
    PraatAnalyzer,
    Visualizer
)


def analyze_subject(subject_folder: Path, output_folder: Path):
    """
    Full analysis of a single subject.

    Args:
        subject_folder: Path to the subject folder
        output_folder: Output folder for results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing subject: {subject_folder.name}")
    print(f"{'='*60}\n")

    # Configuration
    config = create_config(
        data_root=subject_folder.parent,
        output_root=output_folder
    )

    # Initialize components
    loader = DataLoader(subject_folder, config)
    processor = AudioProcessor(config)
    analyzer = PraatAnalyzer(config)
    viz = Visualizer(config)

    print(f"Subject ID: {loader.subject_id}")
    print(f"Date: {loader.date}")

    # List available audio files
    print("\nAvailable audio files:")
    audio_files = loader.list_audio_files()
    for f in audio_files:
        print(f"  - {f.name}")

    # Analyze each audio file found
    results = {}
    
    for audio_file in audio_files:
        task_name = audio_file.stem
        print(f"\n--- Analyse: {task_name} ---")
        
        try:
            # Load audio
            audio, sr = loader.load_audio(audio_file.name)
            print(f"  Duration: {len(audio)/sr:.2f}s, SR: {sr}Hz")

            # Preprocessing
            audio_clean = processor.reduce_noise(audio, sr)
            audio_pe = processor.apply_pre_emphasis(audio_clean)

            # Acoustic analysis
            result = analyzer.analyze_signal(audio_pe, sr)

            # Display results
            print(f"\n  Pitch:")
            print(f"    Mean F0: {result.pitch.mean_f0:.1f} ± {result.pitch.std_f0:.1f} Hz")
            print(f"    F0 range: {result.pitch.min_f0:.1f} - {result.pitch.max_f0:.1f} Hz")

            print(f"\n  Perturbation:")
            print(f"    Jitter: {result.perturbation.local_jitter*100:.3f}%")
            print(f"    Shimmer: {result.perturbation.local_shimmer*100:.2f}%")

            print(f"\n  Voice quality:")
            print(f"    HNR: {result.voice_quality.hnr:.1f} dB")
            print(f"    DSI: {result.voice_quality.dsi:.2f}")

            print(f"\n  Formants:")
            print(f"    F1: {result.formants.f1_mean:.0f} Hz")
            print(f"    F2: {result.formants.f2_mean:.0f} Hz")
            print(f"    F3: {result.formants.f3_mean:.0f} Hz")

            # Save figure
            fig = viz.plot_waveform(audio_pe, sr, title=f"{loader.subject_id} - {task_name}")
            output_path = output_folder / "figures" / loader.subject_id / f"{task_name}_waveform.png"
            viz.save_figure(fig, output_path)
            print(f"\n  Figure saved: {output_path}")

            # Store result
            results[task_name] = result.to_dataframe()

        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Export all results to Excel
    if results:
        import pandas as pd

        df_all = pd.concat(results.values(), ignore_index=True)
        df_all.insert(0, 'subject_id', loader.subject_id)
        df_all.insert(1, 'task', list(results.keys()))

        excel_path = output_folder / f"{loader.subject_id}_results.xlsx"
        df_all.to_excel(excel_path, index=False)
        print(f"\n✅ Results exported: {excel_path}")

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pneumophonic analysis of a single subject"
    )
    parser.add_argument(
        "subject_folder",
        type=Path,
        help="Path to the subject folder"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./output"),
        help="Output folder (default: ./output)"
    )
    
    args = parser.parse_args()
    
    if not args.subject_folder.exists():
        print(f"❌ Error: Folder {args.subject_folder} does not exist")
        sys.exit(1)
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    analyze_subject(args.subject_folder, args.output)


if __name__ == "__main__":
    main()
