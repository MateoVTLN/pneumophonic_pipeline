#!/usr/bin/env python3
"""
Example: Batch analysis of multiple subjects
============================================

This script demonstrates how to use the pipeline to analyze
all subjects in a data directory.

"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add the package to the path if run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumophonic_analysis import (
    PneumophonicPipeline,
    create_config,
    run_pipeline
)


def analyze_batch_simple(data_root: Path, output_root: Path, tasks: list = None):
    """
    Simplified batch analysis using run_pipeline().

    Args:
        data_root: Directory containing subject folders
        output_root: Output directory
        tasks: List of tasks to analyze (None = all)
    """
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS - Simple mode")
    print(f"{'='*60}")
    print(f"Data root: {data_root}")
    print(f"Output: {output_root}")
    print(f"Tasks: {tasks or 'all'}")
    print(f"{'='*60}\n")

    # Run
    results = run_pipeline(
        data_root=data_root,
        output_root=output_root,
        tasks=tasks
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Subjects analyzed: {results.n_subjects}")
    print(f"Successful: {results.n_successful}")
    print(f"Failed: {results.n_failed}")
    
    if results.n_failed > 0:
        print("\nErrors:")
        errors = results.get_errors_summary()
        for _, row in errors[errors['type'] == 'error'].iterrows():
            print(f"  - {row['subject_id']}: {row['message']}")
    
    print(f"{'='*60}\n")


def analyze_batch_detailed(data_root: Path, output_root: Path, tasks: list = None):
    """
    Batch analysis with detailed control.

    Args:
        data_root: Directory containing subject folders
        output_root: Output directory
        tasks: List of tasks to analyze
    """
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS - Detailed mode")
    print(f"{'='*60}\n")

    # Configuration
    config = create_config(
        data_root=data_root,
        output_root=output_root
    )
    
    # Pipeline
    pipeline = PneumophonicPipeline(config)

    # Subject discovery
    subjects = pipeline.discover_subjects()
    print(f"Subjects found: {len(subjects)}")
    for s in subjects:
        print(f"  - {s.name}")

    # Confirmation
    print(f"\nRun analysis on {len(subjects)} subjects?")
    response = input("Continue? [y/N]: ").strip().lower()
    if response not in ['o', 'oui', 'y', 'yes']:
        print("Cancelled.")
        return

    # Batch analysis
    batch_result = pipeline.analyze_batch(
        subjects=subjects,
        tasks=tasks,
        progress=True,
        stop_on_error=False
    )
    
    # Export results
    pipeline.export_results(
        batch_result,
        output_root / "all_results.xlsx"
    )

    # Generate full report
    pipeline.generate_report(
        batch_result,
        output_root / "report"
    )

    # Descriptive statistics
    df = batch_result.to_dataframe()
    if not df.empty:
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS")
        print("="*60)

        # Numeric metrics
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        key_metrics = [c for c in numeric_cols if any(k in c for k in
                       ['mean_f0', 'hnr', 'dsi', 'jitter', 'shimmer'])]

        if key_metrics:
            print("\nKey metrics (all subjects):")
            print(df[key_metrics].describe().round(2).to_string())

        # By task
        if 'task' in df.columns and 'pitch_mean_f0' in df.columns:
            print("\nMean F0 by task:")
            task_stats = df.groupby('task')['pitch_mean_f0'].agg(['mean', 'std', 'count'])
            print(task_stats.round(1).to_string())

    # Final summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results: {output_root / 'all_results.xlsx'}")
    print(f"Report: {output_root / 'report'}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch pneumophonic analysis of multiple subjects"
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Directory containing subject folders"
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Output directory for results"
    )
    parser.add_argument(
        "-t", "--tasks",
        nargs="+",
        choices=['vowel', 'phrase', 'trill', 'glide'],
        help="Tasks to analyze (default: all)"
    )
    parser.add_argument(
        "-s", "--simple",
        action="store_true",
        help="Simple mode (no confirmation prompt)"
    )
    
    args = parser.parse_args()
    
    if not args.data_root.exists():
        print(f"❌ Error: Directory {args.data_root} does not exist")
        sys.exit(1)
    
    args.output_root.mkdir(parents=True, exist_ok=True)
    
    if args.simple:
        analyze_batch_simple(args.data_root, args.output_root, args.tasks)
    else:
        analyze_batch_detailed(args.data_root, args.output_root, args.tasks)


if __name__ == "__main__":
    main()
