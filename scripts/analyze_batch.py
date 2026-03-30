#!/usr/bin/env python3
"""
Exemple: Analyse batch de plusieurs sujets
==========================================

Ce script démontre comment utiliser le pipeline pour analyser
tous les sujets d'un répertoire de données.

Usage:
    python analyze_batch.py /path/to/data_root /path/to/output

Exemple:
    python analyze_batch.py /data/subjects /results
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Ajouter le package au path si exécuté directement
sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumophonic_analysis import (
    PneumophonicPipeline,
    create_config,
    run_pipeline
)


def analyze_batch_simple(data_root: Path, output_root: Path, tasks: list = None):
    """
    Analyse batch simplifiée utilisant run_pipeline().
    
    Args:
        data_root: Répertoire contenant les dossiers sujets
        output_root: Répertoire de sortie
        tasks: Liste des tâches à analyser (None = toutes)
    """
    print(f"\n{'='*60}")
    print("ANALYSE BATCH - Mode simplifié")
    print(f"{'='*60}")
    print(f"Data root: {data_root}")
    print(f"Output: {output_root}")
    print(f"Tâches: {tasks or 'toutes'}")
    print(f"{'='*60}\n")
    
    # Exécution
    results = run_pipeline(
        data_root=data_root,
        output_root=output_root,
        tasks=tasks
    )
    
    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    print(f"Sujets analysés: {results.n_subjects}")
    print(f"Succès: {results.n_successful}")
    print(f"Échecs: {results.n_failed}")
    
    if results.n_failed > 0:
        print("\nErreurs:")
        errors = results.get_errors_summary()
        for _, row in errors[errors['type'] == 'error'].iterrows():
            print(f"  - {row['subject_id']}: {row['message']}")
    
    print(f"{'='*60}\n")


def analyze_batch_detailed(data_root: Path, output_root: Path, tasks: list = None):
    """
    Analyse batch avec contrôle détaillé.
    
    Args:
        data_root: Répertoire contenant les dossiers sujets
        output_root: Répertoire de sortie
        tasks: Liste des tâches à analyser
    """
    print(f"\n{'='*60}")
    print("ANALYSE BATCH - Mode détaillé")
    print(f"{'='*60}\n")
    
    # Configuration
    config = create_config(
        data_root=data_root,
        output_root=output_root
    )
    
    # Pipeline
    pipeline = PneumophonicPipeline(config)
    
    # Découverte des sujets
    subjects = pipeline.discover_subjects()
    print(f"Sujets découverts: {len(subjects)}")
    for s in subjects:
        print(f"  - {s.name}")
    
    # Confirmation
    print(f"\nLancer l'analyse de {len(subjects)} sujets?")
    response = input("Continuer? [o/N]: ").strip().lower()
    if response not in ['o', 'oui', 'y', 'yes']:
        print("Annulé.")
        return
    
    # Analyse batch
    batch_result = pipeline.analyze_batch(
        subjects=subjects,
        tasks=tasks,
        progress=True,
        stop_on_error=False
    )
    
    # Export des résultats
    pipeline.export_results(
        batch_result,
        output_root / "all_results.xlsx"
    )
    
    # Générer le rapport complet
    pipeline.generate_report(
        batch_result,
        output_root / "report"
    )
    
    # Statistiques descriptives
    df = batch_result.to_dataframe()
    if not df.empty:
        print("\n" + "="*60)
        print("STATISTIQUES DESCRIPTIVES")
        print("="*60)
        
        # Métriques numériques
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        key_metrics = [c for c in numeric_cols if any(k in c for k in 
                       ['mean_f0', 'hnr', 'dsi', 'jitter', 'shimmer'])]
        
        if key_metrics:
            print("\nMétriques clés (tous sujets):")
            print(df[key_metrics].describe().round(2).to_string())
        
        # Par tâche
        if 'task' in df.columns and 'pitch_mean_f0' in df.columns:
            print("\nF0 moyen par tâche:")
            task_stats = df.groupby('task')['pitch_mean_f0'].agg(['mean', 'std', 'count'])
            print(task_stats.round(1).to_string())
    
    # Résumé final
    print(f"\n{'='*60}")
    print("ANALYSE TERMINÉE")
    print(f"{'='*60}")
    print(f"Résultats: {output_root / 'all_results.xlsx'}")
    print(f"Rapport: {output_root / 'report'}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse pneumophonique batch de plusieurs sujets"
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Répertoire contenant les dossiers sujets"
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Répertoire de sortie pour les résultats"
    )
    parser.add_argument(
        "-t", "--tasks",
        nargs="+",
        choices=['vowel', 'phrase', 'trill', 'glide'],
        help="Tâches à analyser (défaut: toutes)"
    )
    parser.add_argument(
        "-s", "--simple",
        action="store_true",
        help="Mode simplifié (pas de confirmation)"
    )
    
    args = parser.parse_args()
    
    if not args.data_root.exists():
        print(f"❌ Erreur: Le répertoire {args.data_root} n'existe pas")
        sys.exit(1)
    
    args.output_root.mkdir(parents=True, exist_ok=True)
    
    if args.simple:
        analyze_batch_simple(args.data_root, args.output_root, args.tasks)
    else:
        analyze_batch_detailed(args.data_root, args.output_root, args.tasks)


if __name__ == "__main__":
    main()
