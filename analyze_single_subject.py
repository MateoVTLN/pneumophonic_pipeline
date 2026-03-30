#!/usr/bin/env python3
"""
Exemple: Analyse d'un sujet individuel
======================================

Ce script démontre comment utiliser le pipeline pour analyser
les données d'un seul sujet.

Usage:
    python analyze_single_subject.py /path/to/subject_folder

Exemple:
    python analyze_single_subject.py /data/20260218_GaBa
"""

import sys
import argparse
from pathlib import Path

# Ajouter le package au path si exécuté directement
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
    Analyse complète d'un sujet.
    
    Args:
        subject_folder: Chemin vers le dossier du sujet
        output_folder: Dossier de sortie pour les résultats
    """
    print(f"\n{'='*60}")
    print(f"Analyse du sujet: {subject_folder.name}")
    print(f"{'='*60}\n")
    
    # Configuration
    config = create_config(
        data_root=subject_folder.parent,
        output_root=output_folder
    )
    
    # Initialiser les composants
    loader = DataLoader(subject_folder, config)
    processor = AudioProcessor(config)
    analyzer = PraatAnalyzer(config)
    viz = Visualizer(config)
    
    print(f"Sujet ID: {loader.subject_id}")
    print(f"Date: {loader.date}")
    
    # Lister les fichiers audio disponibles
    print("\nFichiers audio disponibles:")
    audio_files = loader.list_audio_files()
    for f in audio_files:
        print(f"  - {f.name}")
    
    # Analyser chaque fichier audio trouvé
    results = {}
    
    for audio_file in audio_files:
        task_name = audio_file.stem
        print(f"\n--- Analyse: {task_name} ---")
        
        try:
            # Charger l'audio
            audio, sr = loader.load_audio(audio_file.name)
            print(f"  Durée: {len(audio)/sr:.2f}s, SR: {sr}Hz")
            
            # Pré-traitement
            audio_clean = processor.reduce_noise(audio, sr)
            audio_pe = processor.apply_pre_emphasis(audio_clean)
            
            # Analyse acoustique
            result = analyzer.analyze_signal(audio_pe, sr)
            
            # Afficher les résultats
            print(f"\n  Pitch:")
            print(f"    F0 moyen: {result.pitch.mean_f0:.1f} ± {result.pitch.std_f0:.1f} Hz")
            print(f"    F0 range: {result.pitch.min_f0:.1f} - {result.pitch.max_f0:.1f} Hz")
            
            print(f"\n  Perturbation:")
            print(f"    Jitter: {result.perturbation.local_jitter*100:.3f}%")
            print(f"    Shimmer: {result.perturbation.local_shimmer*100:.2f}%")
            
            print(f"\n  Qualité vocale:")
            print(f"    HNR: {result.voice_quality.hnr:.1f} dB")
            print(f"    DSI: {result.voice_quality.dsi:.2f}")
            
            print(f"\n  Formants:")
            print(f"    F1: {result.formants.f1_mean:.0f} Hz")
            print(f"    F2: {result.formants.f2_mean:.0f} Hz")
            print(f"    F3: {result.formants.f3_mean:.0f} Hz")
            
            # Sauvegarder la figure
            fig = viz.plot_waveform(audio_pe, sr, title=f"{loader.subject_id} - {task_name}")
            output_path = output_folder / "figures" / loader.subject_id / f"{task_name}_waveform.png"
            viz.save_figure(fig, output_path)
            print(f"\n  Figure sauvegardée: {output_path}")
            
            # Stocker le résultat
            results[task_name] = result.to_dataframe()
            
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
    
    # Exporter tous les résultats en Excel
    if results:
        import pandas as pd
        
        df_all = pd.concat(results.values(), ignore_index=True)
        df_all.insert(0, 'subject_id', loader.subject_id)
        df_all.insert(1, 'task', list(results.keys()))
        
        excel_path = output_folder / f"{loader.subject_id}_results.xlsx"
        df_all.to_excel(excel_path, index=False)
        print(f"\n✅ Résultats exportés: {excel_path}")
    
    print(f"\n{'='*60}")
    print("Analyse terminée!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse pneumophonique d'un sujet individuel"
    )
    parser.add_argument(
        "subject_folder",
        type=Path,
        help="Chemin vers le dossier du sujet"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./output"),
        help="Dossier de sortie (défaut: ./output)"
    )
    
    args = parser.parse_args()
    
    if not args.subject_folder.exists():
        print(f"❌ Erreur: Le dossier {args.subject_folder} n'existe pas")
        sys.exit(1)
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    analyze_subject(args.subject_folder, args.output)


if __name__ == "__main__":
    main()
