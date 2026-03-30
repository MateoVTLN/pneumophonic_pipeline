#!/usr/bin/env python3
"""
Exemple: Analyse de la roulée R avec fréquence de modulation
============================================================

Ce script démontre l'analyse spécifique de la roulée alvéolaire (trille R)
incluant l'extraction de la fréquence de modulation et la segmentation FRC.

La fréquence de modulation typique de la roulée est de 20-30 Hz,
correspondant à la vibration de la pointe de la langue.

Usage:
    python analyze_trill_modulation.py /path/to/audio.wav

Ou avec fichier de timings:
    python analyze_trill_modulation.py /path/to/audio.wav --frc-time 3.5
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa

# Ajouter le package au path si exécuté directement
sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumophonic_analysis import (
    AudioProcessor,
    PraatAnalyzer,
    TrillAnalyzer,
    ModulationAnalyzer,
    FRCSegmenter,
    Visualizer,
    create_config
)


def analyze_trill(
    audio_path: Path,
    frc_cross_time: float = None,
    output_folder: Path = None,
    show_plots: bool = True
):
    """
    Analyse complète de la roulée R.
    
    Args:
        audio_path: Chemin vers le fichier audio
        frc_cross_time: Temps du crossing FRC (secondes)
        output_folder: Dossier de sortie pour les figures
        show_plots: Afficher les graphiques
    """
    print(f"\n{'='*60}")
    print("ANALYSE DE LA ROULÉE R")
    print(f"{'='*60}")
    print(f"Fichier: {audio_path}")
    print(f"FRC crossing: {frc_cross_time or 'non spécifié'}")
    print(f"{'='*60}\n")
    
    # Charger l'audio
    audio, sr = librosa.load(str(audio_path), sr=48000)
    duration = len(audio) / sr
    print(f"Audio chargé: {duration:.2f}s @ {sr}Hz")
    
    # Initialiser les analyseurs
    config = create_config()
    processor = AudioProcessor(config)
    praat = PraatAnalyzer(config)
    modulation = ModulationAnalyzer(config)
    viz = Visualizer(config)
    
    # Pré-traitement
    print("\n1. Pré-traitement...")
    audio_clean = processor.reduce_noise(audio, sr)
    audio_pe = processor.apply_pre_emphasis(audio_clean)
    
    # Détection des bornes de phonation
    import librosa
    intervals = librosa.effects.split(audio_pe, top_db=45)
    if len(intervals) > 0:
        start_sample = intervals[0, 0]
        end_sample = intervals[-1, 1]
    else:
        start_sample, end_sample = 0, len(audio_pe)
    
    start_time = start_sample / sr
    end_time = end_sample / sr
    audio_segment = audio_pe[start_sample:end_sample]
    
    print(f"   Phonation: {start_time:.2f}s - {end_time:.2f}s ({end_time-start_time:.2f}s)")
    
    # Analyse acoustique globale
    print("\n2. Analyse acoustique...")
    acoustic = praat.analyze_signal(audio_segment, sr)
    
    print(f"   F0: {acoustic.pitch.mean_f0:.1f} ± {acoustic.pitch.std_f0:.1f} Hz")
    print(f"   HNR: {acoustic.voice_quality.hnr:.1f} dB")
    print(f"   Jitter: {acoustic.perturbation.local_jitter*100:.3f}%")
    print(f"   Shimmer: {acoustic.perturbation.local_shimmer*100:.2f}%")
    
    # Analyse de la fréquence de modulation
    print("\n3. Analyse de la fréquence de modulation...")
    
    if frc_cross_time is not None:
        # Segmentation FRC
        frc_segmenter = FRCSegmenter(config)
        frc_segment = frc_segmenter.segment_by_time(
            audio_pe, 
            cross_time=frc_cross_time,
            start_time=start_time,
            end_time=end_time,
            sr=sr
        )
        
        # Modulation par segment
        mod_full = modulation.compute_modulation_frequency(audio_segment, sr)
        mod_above = modulation.compute_modulation_frequency(frc_segment.above_frc, sr)
        mod_below = modulation.compute_modulation_frequency(frc_segment.below_frc, sr)
        
        print(f"\n   Fréquence de modulation:")
        print(f"     Full:      {mod_full:.1f} Hz")
        print(f"     Above FRC: {mod_above:.1f} Hz ({frc_segment.duration_above:.2f}s)")
        print(f"     Below FRC: {mod_below:.1f} Hz ({frc_segment.duration_below:.2f}s)")
        
        # Analyse acoustique par segment
        print("\n   Métriques par segment:")
        if len(frc_segment.above_frc) > sr * 0.1:
            ac_above = praat.analyze_signal(frc_segment.above_frc, sr)
            print(f"     Above FRC - F0: {ac_above.pitch.mean_f0:.1f} Hz, HNR: {ac_above.voice_quality.hnr:.1f} dB")
        
        if len(frc_segment.below_frc) > sr * 0.1:
            ac_below = praat.analyze_signal(frc_segment.below_frc, sr)
            print(f"     Below FRC - F0: {ac_below.pitch.mean_f0:.1f} Hz, HNR: {ac_below.voice_quality.hnr:.1f} dB")
        
    else:
        # Sans segmentation FRC
        mod_full = modulation.compute_modulation_frequency(audio_segment, sr)
        print(f"   Fréquence de modulation: {mod_full:.1f} Hz")
        frc_segment = None
    
    # Comptage d'onsets (proxy pour les cycles)
    onsets = librosa.onset.onset_detect(y=audio_segment, sr=sr)
    print(f"\n   Nombre d'onsets détectés: {len(onsets)}")
    print(f"   Fréquence estimée par onsets: {len(onsets) / (len(audio_segment)/sr):.1f} Hz")
    
    # Visualisation
    print("\n4. Génération des figures...")
    
    # Figure 1: Waveform avec segmentation
    fig1, axes1 = plt.subplots(3, 1, figsize=(14, 10))
    
    # Waveform
    time = np.arange(len(audio_segment)) / sr + start_time
    axes1[0].plot(time, audio_segment, color='#2E86AB', alpha=0.7, linewidth=0.5)
    axes1[0].set_ylabel('Amplitude')
    axes1[0].set_title('Roulée R - Waveform')
    
    if frc_cross_time is not None:
        axes1[0].axvline(frc_cross_time, color='red', linestyle='--', 
                        linewidth=2, label='FRC crossing')
        axes1[0].axvspan(start_time, frc_cross_time, alpha=0.2, color='green',
                        label='Above FRC')
        axes1[0].axvspan(frc_cross_time, end_time, alpha=0.2, color='orange',
                        label='Below FRC')
        axes1[0].legend()
    
    # Spectrogram
    S = librosa.stft(audio_segment)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', 
                            ax=axes1[1], cmap='magma')
    axes1[1].set_title('Spectrogramme')
    
    # Enveloppe RMS et modulation
    rms = librosa.feature.rms(y=audio_segment, frame_length=512, hop_length=128)[0]
    rms_time = np.arange(len(rms)) * 128 / sr + start_time
    axes1[2].plot(rms_time, rms, color='#F18F01', linewidth=1)
    axes1[2].set_xlabel('Time (s)')
    axes1[2].set_ylabel('RMS')
    axes1[2].set_title(f'Enveloppe RMS (Fréq. modulation: {mod_full:.1f} Hz)')
    
    plt.tight_layout()
    
    # Figure 2: Spectre de modulation
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    # FFT de l'enveloppe RMS
    from scipy import signal
    
    # Detrending
    window_len = min(len(rms) // 2 * 2 - 1, 51)
    if window_len < 5:
        window_len = 5
    if window_len % 2 == 0:
        window_len -= 1
    
    if window_len > 5:
        trend = signal.savgol_filter(rms, window_len, 3)
        rms_detrend = rms - trend
    else:
        rms_detrend = rms - np.mean(rms)
    
    # FFT
    n_fft = len(rms_detrend)
    spectrum = np.abs(np.fft.rfft(rms_detrend))
    freqs = np.fft.rfftfreq(n_fft, d=128/sr)
    
    # Plot
    mask = freqs <= 50  # Limiter à 50 Hz
    ax2.plot(freqs[mask], spectrum[mask], color='#21A179', linewidth=1.5)
    ax2.axvline(mod_full, color='red', linestyle='--', linewidth=2,
               label=f'Pic: {mod_full:.1f} Hz')
    ax2.fill_between([10, 35], 0, spectrum.max(), alpha=0.1, color='blue',
                    label='Bande typique (10-35 Hz)')
    ax2.set_xlabel('Fréquence (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Spectre de modulation')
    ax2.legend()
    
    plt.tight_layout()
    
    # Sauvegarder ou afficher
    if output_folder:
        output_folder.mkdir(parents=True, exist_ok=True)
        fig1.savefig(output_folder / "trill_waveform.png", dpi=150, bbox_inches='tight')
        fig2.savefig(output_folder / "trill_modulation_spectrum.png", dpi=150, bbox_inches='tight')
        print(f"   Figures sauvegardées dans: {output_folder}")
    
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    print(f"Durée de phonation: {end_time - start_time:.2f} s")
    print(f"F0 moyen: {acoustic.pitch.mean_f0:.1f} Hz")
    print(f"HNR: {acoustic.voice_quality.hnr:.1f} dB")
    print(f"Fréquence de modulation: {mod_full:.1f} Hz")
    print(f"{'='*60}\n")
    
    return {
        'duration': end_time - start_time,
        'mean_f0': acoustic.pitch.mean_f0,
        'hnr': acoustic.voice_quality.hnr,
        'modulation_frequency': mod_full,
        'onset_count': len(onsets)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyse de la roulée R avec fréquence de modulation"
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Chemin vers le fichier audio de la roulée"
    )
    parser.add_argument(
        "--frc-time",
        type=float,
        default=None,
        help="Temps du crossing FRC en secondes"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Dossier de sortie pour les figures"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Ne pas afficher les graphiques"
    )
    
    args = parser.parse_args()
    
    if not args.audio_path.exists():
        print(f"❌ Erreur: Le fichier {args.audio_path} n'existe pas")
        sys.exit(1)
    
    analyze_trill(
        audio_path=args.audio_path,
        frc_cross_time=args.frc_time,
        output_folder=args.output,
        show_plots=not args.no_show
    )


if __name__ == "__main__":
    main()
