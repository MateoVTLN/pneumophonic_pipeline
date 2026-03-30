"""
Module de visualisation pour le pipeline pneumophonique.

Graphiques:
- Formes d'onde avec sync
- Spectrogrammes et mel-spectrogrammes
- Traces de F0
- Volumes OEP avec zones FRC
- Résultats comparatifs

Référence: Thèse Zocco 2025, Chapitre 4 - Figures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import librosa
import librosa.display
from pathlib import Path
from typing import Optional, List, Tuple, Union, Dict

from .config import PipelineConfig, get_config
from .task_analyzers import TaskResult
from .segmentation import FRCSegment, GlideSegment


# Style par défaut
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'audio': '#2E86AB',
    'sync': '#A23B72',
    'volume': '#F18F01',
    'above_frc': '#C73E1D',
    'below_frc': '#3B1F2B',
    'f0': '#21A179',
    'formants': ['#E63946', '#457B9D', '#1D3557'],
    'p1': '#264653',
    'p2': '#E76F51',
}


class Visualizer:
    """
    Classe principale de visualisation.
    
    Exemple:
    ```python
    viz = Visualizer(config)
    
    # Plot simple
    fig = viz.plot_waveform(audio, sr)
    
    # Plot avec sync
    fig = viz.plot_audio_with_sync(audio, sync, sr)
    
    # Plot complet d'un résultat
    fig = viz.plot_task_result(result)
    
    # Sauvegarder
    viz.save_figure(fig, "output.png")
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_config()
        self.dpi = self.config.output.figure_dpi
        self.format = self.config.output.figure_format
    
    def plot_waveform(
        self,
        audio: np.ndarray,
        sr: int,
        title: str = "Waveform",
        ax: Optional[plt.Axes] = None,
        color: str = None,
        alpha: float = 0.8
    ) -> plt.Figure:
        """
        Affiche la forme d'onde.
        
        Args:
            audio: Signal audio
            sr: Sample rate
            title: Titre du graphique
            ax: Axes matplotlib (créé si None)
            color: Couleur de la courbe
            alpha: Transparence
            
        Returns:
            Figure matplotlib
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        else:
            fig = ax.figure
        
        color = color or COLORS['audio']
        time = np.arange(len(audio)) / sr
        
        ax.plot(time, audio, color=color, alpha=alpha, linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.set_xlim([0, time[-1]])
        
        return fig
    
    def plot_audio_with_sync(
        self,
        audio: np.ndarray,
        sync: np.ndarray,
        sr_audio: int,
        sr_sync: Optional[int] = None,
        falling_edge_sample: Optional[int] = None,
        title: str = "Audio with Sync Signal"
    ) -> plt.Figure:
        """
        Affiche l'audio et le signal de sync.
        
        Args:
            audio: Signal audio
            sync: Signal de synchronisation
            sr_audio: Sample rate audio
            sr_sync: Sample rate sync (= sr_audio si None)
            falling_edge_sample: Position du falling edge
            title: Titre
            
        Returns:
            Figure matplotlib
        """
        sr_sync = sr_sync or sr_audio
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        
        # Audio
        time_audio = np.arange(len(audio)) / sr_audio
        axes[0].plot(time_audio, audio, color=COLORS['audio'], alpha=0.8, linewidth=0.5)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Audio')
        
        # Sync
        time_sync = np.arange(len(sync)) / sr_sync
        axes[1].plot(time_sync, sync, color=COLORS['sync'], alpha=0.8, linewidth=0.5)
        axes[1].set_ylabel('Sync')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title('Sync Signal')
        
        # Marquer le falling edge
        if falling_edge_sample is not None:
            t_edge = falling_edge_sample / sr_audio
            for ax in axes:
                ax.axvline(t_edge, color='red', linestyle='--', 
                          label=f'Falling edge ({t_edge:.3f}s)')
            axes[0].legend()
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_spectrogram(
        self,
        audio: np.ndarray,
        sr: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        title: str = "Spectrogram",
        ax: Optional[plt.Axes] = None,
        y_axis: str = 'log',
        cmap: str = 'magma'
    ) -> plt.Figure:
        """
        Affiche le spectrogramme.
        
        Args:
            audio: Signal audio
            sr: Sample rate
            n_fft: Taille FFT
            hop_length: Hop length
            title: Titre
            ax: Axes matplotlib
            y_axis: Échelle de l'axe y ('log', 'linear', 'mel')
            cmap: Colormap
            
        Returns:
            Figure matplotlib
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.figure
        
        S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        
        img = librosa.display.specshow(
            S_db, sr=sr, hop_length=hop_length,
            x_axis='time', y_axis=y_axis, ax=ax, cmap=cmap
        )
        
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        return fig
    
    def plot_mel_spectrogram(
        self,
        audio: np.ndarray,
        sr: int,
        n_mels: int = 64,
        title: str = "Mel Spectrogram",
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Affiche le mel-spectrogramme."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.figure
        
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma'
        )
        
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        return fig
    
    def plot_f0_trace(
        self,
        f0: np.ndarray,
        hop_length: int,
        sr: int,
        title: str = "F0 Trace",
        ax: Optional[plt.Axes] = None,
        show_stats: bool = True
    ) -> plt.Figure:
        """
        Affiche la trace de F0.
        
        Args:
            f0: Valeurs de F0 (peut contenir NaN)
            hop_length: Hop length utilisé
            sr: Sample rate
            title: Titre
            ax: Axes matplotlib
            show_stats: Afficher les statistiques
            
        Returns:
            Figure matplotlib
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 3))
        else:
            fig = ax.figure
        
        time = np.arange(len(f0)) * hop_length / sr
        
        ax.plot(time, f0, color=COLORS['f0'], linewidth=1.5, marker='o', 
                markersize=2, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('F0 (Hz)')
        ax.set_title(title)
        
        if show_stats:
            mean_f0 = np.nanmean(f0)
            std_f0 = np.nanstd(f0)
            ax.axhline(mean_f0, color='gray', linestyle='--', alpha=0.5,
                      label=f'Mean: {mean_f0:.1f} Hz')
            ax.fill_between(
                time, mean_f0 - std_f0, mean_f0 + std_f0,
                alpha=0.2, color='gray', label=f'±1 STD: {std_f0:.1f} Hz'
            )
            ax.legend(loc='upper right')
        
        return fig
    
    def plot_oep_volume(
        self,
        volume: np.ndarray,
        fs_oep: int,
        frc_level: Optional[float] = None,
        frc_cross_sample: Optional[int] = None,
        title: str = "Chest Wall Volume (Vcw)",
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """
        Affiche le volume de la paroi thoracique avec zones FRC.
        
        Args:
            volume: Volume (Vcw) en litres
            fs_oep: Sample rate OEP
            frc_level: Niveau de la FRC
            frc_cross_sample: Sample du crossing FRC
            title: Titre
            ax: Axes matplotlib
            
        Returns:
            Figure matplotlib
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.figure
        
        time = np.arange(len(volume)) / fs_oep
        
        ax.plot(time, volume, color=COLORS['volume'], linewidth=1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Volume (L)')
        ax.set_title(title)
        
        if frc_level is not None:
            ax.axhline(frc_level, color='black', linestyle='--', 
                      linewidth=2, label='FRC')
            
            # Colorer les zones
            ax.fill_between(
                time, volume, frc_level,
                where=(volume >= frc_level),
                color=COLORS['above_frc'], alpha=0.3, label='Above FRC'
            )
            ax.fill_between(
                time, volume, frc_level,
                where=(volume < frc_level),
                color=COLORS['below_frc'], alpha=0.3, label='Below FRC'
            )
            ax.legend()
        
        if frc_cross_sample is not None:
            t_cross = frc_cross_sample / fs_oep
            ax.axvline(t_cross, color='red', linestyle=':', 
                      linewidth=2, label=f'FRC crossing ({t_cross:.2f}s)')
        
        return fig
    
    def plot_frc_segment(
        self,
        segment: FRCSegment,
        title: str = "FRC Segmentation"
    ) -> plt.Figure:
        """
        Affiche un segment avec sa segmentation FRC.
        
        Args:
            segment: FRCSegment
            title: Titre
            
        Returns:
            Figure matplotlib
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        sr = segment.sample_rate
        
        # Audio complet avec zones
        full_audio = np.concatenate([segment.above_frc, segment.below_frc])
        time = np.arange(len(full_audio)) / sr
        
        axes[0].plot(time, full_audio, color=COLORS['audio'], alpha=0.7, linewidth=0.5)
        
        # Zones colorées
        t_cross = len(segment.above_frc) / sr
        axes[0].axvspan(0, t_cross, alpha=0.2, color=COLORS['above_frc'], 
                       label=f'Above FRC ({segment.duration_above:.2f}s)')
        axes[0].axvspan(t_cross, time[-1], alpha=0.2, color=COLORS['below_frc'],
                       label=f'Below FRC ({segment.duration_below:.2f}s)')
        axes[0].axvline(t_cross, color='red', linestyle='--', linewidth=2)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Full Segment with FRC Zones')
        axes[0].legend(loc='upper right')
        
        # Segments séparés
        time_above = np.arange(len(segment.above_frc)) / sr
        time_below = np.arange(len(segment.below_frc)) / sr + t_cross
        
        axes[1].plot(time_above, segment.above_frc, color=COLORS['above_frc'],
                    alpha=0.8, linewidth=0.5, label='Above FRC')
        axes[1].plot(time_below, segment.below_frc, color=COLORS['below_frc'],
                    alpha=0.8, linewidth=0.5, label='Below FRC')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_title('Separated Segments')
        axes[1].legend()
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_glide_analysis(
        self,
        audio: np.ndarray,
        sr: int,
        segment: GlideSegment,
        novelty: Optional[np.ndarray] = None,
        novelty_time: Optional[np.ndarray] = None,
        title: str = "Glide Analysis"
    ) -> plt.Figure:
        """
        Affiche l'analyse du glissando.
        
        Args:
            audio: Signal audio complet
            sr: Sample rate
            segment: GlideSegment
            novelty: Novelty function (optionnel)
            novelty_time: Axe temporel de la novelty
            title: Titre
            
        Returns:
            Figure matplotlib
        """
        n_rows = 3 if novelty is not None else 2
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows))
        
        # Waveform avec zones P1/P2
        time = np.arange(len(audio)) / sr
        axes[0].plot(time, audio, color=COLORS['audio'], alpha=0.7, linewidth=0.5)
        
        t_start = segment.start_sample / sr
        t_peak = segment.peak_sample / sr
        t_end = segment.end_sample / sr
        
        axes[0].axvspan(t_start, t_peak, alpha=0.3, color=COLORS['p1'],
                       label=f'P1 ({segment.duration_p1:.2f}s)')
        axes[0].axvspan(t_peak, t_end, alpha=0.3, color=COLORS['p2'],
                       label=f'P2 ({segment.duration_p2:.2f}s)')
        axes[0].axvline(t_peak, color='red', linestyle='--', linewidth=2,
                       label=f'Transition ({t_peak:.2f}s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Waveform with P1/P2 Segmentation')
        axes[0].legend()
        
        # Spectrogramme
        S = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        librosa.display.specshow(
            S_db, sr=sr, x_axis='time', y_axis='log', ax=axes[1], cmap='magma'
        )
        axes[1].axvline(t_peak, color='white', linestyle='--', linewidth=2)
        axes[1].set_title('Spectrogram')
        
        # Novelty function
        if novelty is not None and novelty_time is not None:
            axes[2].plot(novelty_time, novelty, color=COLORS['f0'], linewidth=1.5)
            axes[2].axvline(segment.peak_time, color='red', linestyle='--',
                          linewidth=2, label='Peak')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Novelty')
            axes[2].set_title('Novelty Function')
            axes[2].legend()
        
        fig.suptitle(title)
        plt.tight_layout()
        
        return fig
    
    def plot_task_result(
        self,
        result: TaskResult,
        include_spectrogram: bool = True,
        include_f0: bool = True
    ) -> plt.Figure:
        """
        Affiche un résultat de tâche complet.
        
        Args:
            result: TaskResult
            include_spectrogram: Inclure le spectrogramme
            include_f0: Inclure la trace de F0
            
        Returns:
            Figure matplotlib
        """
        n_rows = 1 + int(include_spectrogram) + int(include_f0)
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows))
        
        if n_rows == 1:
            axes = [axes]
        
        row = 0
        sr = result.sample_rate
        
        # Waveform
        if result.audio_processed is not None:
            self.plot_waveform(
                result.audio_processed, sr,
                title=f'{result.subject_id} - {result.task_name}',
                ax=axes[row]
            )
            row += 1
        
        # Spectrogram
        if include_spectrogram and result.audio_processed is not None:
            self.plot_spectrogram(
                result.audio_processed, sr,
                title='Spectrogram',
                ax=axes[row]
            )
            row += 1
        
        # F0
        if include_f0 and result.audio_features is not None:
            self.plot_f0_trace(
                result.audio_features.f0,
                result.audio_features.hop_length,
                sr,
                title='F0 Trace',
                ax=axes[row]
            )
        
        plt.tight_layout()
        
        return fig
    
    def plot_metrics_comparison(
        self,
        df: pd.DataFrame,
        metric: str,
        group_by: str = 'subject_id',
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare une métrique entre groupes.
        
        Args:
            df: DataFrame avec les métriques
            metric: Nom de la colonne métrique
            group_by: Colonne pour le groupement
            title: Titre du graphique
            
        Returns:
            Figure matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        groups = df.groupby(group_by)[metric].mean().sort_values()
        
        bars = ax.barh(range(len(groups)), groups.values, color=COLORS['audio'], alpha=0.7)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups.index)
        ax.set_xlabel(metric)
        ax.set_title(title or f'{metric} by {group_by}')
        
        # Ajouter les valeurs
        for i, (idx, val) in enumerate(groups.items()):
            ax.text(val + 0.01 * groups.max(), i, f'{val:.2f}', va='center')
        
        plt.tight_layout()
        
        return fig
    
    def save_figure(
        self,
        fig: plt.Figure,
        path: Union[str, Path],
        dpi: Optional[int] = None,
        format: Optional[str] = None
    ):
        """
        Sauvegarde une figure.
        
        Args:
            fig: Figure matplotlib
            path: Chemin de sortie
            dpi: Résolution (défaut: config)
            format: Format d'image (défaut: config)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        dpi = dpi or self.dpi
        format = format or self.format
        
        fig.savefig(str(path), dpi=dpi, format=format, bbox_inches='tight')
        plt.close(fig)


def quick_plot_audio(audio: np.ndarray, sr: int, title: str = "Audio"):
    """Fonction utilitaire pour un plot rapide."""
    viz = Visualizer()
    fig = viz.plot_waveform(audio, sr, title)
    plt.show()
    return fig


def quick_plot_spectrogram(audio: np.ndarray, sr: int, title: str = "Spectrogram"):
    """Fonction utilitaire pour un spectrogramme rapide."""
    viz = Visualizer()
    fig = viz.plot_spectrogram(audio, sr, title)
    plt.show()
    return fig
