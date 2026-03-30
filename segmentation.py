"""
Module de segmentation des signaux.

Implémente différentes méthodes de segmentation:
- Crossing FRC (Above/Below Functional Residual Capacity)
- Novelty function pour le glissando (P1/P2)
- Fréquence de modulation pour la roulée R
- Détection de silence et intervalles actifs

Référence: Thèse Zocco 2025, Sections 3.7.2-3.7.4
"""

import numpy as np
import pandas as pd
from scipy import signal
import librosa
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass

from .config import PipelineConfig, get_config


@dataclass
class FRCSegment:
    """Segment relatif à la FRC."""
    
    # Indices (en samples audio)
    start_sample: int
    end_sample: int
    frc_cross_sample: int
    
    # Segments audio
    above_frc: np.ndarray
    below_frc: np.ndarray
    
    # Métadonnées
    duration_above: float  # secondes
    duration_below: float  # secondes
    sample_rate: int
    
    @property
    def above_frc_duration_percent(self) -> float:
        """Pourcentage de temps passé au-dessus de la FRC."""
        total = self.duration_above + self.duration_below
        return (self.duration_above / total * 100) if total > 0 else 0


@dataclass
class GlideSegment:
    """Segmentation du glissando en P1 (basses fréquences) et P2 (hautes)."""
    
    # Indices
    start_sample: int
    peak_sample: int  # Point de transition spectrale maximale
    end_sample: int
    
    # Segments audio
    part1: np.ndarray  # Portion basses fréquences
    part2: np.ndarray  # Portion hautes fréquences
    
    # Temps
    peak_time: float
    duration_p1: float
    duration_p2: float
    
    # Valeur de la novelty function au pic
    peak_novelty_value: float


@dataclass
class ModulationResult:
    """Résultat de l'analyse de fréquence de modulation (roulée R)."""
    
    frequency_full: float      # Fréquence sur tout le segment
    frequency_above_frc: float # Fréquence au-dessus de la FRC
    frequency_below_frc: float # Fréquence en-dessous de la FRC
    
    # Enveloppe et spectre (pour visualisation)
    rms_envelope: Optional[np.ndarray] = None
    modulation_spectrum: Optional[np.ndarray] = None


class FRCSegmenter:
    """
    Segmente la phonation selon le seuil FRC.
    
    La FRC (Functional Residual Capacity) est le volume pulmonaire
    à la fin d'une expiration passive. La phonation au-dessus de
    la FRC utilise le recul élastique, celle en-dessous nécessite
    un effort musculaire actif.
    
    Exemple:
    ```python
    segmenter = FRCSegmenter()
    
    # Avec les données OEP
    segment = segmenter.segment_by_frc(
        audio=audio,
        oep_volume=volume_trace,
        frc_level=frc_value,
        sr_audio=48000,
        sr_oep=50
    )
    
    # Analyse séparée
    result_above = analyze(segment.above_frc)
    result_below = analyze(segment.below_frc)
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_config()
    
    def find_frc_crossing(
        self,
        volume: np.ndarray,
        frc_level: float
    ) -> int:
        """
        Trouve l'instant où le volume croise la FRC (en descendant).
        
        Args:
            volume: Trace de volume (Vcw)
            frc_level: Niveau de la FRC
            
        Returns:
            Index du crossing
        """
        # Chercher où le volume passe en-dessous de la FRC
        below_frc = volume < frc_level
        
        # Trouver le premier passage (descente)
        crossings = np.where(np.diff(below_frc.astype(int)) == 1)[0]
        
        if len(crossings) == 0:
            # Pas de crossing trouvé, retourner le milieu
            return len(volume) // 2
        
        return crossings[0]
    
    def segment_by_frc(
        self,
        audio: np.ndarray,
        oep_volume: np.ndarray,
        frc_level: float,
        sr_audio: int,
        sr_oep: int,
        phonation_start: int,
        phonation_end: int
    ) -> FRCSegment:
        """
        Segmente l'audio selon le seuil FRC.
        
        Args:
            audio: Signal audio
            oep_volume: Volume de la paroi thoracique (Vcw)
            frc_level: Niveau de la FRC
            sr_audio: Sample rate audio
            sr_oep: Sample rate OEP
            phonation_start: Début de phonation (samples audio)
            phonation_end: Fin de phonation (samples audio)
            
        Returns:
            FRCSegment avec les deux parties
        """
        # Extraire la portion de volume correspondant à la phonation
        oep_start = int(phonation_start / sr_audio * sr_oep)
        oep_end = int(phonation_end / sr_audio * sr_oep)
        volume_segment = oep_volume[oep_start:oep_end]
        
        # Trouver le crossing FRC dans le volume
        crossing_oep = self.find_frc_crossing(volume_segment, frc_level)
        
        # Convertir en sample audio
        crossing_audio = int(crossing_oep / sr_oep * sr_audio)
        frc_cross_sample = phonation_start + crossing_audio
        
        # Découper l'audio
        audio_segment = audio[phonation_start:phonation_end]
        above_frc = audio_segment[:crossing_audio]
        below_frc = audio_segment[crossing_audio:]
        
        return FRCSegment(
            start_sample=phonation_start,
            end_sample=phonation_end,
            frc_cross_sample=frc_cross_sample,
            above_frc=above_frc,
            below_frc=below_frc,
            duration_above=len(above_frc) / sr_audio,
            duration_below=len(below_frc) / sr_audio,
            sample_rate=sr_audio
        )
    
    def segment_by_time(
        self,
        audio: np.ndarray,
        cross_time: float,
        start_time: float,
        end_time: float,
        sr: int
    ) -> FRCSegment:
        """
        Segmente l'audio avec un temps de crossing fourni.
        
        Utilisé quand le temps de crossing FRC est déjà connu
        (ex: depuis un fichier Excel).
        
        Args:
            audio: Signal audio
            cross_time: Temps du crossing (secondes)
            start_time: Début de phonation (secondes)
            end_time: Fin de phonation (secondes)
            sr: Sample rate
            
        Returns:
            FRCSegment
        """
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        cross_sample = int(cross_time * sr)
        
        # S'assurer que cross est entre start et end
        cross_sample = max(start_sample, min(cross_sample, end_sample))
        
        above = audio[start_sample:cross_sample]
        below = audio[cross_sample:end_sample]
        
        return FRCSegment(
            start_sample=start_sample,
            end_sample=end_sample,
            frc_cross_sample=cross_sample,
            above_frc=above,
            below_frc=below,
            duration_above=len(above) / sr,
            duration_below=len(below) / sr,
            sample_rate=sr
        )


class GlideSegmenter:
    """
    Segmente le glissando vocal en P1 (basses fréquences) et P2 (hautes).
    
    Utilise la novelty function basée sur le centroïde spectral
    pour détecter le point de transition maximale.
    
    Exemple:
    ```python
    segmenter = GlideSegmenter()
    segment = segmenter.segment_glide(audio, sr)
    
    # Analyser les deux parties
    f0_p1 = analyze(segment.part1)  # Fréquences basses
    f0_p2 = analyze(segment.part2)  # Fréquences hautes
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_config()
    
    def compute_novelty_function(
        self,
        audio: np.ndarray,
        sr: int,
        downsample_factor: int = 32,
        gamma: float = 10.0,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule la novelty function basée sur le centroïde spectral.
        
        La novelty function détecte les changements rapides dans
        le spectre. Pour un glissando, elle aura un pic au moment
        de la transition maximale de fréquence.
        
        Args:
            audio: Signal audio
            sr: Sample rate
            downsample_factor: Facteur de sous-échantillonnage
            gamma: Facteur de compression logarithmique
            frame_length: Taille de la fenêtre STFT
            hop_length: Hop length STFT
            
        Returns:
            Tuple (novelty_function, time_axis)
        """
        # Calcul du centroïde spectral
        S = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        
        # Downsampling et compression log
        centroid_down = centroid[::downsample_factor]
        centroid_log = np.log(1 + gamma * centroid_down)
        
        # Dérivée (changement de fréquence)
        diff = np.diff(centroid_log, prepend=centroid_log[0])
        
        # Half-wave rectification (on garde seulement les augmentations)
        diff[diff < 0] = 0
        
        # Soustraction de la moyenne locale adaptative
        window_size = min(len(diff) // 4, 51)
        if window_size > 5:
            local_mean = signal.savgol_filter(diff, window_size | 1, 3)
            diff = diff - local_mean
            diff[diff < 0] = 0
        
        # Axe temporel
        fps = sr / (hop_length * downsample_factor)
        time_axis = np.arange(len(diff)) / fps
        
        return diff, time_axis
    
    def find_peak_auto(
        self,
        novelty: np.ndarray,
        margin_percent: int = 10
    ) -> int:
        """
        Trouve automatiquement le pic principal de la novelty function.
        
        Ignore les marges (début/fin) pour éviter les faux positifs
        liés aux transitions onset/offset.
        
        Args:
            novelty: Novelty function
            margin_percent: Pourcentage de marge à ignorer
            
        Returns:
            Index du pic
        """
        margin = int(len(novelty) * margin_percent / 100)
        
        if margin >= len(novelty) // 2:
            margin = len(novelty) // 4
        
        cropped = novelty[margin:-margin] if margin > 0 else novelty
        peak_idx = np.argmax(cropped) + margin
        
        return peak_idx
    
    def segment_glide(
        self,
        audio: np.ndarray,
        sr: int,
        phonation_start: Optional[int] = None,
        phonation_end: Optional[int] = None,
        peak_override: Optional[int] = None
    ) -> GlideSegment:
        """
        Segmente le glissando en P1 et P2.
        
        Args:
            audio: Signal audio
            sr: Sample rate
            phonation_start: Début de phonation (samples), auto si None
            phonation_end: Fin de phonation (samples), auto si None
            peak_override: Position du pic manuelle (samples)
            
        Returns:
            GlideSegment avec les deux parties
        """
        # Détection automatique des bornes si non fournies
        if phonation_start is None:
            onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='samples')
            phonation_start = onsets[0] if len(onsets) > 0 else 0
        
        if phonation_end is None:
            intervals = librosa.effects.split(audio, top_db=45)
            phonation_end = intervals[-1, 1] if len(intervals) > 0 else len(audio)
        
        # Extraire le segment de phonation
        audio_segment = audio[phonation_start:phonation_end]
        
        # Calculer la novelty function
        novelty, time_axis = self.compute_novelty_function(audio_segment, sr)
        
        # Trouver le pic
        if peak_override is not None:
            # Convertir le peak_override absolu en relatif
            peak_relative = peak_override - phonation_start
            # Convertir en index de novelty
            fps = sr / (512 * 32)  # Valeurs par défaut
            peak_novelty_idx = int(peak_relative / sr * fps)
            peak_novelty_idx = min(max(0, peak_novelty_idx), len(novelty) - 1)
        else:
            peak_novelty_idx = self.find_peak_auto(novelty)
        
        # Convertir l'index novelty en sample audio
        fps = sr / (512 * 32)
        peak_time = peak_novelty_idx / fps
        peak_sample = phonation_start + int(peak_time * sr)
        
        # Découper
        part1 = audio[phonation_start:peak_sample]
        part2 = audio[peak_sample:phonation_end]
        
        return GlideSegment(
            start_sample=phonation_start,
            peak_sample=peak_sample,
            end_sample=phonation_end,
            part1=part1,
            part2=part2,
            peak_time=peak_time,
            duration_p1=len(part1) / sr,
            duration_p2=len(part2) / sr,
            peak_novelty_value=novelty[peak_novelty_idx] if peak_novelty_idx < len(novelty) else 0
        )


class ModulationAnalyzer:
    """
    Analyse la fréquence de modulation pour la roulée R.
    
    La roulée (trille alvéolaire) produit une modulation périodique
    de l'enveloppe acoustique correspondant à la vibration de la
    pointe de la langue (typiquement 20-30 Hz).
    
    Exemple:
    ```python
    analyzer = ModulationAnalyzer()
    result = analyzer.analyze_modulation(audio, sr)
    print(f"Fréquence de modulation: {result.frequency_full:.1f} Hz")
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_config()
    
    def compute_modulation_frequency(
        self,
        audio: np.ndarray,
        sr: int
    ) -> float:
        """
        Calcule la fréquence de modulation d'un segment audio.
        
        Méthode:
        1. Extraction de l'enveloppe RMS
        2. Suppression du trend lent (Savitzky-Golay)
        3. FFT de l'enveloppe
        4. Recherche du pic dans la bande de modulation
        
        Args:
            audio: Signal audio
            sr: Sample rate
            
        Returns:
            Fréquence de modulation en Hz
        """
        cfg = self.config.modulation
        
        if audio is None or len(audio) < cfg.rms_frame_length:
            return np.nan
        
        # 1. Extraction enveloppe RMS
        rms = librosa.feature.rms(
            y=audio,
            frame_length=cfg.rms_frame_length,
            hop_length=cfg.rms_hop_length
        )[0]
        
        if len(rms) < 15:
            return np.nan
        
        # 2. Suppression du trend lent
        window_len = min(len(rms) // 2 * 2 - 1, cfg.savgol_window_max)
        if window_len < 5:
            window_len = 5
        if window_len % 2 == 0:
            window_len -= 1
        
        if window_len > 5:
            trend = signal.savgol_filter(rms, window_len, cfg.savgol_order)
            rms_clean = rms - trend
        else:
            rms_clean = rms - np.mean(rms)
        
        # 3. FFT de l'enveloppe
        n_fft = len(rms_clean)
        spectrum = np.fft.rfft(rms_clean)
        freqs = np.fft.rfftfreq(n_fft, d=cfg.rms_hop_length / sr)
        
        # 4. Recherche du pic dans la bande typique
        mask = (freqs >= cfg.mod_freq_min) & (freqs <= cfg.mod_freq_max)
        
        if not np.any(mask):
            # Fallback avec bande élargie
            mask = (freqs >= cfg.mod_freq_fallback_min) & (freqs <= cfg.mod_freq_fallback_max)
        
        if not np.any(mask):
            return np.nan
        
        peak_idx = np.argmax(np.abs(spectrum[mask]))
        return freqs[mask][peak_idx]
    
    def analyze_with_frc(
        self,
        audio: np.ndarray,
        sr: int,
        frc_segment: Optional[FRCSegment] = None,
        cross_time: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> ModulationResult:
        """
        Analyse la modulation avec segmentation FRC.
        
        Args:
            audio: Signal audio
            sr: Sample rate
            frc_segment: Segment FRC pré-calculé
            cross_time: Temps de crossing FRC (alternative)
            start_time: Début de phonation
            end_time: Fin de phonation
            
        Returns:
            ModulationResult avec fréquences full, above et below
        """
        if frc_segment is None and cross_time is not None:
            segmenter = FRCSegmenter(self.config)
            frc_segment = segmenter.segment_by_time(
                audio, cross_time, start_time or 0, end_time or len(audio)/sr, sr
            )
        
        # Fréquence sur tout le segment
        if frc_segment is not None:
            full_audio = np.concatenate([frc_segment.above_frc, frc_segment.below_frc])
            above_audio = frc_segment.above_frc
            below_audio = frc_segment.below_frc
        else:
            full_audio = audio
            above_audio = None
            below_audio = None
        
        freq_full = self.compute_modulation_frequency(full_audio, sr)
        freq_above = self.compute_modulation_frequency(above_audio, sr) if above_audio is not None else np.nan
        freq_below = self.compute_modulation_frequency(below_audio, sr) if below_audio is not None else np.nan
        
        return ModulationResult(
            frequency_full=freq_full,
            frequency_above_frc=freq_above,
            frequency_below_frc=freq_below
        )


def detect_non_silent_intervals(
    audio: np.ndarray,
    sr: int,
    top_db: int = 45
) -> np.ndarray:
    """
    Détecte les intervalles non-silencieux.
    
    Args:
        audio: Signal audio
        sr: Sample rate
        top_db: Seuil en dB sous le max
        
    Returns:
        Array de shape (n_intervals, 2) avec [start, end] en samples
    """
    return librosa.effects.split(audio, top_db=top_db)


def detect_phonation_bounds(
    audio: np.ndarray,
    sr: int,
    top_db: int = 45,
    onset_delta: float = 0.05
) -> Tuple[int, int]:
    """
    Détecte le début et la fin de la phonation.
    
    Args:
        audio: Signal audio
        sr: Sample rate
        top_db: Seuil pour la détection de silence
        onset_delta: Sensibilité de détection d'onset
        
    Returns:
        Tuple (start_sample, end_sample)
    """
    # Intervalles non-silencieux
    intervals = detect_non_silent_intervals(audio, sr, top_db)
    
    if len(intervals) == 0:
        return 0, len(audio)
    
    # Prendre le premier intervalle significatif pour le début
    start = intervals[0, 0]
    
    # Prendre la fin du dernier intervalle
    end = intervals[-1, 1]
    
    return start, end
