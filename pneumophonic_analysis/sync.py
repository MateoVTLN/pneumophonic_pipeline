"""
Module de synchronisation pour l'alignement temporel OEP/Audio.

La synchronisation repose sur la détection du signal sync (falling edge)
présent à la fois dans les données OEP et dans l'enregistrement audio.

Référence: Thèse Zocco 2025, Section 3.5 - Data Processing and Analysis
"""

import numpy as np
import pandas as pd
from scipy import signal
import librosa
from typing import Tuple, Optional, Dict, Union
from dataclasses import dataclass

from .config import PipelineConfig, get_config


@dataclass
class SyncResult:
    """Résultat de la synchronisation."""
    
    # Position du falling edge dans le signal sync audio (en samples)
    sync_falling_edge_sample: int
    
    # Position du falling edge dans les données OEP (en samples OEP)
    oep_falling_edge_sample: int
    
    # Offset temporel à appliquer (en secondes)
    time_offset_sec: float
    
    # Timestamps des onsets détectés
    sync_onsets_samples: np.ndarray
    oep_sync_onsets_samples: np.ndarray
    
    # Qualité de la synchronisation
    correlation: float = 0.0


class Synchronizer:
    """
    Synchronise les signaux OEP et audio via le signal sync.
    
    Le protocole utilise un signal de synchronisation (impulsion carrée)
    envoyé simultanément à l'OEP (canal analogique) et à la DAW audio.
    L'alignement se fait sur le falling edge de ce signal.
    
    Exemple:
    ```python
    sync = Synchronizer(config)
    
    # Détecter le falling edge dans l'audio
    audio_offset = sync.detect_sync_in_audio(sync_signal, sr=48000)
    
    # Détecter dans les données OEP
    oep_offset = sync.detect_sync_in_oep(oep_df, take_number=1)
    
    # Aligner les données
    aligned_oep = sync.align_oep_to_audio(oep_df, audio_length, audio_offset, oep_offset)
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialise le synchroniseur.
        
        Args:
            config: Configuration du pipeline
        """
        self.config = config or get_config()
    
    def detect_falling_edge_audio(
        self,
        sync_signal: np.ndarray,
        sr: int,
        onset_index: int = 1
    ) -> int:
        """
        Détecte le falling edge dans le signal sync audio.
        
        Le signal sync est une impulsion carrée. On utilise librosa.onset.onset_detect
        pour trouver les transitions, puis on sélectionne celle correspondant au
        falling edge (typiquement le 2ème onset détecté).
        
        Args:
            sync_signal: Signal de synchronisation audio
            sr: Fréquence d'échantillonnage
            onset_index: Index de l'onset à utiliser (0=premier, 1=deuxième/falling)
            
        Returns:
            Position du falling edge en samples
        """
        # Détecter les onsets
        onsets = librosa.onset.onset_detect(
            y=sync_signal,
            sr=sr,
            units='samples'
        )
        
        if len(onsets) <= onset_index:
            raise ValueError(
                f"Pas assez d'onsets détectés ({len(onsets)}). "
                f"Attendu au moins {onset_index + 1}."
            )
        
        return onsets[onset_index]
    
    def detect_falling_edge_threshold(
        self,
        sync_signal: np.ndarray,
        threshold: float = 0.5
    ) -> int:
        """
        Détecte le falling edge par seuillage.
        
        Alternative à la méthode onset_detect, utile pour les signaux
        avec un rapport signal/bruit élevé.
        
        Args:
            sync_signal: Signal normalisé [-1, 1]
            threshold: Seuil de détection
            
        Returns:
            Position du falling edge en samples
        """
        sync_norm = sync_signal / (np.max(np.abs(sync_signal)) + 1e-9)
        
        # Trouver où le signal passe en dessous du seuil
        falling_edges = np.where(
            (sync_norm[:-1] > threshold) & (sync_norm[1:] < threshold)
        )[0]
        
        if len(falling_edges) == 0:
            raise ValueError("Aucun falling edge détecté avec le seuil spécifié")
        
        return falling_edges[0]
    
    def detect_sync_onsets_oep(
        self,
        oep_df: pd.DataFrame,
        prominence: Optional[float] = None
    ) -> np.ndarray:
        """
        Détecte tous les pics du signal sync dans les données OEP.
        
        Chaque take du protocole génère une paire de pics (rising + falling edge).
        
        Args:
            oep_df: DataFrame avec colonne 'sync'
            prominence: Prominence minimale des pics (défaut: config)
            
        Returns:
            Array des indices des pics
        """
        prominence = prominence or self.config.oep.sync_prominence
        
        sync_signal = np.abs(oep_df['sync'].values)
        peaks, _ = signal.find_peaks(sync_signal, prominence=prominence)
        
        return peaks
    
    def get_take_falling_edge_oep(
        self,
        oep_df: pd.DataFrame,
        take_number: int,
        fs_oep: Optional[int] = None
    ) -> Tuple[int, float]:
        """
        Récupère le falling edge pour un take spécifique.
        
        Convention: Take 1 → pic index 1 (2*1-1=1)
                   Take 2 → pic index 3 (2*2-1=3)
                   etc.
        
        Args:
            oep_df: DataFrame OEP
            take_number: Numéro du take (1-based)
            fs_oep: Fréquence d'échantillonnage OEP
            
        Returns:
            Tuple (index en samples, temps en secondes)
        """
        fs_oep = fs_oep or self.config.oep.fs_kinematic
        
        peaks = self.detect_sync_onsets_oep(oep_df)
        
        # Index du falling edge pour ce take
        peak_index = 2 * take_number - 1
        
        if peak_index >= len(peaks):
            raise ValueError(
                f"Take {take_number} non trouvé. "
                f"Seulement {len(peaks) // 2} takes détectés."
            )
        
        sample = peaks[peak_index]
        time_sec = sample / fs_oep
        
        return sample, time_sec
    
    def synchronize(
        self,
        sync_audio: np.ndarray,
        sr_audio: int,
        oep_df: pd.DataFrame,
        take_number: int = 1,
        fs_oep: Optional[int] = None,
        audio_onset_index: int = 1
    ) -> SyncResult:
        """
        Synchronise complètement les données audio et OEP.
        
        Args:
            sync_audio: Signal sync audio
            sr_audio: Sample rate audio
            oep_df: DataFrame OEP
            take_number: Numéro du take
            fs_oep: Sample rate OEP
            audio_onset_index: Index de l'onset audio à utiliser
            
        Returns:
            SyncResult avec tous les paramètres de synchronisation
        """
        fs_oep = fs_oep or self.config.oep.fs_kinematic
        
        # Falling edge audio
        audio_falling_sample = self.detect_falling_edge_audio(
            sync_audio, sr_audio, audio_onset_index
        )
        audio_falling_time = audio_falling_sample / sr_audio
        
        # Falling edge OEP
        oep_falling_sample, oep_falling_time = self.get_take_falling_edge_oep(
            oep_df, take_number, fs_oep
        )
        
        # Offset temporel
        time_offset = audio_falling_time - oep_falling_time
        
        # Onsets détectés
        audio_onsets = librosa.onset.onset_detect(
            y=sync_audio, sr=sr_audio, units='samples'
        )
        oep_onsets = self.detect_sync_onsets_oep(oep_df)
        
        return SyncResult(
            sync_falling_edge_sample=audio_falling_sample,
            oep_falling_edge_sample=oep_falling_sample,
            time_offset_sec=time_offset,
            sync_onsets_samples=audio_onsets,
            oep_sync_onsets_samples=oep_onsets
        )
    
    def convert_audio_time_to_oep_sample(
        self,
        audio_time_sec: float,
        sync_result: SyncResult,
        fs_oep: Optional[int] = None
    ) -> int:
        """
        Convertit un temps audio en index OEP.
        
        Args:
            audio_time_sec: Temps dans le référentiel audio
            sync_result: Résultat de synchronisation
            fs_oep: Sample rate OEP
            
        Returns:
            Index dans le DataFrame OEP
        """
        fs_oep = fs_oep or self.config.oep.fs_kinematic
        
        # Temps relatif au falling edge audio
        relative_time = audio_time_sec - (sync_result.sync_falling_edge_sample / self.config.audio.sample_rate)
        
        # Convertir en sample OEP
        oep_sample = sync_result.oep_falling_edge_sample + int(relative_time * fs_oep)
        
        return oep_sample
    
    def extract_oep_segment(
        self,
        oep_df: pd.DataFrame,
        audio_start_sec: float,
        audio_end_sec: float,
        sync_result: SyncResult,
        fs_oep: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extrait un segment OEP correspondant à une portion audio.
        
        Args:
            oep_df: DataFrame OEP complet
            audio_start_sec: Début du segment (temps audio)
            audio_end_sec: Fin du segment (temps audio)
            sync_result: Résultat de synchronisation
            fs_oep: Sample rate OEP
            
        Returns:
            DataFrame OEP pour le segment
        """
        fs_oep = fs_oep or self.config.oep.fs_kinematic
        
        start_sample = self.convert_audio_time_to_oep_sample(
            audio_start_sec, sync_result, fs_oep
        )
        end_sample = self.convert_audio_time_to_oep_sample(
            audio_end_sec, sync_result, fs_oep
        )
        
        # Borner aux limites du DataFrame
        start_sample = max(0, start_sample)
        end_sample = min(len(oep_df), end_sample)
        
        return oep_df.iloc[start_sample:end_sample].copy()


def compute_relative_timing(
    audio_sample: int,
    sync_falling_edge_sample: int,
    sr: int
) -> float:
    """
    Calcule le temps relatif au falling edge.
    
    Args:
        audio_sample: Position en samples audio
        sync_falling_edge_sample: Position du falling edge
        sr: Sample rate
        
    Returns:
        Temps en secondes relatif au falling edge
    """
    return (audio_sample - sync_falling_edge_sample) / sr


def detect_onset_in_phonation(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 512,
    delta: float = 0.05,
    onset_index: int = 2
) -> int:
    """
    Détecte le début de la phonation dans un signal audio.
    
    Utilise la détection d'onset de librosa. L'index permet de sauter
    les premiers onsets qui peuvent correspondre au signal sync.
    
    Args:
        audio: Signal audio (noise-reduced recommandé)
        sr: Sample rate
        hop_length: Taille du hop pour l'analyse
        delta: Sensibilité de la détection
        onset_index: Index de l'onset à retourner (pour sauter le sync)
        
    Returns:
        Position du début de phonation en samples
    """
    onsets = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        delta=delta,
        units='samples'
    )
    
    if len(onsets) <= onset_index:
        # Fallback: retourner le dernier onset disponible
        return onsets[-1] if len(onsets) > 0 else 0
    
    return onsets[onset_index]


def detect_end_of_phonation(
    audio: np.ndarray,
    sr: int,
    top_db: int = 45,
    interval_index: int = -1
) -> int:
    """
    Détecte la fin de la phonation.
    
    Utilise librosa.effects.split pour trouver les intervalles non-silencieux.
    
    Args:
        audio: Signal audio
        sr: Sample rate
        top_db: Seuil pour la détection du silence
        interval_index: Index de l'intervalle (-1 = dernier)
        
    Returns:
        Position de fin de phonation en samples
    """
    intervals = librosa.effects.split(audio, top_db=top_db)
    
    if len(intervals) == 0:
        return len(audio)
    
    return intervals[interval_index, 1]
    
def detect_phonation_bounds(audio, sr, top_db=30):
    """
    Détecte les indices de début et de fin de la phonation dans un signal audio.
    Utilise librosa.effects.split pour ignorer les silences au début et à la fin.
    """
    intervals = librosa.effects.split(audio, top_db=top_db)
    
    if len(intervals) == 0:
        # S'il ne trouve rien d'évident, on retourne tout le signal
        return 0, len(audio)
        
    # On prend le début du premier intervalle sonore, et la fin du dernier
    start_idx = intervals[0][0]
    end_idx = intervals[-1][1]
    
    return start_idx, end_idx