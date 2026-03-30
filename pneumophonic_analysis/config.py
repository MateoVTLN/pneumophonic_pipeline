"""
Configuration globale du pipeline d'analyse pneumophonique.

Ce module centralise tous les paramètres utilisés dans l'analyse
pour garantir la reproductibilité et faciliter les ajustements.

Référence: Thèse Zocco 2025 - "Integrated Analysis of Respiratory-Phonatory Functions"
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class AudioConfig:
    """Paramètres pour le traitement audio."""
    
    # Fréquence d'échantillonnage cible
    sample_rate: int = 48000
    
    # Noise reduction (noisereduce library)
    noise_reduction_stationary: bool = True
    noise_reduction_prop_decrease: float = 0.85
    
    # Pre-emphasis filter coefficient
    pre_emphasis_coef: float = 0.97
    
    # STFT parameters
    frame_length_ms: float = 30.0  # Durée de la fenêtre en ms
    hop_length_ratio: float = 0.5   # Ratio hop/frame
    
    # Mel spectrogram
    n_mels: int = 64
    
    # MFCCs
    n_mfcc: int = 13
    normalize_mfcc: bool = True
    
    @property
    def frame_length_samples(self) -> int:
        return int(self.frame_length_ms * self.sample_rate / 1000)
    
    @property
    def hop_length_samples(self) -> int:
        return self.frame_length_samples // 2


@dataclass
class PitchConfig:
    """Paramètres pour l'extraction de F0."""
    
    # Plage de fréquence fondamentale (Hz)
    f0_min: int = 50
    f0_max: int = 500
    
    # Paramètres pyin (librosa)
    pyin_n_thresholds: int = 3
    
    # Unité pour Praat
    unit: str = 'Hertz'


@dataclass  
class OEPConfig:
    """Paramètres pour les données OEP (pléthysmographie optoélectronique)."""
    
    # Fréquences d'échantillonnage typiques
    fs_kinematic: int = 50   # Données cinématiques (Hz)
    fs_analog: int = 200     # Canaux analogiques (Hz)
    
    # Colonnes du fichier .dat
    dat_columns: List[str] = field(default_factory=lambda: [
        'time', 'A', 'B', 'C', 'tot_vol', 'sync',
        'D', 'E', 'F', 'G', 'H', 'I'
    ])
    
    # Paramètres de détection du signal sync
    sync_prominence: float = 0.12
    sync_threshold: float = 0.5


@dataclass
class SegmentationConfig:
    """Paramètres pour la segmentation des signaux."""
    
    # Détection des intervalles non-silencieux (librosa.effects.split)
    silence_top_db: int = 45
    
    # Détection d'onset
    onset_hop_length: int = 512
    onset_delta: float = 0.05
    
    # Novelty function pour glissando
    novelty_window_samples: int = 32
    novelty_margin_percent: int = 10


@dataclass
class FormantConfig:
    """Paramètres pour l'analyse des formants."""
    
    # Praat Formant (burg)
    time_step: float = 0.0025
    max_formants: int = 5
    max_frequency: int = 5000
    window_length: float = 0.025
    pre_emphasis_from: int = 50


@dataclass
class JitterShimmerConfig:
    """Paramètres pour jitter et shimmer (Praat)."""
    
    period_floor: float = 0.0001
    period_ceiling: float = 0.02
    max_period_factor: float = 1.3
    max_amplitude_factor: float = 1.6


@dataclass
class DSIConfig:
    """Paramètres pour le Dysphonia Severity Index."""
    
    # Coefficients de l'équation DSI
    # DSI = 0.13*MPT + 0.0053*F0_high - 0.26*I_low - 1.18*jitter_percent + 12.4
    coef_mpt: float = 0.13
    coef_f0_high: float = 0.0053
    coef_i_low: float = -0.26
    coef_jitter: float = -1.18
    intercept: float = 12.4


@dataclass
class ModulationConfig:
    """Paramètres pour l'analyse de fréquence de modulation (roulée R)."""
    
    # Extraction RMS envelope
    rms_frame_length: int = 512
    rms_hop_length: int = 128
    
    # Filtrage du trend (Savitzky-Golay)
    savgol_window_max: int = 51
    savgol_order: int = 3
    
    # Bande de recherche pour la fréquence de modulation (Hz)
    mod_freq_min: float = 10.0
    mod_freq_max: float = 35.0
    
    # Fallback si hors bande
    mod_freq_fallback_min: float = 5.0
    mod_freq_fallback_max: float = 40.0


@dataclass
class OutputConfig:
    """Configuration des sorties."""
    
    # Format de sortie
    excel_engine: str = 'openpyxl'
    
    # Noms des sheets par défaut
    sheet_metrics: str = 'Metrics'
    sheet_formants: str = 'Formants'
    sheet_respiratory: str = 'Respiratory'
    
    # Figures
    figure_dpi: int = 150
    figure_format: str = 'png'


@dataclass
class PipelineConfig:
    """Configuration complète du pipeline."""
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    oep: OEPConfig = field(default_factory=OEPConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    formant: FormantConfig = field(default_factory=FormantConfig)
    jitter_shimmer: JitterShimmerConfig = field(default_factory=JitterShimmerConfig)
    dsi: DSIConfig = field(default_factory=DSIConfig)
    modulation: ModulationConfig = field(default_factory=ModulationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Chemins par défaut
    data_root: Optional[Path] = None
    output_root: Optional[Path] = None
    
    def __post_init__(self):
        if self.data_root is not None:
            self.data_root = Path(self.data_root)
        if self.output_root is not None:
            self.output_root = Path(self.output_root)


# Configuration par défaut (singleton)
DEFAULT_CONFIG = PipelineConfig()


def get_config() -> PipelineConfig:
    """Retourne la configuration par défaut."""
    return DEFAULT_CONFIG


def create_config(**kwargs) -> PipelineConfig:
    """Crée une configuration personnalisée."""
    return PipelineConfig(**kwargs)
