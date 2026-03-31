"""
Global configuration of the pneumophonic analysis pipeline.

This module centralizes all parameters used in the analysis
to ensure reproducibility and facilitate adjustments.

Reference: Zocco Thesis 2025 - "Integrated Analysis of Respiratory-Phonatory Functions"
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class AudioConfig:
    """Paramètres pour le traitement audio."""
    
    #Target sampling frequency
    sample_rate: int = 48000
    
    # Noise reduction (noisereduce library)
    noise_reduction_stationary: bool = True
    noise_reduction_prop_decrease: float = 0.85
    
    # Pre-emphasis filter coefficient
    pre_emphasis_coef: float = 0.97
    
    # STFT parameters
    frame_length_ms: float = 30.0  # Duration of the window in ms
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
    """Parameters for F0 extraction."""
    
    # Fundamental frequency range (Hz)
    f0_min: int = 50
    f0_max: int = 500
    
    # Parameters for pyin (librosa)
    pyin_n_thresholds: int = 3
    
    # Unit for Praat
    unit: str = 'Hertz'


@dataclass  
class OEPConfig:
    """Parameters for OEP data (opto-electronic plethysmography)."""
    
    # Target sampling frequencies
    fs_kinematic: int = 50   # Kinematic data (Hz)
    fs_analog: int = 200     # Analog channels (Hz)
    
    # Columns in the .dat file (time, A, B, C, tot_vol, sync, D, E, F, G, H, I)
    dat_columns: List[str] = field(default_factory=lambda: [
        'time', 'A', 'B', 'C', 'tot_vol', 'sync',
        'D', 'E', 'F', 'G', 'H', 'I'
    ])
    
    # Parameters for sync signal detection
    sync_prominence: float = 0.12
    sync_threshold: float = 0.5


@dataclass
class SegmentationConfig:
    """Parameters for signal segmentation."""
    
    # Detection of non-silent intervals (librosa.effects.split)
    silence_top_db: int = 45
    
    # Detection of onsets (librosa.onset.onset_detect)
    onset_hop_length: int = 512
    onset_delta: float = 0.05
    
    # Novelty function for glissando
    novelty_window_samples: int = 32
    novelty_margin_percent: int = 10


@dataclass
class FormantConfig:
    """Parameters for formant analysis."""
    
    # Praat Formant (burg)
    time_step: float = 0.0025
    max_formants: int = 5
    max_frequency: int = 5000
    window_length: float = 0.025
    pre_emphasis_from: int = 50


@dataclass
class JitterShimmerConfig:
    """Parameters for jitter and shimmer (Praat)."""
    
    period_floor: float = 0.0001
    period_ceiling: float = 0.02
    max_period_factor: float = 1.3
    max_amplitude_factor: float = 1.6


@dataclass
class DSIConfig:
    """Parameters for the Dysphonia Severity Index."""
    
    # Coefficients of the DSI equation
    # DSI = 0.13*MPT + 0.0053*F0_high - 0.26*I_low - 1.18*jitter_percent + 12.4
    coef_mpt: float = 0.13
    coef_f0_high: float = 0.0053
    coef_i_low: float = -0.26
    coef_jitter: float = -1.18
    intercept: float = 12.4


@dataclass
class ModulationConfig:
    """Parameters for the modulation frequency analysis (glissando R)."""
    
    # Extraction of RMS envelope
    rms_frame_length: int = 512
    rms_hop_length: int = 128
    
    # Filter for the trend (Savitzky-Golay)
    savgol_window_max: int = 51
    savgol_order: int = 3
    
    # Search band for the modulation frequency (Hz)
    mod_freq_min: float = 10.0
    mod_freq_max: float = 35.0
    
    # Fallback if out of band
    mod_freq_fallback_min: float = 5.0
    mod_freq_fallback_max: float = 40.0


@dataclass
class OutputConfig:
    """Parameters for output configuration."""
    
    # Output format
    excel_engine: str = 'openpyxl'
    
    # Default sheet names
    sheet_metrics: str = 'Metrics'
    sheet_formants: str = 'Formants'
    sheet_respiratory: str = 'Respiratory'
    
    # Figures
    figure_dpi: int = 150
    figure_format: str = 'png'


@dataclass
class PipelineConfig:
    """Parameters for the complete pipeline configuration."""
    
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


# Default Configuration (singleton)
DEFAULT_CONFIG = PipelineConfig()


def get_config() -> PipelineConfig:
    """Returns the default configuration."""
    return DEFAULT_CONFIG


def create_config(**kwargs) -> PipelineConfig:
    """Creates a customized configuration."""
    return PipelineConfig(**kwargs)
