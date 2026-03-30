"""
Module d'extraction des paramètres acoustiques via Praat.

Utilise parselmouth (interface Python pour Praat) pour extraire:
- Fréquence fondamentale (F0) et ses statistiques
- Jitter (perturbation de fréquence)
- Shimmer (perturbation d'amplitude)
- HNR (Harmonics-to-Noise Ratio)
- Formants (F1, F2, F3)
- DSI (Dysphonia Severity Index)

Référence: 
- Thèse Zocco 2025, Section 3.5.3
- Praat documentation: https://www.fon.hum.uva.nl/praat/manual/Voice.html
- Scripts: https://github.com/drfeinberg/PraatScripts
"""

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from typing import Optional, Tuple, Dict, Union
from dataclasses import dataclass, field

from .config import PipelineConfig, get_config


@dataclass
class PitchMetrics:
    """Métriques de fréquence fondamentale."""
    mean_f0: float
    std_f0: float
    min_f0: float
    max_f0: float
    range_f0: float = field(init=False)
    
    def __post_init__(self):
        self.range_f0 = self.max_f0 - self.min_f0


@dataclass
class PerturbationMetrics:
    """Métriques de perturbation (jitter/shimmer)."""
    # Jitter
    local_jitter: float
    local_absolute_jitter: float
    rap_jitter: float
    ppq5_jitter: float
    ddp_jitter: float
    
    # Shimmer
    local_shimmer: float
    local_db_shimmer: float
    apq3_shimmer: float
    apq5_shimmer: float
    apq11_shimmer: float
    dda_shimmer: float


@dataclass
class FormantMetrics:
    """Métriques des formants."""
    f1_mean: float
    f2_mean: float
    f3_mean: float
    f1_median: float
    f2_median: float
    f3_median: float


@dataclass
class VoiceQualityMetrics:
    """Métriques de qualité vocale."""
    hnr: float              # Harmonics-to-Noise Ratio
    dsi: float              # Dysphonia Severity Index
    intensity_mean: float   # Intensité moyenne
    intensity_min: float    # Intensité minimale
    mpt: float              # Maximum Phonation Time


@dataclass
class AcousticAnalysisResult:
    """Résultat complet de l'analyse acoustique."""
    
    pitch: PitchMetrics
    perturbation: PerturbationMetrics
    formants: FormantMetrics
    voice_quality: VoiceQualityMetrics
    
    # Traces temporelles (optionnelles)
    f1_trace: Optional[np.ndarray] = None
    f2_trace: Optional[np.ndarray] = None
    f3_trace: Optional[np.ndarray] = None
    time_trace: Optional[np.ndarray] = None
    
    def to_dataframe(self, prefix: str = "") -> pd.DataFrame:
        """Convertit les métriques en DataFrame sur une ligne."""
        data = {}
        
        # Pitch
        for key, val in self.pitch.__dict__.items():
            data[f"{prefix}pitch_{key}"] = val
        
        # Perturbation
        for key, val in self.perturbation.__dict__.items():
            data[f"{prefix}{key}"] = val
        
        # Formants
        for key, val in self.formants.__dict__.items():
            data[f"{prefix}{key}"] = val
        
        # Voice quality
        for key, val in self.voice_quality.__dict__.items():
            data[f"{prefix}{key}"] = val
        
        return pd.DataFrame([data])


class PraatAnalyzer:
    """
    Analyseur acoustique utilisant Praat via Parselmouth.
    
    Exemple:
    ```python
    analyzer = PraatAnalyzer(config)
    
    # Depuis un fichier
    result = analyzer.analyze_file("audio.wav")
    
    # Depuis un array numpy
    result = analyzer.analyze_signal(audio, sr=48000)
    
    # Export en DataFrame
    df = result.to_dataframe()
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialise l'analyseur.
        
        Args:
            config: Configuration du pipeline
        """
        self.config = config or get_config()
    
    def _create_sound(
        self,
        audio: np.ndarray,
        sr: int
    ) -> parselmouth.Sound:
        """Crée un objet Sound Praat depuis un array numpy."""
        return parselmouth.Sound(audio, sampling_frequency=sr)
    
    def _create_pitch(
        self,
        sound: parselmouth.Sound,
        f0_min: Optional[int] = None,
        f0_max: Optional[int] = None
    ) -> parselmouth.Pitch:
        """Crée un objet Pitch."""
        f0_min = f0_min or self.config.pitch.f0_min
        f0_max = f0_max or self.config.pitch.f0_max
        
        return call(sound, "To Pitch", 0.0, f0_min, f0_max)
    
    def _create_point_process(
        self,
        sound: parselmouth.Sound,
        f0_min: Optional[int] = None,
        f0_max: Optional[int] = None
    ) -> parselmouth.Data:
        """Crée un objet PointProcess (instants glottiques)."""
        f0_min = f0_min or self.config.pitch.f0_min
        f0_max = f0_max or self.config.pitch.f0_max
        
        return call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
    
    def extract_pitch_metrics(
        self,
        sound: parselmouth.Sound,
        f0_min: Optional[int] = None,
        f0_max: Optional[int] = None
    ) -> PitchMetrics:
        """
        Extrait les métriques de fréquence fondamentale.
        
        Args:
            sound: Objet Sound Praat
            f0_min: F0 minimum
            f0_max: F0 maximum
            
        Returns:
            PitchMetrics avec mean, std, min, max
        """
        unit = self.config.pitch.unit
        pitch = self._create_pitch(sound, f0_min, f0_max)
        
        mean_f0 = call(pitch, "Get mean", 0, 0, unit)
        std_f0 = call(pitch, "Get standard deviation", 0, 0, unit)
        min_f0 = call(pitch, "Get minimum", 0, 0, unit, "Parabolic")
        max_f0 = call(pitch, "Get maximum", 0, 0, unit, "Parabolic")
        
        return PitchMetrics(
            mean_f0=mean_f0,
            std_f0=std_f0,
            min_f0=min_f0,
            max_f0=max_f0
        )
    
    def extract_perturbation_metrics(
        self,
        sound: parselmouth.Sound,
        f0_min: Optional[int] = None,
        f0_max: Optional[int] = None
    ) -> PerturbationMetrics:
        """
        Extrait les métriques de jitter et shimmer.
        
        Le jitter mesure les perturbations cycle-à-cycle de la période.
        Le shimmer mesure les perturbations d'amplitude.
        
        Args:
            sound: Objet Sound Praat
            f0_min: F0 minimum
            f0_max: F0 maximum
            
        Returns:
            PerturbationMetrics
        """
        cfg = self.config.jitter_shimmer
        point_process = self._create_point_process(sound, f0_min, f0_max)
        
        # Jitter
        local_jitter = call(
            point_process, "Get jitter (local)",
            0, 0, cfg.period_floor, cfg.period_ceiling, cfg.max_period_factor
        )
        local_absolute_jitter = call(
            point_process, "Get jitter (local, absolute)",
            0, 0, cfg.period_floor, cfg.period_ceiling, cfg.max_period_factor
        )
        rap_jitter = call(
            point_process, "Get jitter (rap)",
            0, 0, cfg.period_floor, cfg.period_ceiling, cfg.max_period_factor
        )
        ppq5_jitter = call(
            point_process, "Get jitter (ppq5)",
            0, 0, cfg.period_floor, cfg.period_ceiling, cfg.max_period_factor
        )
        ddp_jitter = call(
            point_process, "Get jitter (ddp)",
            0, 0, cfg.period_floor, cfg.period_ceiling, cfg.max_period_factor
        )
        
        # Shimmer
        local_shimmer = call(
            [sound, point_process], "Get shimmer (local)",
            0, 0, cfg.period_floor, cfg.period_ceiling,
            cfg.max_period_factor, cfg.max_amplitude_factor
        )
        local_db_shimmer = call(
            [sound, point_process], "Get shimmer (local_dB)",
            0, 0, cfg.period_floor, cfg.period_ceiling,
            cfg.max_period_factor, cfg.max_amplitude_factor
        )
        apq3_shimmer = call(
            [sound, point_process], "Get shimmer (apq3)",
            0, 0, cfg.period_floor, cfg.period_ceiling,
            cfg.max_period_factor, cfg.max_amplitude_factor
        )
        apq5_shimmer = call(
            [sound, point_process], "Get shimmer (apq5)",
            0, 0, cfg.period_floor, cfg.period_ceiling,
            cfg.max_period_factor, cfg.max_amplitude_factor
        )
        apq11_shimmer = call(
            [sound, point_process], "Get shimmer (apq11)",
            0, 0, cfg.period_floor, cfg.period_ceiling,
            cfg.max_period_factor, cfg.max_amplitude_factor
        )
        dda_shimmer = call(
            [sound, point_process], "Get shimmer (dda)",
            0, 0, cfg.period_floor, cfg.period_ceiling,
            cfg.max_period_factor, cfg.max_amplitude_factor
        )
        
        return PerturbationMetrics(
            local_jitter=local_jitter,
            local_absolute_jitter=local_absolute_jitter,
            rap_jitter=rap_jitter,
            ppq5_jitter=ppq5_jitter,
            ddp_jitter=ddp_jitter,
            local_shimmer=local_shimmer,
            local_db_shimmer=local_db_shimmer,
            apq3_shimmer=apq3_shimmer,
            apq5_shimmer=apq5_shimmer,
            apq11_shimmer=apq11_shimmer,
            dda_shimmer=dda_shimmer
        )
    
    def extract_hnr(
        self,
        sound: parselmouth.Sound,
        f0_min: Optional[int] = None
    ) -> float:
        """
        Extrait le Harmonics-to-Noise Ratio.
        
        HNR mesure le rapport entre la composante harmonique et le bruit.
        Valeurs typiques: 20+ dB (voix saine), <10 dB (voix pathologique)
        
        Args:
            sound: Objet Sound Praat
            f0_min: F0 minimum
            
        Returns:
            HNR en dB
        """
        f0_min = f0_min or self.config.pitch.f0_min
        
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0_min, 0.1, 1.0)
        return call(harmonicity, "Get mean", 0, 0)
    
    def extract_formants(
        self,
        sound: parselmouth.Sound,
        f0_min: Optional[int] = None,
        f0_max: Optional[int] = None,
        return_traces: bool = False
    ) -> Union[FormantMetrics, Tuple[FormantMetrics, Dict]]:
        """
        Extrait les métriques des formants.
        
        Mesure les formants uniquement aux instants de pulsation glottique
        pour plus de précision.
        
        Args:
            sound: Objet Sound Praat
            f0_min: F0 minimum
            f0_max: F0 maximum
            return_traces: Retourner aussi les traces temporelles
            
        Returns:
            FormantMetrics, optionnellement avec traces
        """
        cfg = self.config.formant
        f0_min = f0_min or self.config.pitch.f0_min
        f0_max = f0_max or self.config.pitch.f0_max
        
        point_process = self._create_point_process(sound, f0_min, f0_max)
        formants = call(
            sound, "To Formant (burg)",
            cfg.time_step, cfg.max_formants, cfg.max_frequency,
            cfg.window_length, cfg.pre_emphasis_from
        )
        
        num_points = call(point_process, "Get number of points")
        
        times, f1_list, f2_list, f3_list = [], [], [], []
        
        for point in range(1, num_points + 1):
            t = call(point_process, "Get time from index", point)
            f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
            f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
            f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
            
            times.append(t)
            f1_list.append(np.nan if f1 is None else f1)
            f2_list.append(np.nan if f2 is None else f2)
            f3_list.append(np.nan if f3 is None else f3)
        
        # Conversion en arrays
        times = np.asarray(times, dtype=float)
        f1_list = np.asarray(f1_list, dtype=float)
        f2_list = np.asarray(f2_list, dtype=float)
        f3_list = np.asarray(f3_list, dtype=float)
        
        # Statistiques (ignorer NaN)
        metrics = FormantMetrics(
            f1_mean=np.nanmean(f1_list),
            f2_mean=np.nanmean(f2_list),
            f3_mean=np.nanmean(f3_list),
            f1_median=np.nanmedian(f1_list),
            f2_median=np.nanmedian(f2_list),
            f3_median=np.nanmedian(f3_list)
        )
        
        if return_traces:
            traces = {
                'times': times,
                'f1': f1_list,
                'f2': f2_list,
                'f3': f3_list
            }
            return metrics, traces
        
        return metrics
    
    def compute_dsi(
        self,
        sound: parselmouth.Sound,
        pitch_metrics: PitchMetrics,
        local_jitter: float
    ) -> VoiceQualityMetrics:
        """
        Calcule le Dysphonia Severity Index.
        
        DSI = 0.13*MPT + 0.0053*F0_high - 0.26*I_low - 1.18*jitter% + 12.4
        
        Interprétation:
        - DSI > 5: Voix normale
        - DSI 1-5: Dysphonie légère
        - DSI < 1: Dysphonie sévère
        
        Args:
            sound: Objet Sound Praat
            pitch_metrics: Métriques de pitch déjà calculées
            local_jitter: Jitter local déjà calculé
            
        Returns:
            VoiceQualityMetrics incluant le DSI
        """
        cfg = self.config.dsi
        
        # Intensité
        intensity = call(sound, "To Intensity", pitch_metrics.mean_f0, 0.0)
        intensity_mean = call(intensity, "Get mean", 0, 0, "energy")
        intensity_min = call(intensity, "Get minimum", 0, 0, "Parabolic")
        
        # Maximum Phonation Time
        mpt = sound.get_total_duration()
        
        # DSI
        jitter_percent = local_jitter * 100
        dsi = (
            cfg.coef_mpt * mpt +
            cfg.coef_f0_high * pitch_metrics.max_f0 +
            cfg.coef_i_low * intensity_min +
            cfg.coef_jitter * jitter_percent +
            cfg.intercept
        )
        
        # HNR
        hnr = self.extract_hnr(sound)
        
        return VoiceQualityMetrics(
            hnr=hnr,
            dsi=dsi,
            intensity_mean=intensity_mean,
            intensity_min=intensity_min,
            mpt=mpt
        )
    
    def analyze_signal(
        self,
        audio: np.ndarray,
        sr: int,
        f0_min: Optional[int] = None,
        f0_max: Optional[int] = None,
        include_traces: bool = False
    ) -> AcousticAnalysisResult:
        """
        Analyse acoustique complète d'un signal.
        
        Args:
            audio: Signal audio (numpy array)
            sr: Fréquence d'échantillonnage
            f0_min: F0 minimum
            f0_max: F0 maximum
            include_traces: Inclure les traces temporelles des formants
            
        Returns:
            AcousticAnalysisResult avec toutes les métriques
        """
        sound = self._create_sound(audio, sr)
        return self._analyze_sound(sound, f0_min, f0_max, include_traces)
    
    def analyze_file(
        self,
        file_path: str,
        f0_min: Optional[int] = None,
        f0_max: Optional[int] = None,
        include_traces: bool = False
    ) -> AcousticAnalysisResult:
        """
        Analyse acoustique complète d'un fichier audio.
        
        Args:
            file_path: Chemin vers le fichier audio
            f0_min: F0 minimum
            f0_max: F0 maximum
            include_traces: Inclure les traces temporelles
            
        Returns:
            AcousticAnalysisResult
        """
        sound = parselmouth.Sound(file_path)
        return self._analyze_sound(sound, f0_min, f0_max, include_traces)
    
    def _analyze_sound(
        self,
        sound: parselmouth.Sound,
        f0_min: Optional[int] = None,
        f0_max: Optional[int] = None,
        include_traces: bool = False
    ) -> AcousticAnalysisResult:
        """Analyse interne d'un objet Sound."""
        
        # Pitch
        pitch_metrics = self.extract_pitch_metrics(sound, f0_min, f0_max)
        
        # Perturbation
        perturbation_metrics = self.extract_perturbation_metrics(sound, f0_min, f0_max)
        
        # Formants
        if include_traces:
            formant_metrics, traces = self.extract_formants(
                sound, f0_min, f0_max, return_traces=True
            )
        else:
            formant_metrics = self.extract_formants(sound, f0_min, f0_max)
            traces = None
        
        # Voice quality et DSI
        voice_quality = self.compute_dsi(
            sound, pitch_metrics, perturbation_metrics.local_jitter
        )
        
        result = AcousticAnalysisResult(
            pitch=pitch_metrics,
            perturbation=perturbation_metrics,
            formants=formant_metrics,
            voice_quality=voice_quality
        )
        
        if include_traces and traces:
            result.f1_trace = traces['f1']
            result.f2_trace = traces['f2']
            result.f3_trace = traces['f3']
            result.time_trace = traces['times']
        
        return result


def quick_analysis(
    audio: np.ndarray,
    sr: int,
    f0_min: int = 50,
    f0_max: int = 500
) -> pd.DataFrame:
    """
    Fonction utilitaire pour une analyse rapide.
    
    Args:
        audio: Signal audio
        sr: Sample rate
        f0_min: F0 minimum
        f0_max: F0 maximum
        
    Returns:
        DataFrame avec les métriques principales
    """
    analyzer = PraatAnalyzer()
    result = analyzer.analyze_signal(audio, sr, f0_min, f0_max)
    return result.to_dataframe()
