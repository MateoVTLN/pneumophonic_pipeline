"""
Task-specific analyzers for each vocal protocol task.

Tasks covered:
- VowelAnalyzer: Sustained vowels (A-Long, 5 vowels)
- PhraseAnalyzer: Phrases and text reading
- TrillAnalyzer: Alveolar trill (R)
- GlideAnalyzer: Vocal glide (A-Glide)

Each analyzer inherits from BaseTaskAnalyzer and implements
the task-specific analyze() method.

Reference: Zocco Thesis 2025, Chapter 4 - Results
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .config import PipelineConfig, get_config
from .io_utils import DataLoader, ResultsWriter
from .sync import Synchronizer, SyncResult, detect_phonation_bounds
from .audio_processing import AudioProcessor, AudioFeatures
from .acoustic_features import PraatAnalyzer, AcousticAnalysisResult
from .segmentation import (
    FRCSegmenter, FRCSegment,
    GlideSegmenter, GlideSegment,
    ModulationAnalyzer, ModulationResult
)


@dataclass
class TaskResult:
    """Generic task analysis result."""

    subject_id: str
    task_name: str

    # Acoustic metrics
    acoustic_result: Optional[AcousticAnalysisResult] = None

    # Spectral features
    audio_features: Optional[AudioFeatures] = None

    # FRC segmentation (if applicable)
    frc_segment: Optional[FRCSegment] = None
    acoustic_above_frc: Optional[AcousticAnalysisResult] = None
    acoustic_below_frc: Optional[AcousticAnalysisResult] = None

    # Metadata
    duration_sec: float = 0.0
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0

    # Raw data (optional)
    audio_processed: Optional[np.ndarray] = None
    sample_rate: int = 48000

    # Additional task-specific metrics
    extra_metrics: Dict = field(default_factory=dict)
    
    def to_dataframe(self, prefix: str = "") -> pd.DataFrame:
        """Converts the result to a DataFrame."""
        data = {
            f'{prefix}subject_id': self.subject_id,
            f'{prefix}task': self.task_name,
            f'{prefix}duration_sec': self.duration_sec,
            f'{prefix}start_time': self.start_time_sec,
            f'{prefix}end_time': self.end_time_sec,
        }

        # Add acoustic metrics
        if self.acoustic_result:
            acoustic_df = self.acoustic_result.to_dataframe(prefix)
            for col in acoustic_df.columns:
                data[col] = acoustic_df[col].iloc[0]

        # Add extra metrics
        for key, val in self.extra_metrics.items():
            data[f'{prefix}{key}'] = val
        
        return pd.DataFrame([data])


class BaseTaskAnalyzer(ABC):
    """
    Base class for task analyzers.

    Provides common methods and defines the interface
    that each specific analyzer must implement.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_config()
        self.audio_processor = AudioProcessor(self.config)
        self.praat_analyzer = PraatAnalyzer(self.config)
        self.synchronizer = Synchronizer(self.config)
    
    @abstractmethod
    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        subject_id: str,
        **kwargs
    ) -> TaskResult:
        """
        Analyzes a specific task.

        Args:
            audio: Audio signal (preprocessed or raw)
            sr: Sample rate
            subject_id: Subject identifier
            **kwargs: Task-specific parameters

        Returns:
            TaskResult with metrics
        """
        pass
    
    def preprocess_audio(
        self,
        audio: np.ndarray,
        sr: int,
        apply_noise_reduction: bool = True,
        apply_pre_emphasis: bool = True
    ) -> np.ndarray:
        """Standard audio preprocessing."""
        processed = audio.copy()
        
        if apply_noise_reduction:
            processed = self.audio_processor.reduce_noise(processed, sr)
        
        if apply_pre_emphasis:
            processed = self.audio_processor.apply_pre_emphasis(processed)
        
        return processed
    
    def detect_bounds(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[int, int]:
        """Detects phonation bounds."""
        return detect_phonation_bounds(audio, sr)


class VowelAnalyzer(BaseTaskAnalyzer):
    """
    Analyzer for sustained vowels.

    Tasks: A-Long (AL), Vowels (5 vowels × 5s)

    Extracted metrics:
    - F0 (mean, std, min, max)
    - Jitter (local, RAP, PPQ5, DDP)
    - Shimmer (local, APQ3, APQ5, APQ11, DDA)
    - HNR
    - DSI
    - Formants (F1, F2, F3)

    Example:
    ```python
    analyzer = VowelAnalyzer()
    result = analyzer.analyze(audio, sr=48000, subject_id="GaBa", vowel="a")
    df = result.to_dataframe()
    ```
    """
    
    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        subject_id: str,
        vowel: str = "a",
        preprocess: bool = True,
        include_frc: bool = False,
        frc_cross_time: Optional[float] = None,
        **kwargs
    ) -> TaskResult:
        """
        Analyzes a sustained vowel.

        Args:
            audio: Audio signal
            sr: Sample rate
            subject_id: Subject ID
            vowel: Vowel being analyzed ('a', 'e', 'i', 'o', 'u')
            preprocess: Apply preprocessing
            include_frc: Include FRC analysis
            frc_cross_time: FRC crossing time (seconds)

        Returns:
            TaskResult
        """
        # Preprocessing
        if preprocess:
            audio_proc = self.preprocess_audio(audio, sr)
        else:
            audio_proc = audio

        # Bounds detection
        start_sample, end_sample = self.detect_bounds(audio_proc, sr)
        audio_segment = audio_proc[start_sample:end_sample]

        # Main acoustic analysis
        acoustic = self.praat_analyzer.analyze_signal(audio_segment, sr)

        # Spectral features
        features = self.audio_processor.extract_features(audio_segment, sr)

        # Base result
        result = TaskResult(
            subject_id=subject_id,
            task_name=f"vowel_{vowel}",
            acoustic_result=acoustic,
            audio_features=features,
            duration_sec=len(audio_segment) / sr,
            start_time_sec=start_sample / sr,
            end_time_sec=end_sample / sr,
            audio_processed=audio_segment,
            sample_rate=sr
        )
        
        # FRC analysis if requested
        if include_frc and frc_cross_time is not None:
            frc_segmenter = FRCSegmenter(self.config)
            frc_segment = frc_segmenter.segment_by_time(
                audio_proc,
                cross_time=frc_cross_time,
                start_time=start_sample / sr,
                end_time=end_sample / sr,
                sr=sr
            )
            result.frc_segment = frc_segment

            # Separate above/below FRC analysis
            if len(frc_segment.above_frc) > sr * 0.1:  # Min 100ms
                result.acoustic_above_frc = self.praat_analyzer.analyze_signal(
                    frc_segment.above_frc, sr
                )
            if len(frc_segment.below_frc) > sr * 0.1:
                result.acoustic_below_frc = self.praat_analyzer.analyze_signal(
                    frc_segment.below_frc, sr
                )
        
        return result


class PhraseAnalyzer(BaseTaskAnalyzer):
    """
    Analyzer for phrases and text reading.

    Tasks: Phrases (5 phrases), TEXT (text reading)

    Specifics:
    - Pause detection
    - Segment-by-segment analysis (optional)
    - Prosodic metrics

    Example:
    ```python
    analyzer = PhraseAnalyzer()
    result = analyzer.analyze(audio, sr=48000, subject_id="AnMa", phrase_id=5)
    ```
    """
    
    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        subject_id: str,
        phrase_id: Optional[int] = None,
        task_name: str = "phrase",
        preprocess: bool = True,
        analyze_segments: bool = False,
        **kwargs
    ) -> TaskResult:
        """
        Analyzes a phrase or text.

        Args:
            audio: Audio signal
            sr: Sample rate
            subject_id: Subject ID
            phrase_id: Phrase number (1-5)
            task_name: Task name ('phrase', 'text')
            preprocess: Apply preprocessing
            analyze_segments: Analyze each segment separately

        Returns:
            TaskResult
        """
        # Preprocessing with stronger noise reduction
        if preprocess:
            audio_proc = self.audio_processor.reduce_noise(
                audio, sr, prop_decrease=0.95
            )
            audio_proc = self.audio_processor.apply_pre_emphasis(audio_proc)
        else:
            audio_proc = audio

        # Bounds detection
        start_sample, end_sample = self.detect_bounds(audio_proc, sr)
        audio_segment = audio_proc[start_sample:end_sample]

        # Acoustic analysis
        acoustic = self.praat_analyzer.analyze_signal(audio_segment, sr)

        # Task name
        if phrase_id:
            full_task_name = f"{task_name}_{phrase_id}"
        else:
            full_task_name = task_name
        
        result = TaskResult(
            subject_id=subject_id,
            task_name=full_task_name,
            acoustic_result=acoustic,
            duration_sec=len(audio_segment) / sr,
            start_time_sec=start_sample / sr,
            end_time_sec=end_sample / sr,
            audio_processed=audio_segment,
            sample_rate=sr
        )
        
        # Additional prosodic metrics
        result.extra_metrics['speech_rate'] = self._estimate_speech_rate(
            audio_segment, sr
        )
        
        return result
    
    def _estimate_speech_rate(
        self,
        audio: np.ndarray,
        sr: int
    ) -> float:
        """
        Estimates speech rate (approximate syllables/second).

        Uses onset detection as a proxy for syllables.
        """
        import librosa
        
        onsets = librosa.onset.onset_detect(y=audio, sr=sr)
        duration = len(audio) / sr
        
        if duration > 0:
            return len(onsets) / duration
        return 0.0


class TrillAnalyzer(BaseTaskAnalyzer):
    """
    Analyzer for the alveolar trill (R trill).

    Task: R

    Specific metrics:
    - Modulation frequency (typically 20-30 Hz)
    - FRC analysis (above/below)
    - Cycle counting (optional)

    Example:
    ```python
    analyzer = TrillAnalyzer()
    result = analyzer.analyze(
        audio, sr=48000, subject_id="RoDi",
        frc_cross_time=3.5  # FRC crossing time
    )
    print(f"Modulation: {result.extra_metrics['mod_freq_full']:.1f} Hz")
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(config)
        self.modulation_analyzer = ModulationAnalyzer(self.config)
        self.frc_segmenter = FRCSegmenter(self.config)
    
    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        subject_id: str,
        preprocess: bool = True,
        frc_cross_time: Optional[float] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        **kwargs
    ) -> TaskResult:
        """
        Analyzes the R trill.

        Args:
            audio: Audio signal
            sr: Sample rate
            subject_id: Subject ID
            preprocess: Apply preprocessing
            frc_cross_time: FRC crossing time
            start_time: Phonation start
            end_time: Phonation end

        Returns:
            TaskResult with modulation frequency
        """
        # Preprocessing
        if preprocess:
            audio_proc = self.preprocess_audio(audio, sr)
        else:
            audio_proc = audio

        # Detect bounds if not provided
        if start_time is None or end_time is None:
            start_sample, end_sample = self.detect_bounds(audio_proc, sr)
            start_time = start_time or (start_sample / sr)
            end_time = end_time or (end_sample / sr)

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_segment = audio_proc[start_sample:end_sample]

        # Basic acoustic analysis
        acoustic = self.praat_analyzer.analyze_signal(audio_segment, sr)

        # Base result
        result = TaskResult(
            subject_id=subject_id,
            task_name="trill_r",
            acoustic_result=acoustic,
            duration_sec=len(audio_segment) / sr,
            start_time_sec=start_time,
            end_time_sec=end_time,
            audio_processed=audio_segment,
            sample_rate=sr
        )
        
        # Modulation analysis
        if frc_cross_time is not None:
            # With FRC segmentation
            mod_result = self.modulation_analyzer.analyze_with_frc(
                audio_proc, sr,
                cross_time=frc_cross_time,
                start_time=start_time,
                end_time=end_time
            )

            # FRC segmentation
            frc_segment = self.frc_segmenter.segment_by_time(
                audio_proc, frc_cross_time, start_time, end_time, sr
            )
            result.frc_segment = frc_segment

            # Acoustic analysis per segment
            if len(frc_segment.above_frc) > sr * 0.1:
                result.acoustic_above_frc = self.praat_analyzer.analyze_signal(
                    frc_segment.above_frc, sr
                )
            if len(frc_segment.below_frc) > sr * 0.1:
                result.acoustic_below_frc = self.praat_analyzer.analyze_signal(
                    frc_segment.below_frc, sr
                )
        else:
            # Without FRC segmentation
            mod_result = ModulationResult(
                frequency_full=self.modulation_analyzer.compute_modulation_frequency(
                    audio_segment, sr
                ),
                frequency_above_frc=np.nan,
                frequency_below_frc=np.nan
            )
        
        # Add modulation metrics
        result.extra_metrics['mod_freq_full'] = mod_result.frequency_full
        result.extra_metrics['mod_freq_above_frc'] = mod_result.frequency_above_frc
        result.extra_metrics['mod_freq_below_frc'] = mod_result.frequency_below_frc

        # Onset count (proxy for trill cycles)
        result.extra_metrics['onset_count'] = self._count_onsets(audio_segment, sr)
        
        return result
    
    def _count_onsets(self, audio: np.ndarray, sr: int) -> int:
        """Counts the number of onsets (trill cycles)."""
        import librosa
        onsets = librosa.onset.onset_detect(y=audio, sr=sr)
        return len(onsets)


class GlideAnalyzer(BaseTaskAnalyzer):
    """
    Analyzer for the vocal glide (A-Glide).

    Task: AG (A-Glide)

    The glide consists of a progressive rise in F0
    from the low register to the high register.

    Specific metrics:
    - P1 (low frequencies) / P2 (high frequencies) separation
    - Separate analysis of each part
    - F0 range

    Example:
    ```python
    analyzer = GlideAnalyzer()
    result = analyzer.analyze(audio, sr=48000, subject_id="CaBl")

    # Access P1/P2 metrics
    print(f"F0 P1: {result.extra_metrics['P1_meanF0']:.1f} Hz")
    print(f"F0 P2: {result.extra_metrics['P2_meanF0']:.1f} Hz")
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__(config)
        self.glide_segmenter = GlideSegmenter(self.config)
    
    def analyze(
        self,
        audio: np.ndarray,
        sr: int,
        subject_id: str,
        preprocess: bool = True,
        peak_time_override: Optional[float] = None,
        **kwargs
    ) -> TaskResult:
        """
        Analyzes the glide.

        Args:
            audio: Audio signal
            sr: Sample rate
            subject_id: Subject ID
            preprocess: Apply preprocessing
            peak_time_override: Manual position of the transition peak

        Returns:
            TaskResult with P1/P2 metrics
        """
        # Preprocessing
        if preprocess:
            audio_proc = self.preprocess_audio(audio, sr)
        else:
            audio_proc = audio

        # Glide segmentation
        peak_sample = int(peak_time_override * sr) if peak_time_override else None
        glide_segment = self.glide_segmenter.segment_glide(
            audio_proc, sr, peak_override=peak_sample
        )

        # Global acoustic analysis
        full_segment = np.concatenate([glide_segment.part1, glide_segment.part2])
        acoustic_full = self.praat_analyzer.analyze_signal(full_segment, sr)

        # Base result
        result = TaskResult(
            subject_id=subject_id,
            task_name="glide",
            acoustic_result=acoustic_full,
            duration_sec=glide_segment.duration_p1 + glide_segment.duration_p2,
            start_time_sec=glide_segment.start_sample / sr,
            end_time_sec=glide_segment.end_sample / sr,
            audio_processed=full_segment,
            sample_rate=sr
        )
        
        # P1 analysis (low frequencies)
        if len(glide_segment.part1) > sr * 0.1:
            acoustic_p1 = self.praat_analyzer.analyze_signal(glide_segment.part1, sr)
            result.extra_metrics['P1_meanF0'] = acoustic_p1.pitch.mean_f0
            result.extra_metrics['P1_maxF0'] = acoustic_p1.pitch.max_f0
            result.extra_metrics['P1_duration'] = glide_segment.duration_p1
            result.extra_metrics['P1_HNR'] = acoustic_p1.voice_quality.hnr
            result.extra_metrics['P1_jitter'] = acoustic_p1.perturbation.local_jitter
        
        # P2 analysis (high frequencies)
        if len(glide_segment.part2) > sr * 0.1:
            acoustic_p2 = self.praat_analyzer.analyze_signal(glide_segment.part2, sr)
            result.extra_metrics['P2_meanF0'] = acoustic_p2.pitch.mean_f0
            result.extra_metrics['P2_minF0'] = acoustic_p2.pitch.min_f0
            result.extra_metrics['P2_duration'] = glide_segment.duration_p2
            result.extra_metrics['P2_HNR'] = acoustic_p2.voice_quality.hnr
            result.extra_metrics['P2_jitter'] = acoustic_p2.perturbation.local_jitter
        
        # Glide metrics
        result.extra_metrics['peak_time'] = glide_segment.peak_time
        result.extra_metrics['peak_novelty'] = glide_segment.peak_novelty_value

        # F0 range
        if 'P1_meanF0' in result.extra_metrics and 'P2_meanF0' in result.extra_metrics:
            result.extra_metrics['F0_range'] = (
                result.extra_metrics['P2_meanF0'] - result.extra_metrics['P1_meanF0']
            )
        
        return result


def get_analyzer_for_task(task_name: str, config: Optional[PipelineConfig] = None) -> BaseTaskAnalyzer:
    """
    Factory function to get the appropriate analyzer.

    Args:
        task_name: Task name ('vowel', 'phrase', 'trill', 'glide', 'text')
        config: Pipeline configuration

    Returns:
        Instance of the appropriate analyzer
    """
    analyzers = {
        'vowel': VowelAnalyzer,
        'al': VowelAnalyzer,
        'a_long': VowelAnalyzer,
        'phrase': PhraseAnalyzer,
        'text': PhraseAnalyzer,
        'txt': PhraseAnalyzer,
        'trill': TrillAnalyzer,
        'r': TrillAnalyzer,
        'glide': GlideAnalyzer,
        'ag': GlideAnalyzer,
        'a_glide': GlideAnalyzer,
    }
    
    task_lower = task_name.lower().replace('-', '_')
    
    if task_lower not in analyzers:
        raise ValueError(f"Unknown task: {task_name}. "
                        f"Supported tasks: {list(analyzers.keys())}")
    
    return analyzers[task_lower](config)
