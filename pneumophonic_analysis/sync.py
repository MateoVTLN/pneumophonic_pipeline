"""
Synchronisation Module for the temporal alignment OEP/Audio.

The synchronisation is based on the detection of the sync signal (falling edge)
present in both the OEP data and the audio recording.

Reference: Zocco Thesis 2025, Section 3.5 - Data Processing and Analysis
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
    """Result of the synchronization."""

    # Position of the falling edge in the audio sync signal (in samples)
    sync_falling_edge_sample: int

    # Position of the falling edge in the OEP data (in OEP samples)
    oep_falling_edge_sample: int

    # Time offset to apply (in seconds)
    time_offset_sec: float

    # Timestamps of detected onsets
    sync_onsets_samples: np.ndarray
    oep_sync_onsets_samples: np.ndarray

    # Synchronization quality
    correlation: float = 0.0


class Synchronizer:
    """
    Synchronizes OEP and audio signals via the sync signal.

    The protocol uses a synchronization signal (square pulse)
    sent simultaneously to the OEP (analog channel) and the audio DAW.
    Alignment is performed on the falling edge of this signal.

    Example:
    ```python
    sync = Synchronizer(config)

    # Detect the falling edge in the audio
    audio_offset = sync.detect_sync_in_audio(sync_signal, sr=48000)

    # Detect in the OEP data
    oep_offset = sync.detect_sync_in_oep(oep_df, take_number=1)

    # Align the data
    aligned_oep = sync.align_oep_to_audio(oep_df, audio_length, audio_offset, oep_offset)
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initializes the synchronizer.

        Args:
            config: Pipeline configuration
        """
        self.config = config or get_config()
    
    def detect_falling_edge_audio(
        self,
        sync_signal: np.ndarray,
        sr: int,
        onset_index: int = 1
    ) -> int:
        """
        Detects the falling edge in the audio sync signal.

        The sync signal is a square pulse. librosa.onset.onset_detect is used
        to find the transitions, then the one corresponding to the
        falling edge is selected (typically the 2nd detected onset).

        Args:
            sync_signal: Audio synchronization signal
            sr: Sample rate
            onset_index: Index of the onset to use (0=first, 1=second/falling)

        Returns:
            Position of the falling edge in samples
        """
        # Detect onsets
        onsets = librosa.onset.onset_detect(
            y=sync_signal,
            sr=sr,
            units='samples'
        )
        
        if len(onsets) <= onset_index:
            raise ValueError(
                f"Not enough onsets detected ({len(onsets)}). "
                f"Expected at least {onset_index + 1}."
            )
        
        return onsets[onset_index]
    
    def detect_falling_edge_threshold(
        self,
        sync_signal: np.ndarray,
        threshold: float = 0.5
    ) -> int:
        """
        Detects the falling edge by thresholding.

        Alternative to the onset_detect method, useful for signals
        with a high signal-to-noise ratio.

        Args:
            sync_signal: Normalized signal [-1, 1]
            threshold: Detection threshold

        Returns:
            Position of the falling edge in samples
        """
        sync_norm = sync_signal / (np.max(np.abs(sync_signal)) + 1e-9)

        # Find where the signal drops below the threshold
        falling_edges = np.where(
            (sync_norm[:-1] > threshold) & (sync_norm[1:] < threshold)
        )[0]
        
        if len(falling_edges) == 0:
            raise ValueError("No falling edge detected with the specified threshold")
        
        return falling_edges[0]
    
    def detect_sync_onsets_oep(
        self,
        oep_df: pd.DataFrame,
        prominence: Optional[float] = None
    ) -> np.ndarray:
        """
        Detects all peaks of the sync signal in the OEP data.

        Each take in the protocol generates a pair of peaks (rising + falling edge).

        Args:
            oep_df: DataFrame with a 'sync' column
            prominence: Minimum peak prominence (default: config)

        Returns:
            Array of peak indices
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
        Retrieves the falling edge for a specific take.

        Convention: Take 1 → peak index 1 (2*1-1=1)
                   Take 2 → peak index 3 (2*2-1=3)
                   etc.

        Args:
            oep_df: OEP DataFrame
            take_number: Take number (1-based)
            fs_oep: OEP sample rate

        Returns:
            Tuple (index in samples, time in seconds)
        """
        fs_oep = fs_oep or self.config.oep.fs_kinematic
        
        peaks = self.detect_sync_onsets_oep(oep_df)
        
        # Falling edge index for this take
        peak_index = 2 * take_number - 1
        
        if peak_index >= len(peaks):
            raise ValueError(
                f"Take {take_number} not found. "
                f"Only {len(peaks) // 2} takes detected."
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
        Fully synchronizes the audio and OEP data.

        Args:
            sync_audio: Audio sync signal
            sr_audio: Audio sample rate
            oep_df: OEP DataFrame
            take_number: Take number
            fs_oep: OEP sample rate
            audio_onset_index: Index of the audio onset to use

        Returns:
            SyncResult with all synchronization parameters
        """
        fs_oep = fs_oep or self.config.oep.fs_kinematic
        
        # Audio falling edge
        audio_falling_sample = self.detect_falling_edge_audio(
            sync_audio, sr_audio, audio_onset_index
        )
        audio_falling_time = audio_falling_sample / sr_audio

        # OEP falling edge
        oep_falling_sample, oep_falling_time = self.get_take_falling_edge_oep(
            oep_df, take_number, fs_oep
        )

        # Time offset
        time_offset = audio_falling_time - oep_falling_time

        # Detected onsets
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
        Converts an audio time to an OEP index.

        Args:
            audio_time_sec: Time in the audio reference frame
            sync_result: Synchronization result
            fs_oep: OEP sample rate

        Returns:
            Index in the OEP DataFrame
        """
        fs_oep = fs_oep or self.config.oep.fs_kinematic
        
        # Time relative to the audio falling edge
        relative_time = audio_time_sec - (sync_result.sync_falling_edge_sample / self.config.audio.sample_rate)

        # Convert to OEP sample
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
        Extracts an OEP segment corresponding to an audio portion.

        Args:
            oep_df: Full OEP DataFrame
            audio_start_sec: Segment start (audio time)
            audio_end_sec: Segment end (audio time)
            sync_result: Synchronization result
            fs_oep: OEP sample rate

        Returns:
            OEP DataFrame for the segment
        """
        fs_oep = fs_oep or self.config.oep.fs_kinematic
        
        start_sample = self.convert_audio_time_to_oep_sample(
            audio_start_sec, sync_result, fs_oep
        )
        end_sample = self.convert_audio_time_to_oep_sample(
            audio_end_sec, sync_result, fs_oep
        )
        
        # Clamp to DataFrame bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(oep_df), end_sample)
        
        return oep_df.iloc[start_sample:end_sample].copy()


def compute_relative_timing(
    audio_sample: int,
    sync_falling_edge_sample: int,
    sr: int
) -> float:
    """
    Computes the time relative to the falling edge.

    Args:
        audio_sample: Position in audio samples
        sync_falling_edge_sample: Position of the falling edge
        sr: Sample rate

    Returns:
        Time in seconds relative to the falling edge
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
    Detects the onset of phonation in an audio signal.

    Uses librosa onset detection. The index allows skipping
    the first onsets that may correspond to the sync signal.

    Args:
        audio: Audio signal (noise-reduced recommended)
        sr: Sample rate
        hop_length: Hop size for analysis
        delta: Detection sensitivity
        onset_index: Index of the onset to return (to skip the sync)

    Returns:
        Position of phonation onset in samples
    """
    onsets = librosa.onset.onset_detect(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        delta=delta,
        units='samples'
    )
    
    if len(onsets) <= onset_index:
        # Fallback: return the last available onset
        return onsets[-1] if len(onsets) > 0 else 0
    
    return onsets[onset_index]


def detect_end_of_phonation(
    audio: np.ndarray,
    sr: int,
    top_db: int = 45,
    interval_index: int = -1
) -> int:
    """
    Detects the end of phonation.

    Uses librosa.effects.split to find non-silent intervals.

    Args:
        audio: Audio signal
        sr: Sample rate
        top_db: Threshold for silence detection
        interval_index: Interval index (-1 = last)

    Returns:
        Position of phonation end in samples
    """
    intervals = librosa.effects.split(audio, top_db=top_db)
    
    if len(intervals) == 0:
        return len(audio)
    
    return intervals[interval_index, 1]

def detect_phonation_bounds(audio, sr, top_db=30):
    """
    Detects the start and end indices of phonation in an audio signal.
    Uses librosa.effects.split to ignore silence at the beginning and end.
    """
    intervals = librosa.effects.split(audio, top_db=top_db)
    
    if len(intervals) == 0:
        # If nothing obvious is found, return the full signal
        return 0, len(audio)

    # Take the start of the first non-silent interval and the end of the last
    start_idx = intervals[0][0]
    end_idx = intervals[-1][1]
    
    return start_idx, end_idx