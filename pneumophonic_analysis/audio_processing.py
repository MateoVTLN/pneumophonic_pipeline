"""
Audio signal processing module.

Features:
- Noise reduction (noisereduce)
- Pre-emphasis filter
- Spectral feature extraction (STFT, Mel, MFCCs)
- F0 estimation (pyin)

Reference: Zocco Thesis 2025, Section 3.5.3 - Acoustic Signal Processing
"""

import numpy as np
import librosa
import noisereduce as nr
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .config import PipelineConfig, get_config


@dataclass
class AudioFeatures:
    """Container pour les features audio extraites."""
    
    # Spectral representations
    stft: np.ndarray                  # Power spectrum
    mel_spectrogram: np.ndarray       # Mel spectrogram
    mfcc: np.ndarray                  # MFCC coefficients
    
    # Pitch
    f0: np.ndarray                    # Fundamental frequency
    voiced_flag: np.ndarray           # Voiced/unvoiced indicator
    voiced_probs: np.ndarray          # Voicing probabilities
    
    # Harmonics
    f0_harmonics: Optional[np.ndarray] = None
    
    # Metadata
    sample_rate: int = 48000
    hop_length: int = 720
    frame_length: int = 1440
    
    @property
    def time_axis(self) -> np.ndarray:
        """Returns the temporal axis in seconds."""
        n_frames = self.f0.shape[0]
        return np.arange(n_frames) * self.hop_length / self.sample_rate


class AudioProcessor:
    """
    Audio processor for the pneumophonic pipeline.
    
    Example usage:
    ```python
    processor = AudioProcessor(config)
    
    # Noise reduction
    audio_clean = processor.reduce_noise(audio, sr)
    
    # Pre-emphasis
    audio_pe = processor.apply_pre_emphasis(audio_clean)
    
    # Complete feature extraction
    features = processor.extract_features(audio_pe, sr)
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initializes the processor.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or get_config()
    
    def reduce_noise(
        self,
        audio: np.ndarray,
        sr: int,
        stationary: Optional[bool] = None,
        prop_decrease: Optional[float] = None
    ) -> np.ndarray:
        """
        Applies noise reduction to the audio signal.
        
        Uses the spectral algorithm from the noisereduce library.
        OEP noise (IR cameras, ventilation) is primarily stationary.
        
        Args:
            audio: Raw audio signal
            sr: Sampling frequency
            stationary: Use stationary mode (default: config)
            prop_decrease: Proportion of reduction (0-1, default: config)
            
        Returns:
            Denoised audio signal
        """
        stationary = stationary if stationary is not None else \
            self.config.audio.noise_reduction_stationary
        prop_decrease = prop_decrease if prop_decrease is not None else \
            self.config.audio.noise_reduction_prop_decrease
        
        audio_nr = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=stationary,
            prop_decrease=prop_decrease
        )
        
        # Normalization
        max_val = np.max(np.abs(audio_nr))
        if max_val > 0:
            audio_nr = audio_nr / max_val
        
        return audio_nr
    
    def apply_pre_emphasis(
        self,
        audio: np.ndarray,
        coef: Optional[float] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Applies a pre-emphasis filter.
        
        The pre-emphasis filter amplifies high frequencies
        to compensate for the natural attenuation of the vocal spectrum.
        Equation: y[n] = x[n] - coef * x[n-1]
        
        Args:
            audio: Audio signal
            coef: Pre-emphasis coefficient (default: 0.97)
            normalize: Normalize energy after filtering
            
        Returns:
            Filtered signal
        """
        coef = coef if coef is not None else self.config.audio.pre_emphasis_coef
        
        # Pre-emphasis filter implementation
        audio_pe = np.append(audio[0], audio[1:] - coef * audio[:-1])
        
        if normalize:
            # Normalization by L2 norm (unit energy)
            norm = np.linalg.norm(audio_pe)
            if norm > 0:
                audio_pe = audio_pe / norm
        
        return audio_pe
    
    def compute_stft(
        self,
        audio: np.ndarray,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        power: float = 2.0
    ) -> np.ndarray:
        """
        Computes the power spectrogram.
        
        Args:
            audio: Audio signal
            n_fft: FFT size (default: config frame_length)
            hop_length: Hop size (default: config)
            power: Exponent for power spectrum
            
        Returns:
            Power spectrogram |STFT|^power
        """
        n_fft = n_fft or self.config.audio.frame_length_samples
        hop_length = hop_length or self.config.audio.hop_length_samples
        
        stft = librosa.stft(
            y=audio,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length
        )
        
        return np.abs(stft) ** power
    
    def compute_mel_spectrogram(
        self,
        audio: np.ndarray,
        sr: int,
        n_mels: Optional[int] = None,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Computes the mel-spectrogram.
        
        Args:
            audio: Audio signal
            sr: Sampling frequency
            n_mels: Number of mel bands (default: config)
            n_fft: FFT size (default: config)
            hop_length: Hop size (default: config)
            
        Returns:
            Mel-spectrogram
        """
        n_mels = n_mels or self.config.audio.n_mels
        n_fft = n_fft or self.config.audio.frame_length_samples
        hop_length = hop_length or self.config.audio.hop_length_samples
        
        return librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
    
    def compute_mfcc(
        self,
        audio: np.ndarray,
        sr: int,
        n_mfcc: Optional[int] = None,
        normalize: Optional[bool] = None
    ) -> np.ndarray:
        """
            Calcule les coefficients MFCC.
        
        Args:
            audio:  Audio signal
            sr: Sampling frequency
            n_mfcc: Number of coefficients (default: 13)
            normalize: Normalize (mean=0, std=1) by coefficient
            
        Returns:
            Matrix MFCC (n_mfcc x n_frames)
        """
        n_mfcc = n_mfcc or self.config.audio.n_mfcc
        normalize = normalize if normalize is not None else self.config.audio.normalize_mfcc
        
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        if normalize:
            # Normalisation z-score by coefficient
            mean = np.mean(mfcc, axis=1, keepdims=True)
            std = np.std(mfcc, axis=1, keepdims=True)
            std[std == 0] = 1  # Avoid division by zero
            mfcc = (mfcc - mean) / std
        
        return mfcc
    
    def estimate_f0(
        self,
        audio: np.ndarray,
        sr: int,
        fmin: Optional[int] = None,
        fmax: Optional[int] = None,
        hop_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimates the fundamental frequency using pYIN.
        
        Args:
            audio: Audio signal
            sr: Sampling frequency
            fmin: Minimum F0 (default: 50 Hz)
            fmax: Maximum F0 (default: 500 Hz)
            hop_length: Hop size
            
        Returns:
            Tuple (f0, voiced_flag, voiced_probs)
            - f0: Fundamental frequency (NaN if unvoiced)
            - voiced_flag: Voiced/unvoiced boolean
            - voiced_probs: Voicing probability
        """
        fmin = fmin or self.config.pitch.f0_min
        fmax = fmax or self.config.pitch.f0_max
        hop_length = hop_length or self.config.audio.hop_length_samples
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y=audio,
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            n_thresholds=self.config.pitch.pyin_n_thresholds
        )
        
        return f0, voiced_flag, voiced_probs
    
    def compute_f0_harmonics(
        self,
        stft_power: np.ndarray,
        f0: np.ndarray,
        sr: int,
        n_fft: int,
        harmonics: np.ndarray = None
    ) -> np.ndarray:
        """
        Extracts the energy of F0 harmonics.
        
        Args:
            stft_power: Power spectrogram
            f0: Fundamental frequency
            sr: Sample rate
            n_fft: FFT size used
            harmonics: Harmonic indices (default: [1, 2, 3])
            
        Returns:
            Energy per harmonic (n_harmonics x n_frames)
        """
        if harmonics is None:
            harmonics = np.arange(1, 4)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        return librosa.f0_harmonics(
            stft_power,
            freqs=freqs,
            f0=f0,
            harmonics=harmonics
        )
    
    def extract_features(
        self,
        audio: np.ndarray,
        sr: int,
        include_harmonics: bool = True
    ) -> AudioFeatures:
        """
        Extracts all audio features.
        
        Complete pipeline:
        1. STFT → Power spectrogram
        2. Mel spectrogram
        3. MFCCs
        4. F0 (pYIN)
        5. F0 harmonics (optional)
        
        Args:
            audio: Audio signal (pre-processed recommended)
            sr: Sampling frequency
            include_harmonics: Compute F0 harmonics
            
        Returns:
            AudioFeatures with all representations
        """
        n_fft = self.config.audio.frame_length_samples
        hop_length = self.config.audio.hop_length_samples
        
        # Spectrograms
        stft = self.compute_stft(audio, n_fft=n_fft, hop_length=hop_length)
        mel = self.compute_mel_spectrogram(audio, sr, n_fft=n_fft, hop_length=hop_length)
        mfcc = self.compute_mfcc(audio, sr)
        
        # Pitch
        f0, voiced_flag, voiced_probs = self.estimate_f0(audio, sr, hop_length=hop_length)
        
        # Harmonics
        f0_harm = None
        if include_harmonics:
            f0_harm = self.compute_f0_harmonics(stft, f0, sr, n_fft)
        
        return AudioFeatures(
            stft=stft,
            mel_spectrogram=mel,
            mfcc=mfcc,
            f0=f0,
            voiced_flag=voiced_flag,
            voiced_probs=voiced_probs,
            f0_harmonics=f0_harm,
            sample_rate=sr,
            hop_length=hop_length,
            frame_length=n_fft
        )
    
    def process_full_pipeline(
        self,
        audio: np.ndarray,
        sr: int,
        apply_noise_reduction: bool = True,
        apply_pre_emphasis: bool = True
    ) -> Tuple[np.ndarray, AudioFeatures]:
        """
        Complete processing pipeline.
        
        Args:
            audio: Raw audio signal
            sr: Sampling frequency
            apply_noise_reduction: Apply noise reduction
            apply_pre_emphasis: Apply pre-emphasis
            
        Returns:
            Tuple (processed audio, features)
        """
        processed = audio.copy()
        
        if apply_noise_reduction:
            processed = self.reduce_noise(processed, sr)
        
        if apply_pre_emphasis:
            processed = self.apply_pre_emphasis(processed)
        
        features = self.extract_features(processed, sr)
        
        return processed, features


def to_db(
    spectrogram: np.ndarray,
    ref: float = 1.0,
    amin: float = 1e-10
) -> np.ndarray:
    """
    Converts a spectrogram to dB.
    
    Args:
        spectrogram: Linear spectrogram
        ref: Reference value
        amin: Minimum amplitude (avoids log(0))
        
    Returns:
        Spectrogram in dB
    """
    return 10 * np.log10(np.maximum(spectrogram, amin) / ref)


def compute_spectral_centroid(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
        Calcule le centroïde spectral.
    
    The spectral centroid represents the "center of mass" of the spectrum.
    Used for the novelty function in glissando analysis.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        
    Returns:
        Spectral centroid (Hz) per frame
    """
    stft = np.abs(librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length))
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    return centroid


def compute_rms_envelope(
    audio: np.ndarray,
    frame_length: int = 512,
    hop_length: int = 128
) -> np.ndarray:
    """
    Computes the RMS envelope of the signal.
    
    Args:
        audio: Audio signal
        frame_length: Window size
        hop_length: Hop size
        
    Returns:
        RMS envelope
    """
    return librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
