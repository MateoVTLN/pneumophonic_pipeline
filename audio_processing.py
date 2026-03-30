"""
Module de traitement du signal audio.

Fonctionnalités:
- Réduction du bruit (noisereduce)
- Filtre de pré-emphasis
- Extraction de features spectrales (STFT, Mel, MFCCs)
- Estimation de F0 (pyin)

Référence: Thèse Zocco 2025, Section 3.5.3 - Acoustic Signal Processing
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
    
    # Représentations spectrales
    stft: np.ndarray                  # Spectre de puissance
    mel_spectrogram: np.ndarray       # Mel-spectrogramme
    mfcc: np.ndarray                  # Coefficients MFCC
    
    # Pitch
    f0: np.ndarray                    # Fréquence fondamentale
    voiced_flag: np.ndarray           # Indicateur voix/non-voix
    voiced_probs: np.ndarray          # Probabilités de voisement
    
    # Harmoniques
    f0_harmonics: Optional[np.ndarray] = None
    
    # Métadonnées
    sample_rate: int = 48000
    hop_length: int = 720
    frame_length: int = 1440
    
    @property
    def time_axis(self) -> np.ndarray:
        """Retourne l'axe temporel en secondes."""
        n_frames = self.f0.shape[0]
        return np.arange(n_frames) * self.hop_length / self.sample_rate


class AudioProcessor:
    """
    Processeur audio pour le pipeline pneumophonique.
    
    Exemple:
    ```python
    processor = AudioProcessor(config)
    
    # Réduction du bruit
    audio_clean = processor.reduce_noise(audio, sr)
    
    # Pré-emphasis
    audio_pe = processor.apply_pre_emphasis(audio_clean)
    
    # Extraction complète
    features = processor.extract_features(audio_pe, sr)
    ```
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialise le processeur.
        
        Args:
            config: Configuration du pipeline
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
        Applique la réduction de bruit au signal.
        
        Utilise l'algorithme spectral de la librairie noisereduce.
        Le bruit OEP (caméras IR, ventilation) est principalement stationnaire.
        
        Args:
            audio: Signal audio brut
            sr: Fréquence d'échantillonnage
            stationary: Utiliser le mode stationnaire (défaut: config)
            prop_decrease: Proportion de réduction (0-1, défaut: config)
            
        Returns:
            Signal audio débruité
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
        
        # Normalisation
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
        Applique un filtre de pré-emphasis.
        
        Le filtre de pré-emphasis amplifie les hautes fréquences
        pour compenser l'atténuation naturelle du spectre vocal.
        Équation: y[n] = x[n] - coef * x[n-1]
        
        Args:
            audio: Signal audio
            coef: Coefficient de pré-emphasis (défaut: 0.97)
            normalize: Normaliser l'énergie après filtrage
            
        Returns:
            Signal filtré
        """
        coef = coef if coef is not None else self.config.audio.pre_emphasis_coef
        
        # Filtre de pré-emphasis
        audio_pe = np.append(audio[0], audio[1:] - coef * audio[:-1])
        
        if normalize:
            # Normalisation par la norme L2 (énergie unitaire)
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
        Calcule le spectrogramme de puissance.
        
        Args:
            audio: Signal audio
            n_fft: Taille de la FFT (défaut: config frame_length)
            hop_length: Taille du hop (défaut: config)
            power: Exposant pour le spectre de puissance
            
        Returns:
            Spectrogramme de puissance |STFT|^power
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
        Calcule le mel-spectrogramme.
        
        Args:
            audio: Signal audio
            sr: Fréquence d'échantillonnage
            n_mels: Nombre de bandes mel (défaut: config)
            n_fft: Taille de la FFT
            hop_length: Taille du hop
            
        Returns:
            Mel-spectrogramme
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
            audio: Signal audio
            sr: Fréquence d'échantillonnage
            n_mfcc: Nombre de coefficients (défaut: 13)
            normalize: Normaliser (mean=0, std=1) par coefficient
            
        Returns:
            Matrice MFCC (n_mfcc x n_frames)
        """
        n_mfcc = n_mfcc or self.config.audio.n_mfcc
        normalize = normalize if normalize is not None else self.config.audio.normalize_mfcc
        
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        if normalize:
            # Normalisation z-score par coefficient
            mean = np.mean(mfcc, axis=1, keepdims=True)
            std = np.std(mfcc, axis=1, keepdims=True)
            std[std == 0] = 1  # Éviter division par zéro
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
        Estime la fréquence fondamentale avec pYIN.
        
        Args:
            audio: Signal audio
            sr: Fréquence d'échantillonnage
            fmin: F0 minimum (défaut: 50 Hz)
            fmax: F0 maximum (défaut: 500 Hz)
            hop_length: Taille du hop
            
        Returns:
            Tuple (f0, voiced_flag, voiced_probs)
            - f0: Fréquence fondamentale (NaN si non-voisé)
            - voiced_flag: Booléen voisé/non-voisé
            - voiced_probs: Probabilité de voisement
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
        Extrait l'énergie des harmoniques de F0.
        
        Args:
            stft_power: Spectrogramme de puissance
            f0: Fréquence fondamentale
            sr: Sample rate
            n_fft: Taille FFT utilisée
            harmonics: Indices des harmoniques (défaut: [1, 2, 3])
            
        Returns:
            Énergie par harmonique (n_harmonics x n_frames)
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
        Extrait toutes les features audio.
        
        Pipeline complet:
        1. STFT → Spectrogramme de puissance
        2. Mel-spectrogram
        3. MFCCs
        4. F0 (pYIN)
        5. Harmoniques F0 (optionnel)
        
        Args:
            audio: Signal audio (pré-traité recommandé)
            sr: Fréquence d'échantillonnage
            include_harmonics: Calculer les harmoniques de F0
            
        Returns:
            AudioFeatures avec toutes les représentations
        """
        n_fft = self.config.audio.frame_length_samples
        hop_length = self.config.audio.hop_length_samples
        
        # Spectrogrammes
        stft = self.compute_stft(audio, n_fft=n_fft, hop_length=hop_length)
        mel = self.compute_mel_spectrogram(audio, sr, n_fft=n_fft, hop_length=hop_length)
        mfcc = self.compute_mfcc(audio, sr)
        
        # Pitch
        f0, voiced_flag, voiced_probs = self.estimate_f0(audio, sr, hop_length=hop_length)
        
        # Harmoniques
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
        Pipeline complet de traitement.
        
        Args:
            audio: Signal audio brut
            sr: Fréquence d'échantillonnage
            apply_noise_reduction: Appliquer réduction de bruit
            apply_pre_emphasis: Appliquer pré-emphasis
            
        Returns:
            Tuple (audio traité, features)
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
    Convertit un spectrogramme en dB.
    
    Args:
        spectrogram: Spectrogramme linéaire
        ref: Valeur de référence
        amin: Amplitude minimale (évite log(0))
        
    Returns:
        Spectrogramme en dB
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
    
    Le centroïde spectral représente le "centre de masse" du spectre.
    Utilisé pour la novelty function dans l'analyse du glissando.
    
    Args:
        audio: Signal audio
        sr: Sample rate
        n_fft: Taille FFT
        hop_length: Hop length
        
    Returns:
        Centroïde spectral (Hz) par frame
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
    Calcule l'enveloppe RMS du signal.
    
    Args:
        audio: Signal audio
        frame_length: Taille de la fenêtre
        hop_length: Taille du hop
        
    Returns:
        Enveloppe RMS
    """
    return librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
