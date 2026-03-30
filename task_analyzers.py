"""
Analyseurs spécifiques pour chaque tâche du protocole vocal.

Tâches couvertes:
- VowelAnalyzer: Voyelles soutenues (A-Long, 5 vowels)
- PhraseAnalyzer: Phrases et lecture de texte
- TrillAnalyzer: Roulée alvéolaire (R)
- GlideAnalyzer: Glissando vocal (A-Glide)

Chaque analyseur hérite de BaseTaskAnalyzer et implémente
la méthode analyze() spécifique à la tâche.

Référence: Thèse Zocco 2025, Chapitre 4 - Results
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
    """Résultat générique d'analyse d'une tâche."""
    
    subject_id: str
    task_name: str
    
    # Métriques acoustiques
    acoustic_result: Optional[AcousticAnalysisResult] = None
    
    # Features spectrales
    audio_features: Optional[AudioFeatures] = None
    
    # Segmentation FRC (si applicable)
    frc_segment: Optional[FRCSegment] = None
    acoustic_above_frc: Optional[AcousticAnalysisResult] = None
    acoustic_below_frc: Optional[AcousticAnalysisResult] = None
    
    # Métadonnées
    duration_sec: float = 0.0
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    
    # Données brutes (optionnel)
    audio_processed: Optional[np.ndarray] = None
    sample_rate: int = 48000
    
    # Métriques additionnelles spécifiques à la tâche
    extra_metrics: Dict = field(default_factory=dict)
    
    def to_dataframe(self, prefix: str = "") -> pd.DataFrame:
        """Convertit le résultat en DataFrame."""
        data = {
            f'{prefix}subject_id': self.subject_id,
            f'{prefix}task': self.task_name,
            f'{prefix}duration_sec': self.duration_sec,
            f'{prefix}start_time': self.start_time_sec,
            f'{prefix}end_time': self.end_time_sec,
        }
        
        # Ajouter les métriques acoustiques
        if self.acoustic_result:
            acoustic_df = self.acoustic_result.to_dataframe(prefix)
            for col in acoustic_df.columns:
                data[col] = acoustic_df[col].iloc[0]
        
        # Ajouter les métriques extra
        for key, val in self.extra_metrics.items():
            data[f'{prefix}{key}'] = val
        
        return pd.DataFrame([data])


class BaseTaskAnalyzer(ABC):
    """
    Classe de base pour les analyseurs de tâches.
    
    Fournit les méthodes communes et définit l'interface
    que chaque analyseur spécifique doit implémenter.
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
        Analyse une tâche spécifique.
        
        Args:
            audio: Signal audio (pré-traité ou brut)
            sr: Sample rate
            subject_id: Identifiant du sujet
            **kwargs: Paramètres spécifiques à la tâche
            
        Returns:
            TaskResult avec les métriques
        """
        pass
    
    def preprocess_audio(
        self,
        audio: np.ndarray,
        sr: int,
        apply_noise_reduction: bool = True,
        apply_pre_emphasis: bool = True
    ) -> np.ndarray:
        """Pré-traitement standard de l'audio."""
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
        """Détecte les bornes de phonation."""
        return detect_phonation_bounds(audio, sr)


class VowelAnalyzer(BaseTaskAnalyzer):
    """
    Analyseur pour les voyelles soutenues.
    
    Tâches: A-Long (AL), Voyelles (5 vowels × 5s)
    
    Métriques extraites:
    - F0 (mean, std, min, max)
    - Jitter (local, RAP, PPQ5, DDP)
    - Shimmer (local, APQ3, APQ5, APQ11, DDA)
    - HNR
    - DSI
    - Formants (F1, F2, F3)
    
    Exemple:
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
        Analyse une voyelle soutenue.
        
        Args:
            audio: Signal audio
            sr: Sample rate
            subject_id: ID du sujet
            vowel: Voyelle analysée ('a', 'e', 'i', 'o', 'u')
            preprocess: Appliquer le pré-traitement
            include_frc: Inclure l'analyse FRC
            frc_cross_time: Temps de crossing FRC (secondes)
            
        Returns:
            TaskResult
        """
        # Pré-traitement
        if preprocess:
            audio_proc = self.preprocess_audio(audio, sr)
        else:
            audio_proc = audio
        
        # Détection des bornes
        start_sample, end_sample = self.detect_bounds(audio_proc, sr)
        audio_segment = audio_proc[start_sample:end_sample]
        
        # Analyse acoustique principale
        acoustic = self.praat_analyzer.analyze_signal(audio_segment, sr)
        
        # Features spectrales
        features = self.audio_processor.extract_features(audio_segment, sr)
        
        # Résultat de base
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
        
        # Analyse FRC si demandée
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
            
            # Analyse séparée above/below FRC
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
    Analyseur pour les phrases et la lecture de texte.
    
    Tâches: Phrases (5 phrases), TEXT (lecture de texte)
    
    Particularités:
    - Détection de pause
    - Analyse par segments (optionnel)
    - Métriques prosodiques
    
    Exemple:
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
        Analyse une phrase ou texte.
        
        Args:
            audio: Signal audio
            sr: Sample rate
            subject_id: ID du sujet
            phrase_id: Numéro de la phrase (1-5)
            task_name: Nom de la tâche ('phrase', 'text')
            preprocess: Appliquer le pré-traitement
            analyze_segments: Analyser chaque segment séparément
            
        Returns:
            TaskResult
        """
        # Pré-traitement avec réduction de bruit plus forte
        if preprocess:
            audio_proc = self.audio_processor.reduce_noise(
                audio, sr, prop_decrease=0.95
            )
            audio_proc = self.audio_processor.apply_pre_emphasis(audio_proc)
        else:
            audio_proc = audio
        
        # Détection des bornes
        start_sample, end_sample = self.detect_bounds(audio_proc, sr)
        audio_segment = audio_proc[start_sample:end_sample]
        
        # Analyse acoustique
        acoustic = self.praat_analyzer.analyze_signal(audio_segment, sr)
        
        # Nom de la tâche
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
        
        # Métriques prosodiques additionnelles
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
        Estime le débit de parole (syllabes/seconde approximatif).
        
        Utilise la détection d'onset comme proxy pour les syllabes.
        """
        import librosa
        
        onsets = librosa.onset.onset_detect(y=audio, sr=sr)
        duration = len(audio) / sr
        
        if duration > 0:
            return len(onsets) / duration
        return 0.0


class TrillAnalyzer(BaseTaskAnalyzer):
    """
    Analyseur pour la roulée alvéolaire (trille R).
    
    Tâche: R
    
    Métriques spécifiques:
    - Fréquence de modulation (20-30 Hz typiquement)
    - Analyse FRC (above/below)
    - Comptage de cycles (optionnel)
    
    Exemple:
    ```python
    analyzer = TrillAnalyzer()
    result = analyzer.analyze(
        audio, sr=48000, subject_id="RoDi",
        frc_cross_time=3.5  # Temps du crossing FRC
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
        Analyse la roulée R.
        
        Args:
            audio: Signal audio
            sr: Sample rate
            subject_id: ID du sujet
            preprocess: Appliquer le pré-traitement
            frc_cross_time: Temps du crossing FRC
            start_time: Début de phonation
            end_time: Fin de phonation
            
        Returns:
            TaskResult avec fréquence de modulation
        """
        # Pré-traitement
        if preprocess:
            audio_proc = self.preprocess_audio(audio, sr)
        else:
            audio_proc = audio
        
        # Détection des bornes si non fournies
        if start_time is None or end_time is None:
            start_sample, end_sample = self.detect_bounds(audio_proc, sr)
            start_time = start_time or (start_sample / sr)
            end_time = end_time or (end_sample / sr)
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        audio_segment = audio_proc[start_sample:end_sample]
        
        # Analyse acoustique de base
        acoustic = self.praat_analyzer.analyze_signal(audio_segment, sr)
        
        # Résultat de base
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
        
        # Analyse de modulation
        if frc_cross_time is not None:
            # Avec segmentation FRC
            mod_result = self.modulation_analyzer.analyze_with_frc(
                audio_proc, sr,
                cross_time=frc_cross_time,
                start_time=start_time,
                end_time=end_time
            )
            
            # Segmentation FRC
            frc_segment = self.frc_segmenter.segment_by_time(
                audio_proc, frc_cross_time, start_time, end_time, sr
            )
            result.frc_segment = frc_segment
            
            # Analyse acoustique par segment
            if len(frc_segment.above_frc) > sr * 0.1:
                result.acoustic_above_frc = self.praat_analyzer.analyze_signal(
                    frc_segment.above_frc, sr
                )
            if len(frc_segment.below_frc) > sr * 0.1:
                result.acoustic_below_frc = self.praat_analyzer.analyze_signal(
                    frc_segment.below_frc, sr
                )
        else:
            # Sans segmentation FRC
            mod_result = ModulationResult(
                frequency_full=self.modulation_analyzer.compute_modulation_frequency(
                    audio_segment, sr
                ),
                frequency_above_frc=np.nan,
                frequency_below_frc=np.nan
            )
        
        # Ajouter les métriques de modulation
        result.extra_metrics['mod_freq_full'] = mod_result.frequency_full
        result.extra_metrics['mod_freq_above_frc'] = mod_result.frequency_above_frc
        result.extra_metrics['mod_freq_below_frc'] = mod_result.frequency_below_frc
        
        # Comptage d'onsets (proxy pour cycles de roulée)
        result.extra_metrics['onset_count'] = self._count_onsets(audio_segment, sr)
        
        return result
    
    def _count_onsets(self, audio: np.ndarray, sr: int) -> int:
        """Compte le nombre d'onsets (cycles de roulée)."""
        import librosa
        onsets = librosa.onset.onset_detect(y=audio, sr=sr)
        return len(onsets)


class GlideAnalyzer(BaseTaskAnalyzer):
    """
    Analyseur pour le glissando vocal (A-Glide).
    
    Tâche: AG (A-Glide)
    
    Le glissando consiste en une montée progressive de F0
    du registre grave au registre aigu.
    
    Métriques spécifiques:
    - Séparation P1 (basses fréquences) / P2 (hautes fréquences)
    - Analyse séparée des deux parties
    - Range de F0
    
    Exemple:
    ```python
    analyzer = GlideAnalyzer()
    result = analyzer.analyze(audio, sr=48000, subject_id="CaBl")
    
    # Accès aux métriques P1/P2
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
        Analyse le glissando.
        
        Args:
            audio: Signal audio
            sr: Sample rate
            subject_id: ID du sujet
            preprocess: Appliquer le pré-traitement
            peak_time_override: Position manuelle du pic de transition
            
        Returns:
            TaskResult avec métriques P1/P2
        """
        # Pré-traitement
        if preprocess:
            audio_proc = self.preprocess_audio(audio, sr)
        else:
            audio_proc = audio
        
        # Segmentation du glissando
        peak_sample = int(peak_time_override * sr) if peak_time_override else None
        glide_segment = self.glide_segmenter.segment_glide(
            audio_proc, sr, peak_override=peak_sample
        )
        
        # Analyse acoustique globale
        full_segment = np.concatenate([glide_segment.part1, glide_segment.part2])
        acoustic_full = self.praat_analyzer.analyze_signal(full_segment, sr)
        
        # Résultat de base
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
        
        # Analyse P1 (basses fréquences)
        if len(glide_segment.part1) > sr * 0.1:
            acoustic_p1 = self.praat_analyzer.analyze_signal(glide_segment.part1, sr)
            result.extra_metrics['P1_meanF0'] = acoustic_p1.pitch.mean_f0
            result.extra_metrics['P1_maxF0'] = acoustic_p1.pitch.max_f0
            result.extra_metrics['P1_duration'] = glide_segment.duration_p1
            result.extra_metrics['P1_HNR'] = acoustic_p1.voice_quality.hnr
            result.extra_metrics['P1_jitter'] = acoustic_p1.perturbation.local_jitter
        
        # Analyse P2 (hautes fréquences)
        if len(glide_segment.part2) > sr * 0.1:
            acoustic_p2 = self.praat_analyzer.analyze_signal(glide_segment.part2, sr)
            result.extra_metrics['P2_meanF0'] = acoustic_p2.pitch.mean_f0
            result.extra_metrics['P2_minF0'] = acoustic_p2.pitch.min_f0
            result.extra_metrics['P2_duration'] = glide_segment.duration_p2
            result.extra_metrics['P2_HNR'] = acoustic_p2.voice_quality.hnr
            result.extra_metrics['P2_jitter'] = acoustic_p2.perturbation.local_jitter
        
        # Métriques du glissando
        result.extra_metrics['peak_time'] = glide_segment.peak_time
        result.extra_metrics['peak_novelty'] = glide_segment.peak_novelty_value
        
        # Range de F0
        if 'P1_meanF0' in result.extra_metrics and 'P2_meanF0' in result.extra_metrics:
            result.extra_metrics['F0_range'] = (
                result.extra_metrics['P2_meanF0'] - result.extra_metrics['P1_meanF0']
            )
        
        return result


def get_analyzer_for_task(task_name: str, config: Optional[PipelineConfig] = None) -> BaseTaskAnalyzer:
    """
    Factory function pour obtenir l'analyseur approprié.
    
    Args:
        task_name: Nom de la tâche ('vowel', 'phrase', 'trill', 'glide', 'text')
        config: Configuration du pipeline
        
    Returns:
        Instance de l'analyseur approprié
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
        raise ValueError(f"Tâche inconnue: {task_name}. "
                        f"Tâches supportées: {list(analyzers.keys())}")
    
    return analyzers[task_lower](config)
