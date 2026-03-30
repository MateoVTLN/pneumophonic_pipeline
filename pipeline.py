"""
Pipeline principal d'orchestration pour l'analyse pneumophonique.

Ce module coordonne l'analyse complète:
- Découverte des sujets
- Chargement des données
- Synchronisation OEP/Audio
- Analyse par tâche
- Export des résultats

Référence: Thèse Zocco 2025 - Protocole d'analyse
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from tqdm import tqdm

from .config import PipelineConfig, get_config, create_config
from .io_utils import DataLoader, ResultsWriter, discover_subjects, load_master_excel
from .sync import Synchronizer, SyncResult
from .audio_processing import AudioProcessor
from .acoustic_features import PraatAnalyzer
from .task_analyzers import (
    TaskResult, BaseTaskAnalyzer,
    VowelAnalyzer, PhraseAnalyzer, TrillAnalyzer, GlideAnalyzer,
    get_analyzer_for_task
)
from .visualization import Visualizer


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SubjectAnalysis:
    """Résultat de l'analyse complète d'un sujet."""
    
    subject_id: str
    subject_folder: Path
    
    # Résultats par tâche
    results: Dict[str, TaskResult] = field(default_factory=dict)
    
    # Synchronisation
    sync_result: Optional[SyncResult] = None
    
    # Erreurs éventuelles
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Combine tous les résultats en un DataFrame."""
        dfs = []
        for task_name, result in self.results.items():
            df = result.to_dataframe()
            dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()


@dataclass
class BatchAnalysis:
    """Résultat d'une analyse batch sur plusieurs sujets."""
    
    subjects: Dict[str, SubjectAnalysis] = field(default_factory=dict)
    
    @property
    def n_subjects(self) -> int:
        return len(self.subjects)
    
    @property
    def n_successful(self) -> int:
        return sum(1 for s in self.subjects.values() if s.success)
    
    @property
    def n_failed(self) -> int:
        return self.n_subjects - self.n_successful
    
    def to_dataframe(self) -> pd.DataFrame:
        """Combine tous les résultats en un seul DataFrame."""
        dfs = []
        for subject_id, analysis in self.subjects.items():
            df = analysis.to_dataframe()
            if not df.empty:
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def get_errors_summary(self) -> pd.DataFrame:
        """Résumé des erreurs."""
        data = []
        for subject_id, analysis in self.subjects.items():
            for error in analysis.errors:
                data.append({
                    'subject_id': subject_id,
                    'type': 'error',
                    'message': error
                })
            for warning in analysis.warnings:
                data.append({
                    'subject_id': subject_id,
                    'type': 'warning',
                    'message': warning
                })
        return pd.DataFrame(data)


class PneumophonicPipeline:
    """
    Pipeline principal pour l'analyse pneumophonique.
    
    Orchestre l'analyse complète d'un ou plusieurs sujets,
    coordonnant les différents modules du package.
    
    Exemple d'utilisation:
    ```python
    # Configuration
    config = create_config(
        data_root=Path("/data/subjects"),
        output_root=Path("/results")
    )
    
    # Pipeline
    pipeline = PneumophonicPipeline(config)
    
    # Analyse d'un sujet
    result = pipeline.analyze_subject("20260218_GaBa")
    
    # Analyse batch
    batch_result = pipeline.analyze_all()
    
    # Export
    pipeline.export_results(batch_result, "all_results.xlsx")
    ```
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        data_root: Optional[Path] = None,
        output_root: Optional[Path] = None
    ):
        """
        Initialise le pipeline.
        
        Args:
            config: Configuration du pipeline
            data_root: Répertoire racine des données (override config)
            output_root: Répertoire de sortie (override config)
        """
        self.config = config or get_config()
        
        if data_root:
            self.config.data_root = Path(data_root)
        if output_root:
            self.config.output_root = Path(output_root)
        
        # Composants
        self.audio_processor = AudioProcessor(self.config)
        self.praat_analyzer = PraatAnalyzer(self.config)
        self.synchronizer = Synchronizer(self.config)
        self.visualizer = Visualizer(self.config)
        
        # Analyseurs par tâche
        self.analyzers: Dict[str, BaseTaskAnalyzer] = {
            'vowel': VowelAnalyzer(self.config),
            'phrase': PhraseAnalyzer(self.config),
            'trill': TrillAnalyzer(self.config),
            'glide': GlideAnalyzer(self.config),
        }
        
        logger.info(f"Pipeline initialisé")
        logger.info(f"  Data root: {self.config.data_root}")
        logger.info(f"  Output root: {self.config.output_root}")
    
    def discover_subjects(self) -> List[Path]:
        """
        Découvre les sujets disponibles.
        
        Returns:
            Liste des chemins vers les dossiers sujets
        """
        if self.config.data_root is None:
            raise ValueError("data_root non configuré")
        
        subjects = discover_subjects(self.config.data_root)
        logger.info(f"Découvert {len(subjects)} sujets")
        
        return subjects
    
    def analyze_subject(
        self,
        subject_folder: Union[str, Path],
        tasks: Optional[List[str]] = None,
        timing_excel: Optional[Path] = None,
        save_figures: bool = False
    ) -> SubjectAnalysis:
        """
        Analyse complète d'un sujet.
        
        Args:
            subject_folder: Chemin vers le dossier du sujet
            tasks: Liste des tâches à analyser (None = toutes)
            timing_excel: Fichier Excel avec les timings
            save_figures: Sauvegarder les figures
            
        Returns:
            SubjectAnalysis avec tous les résultats
        """
        subject_folder = Path(subject_folder)
        loader = DataLoader(subject_folder, self.config)
        
        analysis = SubjectAnalysis(
            subject_id=loader.subject_id,
            subject_folder=subject_folder
        )
        
        logger.info(f"Analyse du sujet {loader.subject_id}")
        
        # Charger les timings si disponibles
        timing_data = None
        if timing_excel is not None:
            try:
                timing_data = pd.read_excel(timing_excel)
            except Exception as e:
                analysis.warnings.append(f"Impossible de charger les timings: {e}")
        
        # Synchronisation (si sync_signal.wav existe)
        try:
            sync_audio, sr_sync = loader.load_sync_signal()
            # Note: la sync complète nécessite aussi les données OEP
            logger.debug(f"Signal sync chargé: {len(sync_audio)} samples")
        except FileNotFoundError:
            analysis.warnings.append("Signal sync non trouvé")
            sync_audio = None
        
        # Analyser chaque tâche
        tasks_to_analyze = tasks or ['vowel', 'phrase', 'trill', 'glide']
        
        for task_name in tasks_to_analyze:
            try:
                result = self._analyze_task(
                    loader, task_name, timing_data
                )
                if result:
                    analysis.results[task_name] = result
                    
                    if save_figures and self.config.output_root:
                        self._save_task_figure(result, loader.subject_id)
                        
            except Exception as e:
                error_msg = f"Erreur sur tâche {task_name}: {str(e)}"
                analysis.errors.append(error_msg)
                logger.error(error_msg)
        
        return analysis
    
    def _analyze_task(
        self,
        loader: DataLoader,
        task_name: str,
        timing_data: Optional[pd.DataFrame] = None
    ) -> Optional[TaskResult]:
        """Analyse une tâche spécifique."""
        
        # Mapping des fichiers audio par tâche
        task_files = {
            'vowel': ['a.wav', 'vocali_a.wav', 'AL.wav'],
            'phrase': ['phrase_1.wav', 'phrase_5.wav', 'frase_1.wav'],
            'trill': ['r.wav', 'R.wav', 'trill.wav'],
            'glide': ['glide.wav', 'AG.wav', 'phonema_a_7.wav'],
        }
        
        # Trouver le fichier audio
        audio_file = None
        for candidate in task_files.get(task_name, []):
            try:
                audio, sr = loader.load_audio(candidate)
                audio_file = candidate
                break
            except FileNotFoundError:
                continue
        
        if audio_file is None:
            logger.warning(f"Fichier audio non trouvé pour {task_name}")
            return None
        
        # Obtenir l'analyseur
        analyzer = self.analyzers.get(task_name)
        if analyzer is None:
            analyzer = get_analyzer_for_task(task_name, self.config)
        
        # Paramètres additionnels depuis les timings
        kwargs = {}
        if timing_data is not None:
            # Chercher les timings pour cette tâche
            # (Structure dépend du format du fichier Excel)
            pass
        
        # Analyser
        result = analyzer.analyze(
            audio=audio,
            sr=sr,
            subject_id=loader.subject_id,
            **kwargs
        )
        
        logger.info(f"  {task_name}: {result.duration_sec:.2f}s")
        
        return result
    
    def _save_task_figure(self, result: TaskResult, subject_id: str):
        """Sauvegarde la figure d'un résultat."""
        if self.config.output_root is None:
            return
        
        fig = self.visualizer.plot_task_result(result)
        
        output_path = (
            self.config.output_root / 
            "figures" / 
            subject_id / 
            f"{result.task_name}.png"
        )
        
        self.visualizer.save_figure(fig, output_path)
    
    def analyze_batch(
        self,
        subjects: Optional[List[Path]] = None,
        tasks: Optional[List[str]] = None,
        progress: bool = True,
        stop_on_error: bool = False
    ) -> BatchAnalysis:
        """
        Analyse un lot de sujets.
        
        Args:
            subjects: Liste des sujets (None = tous)
            tasks: Tâches à analyser
            progress: Afficher la barre de progression
            stop_on_error: Arrêter en cas d'erreur
            
        Returns:
            BatchAnalysis avec tous les résultats
        """
        if subjects is None:
            subjects = self.discover_subjects()
        
        batch = BatchAnalysis()
        
        iterator = tqdm(subjects, desc="Analyse") if progress else subjects
        
        for subject_folder in iterator:
            try:
                analysis = self.analyze_subject(
                    subject_folder,
                    tasks=tasks
                )
                batch.subjects[analysis.subject_id] = analysis
                
            except Exception as e:
                logger.error(f"Erreur sujet {subject_folder}: {e}")
                if stop_on_error:
                    raise
        
        logger.info(f"Batch terminé: {batch.n_successful}/{batch.n_subjects} réussis")
        
        return batch
    
    def export_results(
        self,
        results: Union[SubjectAnalysis, BatchAnalysis],
        output_path: Union[str, Path],
        include_errors: bool = True
    ):
        """
        Exporte les résultats vers Excel.
        
        Args:
            results: Résultats à exporter
            output_path: Chemin du fichier de sortie
            include_errors: Inclure la feuille des erreurs
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Données principales
            df_main = results.to_dataframe()
            if not df_main.empty:
                df_main.to_excel(writer, sheet_name='Results', index=False)
            
            # Erreurs
            if include_errors and isinstance(results, BatchAnalysis):
                df_errors = results.get_errors_summary()
                if not df_errors.empty:
                    df_errors.to_excel(writer, sheet_name='Errors', index=False)
        
        logger.info(f"Résultats exportés: {output_path}")
    
    def generate_report(
        self,
        results: BatchAnalysis,
        output_folder: Union[str, Path],
        include_figures: bool = True
    ):
        """
        Génère un rapport complet avec figures.
        
        Args:
            results: Résultats de l'analyse batch
            output_folder: Dossier de sortie
            include_figures: Inclure les figures
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Export Excel
        self.export_results(results, output_folder / "results.xlsx")
        
        # Figures de synthèse
        if include_figures:
            df = results.to_dataframe()
            
            if not df.empty and 'pitch_mean_f0' in df.columns:
                fig = self.visualizer.plot_metrics_comparison(
                    df, 'pitch_mean_f0', 'subject_id', 'Mean F0 by Subject'
                )
                self.visualizer.save_figure(
                    fig, output_folder / "figures" / "mean_f0_comparison.png"
                )
        
        logger.info(f"Rapport généré: {output_folder}")


def run_pipeline(
    data_root: Union[str, Path],
    output_root: Union[str, Path],
    tasks: Optional[List[str]] = None,
    subjects: Optional[List[str]] = None
) -> BatchAnalysis:
    """
    Fonction de commodité pour exécuter le pipeline complet.
    
    Args:
        data_root: Répertoire des données
        output_root: Répertoire de sortie
        tasks: Tâches à analyser
        subjects: Sujets spécifiques (None = tous)
        
    Returns:
        BatchAnalysis
    
    Exemple:
    ```python
    results = run_pipeline(
        data_root="/data/subjects",
        output_root="/results",
        tasks=['vowel', 'trill']
    )
    print(f"Analysé {results.n_subjects} sujets")
    ```
    """
    config = create_config(
        data_root=Path(data_root),
        output_root=Path(output_root)
    )
    
    pipeline = PneumophonicPipeline(config)
    
    # Filtrer les sujets si spécifiés
    if subjects:
        subject_folders = [
            Path(data_root) / s for s in subjects
        ]
    else:
        subject_folders = None
    
    # Analyser
    batch = pipeline.analyze_batch(subjects=subject_folders, tasks=tasks)
    
    # Exporter
    pipeline.generate_report(batch, output_root)
    
    return batch
