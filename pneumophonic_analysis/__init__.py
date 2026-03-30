"""
Pneumophonic Analysis Pipeline
==============================

Package Python pour l'analyse intégrée des fonctions respiratoires-phonatoires.

Basé sur les travaux de la thèse:
"Integrated Analysis of Respiratory–Phonatory Functions: Normative Patterns Across Sex and Age"
Bianca Zocco, Politecnico di Milano, 2024-2025

Modules principaux
------------------
- config : Configuration globale du pipeline
- io_utils : Lecture/écriture des fichiers (OEP, audio, Excel)
- sync : Synchronisation entre signaux OEP et audio
- audio_processing : Traitement du signal audio (noise reduction, features)
- acoustic_features : Extraction des paramètres acoustiques via Praat
- segmentation : Segmentation des signaux (FRC, novelty, modulation)
- task_analyzers : Analyseurs spécifiques par tâche vocale
- visualization : Visualisation des signaux et résultats
- pipeline : Orchestration du pipeline complet

Exemple d'utilisation
---------------------
```python
from pneumophonic_analysis import PneumophonicPipeline, create_config
from pathlib import Path

# Configuration
config = create_config(
    data_root=Path("/path/to/data"),
    output_root=Path("/path/to/output")
)

# Pipeline
pipeline = PneumophonicPipeline(config)

# Analyse d'un sujet
result = pipeline.analyze_subject("20260218_GaBa")

# Analyse batch
batch = pipeline.analyze_batch()
pipeline.export_results(batch, "results.xlsx")
```

Utilisation rapide
------------------
```python
from pneumophonic_analysis import run_pipeline

results = run_pipeline(
    data_root="/data/subjects",
    output_root="/results",
    tasks=['vowel', 'trill']
)
```
"""

__version__ = "1.0.0"
__author__ = "Pipeline basé sur les travaux de Bianca Zocco"
__email__ = ""

# Configuration
from .config import (
    PipelineConfig,
    AudioConfig,
    PitchConfig,
    OEPConfig,
    get_config,
    create_config,
    DEFAULT_CONFIG,
)

# I/O
from .io_utils import (
    DataLoader,
    ResultsWriter,
    discover_subjects,
    load_master_excel,
    save_audio,
)

# Synchronisation
from .sync import (
    Synchronizer,
    SyncResult,
    compute_relative_timing,
    detect_onset_in_phonation,
    detect_end_of_phonation,
)

# Audio processing
from .audio_processing import (
    AudioProcessor,
    AudioFeatures,
    to_db,
    compute_spectral_centroid,
    compute_rms_envelope,
)

# Acoustic features (Praat)
from .acoustic_features import (
    PraatAnalyzer,
    AcousticAnalysisResult,
    PitchMetrics,
    PerturbationMetrics,
    FormantMetrics,
    VoiceQualityMetrics,
    quick_analysis,
)

# Segmentation
from .segmentation import (
    FRCSegmenter,
    FRCSegment,
    GlideSegmenter,
    GlideSegment,
    ModulationAnalyzer,
    ModulationResult,
    detect_non_silent_intervals,
    detect_phonation_bounds,
)

# Task analyzers
from .task_analyzers import (
    TaskResult,
    BaseTaskAnalyzer,
    VowelAnalyzer,
    PhraseAnalyzer,
    TrillAnalyzer,
    GlideAnalyzer,
    get_analyzer_for_task,
)

# Visualization
from .visualization import (
    Visualizer,
    quick_plot_audio,
    quick_plot_spectrogram,
)

# Pipeline
from .pipeline import (
    PneumophonicPipeline,
    SubjectAnalysis,
    BatchAnalysis,
    run_pipeline,
)


# API publique
__all__ = [
    # Version
    "__version__",
    
    # Configuration
    "PipelineConfig",
    "AudioConfig",
    "PitchConfig",
    "OEPConfig",
    "get_config",
    "create_config",
    "DEFAULT_CONFIG",
    
    # I/O
    "DataLoader",
    "ResultsWriter",
    "discover_subjects",
    "load_master_excel",
    "save_audio",
    
    # Sync
    "Synchronizer",
    "SyncResult",
    
    # Audio
    "AudioProcessor",
    "AudioFeatures",
    
    # Praat
    "PraatAnalyzer",
    "AcousticAnalysisResult",
    "quick_analysis",
    
    # Segmentation
    "FRCSegmenter",
    "FRCSegment",
    "GlideSegmenter",
    "GlideSegment",
    "ModulationAnalyzer",
    "ModulationResult",
    
    # Analyzers
    "TaskResult",
    "VowelAnalyzer",
    "PhraseAnalyzer",
    "TrillAnalyzer",
    "GlideAnalyzer",
    "get_analyzer_for_task",
    
    # Visualization
    "Visualizer",
    "quick_plot_audio",
    "quick_plot_spectrogram",
    
    # Pipeline
    "PneumophonicPipeline",
    "SubjectAnalysis",
    "BatchAnalysis",
    "run_pipeline",
]
