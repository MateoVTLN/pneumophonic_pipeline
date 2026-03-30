"""
Module d'entrées/sorties pour le pipeline pneumophonique.

Gère la lecture des fichiers:
- .dat : Données OEP (volumes, sync)
- .wav : Signaux audio
- .xlsx : Fichiers Excel de résultats et timing

Auteur: Pipeline basé sur les travaux de Bianca Zocco (2025)
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

from .config import PipelineConfig, get_config


class DataLoader:
    """
    Classe pour charger les données d'un sujet.
    
    Exemple d'utilisation:
    ```python
    loader = DataLoader(subject_folder="20260218_GaBa")
    audio, sr = loader.load_audio("a.wav")
    sync, sr_sync = loader.load_sync_signal()
    oep_data = loader.load_oep_data("csv/GaBaVocali.csv")
    ```
    """
    
    def __init__(
        self,
        subject_folder: Union[str, Path],
        config: Optional[PipelineConfig] = None,
        renders_subfolder: str = "renders"
    ):
        """
        Initialise le loader pour un sujet.
        
        Args:
            subject_folder: Chemin vers le dossier du sujet (ex: "20260218_GaBa")
            config: Configuration du pipeline (utilise DEFAULT_CONFIG si None)
            renders_subfolder: Nom du sous-dossier contenant les fichiers audio rendus
        """
        self.config = config or get_config()
        self.subject_folder = Path(subject_folder)
        self.renders_folder = self.subject_folder / renders_subfolder
        
        # Extraire l'ID du sujet depuis le nom du dossier
        # Format attendu: "YYYYMMDD_SubjectID"
        folder_name = self.subject_folder.name
        parts = folder_name.split('_')
        self.subject_id = parts[1] if len(parts) > 1 else folder_name
        self.date = parts[0] if len(parts) > 1 else None
        
    def load_audio(
        self,
        filename: str,
        sr: Optional[int] = None,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Charge un fichier audio.
        
        Args:
            filename: Nom du fichier (cherché dans renders_folder puis subject_folder)
            sr: Fréquence d'échantillonnage cible (None = originale, défaut = config)
            normalize: Normaliser l'amplitude entre -1 et 1
            
        Returns:
            Tuple (signal audio, sample rate)
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
        """
        sr = sr or self.config.audio.sample_rate
        
        # Chercher le fichier
        audio_path = self._find_file(filename, [self.renders_folder, self.subject_folder])
        
        # Charger avec librosa
        audio, sample_rate = librosa.load(str(audio_path), sr=sr)
        
        if normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
                
        return audio, sample_rate
    
    def load_sync_signal(
        self,
        filename: str = "sync_signal.wav",
        sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Charge le signal de synchronisation.
        
        Args:
            filename: Nom du fichier sync (défaut: "sync_signal.wav")
            sr: Fréquence d'échantillonnage cible
            
        Returns:
            Tuple (signal sync, sample rate)
        """
        return self.load_audio(filename, sr=sr, normalize=True)
    
    def load_oep_data(
        self,
        csv_path: str,
        fs_dat: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Charge les données OEP depuis un fichier .csv/.dat.
        
        Args:
            csv_path: Chemin relatif vers le fichier CSV
            fs_dat: Fréquence d'échantillonnage des données (pour vérification)
            
        Returns:
            DataFrame avec les colonnes OEP standard
        """
        full_path = self.subject_folder / csv_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Fichier OEP non trouvé: {full_path}")
        
        # Charger avec les noms de colonnes standard
        df = pd.read_csv(
            full_path,
            sep=' ',
            names=self.config.oep.dat_columns,
            index_col=False
        )
        
        # Vérifier la cohérence temporelle si fs_dat fourni
        if fs_dat is not None and len(df) > 1:
            dt_expected = 1 / fs_dat
            dt_actual = df['time'].iloc[1] - df['time'].iloc[0]
            if not np.isclose(dt_actual, dt_expected, rtol=0.1):
                print(f"⚠️  Attention: fs_dat attendu={fs_dat}Hz, "
                      f"calculé={1/dt_actual:.1f}Hz")
        
        return df
    
    def load_timing_excel(
        self,
        sheet_name: str = "Timing"
    ) -> pd.DataFrame:
        """
        Charge les timings depuis le fichier Excel du sujet.
        
        Args:
            sheet_name: Nom de la feuille contenant les timings
            
        Returns:
            DataFrame avec les timings par tâche
        """
        excel_path = self.subject_folder / f"{self.subject_id}_audio.xlsx"
        
        if not excel_path.exists():
            raise FileNotFoundError(f"Fichier timing non trouvé: {excel_path}")
        
        return pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    
    def get_task_timing(
        self,
        task_name: str,
        timing_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Récupère les timings pour une tâche spécifique.
        
        Args:
            task_name: Nom de la tâche (ex: "a", "r", "phrase_1")
            timing_df: DataFrame des timings (chargé automatiquement si None)
            
        Returns:
            Dict avec 't_start', 't_stop', et autres timings disponibles
        """
        if timing_df is None:
            timing_df = self.load_timing_excel()
        
        # Chercher la ligne correspondante
        mask = timing_df[0].astype(str).str.lower() == task_name.lower()
        row = timing_df[mask]
        
        if row.empty:
            raise ValueError(f"Tâche '{task_name}' non trouvée dans les timings")
        
        row = row.iloc[0]
        
        result = {
            't_start': float(row.iloc[1]) if pd.notna(row.iloc[1]) else None,
            't_stop': float(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else None,
        }
        
        # Ajouter d'autres colonnes si présentes
        for i in range(3, len(row)):
            if pd.notna(row.iloc[i]):
                result[f'col_{i}'] = row.iloc[i]
        
        return result
    
    def list_audio_files(self, pattern: str = "*.wav") -> List[Path]:
        """Liste tous les fichiers audio dans le dossier renders."""
        return list(self.renders_folder.glob(pattern))
    
    def _find_file(self, filename: str, search_paths: List[Path]) -> Path:
        """Cherche un fichier dans plusieurs chemins."""
        for path in search_paths:
            full_path = path / filename
            if full_path.exists():
                return full_path
        
        # Si le filename est déjà un chemin absolu ou relatif qui existe
        if Path(filename).exists():
            return Path(filename)
        
        raise FileNotFoundError(
            f"Fichier '{filename}' non trouvé dans: {[str(p) for p in search_paths]}"
        )


class ResultsWriter:
    """
    Classe pour sauvegarder les résultats d'analyse.
    
    Exemple:
    ```python
    writer = ResultsWriter("results/subject_analysis.xlsx")
    writer.write_metrics(df_metrics, sheet="Metrics")
    writer.write_formants(df_formants, sheet="Formants")
    writer.close()
    ```
    """
    
    def __init__(
        self,
        output_path: Union[str, Path],
        config: Optional[PipelineConfig] = None
    ):
        """
        Initialise le writer.
        
        Args:
            output_path: Chemin du fichier de sortie (.xlsx)
            config: Configuration du pipeline
        """
        self.config = config or get_config()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = None
        
    def write_dataframe(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        index: bool = True,
        startrow: int = 0,
        header: bool = True
    ):
        """
        Écrit un DataFrame dans une feuille Excel.
        
        Args:
            df: DataFrame à écrire
            sheet_name: Nom de la feuille
            index: Inclure l'index
            startrow: Ligne de départ
            header: Inclure les en-têtes
        """
        mode = 'a' if self.output_path.exists() else 'w'
        
        with pd.ExcelWriter(
            self.output_path,
            engine=self.config.output.excel_engine,
            mode=mode,
            if_sheet_exists='overlay' if mode == 'a' else None
        ) as writer:
            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=index,
                startrow=startrow,
                header=header
            )
    
    def append_row(
        self,
        df_row: pd.DataFrame,
        sheet_name: str,
        row_index: int
    ):
        """
        Ajoute une ligne à une feuille existante.
        
        Args:
            df_row: DataFrame d'une seule ligne
            sheet_name: Nom de la feuille
            row_index: Index de la ligne (1-based, comme Excel)
        """
        self.write_dataframe(
            df_row,
            sheet_name=sheet_name,
            index=False,
            startrow=row_index - 1,  # Conversion vers 0-based
            header=False
        )


def save_audio(
    audio: np.ndarray,
    path: Union[str, Path],
    sr: int,
    normalize: bool = True
):
    """
    Sauvegarde un signal audio en fichier WAV.
    
    Args:
        audio: Signal audio (1D numpy array)
        path: Chemin de sortie
        sr: Fréquence d'échantillonnage
        normalize: Normaliser avant sauvegarde
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if normalize:
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    
    sf.write(str(path), audio, sr)


def discover_subjects(
    data_root: Union[str, Path],
    pattern: str = "*_*"
) -> List[Path]:
    """
    Découvre tous les dossiers de sujets dans un répertoire.
    
    Args:
        data_root: Répertoire racine contenant les dossiers sujets
        pattern: Pattern glob pour les dossiers (défaut: format "DATE_ID")
        
    Returns:
        Liste des chemins vers les dossiers sujets
    """
    data_root = Path(data_root)
    
    subjects = []
    for folder in sorted(data_root.glob(pattern)):
        if folder.is_dir():
            # Vérifier que c'est bien un dossier sujet (contient renders/)
            if (folder / "renders").exists() or (folder / "sync_signal.wav").exists():
                subjects.append(folder)
    
    return subjects


def load_master_excel(
    path: Union[str, Path],
    sheet_name: str = "cross"
) -> pd.DataFrame:
    """
    Charge le fichier Excel master contenant les données de tous les sujets.
    
    Args:
        path: Chemin vers le fichier Excel master
        sheet_name: Nom de la feuille à charger
        
    Returns:
        DataFrame avec les données master
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    return df
