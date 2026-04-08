"""
Quick test script for paired feature extraction. Adjust paths and parameters as needed, then run to verify end-to-end functionality of the PairedFeatureExtractor module.
"""
import logging
from pathlib import Path
from pneumophonic_analysis import create_config
from pneumophonic_analysis.paired_features import PairedFeatureExtractor

logging.basicConfig(level=logging.INFO)

# ---- ADAPTE CES VALEURS ----
SUBJECT_FOLDER = Path(r"C:\Users\Matéo\...\data\20260218_GaBa")  # ton dossier sujet
OEP_CSV        = "csv/GaBaVocali.csv"   # chemin relatif au dossier sujet
AUDIO_FILE     = "a.wav"                 # un fichier audio dans renders/
TASK_NAME      = "a"
TAKE_NUMBER    = 1                       # ajuste selon le take voulu
# -----------------------------

config = create_config(
    data_root=SUBJECT_FOLDER.parent,
    output_root=Path("output")
)

extractor = PairedFeatureExtractor(config)

paired = extractor.extract(
    subject_folder=SUBJECT_FOLDER,
    task_name=TASK_NAME,
    audio_filename=AUDIO_FILE,
    oep_csv_path=OEP_CSV,
    take_number=TAKE_NUMBER,
)

df = paired.dataframe
print(f"\n✅ Matrice alignée: {df.shape[0]} frames × {df.shape[1]} features")
print(f"\nColonnes:\n{df.columns.tolist()}")
print(f"\nAperçu (5 premières lignes):\n{df.head()}")

# Vérification rapide: Vrc + Vab ≈ Vcw ?
err = (df['vrc'] + df['vab'] - df['vcw']).abs().mean()
print(f"\n🔍 Erreur moyenne |Vrc + Vab - Vcw| = {err:.6f} L")
if err < 0.01:
    print("   → Mapping compartimentaire OK ✓")
else:
    print("   → ⚠️ Vérifier le mapping des colonnes")

# Sauvegarde HDF5
h5_path = PairedFeatureExtractor.save_hdf5(paired, f"output/paired/{paired.subject_id}_{TASK_NAME}.h5")
print(f"\n💾 Sauvegardé: {h5_path}")