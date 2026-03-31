# Pneumophonic Analysis Pipeline (not definitive : will include other methods, analysis in the future)

Modular Python pipeline for the integrated analysis of respiratory-phonatory functions + speech signals.
Opto-electronic Plethysmography + Audio Acquisitions made by Bianca ZOCCO
## Description

This pipeline is based on the Master's thesis work:
> **"Integrated Analysis of Respiratory–Phonatory Functions: Normative Patterns Across Sex and Age"** > Bianca Zocco, Politecnico di Milano, aa : 2024-2025

It allows the analysis of combined data from:
- **OEP (Optoelectronic Plethysmography)**: chest wall kinematics
- **Acoustic signals**: voice recorded during vocal tasks

### Supported Vocal Tasks

| Task | Code | Description |
|-------|------|-------------|
| A-Long | AL | Sustained vowel /a/ |
| Vowels | - | 5 vowels × 5 seconds |
| Sentences | - | 5 predefined sentences |
| Text | TXT | Balanced text reading |
| Rolled | R | Sustained alveolar trill |
| Glissando | AG | Vocal glide (low → high) |
| A-Softly | - | Soft emission of /a/ |

## Installation

#### -> Clone the repository
git clone <repo-url>
cd pneumophonic_pipeline

#### -> Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
-> or: venv\Scripts\activate  # Windows

#### -> Install dependencies
pip install -r requirements.txt

#### -> Install the package in development mode
pip install -e .


## Main Dependencies
- **numpy, scipy, pandas: scientific computing**
- **librosa: audio processing**
- 
-**noisereduce: noise reduction**
  
-**praat-parselmouth:** Python interface for Praat
  
-**matplotlib:** visualization
  
-**openpyxl:** Excel read/write

📁 Expected Data Structure
```
data_root/
├── 20260218_GaBa/           # Format: YYYYMMDD_SubjectID
│   ├── csv/
│   │   └── GaBaVocali.csv   # Données OEP
│   ├── renders/
│   │   ├── a.wav            # Voyelle A
│   │   ├── r.wav            # Roulée R
│   │   └── ...
│   ├── sync_signal.wav      # Signal de synchronisation
│   └── GaBa_audio.xlsx      # Timings (optionnel)
├── 20260226_AnMa/
│   └── ...
└── ...
```

## Usage

### Quick Analysis

```python
from pneumophonic_analysis import run_pipeline

results = run_pipeline(
    data_root="/path/to/data",
    output_root="/path/to/output",
    tasks=['vowel', 'trill', 'glide']
)

print(f"Analyzed {results.n_subjects} subjects")
print(f"Success: {results.n_successful}, Failures: {results.n_failed}")
```

### Detailed Usage

```python
from pneumophonic_analysis import (
    PneumophonicPipeline, 
    create_config,
    VowelAnalyzer,
    Visualizer
)
from pathlib import Path

# 1. config
config = create_config(
    data_root=Path("/path/to/data"),
    output_root=Path("/path/to/output")
)

# 2. Init piepline
pipeline = PneumophonicPipeline(config)

# 3. Analyze subject specific
subject_result = pipeline.analyze_subject(
    "20260218_GaBa",
    tasks=['vowel', 'trill']
)

# results
for task_name, result in subject_result.results.items():
    print(f"\n{task_name}:")
    print(f"  duration: {result.duration_sec:.2f}s")
    print(f"  F0 mean: {result.acoustic_result.pitch.mean_f0:.1f} Hz")
    print(f"  HNR: {result.acoustic_result.voice_quality.hnr:.1f} dB")

# 4. analysis batch
batch_result = pipeline.analyze_batch(progress=True)

# 5. Export
pipeline.export_results(batch_result, "all_results.xlsx")
pipeline.generate_report(batch_result, "report/")
```

### Individual Audio File Analysis

```python
from pneumophonic_analysis import (
    AudioProcessor, 
    PraatAnalyzer,
    quick_analysis
)
import librosa

#  audio loading
audio, sr = librosa.load("vowel_a.wav", sr=48000)

# method 1: fast analysis
df = quick_analysis(audio, sr)
print(df)

# method 2: detailed analysis
processor = AudioProcessor()
analyzer = PraatAnalyzer()

# pre-procesing
audio_clean = processor.reduce_noise(audio, sr)
audio_pe = processor.apply_pre_emphasis(audio_clean)

# feature extraction
result = analyzer.analyze_signal(audio_pe, sr)

print(f"F0: {result.pitch.mean_f0:.1f} ± {result.pitch.std_f0:.1f} Hz")
print(f"Jitter: {result.perturbation.local_jitter*100:.2f}%")
print(f"Shimmer: {result.perturbation.local_shimmer*100:.2f}%")
print(f"HNR: {result.voice_quality.hnr:.1f} dB")
print(f"DSI: {result.voice_quality.dsi:.2f}")
```

### Visualization

```python
from pneumophonic_analysis import Visualizer, DataLoader
import librosa

# load data
loader = DataLoader("20260218_GaBa")
audio, sr = loader.load_audio("a.wav")
# Visualize
viz = Visualizer()
# Waveform
fig = viz.plot_waveform(audio, sr, title="Vowel /a/ - GaBa")
# Spectrogram
fig = viz.plot_spectrogram(audio, sr)
# Mel-spectrogram
fig = viz.plot_mel_spectrogram(audio, sr)
# Save
viz.save_figure(fig, "output/spectrogram.png")
```

### FRC Segmentation

```python
from pneumophonic_analysis import FRCSegmenter, TrillAnalyzer

# Analyse de la roulée avec segmentation FRC
analyzer = TrillAnalyzer()

result = analyzer.analyze(
    audio=audio,
    sr=sr,
    subject_id="RoDi",
    frc_cross_time=3.5  # Temps du crossing FRC en secondes
)

print(f"Fréquence de modulation:")
print(f"  Full: {result.extra_metrics['mod_freq_full']:.1f} Hz")
print(f"  Above FRC: {result.extra_metrics['mod_freq_above_frc']:.1f} Hz")
print(f"  Below FRC: {result.extra_metrics['mod_freq_below_frc']:.1f} Hz")
```

## Extracted Metrics

### Pitch Parameters (F0)
-`mean_f0:` Mean fundamental frequency (Hz)
-``std_f0: Standard deviation of F0
-`min_f0, max_f0:` F0 extrema
-`range_f0:` F0 range
### Frequency Perturbation (Jitter)
-`local_jitter:` Local jitter (%)
-`rap_jitter:` Relative Average Perturbation
-`ppq5_jitter:` 5-point Period Perturbation Quotient
-`ddp_jitter:` Difference of Differences of Periods
### Amplitude Perturbation (Shimmer)
-`local_shimmer:` Local shimmer (%)
-`apq3_shimmer, apq5_shimmer, apq11_shimmer:` Amplitude Perturbation Quotients
-`dda_shimmer:` Difference of Differences of Amplitudes
### Voice Quality
-`hnr:` Harmonics-to-Noise Ratio (dB)
-`dsi:` Dysphonia Severity Index
-`intensity_mean, intensity_min:` Intensity (dB)
-`mpt:` Maximum Phonation Time (s)
### Formants
-`f1_mean, f2_mean, f3_mean:` Mean formants (Hz)
-`f1_median, f2_median, f3_median:` Median formants
### Specific Metrics
-`Rolled R:` mod_freq_full/above/below (modulation frequency)
-`Glissando:` P1_meanF0, P2_meanF0, F0_range

## Pipeline Architecture

```
pneumophonic_analysis/
├── config.py           # Centralized configuration
├── io_utils.py         # File read/write
├── sync.py             # OEP/Audio synchronization
├── audio_processing.py # Audio processing
├── acoustic_features.py # Praat extraction
├── segmentation.py     # FRC/novelty segmentation
├── task_analyzers.py   # Task-specific analyzers
├── visualization.py    # Visualization
└── pipeline.py         # Orchestration
```

## References

If you use this pipeline, please cite:

```bibtex
@mastersthesis{zocco2025pneumophonic,
  title={Integrated Analysis of Respiratory-Phonatory Functions: 
         Normative Patterns Across Sex and Age},
  author={Zocco, Bianca},
  year={2025},
  school={Politecnico di Milano},
  advisor={Lo Mauro, Antonella}
}
```

## Licence

This project is distributed under the MIT license.

## Contributors

Contributions are welcome! Please open an issue to discuss major changes.