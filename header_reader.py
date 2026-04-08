import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
from pathlib import Path


from pneumophonic_analysis.io_utils import DataLoader


loader = DataLoader("data_root/healthy_subjects/20251205_MaCa")
oep = loader.load_oep_data("csv/MaCa_Vocali.csv")
print(oep.head())
print(oep.columns.tolist())