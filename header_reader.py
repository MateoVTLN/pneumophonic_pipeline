import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
from pathlib import Path


from pneumophonic_analysis.io_utils import DataLoader


loader = DataLoader("data_root/healthy_subjects/YYYYMMDD_Subject_ID/phonema_a_2.wav")
oep = loader.load_oep_data("csv/Subject_ID_Vocali.csv")
print(oep.head())
print(oep.columns.tolist())