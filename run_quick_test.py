from pneumophonic_analysis import run_pipeline
import logging
logging.basicConfig(level=logging.DEBUG)
results = run_pipeline(
    data_root="C:/Users/Matéo/OneDrive/Documents/GitHub/pneumophonic_pipeline/data_root/healthy_subjects",   
    output_root="C:/Users/Matéo/OneDrive/Documents/GitHub/pneumophonic_pipeline/data_target",
    tasks=['vowel', 'trill', 'glide'] 
print(f"Analyzed {results.n_subjects} subjects")
    )
"""
from pneumophonic_analysis import run_pipeline
import logging
logging.basicConfig(level=logging.DEBUG)
results = run_pipeline(
    data_root="C:/Users/Matéo/OneDrive/Documents/GitHub/pneumophonic_pipeline/data_root",    
    output_root="C:/Users/Matéo/OneDrive/Documents/GitHub/pneumophonic_pipeline/data_target", 
    tasks=['vowel', 'trill', 'glide'] 
print(f"Analyzed {results.n_subjects} subjects")
)
"""