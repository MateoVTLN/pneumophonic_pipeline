from pneumophonic_analysis import run_pipeline
import logging
logging.basicConfig(level=logging.DEBUG)
# Run the complete analysis with automatic export
results = run_pipeline(
<<<<<<< Updated upstream
    data_root="C:/Users/Matéo/OneDrive/Documents/GitHub/pneumophonic_pipeline/data_root",    
    output_root="C:/Users/Matéo/OneDrive/Documents/GitHub/pneumophonic_pipeline/data_target", 
    tasks=['vowel', 'trill', 'glide'] 
=======
    data_root="C:/Users/Matéo/OneDrive/Documents/GitHub/pneumophonic_pipeline/data_root/healthy_subjects",    # <-- UPDATE THIS PATH
    output_root="C:/Users/Matéo/OneDrive/Documents/GitHub/pneumophonic_pipeline/data_target", # <-- UPDATE THIS PATH
    tasks=['vowel', 'trill', 'glide'] # You can change these based on what you want to analyze
>>>>>>> Stashed changes
)

print(f"Analyzed {results.n_subjects} subjects")
print(f"Successes: {results.n_successful}, Failures: {results.n_failed}")
