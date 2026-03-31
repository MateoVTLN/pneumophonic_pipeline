from pneumophonic_analysis import run_pipeline

# Run the complete analysis with automatic export
results = run_pipeline(
    data_root="C:/Users/Matéo/OneDrive/Documents/GitHub/pneumophonic_pipeline/data_root",    
    output_root="C:/Users/Matéo/OneDrive/Documents/GitHub/pneumophonic_pipeline/data_target", 
    tasks=['vowel', 'trill', 'glide'] 
)

print(f"Analyzed {results.n_subjects} subjects")
print(f"Successes: {results.n_successful}, Failures: {results.n_failed}")
