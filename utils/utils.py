"""
utils.py

This module contains utility variables  including arrhythmia classes,
data reduction limits, and file paths for saving models and results.
"""
from config import CSV_PATH,BASE_MODEL_DIR
import os

# List of arrhythmia classes order as per research paper

arrhythmia_classes_cpsc = [
    'SNR',
    'AF',
    'IAVB',
    'LBBB',
    'RBBB',
    'PAC',
    'PVC',
    'STD',
    'STE',
]

# Data reduction limit 
data_reduction = 10000    #10000
samples_limit = 10000    #10000

# Define paths for saving models, results, and plots
BASE_MODEL_DIR = BASE_MODEL_DIR

model_save_path = os.path.join(BASE_MODEL_DIR, "model_paths")
model_results_path = os.path.join(BASE_MODEL_DIR, "model_results")
model_plots_path = os.path.join(BASE_MODEL_DIR, "model_plots")
misclassified_samples = os.path.join(BASE_MODEL_DIR,"misclassified_samples")
input_samples = os.path.join(BASE_MODEL_DIR,"input_samples")
spectrogram_save_path = os.path.join(BASE_MODEL_DIR,"Spectrograms")

# CSV file paths
csv_path = CSV_PATH

# Ensure directories exist
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(model_results_path, exist_ok=True)
os.makedirs(model_plots_path, exist_ok=True)
os.makedirs(csv_path, exist_ok=True)
os.makedirs(misclassified_samples,exist_ok=True)
os.makedirs(spectrogram_save_path, exist_ok = True)


# Input sizes
input_length = 15000 #15000
num_channels = 12  #12 for 12 lead and 1 for single lead
num_classes = len(arrhythmia_classes_cpsc)
