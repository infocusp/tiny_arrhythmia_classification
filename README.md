# 🫀 Arrhythmia Classification on CPSC 2018 Dataset

This project focuses on **multi-class arrhythmia classification** using the **CPSC 2018** 12-lead ECG dataset.
It supports training and evaluation on both **12-lead signals** and **single-lead signals**, with configurable model architectures and MLflow-based experiment tracking.

## 📌 Table of Contents
1. [Project Structure](#-project-structure)
2. [CPSC Dataset](#-cpsc-dataset)
3. [Installation](#-installation)
4. [Modules Breakdown](#-modules-breakdown)
5. [Installation](#-installation)
6. [How to use](#-how-to-use)

## 🧠 Project Structure

```
├── config.py                                   # Configuration file for paths and settings
├── cpsc_data_prep_csv.py                       # Script to prepare CSVs from raw data
├── data_processing
│   ├── cpsc_signal_loader.py                   # Signal loading and preprocessing
│   ├── data_generator.py                       # Data generator for model input
│   ├── data_loading.py                         # Load and preprocess dataset
│   ├── data_reduction.py                       # Dataset size reduction scripts
│   ├── data_splitter.py                        # Train/Val/test splitting utilities
├── models
│   ├── checkpoints                             # Model checkpoints (saved weights)        
│   ├── final                                   # Final trained models
│   ├── misclassified_samples                   # Incorrect predictions for analysis
│   ├── model
│   │   ├── cnn1d_resent_18.py                  # resnet model  
│   │   └── cnn_attentaion_bilstm_improve.py    # cnn_attentaion_bilstm_model
│   ├── model_paths                             # Model path utilities
│   ├── model_plots                             # Training and evaluation plots
│   ├── model_results                           # Evaluation results (metrics)
│   └── Spectrograms                            # Spectrogram representations of signals
├── README.md                                   # Project documentation
├── requirements.txt                            # Python package dependencies
├── test_cpsc.py                                # Testing script
├── train_custom_log_cpsc.py                    # Training script with MLflow logging
├── utils
│   ├── model_selector.py                       # Model selection utilities
│   ├── utils.py                                # Utility functions and global configs and input sizes paths
│   └── visualization.py                        # Visulization utility
└── visualize_signal.py                         # ECG signal visulization script




```
## 🛢️ CPSC Dataset
CPSC dataset is opensource dataset avaliable [here](https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018/)

## ⚙️ Installation
1. Colne the repository:
```bash
   clone repo
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3.Download the datasets:
  - place the datasets as per your convinent give the correspoding path in the config file and utils file


## 🚀 How to use

1. To train a model
```bash
python3 train_custom_log_cpsc.py
```
- config.py:
   Contains model selection, MLflow enable/disable settings, number of epochs, and all other required configuration parameters.

- utils.py:
  Contains file paths, input sizes, and other utility constants needed across the project.

   - Models will be saved in models/checkpoints/.
   - Training history will be saved in mlflow server



2. To test a model
```bash
python3 test_cpsc.py
```
   - training logs ,tesing logs, model performace metrics and model are saving in mflow server

3. To Visualization
```bash
python visualize_signal.py --file_path data/test_data.pkl --plot_signal --sampling_rate 250
```
   - provide reqiured args for visulization of the signal
