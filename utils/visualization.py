# visualization.py

"""
Visualization module for ECG signal and spectrogram plotting.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def generate_spectrogram(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Generate a log-scaled spectrogram from a signal.

    Args:
        signal (np.ndarray): 1D ECG signal.
        sampling_rate (int): Sampling frequency.

    Returns:
        np.ndarray: Log-scaled spectrogram.
    """
    f, t, Sxx = spectrogram(signal, fs=sampling_rate, nperseg=256, noverlap=128)
    log_sxx = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)
    return log_sxx

def save_spectrogram_plot(log_sxx: np.ndarray, channel_idx: int, dump_dir: str):
    """
    Save a spectrogram plot to disk.

    Args:
        log_sxx (np.ndarray): Log-scaled spectrogram.
        channel_idx (int): Channel number (0-indexed).
        dump_dir (str): Directory to save the plot.
    """
    os.makedirs(dump_dir, exist_ok=True)
    plt.figure()
    plt.pcolormesh(log_sxx, shading='gouraud')
    plt.title(f"Spectrogram - Channel {channel_idx + 1}")
    plt.ylabel("Frequency bin")
    plt.xlabel("Time bin")
    plt.colorbar(label="Power (dB/Hz)")
    plt.tight_layout()
    save_path = os.path.join(dump_dir, f"spectrogram_ch_{channel_idx + 1}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved spectrogram: {save_path}")

def plot_ecg_signal(signal: np.ndarray, sampling_rate: int, title: str = "ECG Signal"):
    """
    Plot a single ECG signal.

    Args:
        signal (np.ndarray): 1D ECG signal.
        sampling_rate (int): Sampling frequency.
        title (str): Title of the plot.
    """
    time_axis = np.arange(len(signal)) / sampling_rate
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, signal)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
