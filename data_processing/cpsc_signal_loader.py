"""
cpsc_signal_loader.py
This module for loading, preprocessing of ECG signals from .mat files. It supports both 12-lead (multi-lead)
and single-lead ECG signals based on a configuration flag.
"""
import os
from typing import Optional, List, Tuple
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt

class RawSignalLoader:
    """
    RawSignalLoader class for loading ECG signals from .mat files.
    """

    def __init__(
        self,
        file_path: str,
        lead_mode: str = "multi",  # "single" or "multi"
        desired_lead: Optional[int] = None,
    ):
        self.file_path = file_path
        self.lead_mode = lead_mode
        self.desired_lead = desired_lead

    def load_signal(self) -> Tuple[np.ndarray, List[str]]:
        """
        Load the raw ECG signal from the file.

        Returns:
            Tuple:
                - Raw signal (np.ndarray)
                - List of channel names (List[str])
        """
        mat_data = sio.loadmat(self.file_path)
        signal_key = next(k for k in mat_data if not k.startswith("__"))
        signals = mat_data[signal_key]
        if signals.shape[0] < signals.shape[1]:
            signals = signals.T  # Ensure shape (samples, channels)

        num_channels = signals.shape[1]
        if num_channels != 12:
            raise ValueError(f"Expected 12 channels, got {num_channels}")

        # Replace NaNs
        for ch in range(num_channels):
            ch_data = signals[:, ch]
            if np.any(np.isnan(ch_data)):
                signals[:, ch] = np.nan_to_num(ch_data, nan=np.nanmean(ch_data))

        return signals, [f"Lead {i+1}" for i in range(num_channels)]


class SignalProcessor:
    """
    SignalProcessor class for filtering and processing ECG signals.
    """

    def __init__(
        self,
        sampling_rate: int = 250,
        target_length: int = 15000,
        lead_mode: str = "multi",  # "single" or "multi"
        desired_lead: Optional[int] = None,
    ):
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.lead_mode = lead_mode
        self.desired_lead = desired_lead

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance."""
        mean = np.mean(signal)
        std = np.std(signal)
        return signal - mean if std == 0 else (signal - mean) / std

    def _pad_or_trim(self, signal: np.ndarray) -> np.ndarray:
        """Pad or trim signal to the target length."""
        if len(signal) < self.target_length:
            return np.pad(signal, (0, self.target_length - len(signal)), 'constant')
        return signal[:self.target_length]

    def _high_pass_filter(self, signal: np.ndarray, cutoff: float = 0.5, order: int = 2) -> np.ndarray:
        """Apply Butterworth high-pass filter to the signal."""
        nyquist = 0.5 * self.sampling_rate
        b, a = butter(order, cutoff / nyquist, btype='high')
        return filtfilt(b, a, signal)

    def process_signal(self, raw_signals: np.ndarray) -> np.ndarray:
        """
        Process the raw ECG signals by padding, filtering, and normalization.

        Returns:
            Processed signal (np.ndarray)
        """
        if self.lead_mode == "single":
            if self.desired_lead is None or not (1 <= self.desired_lead <= raw_signals.shape[1]):
                raise ValueError("Invalid desired_lead index.")
            signal = raw_signals[:, self.desired_lead - 1]
            signal = self._pad_or_trim(signal)
            signal = self._high_pass_filter(signal)
            processed = np.expand_dims(signal, axis=-1)
            return processed

        # Multi-lead mode
        processed_signals = np.zeros((self.target_length, raw_signals.shape[1]))
        for ch in range(raw_signals.shape[1]):
            signal = raw_signals[:, ch]
            signal = self._pad_or_trim(signal)
            signal = self._high_pass_filter(signal)
            processed_signals[:, ch] = signal

        return processed_signals
