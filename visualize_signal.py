# for for data visulization
import argparse
from data_processing.cpsc_signal_loader import RawSignalLoader, SignalProcessor
from utils.visualization import generate_spectrogram, save_spectrogram_plot, plot_ecg_signal
import os

def main(args):
    # Initialize the RawSignalLoader to load the raw ECG signals
    raw_loader = RawSignalLoader(file_path=args.file_path, lead_mode=args.lead_mode, desired_lead=args.desired_lead)

    # Load raw signal data
    raw_signals, channels = raw_loader.load_signal()

    # Initialize the SignalProcessor to preprocess the signal
    signal_processor = SignalProcessor(
        sampling_rate=args.sampling_rate,
        target_length=args.target_length,
        lead_mode=args.lead_mode,
        desired_lead=args.desired_lead
    )
    
    # Process the raw signal
    processed_signals = signal_processor.process_signal(raw_signals)

    # Handle multi-lead or single-lead signals
    if processed_signals.ndim == 2:  # Multi-lead ECG signal
        for idx, signal in enumerate(processed_signals.T):
            if args.plot_signal:
                plot_ecg_signal(signal, args.sampling_rate, title=f"Lead {idx + 1}")
            log_sxx = generate_spectrogram(signal, args.sampling_rate)
            save_spectrogram_plot(log_sxx, idx, args.dump_dir)
    else:  # Single-lead ECG signal
        if args.plot_signal:
            plot_ecg_signal(processed_signals.squeeze(), args.sampling_rate, title=f"Lead {args.desired_lead}")
        log_sxx = generate_spectrogram(processed_signals.squeeze(), args.sampling_rate)
        save_spectrogram_plot(log_sxx, 0, args.dump_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ECG signals and spectrograms.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to .mat ECG file.")
    parser.add_argument("--lead_mode", type=str, choices=["single", "multi"], default="multi", help="Lead mode.")
    parser.add_argument("--desired_lead", type=int, default=None, help="Lead number if single mode.")
    parser.add_argument("--dump_dir", type=str, default="saved_spectrograms", help="Directory to save spectrograms.")
    parser.add_argument("--plot_signal", action="store_true", help="Plot ECG signals.")
    parser.add_argument("--sampling_rate", type=int, default=250, help="Sampling rate.")
    parser.add_argument("--target_length", type=int, default=15000, help="Target length for signal padding or trimming.")
    
    args = parser.parse_args()
    main(args)
