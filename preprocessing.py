"""
EEG Data Preprocessing Script (V.1.1)
- Downloads EEG data from PhysioNet for a specified subject and run
- Visualizes raw data and PSD before filtering
- Applies bandpass filtering (8-30 Hz) and visualizes filtered data and PSD after filtering
- Extracts events and creates epochs (-0.5 to 3.0 seconds)
- Extracts features using a Morlet wavelet transform
- Saves preprocessed epochs to data/sub-{subject}_run-{run}_preprocessed-epo.fif
"""

import os
import argparse
import mne
from mne.datasets import eegbci
import matplotlib.pyplot as plt
import numpy as np

def extract_wavelet_features(epochs, freqs=np.arange(8, 31), n_cycles=7):
    """
    Extract features using Morlet wavelet transform.
    
    Parameters:
    - epochs: MNE Epochs object.
    - freqs: Array of frequencies of interest (default: 8-30 Hz).
    - n_cycles: Number of cycles for each Morlet wavelet.
    
    Returns:
    - features: Array of shape (n_epochs, n_freqs) containing mean power per frequency.
    """
    power = epochs.compute_tfr(
        method='morlet',
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        decim=3,
        average=False
    )
    power_data = np.mean(power.data, axis=-1)
    print(f"Wavelet transform computed. Power data shape: {power_data.shape}")
    features = np.mean(power_data, axis=1)
    return features

def load_and_preprocess(subject, run):
    """
    Main preprocessing pipeline:
    1. Download and read raw EEG data (EDF) for the given subject and run.
    2. Rename channels and set a standard montage.
    3. Visualize raw data and PSD before filtering.
    4. Apply bandpass filter (8-30 Hz), visualize filtered data and PSD after filtering.
    5. Extract events and create epochs (from -0.5 to 3.0 seconds).
    6. Extract wavelet-based features.
    7. Save preprocessed epochs to disk.
    
    Returns:
    - epochs: MNE Epochs object.
    - features: Array of extracted wavelet features (n_epochs, n_freqs).
    """
    try:
        data_path = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_path, exist_ok=True)

        eegbci.load_data(subject, [run], path=data_path)
        raw_file = f'S{subject:03d}R{run:02d}.edf'
        raw_path = os.path.join(
            data_path, 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0',
            f'S{subject:03d}', raw_file
        )
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"EEG file not found: {raw_path}")
        raw = mne.io.read_raw_edf(raw_path, preload=True)

        new_names = {ch: ch.rstrip('.').upper() for ch in raw.info['ch_names']}
        raw.rename_channels(new_names)

        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')

        raw.plot(title='Raw EEG Data (Unfiltered)')
        plt.show(block=True)

        raw_before = raw.copy()
        raw_before.compute_psd(fmax=50).plot(spatial_colors=False)
        plt.title('PSD Before Filtering')
        plt.show(block=True)

        raw.filter(8.0, 30.0, fir_design='firwin')
        raw.plot(title='Filtered EEG Data (8-30 Hz)')
        plt.show(block=True)

        raw.compute_psd(fmax=50).plot(spatial_colors=False)
        plt.title('PSD After Filtering')
        plt.show(block=True)

        events, event_id = mne.events_from_annotations(raw)
        print(f"Found events: {event_id}")
        epochs = mne.Epochs(
            raw, events, event_id,
            tmin=-0.5, tmax=3.0,
            baseline=None,
            preload=True
        )
        print(f"Epochs shape: {epochs.get_data().shape}")
        print(f"Channel names: {raw.info['ch_names']}")

        features = extract_wavelet_features(epochs)
        print(f"Extracted features (shape: {features.shape})")

        epochs_fname = os.path.join(
            data_path, f'sub-{subject}_run-{run}_preprocessed-epo.fif'
        )
        epochs.save(epochs_fname, overwrite=True)
        print(f"Preprocessed epochs saved to {epochs_fname}")

        return epochs, features

    except Exception as err:
        print(f"Error in preprocessing: {err}")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess EEG data for a specified subject and run"
    )
    parser.add_argument('subject', type=int, help="Subject ID (e.g., 4)")
    parser.add_argument('run',     type=int, help="Run number (e.g., 14)")
    args = parser.parse_args()

    load_and_preprocess(args.subject, args.run)
