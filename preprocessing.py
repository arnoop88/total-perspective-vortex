"""
EEG Data Preprocessing Script (V.1.2)
- Simplified to focus on core preprocessing
- Saves filtered epochs for later processing
"""

import os
import argparse
import mne
from mne.datasets import eegbci
import matplotlib.pyplot as plt

def load_and_preprocess(subject, run, visualize=False):
    """
    Main preprocessing pipeline:
    1. Download and read raw EEG data (EDF) for the given subject and run.
    2. Rename channels and set standard montage.
    3. Apply bandpass filter (8-30 Hz).
    4. Extract events and create epochs (-0.5 to 3.0 seconds).
    5. Save preprocessed epochs to disk.
    
    Returns:
    - epochs: MNE Epochs object
    """
    try:
        data_path = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_path, exist_ok=True)

        # Download data if needed
        eegbci.load_data(subject, [run], path=data_path)
        
        # Find raw file path
        raw_file = f'S{subject:03d}R{run:02d}.edf'
        raw_path = os.path.join(
            data_path, 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0',
            f'S{subject:03d}', raw_file
        )
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"EEG file not found: {raw_path}")
            
        # Load and preprocess data
        raw = mne.io.read_raw_edf(raw_path, preload=True)
        
        # Standardize channel names
        new_names = {ch: ch.rstrip('.').upper() for ch in raw.info['ch_names']}
        raw.rename_channels(new_names)
        
        # Set montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')

        # Visualization before filtering
        if visualize:
            raw.plot(title='Raw EEG Data (Unfiltered)')
            plt.show(block=True)
            raw.compute_psd(fmax=50).plot(spatial_colors=False)
            plt.title('PSD Before Filtering')
            plt.show(block=True)

        # Apply bandpass filter
        raw.filter(8.0, 30.0, fir_design='firwin')
        
        # Visualization after filtering
        if visualize:
            raw.plot(title='Filtered EEG Data (8-30 Hz)')
            plt.show(block=True)
            raw.compute_psd(fmax=50).plot(spatial_colors=False)
            plt.title('PSD After Filtering')
            plt.show(block=True)

        # Extract events and create epochs
        events, event_id = mne.events_from_annotations(raw)
        print(f"Found events: {event_id}")
        epochs = mne.Epochs(
            raw, events, event_id,
            tmin=-0.5, tmax=3.0,
            baseline=None,
            preload=True
        )
        print(f"Epochs shape: {epochs.get_data().shape}")

        # Save preprocessed epochs
        epochs_fname = os.path.join(
            data_path, f'sub-{subject}_run-{run}_preprocessed-epo.fif'
        )
        epochs.save(epochs_fname, overwrite=True)
        print(f"Preprocessed epochs saved to {epochs_fname}")

        return epochs

    except Exception as err:
        print(f"Error in preprocessing: {err}")
        raise

def load_preprocessed_epochs(subject, run):
    """Loads saved epochs from disk for a given subject and run."""
    data_path = os.path.join(os.getcwd(), 'data')
    fname = f'sub-{subject}_run-{run}_preprocessed-epo.fif'
    epochs_save_path = os.path.join(data_path, fname)
    if os.path.exists(epochs_save_path):
        return mne.read_epochs(epochs_save_path, preload=True)
    else:
        raise FileNotFoundError(f"Preprocessed file not found: {epochs_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess EEG data for a specified subject and run"
    )
    parser.add_argument('subject', type=int, help="Subject ID (e.g., 4)")
    parser.add_argument('run',     type=int, help="Run number (e.g., 14)")
    parser.add_argument('--visualize', action='store_true', 
                        help="Show visualization plots")
    args = parser.parse_args()

    load_and_preprocess(args.subject, args.run, args.visualize)