"""
EEG Data Preprocessing Script (V.1.1)
- Loads EEG data from PhysioNet
- Visualizes raw and filtered data
- Applies bandpass filtering
- Creates epochs around events
- Computes and visualizes power spectral density (PSD) before and after filtering
- Extracts features using wavelet transform (Morlet)
- Saves preprocessed data
"""

import os
import mne
from mne.datasets import eegbci
import matplotlib.pyplot as plt
import numpy as np

def extract_wavelet_features(epochs, freqs=np.arange(8, 31), n_cycles=7):
    """
    Extracts features using Morlet wavelet transform.
    
    Parameters:
    - epochs: The MNE Epochs object.
    - freqs: Array of frequencies of interest (default: 8-30 Hz covering mu and beta bands).
    - n_cycles: Number of cycles in each Morlet wavelet.
    
    Returns:
    - features: Array of features computed as the mean power over time and channels.
    """
    # Use new compute_tfr method (replacing legacy tfr_morlet)
    power = epochs.compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles,
                                 return_itc=False, decim=3, average=False)
    
    # power.data shape: (n_epochs, n_channels, n_freqs, n_times)
    # Average over the time dimension
    power_data = np.mean(power.data, axis=-1)
    print(f"Wavelet transform computed. Power data shape: {power_data.shape}")
    
    # Average over channels for a simple feature vector per epoch
    features = np.mean(power_data, axis=1)  # shape: (n_epochs, n_freqs)
    return features

def load_and_preprocess(subject=4, run=14):
    try:
        # File setup: create local data storage
        data_path = os.path.join(os.getcwd(), 'data')
        os.makedirs(data_path, exist_ok=True)
        
        # Load raw EEG data using MNE's EEGBCI dataset
        eegbci.load_data(subject, [run], path=data_path)
        raw_file = f'S{subject:03d}R{run:02d}.edf'
        raw_path = os.path.join(data_path, 'MNE-eegbci-data', 'files', 'eegmmidb', '1.0.0',
                                f'S{subject:03d}', raw_file)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"EEG file not found at {raw_path}")
        
        raw = mne.io.read_raw_edf(raw_path, preload=True)
        
        # Rename channels to remove trailing dots and standardize names
        new_names = {ch: ch.rstrip('.').upper() for ch in raw.info['ch_names']}
        raw.rename_channels(new_names)
        
        # Set a standard EEG montage to provide channel locations and remove spatial colors warning
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        
        # Visualize raw EEG data
        raw.plot(title='Raw EEG Data (Unfiltered)')
        plt.show(block=True)
        
        # Save a copy of raw data for PSD before filtering
        raw_before_filter = raw.copy()
        raw_before_filter.compute_psd(fmax=50).plot(spatial_colors=False)
        plt.title('PSD Before Filtering')
        plt.show(block=True)
        
        # Apply bandpass filter (8-30 Hz) to focus on relevant frequency bands for motor imagery
        raw.filter(8.0, 30.0, fir_design='firwin')
        raw.plot(title='Filtered EEG Data (8-30 Hz)')
        plt.show(block=True)
        
        # Compute and visualize PSD after filtering
        raw.compute_psd(fmax=50).plot(spatial_colors=False)
        plt.title('PSD After Filtering')
        plt.show(block=True)
        
        # Extract events from annotations in the filtered data
        events, event_id = mne.events_from_annotations(raw)
        print(f"Found events: {event_id}")
        
        # Create epochs around events with a window from -0.5 to 3.0 seconds
        epochs = mne.Epochs(raw, events, event_id,
                            tmin=-0.5, tmax=3.0,
                            baseline=None, preload=True)
        print(f"Epochs shape: {epochs.get_data().shape}")
        print(f"Channels: {raw.info['ch_names']}")
        
        # Extract features using wavelet transform (Morlet)
        features = extract_wavelet_features(epochs)
        print(f"Extracted features (shape: {features.shape})")
        
        # Save preprocessed epochs data to a file
        epochs_save_path = os.path.join(data_path, f'sub-{subject}_preprocessed-epo.fif')
        epochs.save(epochs_save_path, overwrite=True)
        print(f"Preprocessed data saved to {epochs_save_path}")
        
        return epochs, features
    
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None, None

if __name__ == "__main__":
    preprocessed_data, wavelet_features = load_and_preprocess(subject=4, run=14)
    if preprocessed_data is not None:
        print("\nPreprocessing completed successfully!")
        print(preprocessed_data)
