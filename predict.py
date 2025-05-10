import os
import time
import numpy as np
import mne
from train import build_pipeline

def load_preprocessed_epochs(subject=4):
    """Loads saved epochs from disk."""
    data_path = os.path.join(os.getcwd(), 'data')
    epochs_save_path = os.path.join(data_path, f'sub-{subject}_preprocessed-epo.fif')
    if os.path.exists(epochs_save_path):
        return mne.read_epochs(epochs_save_path, preload=True)
    else:
        raise FileNotFoundError(f"Preprocessed file not found at {epochs_save_path}")

def predict_stream(pipeline, X, y, delay=2.0):
    """Simulate real-time prediction with a delay per epoch."""
    print("Starting real-time simulation prediction...")
    for i, x in enumerate(X):
        pred = pipeline.predict(x[np.newaxis, ...])[0]
        print(f"Epoch {i:02d}: Prediction = {pred}, Truth = {y[i]}, Correct = {pred==y[i]}")
        time.sleep(delay)
    overall_acc = np.mean(pipeline.predict(X) == y)
    print("\nOverall simulated streaming accuracy: {:.4f}".format(overall_acc))

def predict_model(subject=4):
    """Load epochs, train CSP+LDA, and run real-time simulation."""
    epochs = load_preprocessed_epochs(subject)
    X = epochs.get_data()               # raw epoch data
    y = epochs.events[:, -1]            # labels
    
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    predict_stream(pipeline, X, y)

if __name__ == "__main__":
    predict_model(subject=4)
