import os
import time
import numpy as np
import mne
from preprocessing import extract_wavelet_features
from train import build_pipeline

def load_preprocessed_epochs(subject=4):
    data_path = os.path.join(os.getcwd(), 'data')
    epochs_save_path = os.path.join(data_path, f'sub-{subject}_preprocessed-epo.fif')
    if os.path.exists(epochs_save_path):
        epochs = mne.read_epochs(epochs_save_path, preload=True)
        return epochs
    else:
        raise FileNotFoundError(f"Preprocessed file not found at {epochs_save_path}")

def predict_stream(trained_model, epochs, features, delay=2.0):
    labels = epochs.events[:, -1]
    predictions = []
    print("Starting real-time simulation prediction...")
    
    for i, feat in enumerate(features):
        feat = feat.reshape(1, -1)
        pred = trained_model.predict(feat)
        predictions.append(pred[0])
        truth = labels[i]
        print(f"Epoch {i:02d}: Prediction = {pred[0]}, Truth = {truth}, Correct: {pred[0]==truth}")
        time.sleep(delay)
    
    accuracy = np.mean(np.array(predictions) == labels)
    print("\nOverall simulated streaming accuracy: {:.4f}".format(accuracy))

def predict_model(subject=4):
    epochs = load_preprocessed_epochs(subject)
    features = extract_wavelet_features(epochs)
    labels = epochs.events[:, -1]
    # In this example, we train a pipeline on the fly (or you could load a saved model)
    pipeline = build_pipeline()
    pipeline.fit(features, labels)
    predict_stream(pipeline, epochs, features)

if __name__ == "__main__":
    predict_model(subject=4)
