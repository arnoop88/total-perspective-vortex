import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import mne
from preprocessing import extract_wavelet_features  # Import function from preprocessing.py

def load_preprocessed_epochs(subject=4):
    data_path = os.path.join(os.getcwd(), 'data')
    epochs_save_path = os.path.join(data_path, f'sub-{subject}_preprocessed-epo.fif')
    if os.path.exists(epochs_save_path):
        epochs = mne.read_epochs(epochs_save_path, preload=True)
        return epochs
    else:
        raise FileNotFoundError(f"Preprocessed file not found at {epochs_save_path}")

def build_pipeline(n_components=5):
    pipeline = Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('lda', LDA())
    ])
    return pipeline

def train_model(subject=4):
    # Load saved epochs
    epochs = load_preprocessed_epochs(subject)
    # Recompute features from the loaded epochs
    features = extract_wavelet_features(epochs)
    # Assume labels are obtained from the epochs events (last column in the events array)
    labels = epochs.events[:, -1]
    
    pipeline = build_pipeline()
    scores = cross_val_score(pipeline, features, labels, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean accuracy: {:.4f}".format(np.mean(scores)))
    
    # Optionally, fit the pipeline on the entire dataset and return it
    pipeline.fit(features, labels)
    return pipeline

if __name__ == "__main__":
    trained_model = train_model(subject=4)
