import os
import numpy as np
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def load_preprocessed_epochs(subject=4):
    """Loads saved epochs from disk."""
    data_path = os.path.join(os.getcwd(), 'data')
    epochs_save_path = os.path.join(data_path, f'sub-{subject}_preprocessed-epo.fif')
    if os.path.exists(epochs_save_path):
        return mne.read_epochs(epochs_save_path, preload=True)
    else:
        raise FileNotFoundError(f"Preprocessed file not found at {epochs_save_path}")

def build_pipeline(n_components=4, reg=None, log=True):
    """
    CSP + LDA pipeline.
      - n_components: how many spatial filters to keep
      - reg: CSP regularization (None by default)
      - log: whether to log-transform the variances
    """
    return Pipeline([
        ('csp', CSP(n_components=n_components, reg=reg, log=log)),
        ('lda', LDA())
    ])

def train_model(subject=4):
    """Load epochs, train CSP+LDA, and report 5-fold CV accuracy."""
    epochs = load_preprocessed_epochs(subject)
    X = epochs.get_data()                 # shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]              # labels
    
    pipeline = build_pipeline()
    scores = cross_val_score(pipeline, X, y, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean accuracy: {:.4f}".format(np.mean(scores)))
    
    # Fit on full data for later use
    pipeline.fit(X, y)
    return pipeline

if __name__ == "__main__":
    train_model(subject=4)
