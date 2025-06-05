import os
import argparse
import numpy as np
import mne
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

mne.set_log_level('ERROR')

def load_preprocessed_epochs(subject, run):
    """Loads saved epochs from disk for a given subject and run."""
    data_path = os.path.join(os.getcwd(), 'data')
    fname = f'sub-{subject}_run-{run}_preprocessed-epo.fif'
    epochs_save_path = os.path.join(data_path, fname)
    if os.path.exists(epochs_save_path):
        return mne.read_epochs(epochs_save_path, preload=True)
    else:
        raise FileNotFoundError(f"Preprocessed file not found: {epochs_save_path}")

def build_pipeline(n_components=4, reg=None, log=True):
    """CSP + LDA pipeline."""
    return Pipeline([
        ('csp', CSP(n_components=n_components, reg=reg, log=log)),
        ('lda', LDA())
    ])

def train_model(subject, run):
    """Load epochs, run 5-fold CV with CSP+LDA, report mean accuracy."""
    epochs = load_preprocessed_epochs(subject, run)
    X = epochs.get_data()
    y = epochs.events[:, -1]

    pipeline = build_pipeline()
    scores = cross_val_score(pipeline, X, y, cv=5)
    print("Cross-validation scores:", np.round(scores, 4))
    print("Mean accuracy: {:.4f}".format(np.mean(scores)))

    pipeline.fit(X, y)
    return pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CSP+LDA on preprocessed EEG epochs"
    )
    parser.add_argument('subject', type=int, help="Subject ID (e.g. 4)")
    parser.add_argument('run',     type=int, help="Run number (e.g. 14)")
    args = parser.parse_args()

    train_model(args.subject, args.run)
