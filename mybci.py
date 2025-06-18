"""
Main BCI Processing Script (Total Perspective Vortex)
Implements the full processing pipeline with CLI interface
"""

import argparse
import time
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh
from preprocessing import load_preprocessed_epochs, load_and_preprocess

# Custom CSP Implementation (from dimensionality_reduction.py)
class CSPTransformer(BaseEstimator, TransformerMixin):
    """Custom CSP Implementation for EEG Dimensionality Reduction"""
    def __init__(self, n_components=4, reg=0.1):
        self.n_components = n_components
        self.reg = reg
        self.filters_ = None
        
    def fit(self, X, y):
        # X shape: (n_trials, n_channels, n_times)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly two classes")
        
        # Compute class covariance matrices
        covs = [self._compute_cov(X[y == cls]) for cls in classes]
        
        # Solve generalized eigenvalue problem
        evals, evecs = eigh(covs[0], covs[0] + covs[1])
        ix = np.argsort(evals)[::-1]  # Sort descending
        
        # Store filters (first and last components)
        self.filters_ = evecs[:, ix[:self.n_components//2]]
        self.filters_ = np.hstack([self.filters_, evecs[:, ix[-self.n_components//2:]]])
        return self
    
    def transform(self, X):
        if self.filters_ is None:
            raise RuntimeError("Fit CSP before transforming")
        return np.array([self.filters_.T @ epoch for epoch in X])
    
    def _compute_cov(self, trials):
        # Regularize covariance matrix
        cov = np.mean(np.array([epoch @ epoch.T for epoch in trials]), axis=0)
        cov /= np.trace(cov)
        if self.reg:
            cov += self.reg * np.eye(cov.shape[0])
        return cov

def build_pipeline(n_components=4):
    """Create sklearn pipeline with CSP and LDA"""
    return Pipeline([
        ('csp', CSPTransformer(n_components=n_components)),
        ('lda', LDA())
    ])

def train_mode(subject, run):
    """Train and validate model for given subject/run"""
    print(f"Training model for subject {subject}, run {run}")
    
    # Load preprocessed data
    epochs = load_preprocessed_epochs(subject, run)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    
    # Build and evaluate pipeline
    pipeline = build_pipeline()
    scores = cross_val_score(pipeline, X, y, cv=5, n_jobs=-1)
    
    # Print results
    print("Cross-validation scores:", np.round(scores, 4))
    print("Mean accuracy: {:.4f}".format(np.mean(scores)))
    
    # Full training for later prediction
    pipeline.fit(X, y)
    return pipeline

def predict_mode(subject, run):
    """Simulate real-time prediction with <2s constraint"""
    print(f"Predicting for subject {subject}, run {run}")
    
    # Load data and split
    epochs = load_preprocessed_epochs(subject, run)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    
    # Simulate real-time prediction
    correct = []
    print("\nStarting real-time prediction simulation...")
    for i, (epoch, true_label) in enumerate(zip(X_test, y_test)):
        start_time = time.time()
        
        # Predict with timing constraint
        pred = pipeline.predict(epoch[np.newaxis, ...])[0]
        proc_time = time.time() - start_time
        
        # Check timing constraint
        time_ok = proc_time < 2.0
        correct.append(pred == true_label)
        
        # Print results
        status = "OK" if time_ok else f"TOO SLOW ({proc_time:.2f}s)"
        result = "CORRECT" if pred == true_label else "WRONG"
        print(f"Epoch {i:03d}: Pred={pred}, True={true_label} | {result} | {status}")
    
    # Final accuracy
    accuracy = np.mean(correct)
    print(f"\nPrediction accuracy: {accuracy:.4f}")
    return accuracy

def full_evaluation_mode():
    """Run full evaluation as shown in project example"""
    print("Running full evaluation...")
    
    # Example subject/runs (modify with actual dataset parameters)
    subjects = [1, 2, 3]  # Replace with actual subject IDs
    runs = [3, 7, 11]     # Replace with actual run numbers
    
    results = []
    for subject in subjects:
        subject_results = []
        for run in runs:
            try:
                # Preprocess if needed
                if not os.path.exists(f'data/sub-{subject}_run-{run}_preprocessed-epo.fif'):
                    load_and_preprocess(subject, run, visualize=False)
                
                # Train and evaluate
                epochs = load_preprocessed_epochs(subject, run)
                X = epochs.get_data()
                y = epochs.events[:, -1]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                
                # Build and evaluate pipeline
                pipeline = build_pipeline()
                pipeline.fit(X_train, y_train)
                accuracy = pipeline.score(X_test, y_test)
                subject_results.append(accuracy)
                
                print(f"Subject {subject}, Run {run}: Accuracy = {accuracy:.4f}")
                
            except Exception as e:
                print(f"Error processing subject {subject}, run {run}: {str(e)}")
                subject_results.append(0.0)
        
        # Store subject results
        avg_accuracy = np.mean(subject_results)
        results.append(avg_accuracy)
        print(f"Subject {subject} average: {avg_accuracy:.4f}")
    
    # Final report
    print("\n===== FINAL RESULTS =====")
    for i, acc in enumerate(results):
        print(f"Subject {subjects[i]}: {acc:.4f}")
    print(f"Overall mean accuracy: {np.mean(results):.4f}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Total Perspective Vortex - BCI Processing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('subject', type=int, nargs='?', 
                        help='Subject ID (e.g. 4)')
    parser.add_argument('run', type=int, nargs='?', 
                        help='Run number (e.g. 14)')
    parser.add_argument('mode', nargs='?', choices=['train', 'predict'], 
                        help='Operation mode')
    
    args = parser.parse_args()
    
    if args.subject is not None and args.run is not None and args.mode:
        if args.mode == 'train':
            train_mode(args.subject, args.run)
        else:
            predict_mode(args.subject, args.run)
    else:
        full_evaluation_mode()

if __name__ == '__main__':
    main()