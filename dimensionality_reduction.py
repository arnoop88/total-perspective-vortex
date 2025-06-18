import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

class CSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, reg=None):
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