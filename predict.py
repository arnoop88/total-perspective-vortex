import time
import argparse
import numpy as np
from train import build_pipeline, load_preprocessed_epochs

def predict_stream(pipeline, X, y, delay=2.0):
    """Simulate real-time prediction with a delay per epoch."""
    print("Starting real-time simulation prediction...")
    for i, x in enumerate(X):
        pred = pipeline.predict(x[np.newaxis, ...])[0]
        correct = pred == y[i]
        print(f"Epoch {i:02d}: Pred = {pred}, True = {y[i]}, Correct = {correct}")
        time.sleep(delay)
    overall_acc = np.mean(pipeline.predict(X) == y)
    print("\nOverall simulated streaming accuracy: {:.4f}".format(overall_acc))

def predict_model(subject, run):
    """Train (or load) CSP+LDA and run the real-time prediction simulation."""
    epochs = load_preprocessed_epochs(subject, run)
    X = epochs.get_data()
    y = epochs.events[:, -1]

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    predict_stream(pipeline, X, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate real-time CSP+LDA predictions on preprocessed EEG epochs"
    )
    parser.add_argument('subject', type=int, help="Subject ID (e.g. 4)")
    parser.add_argument('run',     type=int, help="Run number (e.g. 14)")
    args = parser.parse_args()

    predict_model(args.subject, args.run)
