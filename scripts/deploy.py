"""
deploy.py
=========
Model persistence module for Credit Score Classification project.
Supports saving and loading trained models using pickle.
"""

import pickle
import os


class DeployClassifier:
    """Save and load trained classification models."""

    def __init__(self, path):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def save_model(self, filename, model):
        """Save a trained model to disk."""
        file_path = os.path.join(self.path, filename)
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        print(f"  Model saved to: {file_path}")

    def load_model(self, filename):
        """Load a trained model from disk."""
        file_path = os.path.join(self.path, filename)
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        print(f"  Model loaded from: {file_path}")
        return model
