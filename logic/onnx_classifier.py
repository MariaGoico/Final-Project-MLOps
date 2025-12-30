"""
ONNX classifier wrapper for tabular XGBoost inference.
"""

import pickle
import numpy as np
import onnxruntime as ort
from pathlib import Path


class ONNXBreastCancerClassifier:
    """
    Wrapper for ONNX XGBoost breast cancer classifier.
    """

    def __init__(self, model_path: str, threshold_path: str):
        """
        Args:
            model_path: Path to ONNX model
            threshold_path: Path to pickled threshold
        """
        if not Path(model_path).exists():  # pragma: no cover
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not Path(threshold_path).exists():  # pragma: no cover
            raise FileNotFoundError(f"Threshold file not found: {threshold_path}")

        # Load ONNX session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

        # Get input name dynamically
        self.input_name = self.session.get_inputs()[0].name

        # Load threshold
        with open(threshold_path, "rb") as f:
            self.threshold = pickle.load(f)["threshold"]

        print("ONNX Breast Cancer Classifier initialized")
        print(f"Model: {model_path}")
        print(f"Threshold: {self.threshold:.4f}")

    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """
        Ensure correct shape and dtype.

        Args:
            features: shape (n_features,) or (1, n_features)

        Returns:
            np.ndarray of shape (1, n_features), float32
        """
        features = np.asarray(features, dtype=np.float32)

        if features.ndim == 1:
            features = np.expand_dims(features, axis=0)

        return features

    def predict_proba(self, features: np.ndarray) -> float:
        """
        Predict probability of positive class.

        Returns:
            float: probability
        """
        inputs = {
            self.input_name: self.preprocess(features)
        }

        outputs = self.session.run(None, inputs)

        # XGBoost ONNX returns probability directly
        proba = float(outputs[0].ravel()[0])
        return proba

    def predict(self, features: np.ndarray) -> int:
        """
        Predict binary class using stored threshold.

        Returns:
            0 or 1
        """
        proba = self.predict_proba(features)
        return int(proba >= self.threshold)

    def predict_with_confidence(self, features: np.ndarray) -> tuple:
        """
        Returns:
            (prediction, probability)
        """
        proba = self.predict_proba(features)
        return int(proba >= self.threshold), proba
