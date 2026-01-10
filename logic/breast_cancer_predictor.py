import json
import pickle
import numpy as np
import xgboost as xgb
import onnxruntime as ort
import os

class BreastCancerPredictor:
    def __init__(self, artifact_dir="artifacts"):
        # 1. Load Metadata to determine model type
        metadata_path = os.path.join(artifact_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
            self.model_type = self.metadata.get("model_type", "xgboost")

        # 2. Load Preprocessor (Common to both)
        with open(os.path.join(artifact_dir, "preprocessor.pkl"), "rb") as f:
            self.processor = pickle.load(f)

        # 3. Load Threshold (Common to both)
        with open(os.path.join(artifact_dir, "threshold.json"), "r") as f:
            self.threshold = float(json.load(f)["threshold"])

        # 4. Load Specific Model Backend
        if self.model_type == "xgboost":
            self.model = xgb.Booster()
            self.model.load_model(os.path.join(artifact_dir, "model.json"))
            self.feature_count = self.model.num_features()
            
        elif self.model_type == "tabnet":
            model_path = os.path.join(artifact_dir, "tabnet.onnx")
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            # Get expected feature count from metadata or metadata.json
            self.feature_count = self.metadata.get("n_features", 30)
            
        else:
            raise ValueError(f"Unknown model type in metadata: {self.model_type}")

    def _prepare(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_proc = self.processor.transform(X)

        # Validation check
        if X_proc.shape[1] != self.feature_count:
            raise ValueError(
                f"Expected {self.feature_count} features, "
                f"got {X_proc.shape[1]}"
            )

        # Convert to float32 for ONNX compatibility
        return X_proc.astype(np.float32)

    def predict_proba(self, X):
        X_proc = self._prepare(X)

        if self.model_type == "xgboost":
            dmatrix = xgb.DMatrix(X_proc)
            return self.model.predict(dmatrix)

        elif self.model_type == "tabnet":
            # ONNX Inference
            probs = self.session.run(None, {self.input_name: X_proc})[0]
            
            # TabNet ONNX output is usually (N, 2) [prob_0, prob_1]
            if probs.ndim == 2 and probs.shape[1] == 2:
                return probs[:, 1]
            return probs.flatten()

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def predict_with_confidence(self, X):
        probs = self.predict_proba(X)
        prob_positive = float(probs[0])
        return int(prob_positive >= self.threshold), prob_positive
    
    # Helper for API to query expected features without crashing
    @property
    def num_features(self):
        return self.feature_count