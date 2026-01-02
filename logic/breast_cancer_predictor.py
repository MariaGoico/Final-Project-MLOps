import json
import pickle
import numpy as np
import xgboost as xgb


class BreastCancerPredictor:
    def __init__(self, artifact_dir="artifacts"):
        self.model = xgb.Booster()
        self.model.load_model(f"{artifact_dir}/model.json")

        with open(f"{artifact_dir}/preprocessor.pkl", "rb") as f:
            self.processor = pickle.load(f)

        with open(f"{artifact_dir}/threshold.json") as f:
            self.threshold = float(json.load(f)["threshold"])

    def _prepare(self, X):
        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_proc = self.processor.transform(X)

        if X_proc.shape[1] != self.model.num_features():
            raise ValueError(
                f"Expected {self.model.num_features()} features, "
                f"got {X_proc.shape[1]}"
            )

        return xgb.DMatrix(X_proc)

    def predict_proba(self, X):
        dmatrix = self._prepare(X)
        return self.model.predict(dmatrix)

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def predict_with_confidence(self, X):
        probs = self.predict_proba(X)
        return int(probs[0] >= self.threshold), float(probs[0])

# if __name__ == "__main__":
#     predictor = BreastCancerPredictor("artifacts")


#     # Example fake input (replace length if needed)
#     n_features = predictor.model.num_features()
#     x = np.random.rand(n_features)

#     pred, prob = predictor.predict_with_confidence(x)

#     print("Prediction:", pred)
#     print("Probability:", prob)

#     assert 0.0 <= prob <= 1.0
#     assert pred in (0, 1)

#     print("✅ Smoke test passed")
