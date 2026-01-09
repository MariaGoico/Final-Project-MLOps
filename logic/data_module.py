import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from logic.utilities import DataValidator
import pickle
import os
from logic.utilities import set_seed
from sklearn.feature_selection import VarianceThreshold


class CancerDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def load_data(self, csv_file):
        data = pd.read_csv(csv_file)

        if "diagnosis" not in data.columns:
            raise ValueError("Target column 'diagnosis' not found")

        y = data["diagnosis"].map({"M": 1, "B": 0}).values
        X = data.drop(columns=["diagnosis", "id"])

        return X.values, y

    def fit_transform(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted")
        return self.scaler.transform(X)


class CancerDataProcessorTabNet:
    """
    Preprocessing pipeline for TabNet models.

    Guarantees:
    - No NaNs or Infs after preprocessing
    - No zero-variance features
    - Deterministic behavior across train/val/test
    - ONNX-safe numerical outputs
    """

    def __init__(self):
        self.var_thresh = VarianceThreshold(threshold=0.0)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names_ = None

    def load_data(self, csv_file):
        data = pd.read_csv(csv_file)

        if "diagnosis" not in data.columns:
            raise ValueError("Target column 'diagnosis' not found")

        y = data["diagnosis"].map({"M": 1, "B": 0}).values

        # Drop target and ID
        X = data.drop(columns=["diagnosis", "id"])

        self.feature_names_ = X.columns.tolist()

        return X.values.astype(np.float32), y.astype(np.int64)

    def fit_transform(self, X):
        """
        Fit preprocessing on training data only.
        """
        # 1. Remove zero-variance features
        X = self.var_thresh.fit_transform(X)

        # Keep feature names after filtering
        self.feature_names_ = [
            name for name, keep in zip(
                self.feature_names_, self.var_thresh.get_support()
            ) if keep
        ]

        # 2. Scale features
        X = self.scaler.fit_transform(X)

        # 3. Final safety check
        if not np.isfinite(X).all():
            raise ValueError("Non-finite values detected after preprocessing")

        self.is_fitted = True
        return X.astype(np.float32)

    def transform(self, X):
        """
        Apply preprocessing to validation / test / inference data.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before calling transform")

        X = self.var_thresh.transform(X)
        X = self.scaler.transform(X)

        if not np.isfinite(X).all():
            raise ValueError("Non-finite values detected during transform")

        return X.astype(np.float32)

    def save(self, path):
        """
        Persist preprocessing pipeline for inference.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Load preprocessing pipeline for inference.
        """
        with open(path, "rb") as f:
            return pickle.load(f)
