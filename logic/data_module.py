import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from logic.utilities import DataValidator
import pickle
import os
from logic.utilities import set_seed


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