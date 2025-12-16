import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from logic.utilities import DataValidator
import pickle
import os


class StrokeDataProcessor:
    """
    Data processor for stroke prediction dataset.
    Handles preprocessing, encoding, and scaling for imbalanced binary classification.
    """
    
    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        self.num_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        self.is_fitted = False
        
    def load_data(self, csv_file):
        """Load data from CSV file"""
        data = pd.read_csv(csv_file)
        
        # Separate features and target
        if 'stroke' in data.columns:
            y = data['stroke'].values
            X = data.drop(['stroke'], axis=1)
            if 'id' in X.columns:
                X = X.drop(['id'], axis=1)
        else:
            y = None
            X = data.drop(['id'], axis=1) if 'id' in data.columns else data
            
        return X, y
    
    def preprocess(self, X, fit=True):
        
        X = X.copy()
        
        # Handle missing values
        X['smoking_status'] = X['smoking_status'].fillna('Unknown')
        
        if fit:
            # Store median for BMI imputation
            self.bmi_median = X['bmi'].median()
        X['bmi'] = X['bmi'].fillna(self.bmi_median)
        
        # Encode categorical variables
        if fit:
            self.encoders = {}
            for col in self.cat_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
        else:
            for col in self.cat_cols:
                X[col] = self.encoders[col].transform(X[col].astype(str))
        
        # Scale numerical variables
        if fit:
            X[self.num_cols] = self.scaler.fit_transform(X[self.num_cols])
            self.is_fitted = True
        else:
            X[self.num_cols] = self.scaler.transform(X[self.num_cols])
        
        return X.values
    
    def fit_transform(self, X):
        """Fit preprocessor and transform data"""
        return self.preprocess(X, fit=True)
    
    def transform(self, X):
        """Transform data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Use fit_transform first.")
        return self.preprocess(X, fit=False)
    
    def save(self, filepath):
        """Save the fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'encoders': self.encoders,
                'scaler': self.scaler,
                'bmi_median': self.bmi_median,
                'cat_cols': self.cat_cols,
                'num_cols': self.num_cols
            }, f)
    
    def load(self, filepath):
        """Load a fitted preprocessor"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.encoders = data['encoders']
        self.scaler = data['scaler']
        self.bmi_median = data['bmi_median']
        self.cat_cols = data['cat_cols']
        self.num_cols = data['num_cols']
        self.is_fitted = True


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = StrokeDataProcessor()
    
    # Load and preprocess training data
    X_train, y_train = processor.load_data('data/dataset.csv')
    X_train_processed = processor.fit_transform(X_train)
    
    # Save preprocessor
    processor.save('models/preprocessor.pkl')
    
    # Load and preprocess test data
    X_test, y_test = processor.load_data('data/dataset.csv')
    X_test_processed = processor.transform(X_test)
    
    # Get validation strategy
    validator = DataValidator()
    cv = validator.get_stratified_kfold(n_splits=5)
    scale_pos_weight = validator.get_scale_pos_weight(y_train)
    
    print(f"Training samples: {len(X_train_processed)}")
    print(f"Test samples: {len(X_test_processed)}")
    print(f"Scale pos weight for XGBoost: {scale_pos_weight:.2f}")
    print(f"Class distribution - 0: {np.sum(y_train==0)}, 1: {np.sum(y_train==1)}")