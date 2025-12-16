import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np


class StrokeDataset(Dataset):
    def __init__(self, csv_file, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            train (bool): If True, fit encoders/scalers. If False, use pre-fitted ones.
        """
        self.data = pd.read_csv(csv_file)
        self.train = train
        
        # Separate features and target
        self.target = self.data['stroke'].values
        self.data = self.data.drop(['id', 'stroke'], axis=1)
        
        # Define categorical and numerical columns
        self.cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        self.num_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        
        # Handle missing values in smoking_status (empty strings)
        self.data['smoking_status'] = self.data['smoking_status'].fillna('Unknown')
        self.data['bmi'] = self.data['bmi'].fillna(self.data['bmi'].median())
        
        # Initialize encoders and scalers
        if train:
            self.encoders = {}
            for col in self.cat_cols:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.encoders[col] = le
            
            self.scaler = StandardScaler()
            self.data[self.num_cols] = self.scaler.fit_transform(self.data[self.num_cols])
        
    def set_encoders_scalers(self, encoders, scaler):
        """Set pre-fitted encoders and scaler for test set"""
        self.encoders = encoders
        self.scaler = scaler
        
        for col in self.cat_cols:
            self.data[col] = self.encoders[col].transform(self.data[col].astype(str))
        
        self.data[self.num_cols] = self.scaler.transform(self.data[self.num_cols])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get features as tensor
        features = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)
        # Get target
        target = torch.tensor(self.target[idx], dtype=torch.float32)
        
        return features, target
    
    def get_encoders_scalers(self):
        """Return fitted encoders and scaler for use with test set"""
        return self.encoders, self.scaler


# Usage example
if __name__ == "__main__":
    # Create dataset
    train_dataset = StrokeDataset('data/dataset.csv', train=True)
    
    # Get encoders and scalers for potential test set
    encoders, scaler = train_dataset.get_encoders_scalers()
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # Print dataset info
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of features: {train_dataset.data.shape[1]}")
    print(f"Feature names: {list(train_dataset.data.columns)}")
    
    # Test loading a batch
    for features, targets in train_loader:
        print(f"\nBatch features shape: {features.shape}")
        print(f"Batch targets shape: {targets.shape}")
        print(f"Sample features: {features[0]}")
        print(f"Sample target: {targets[0]}")
        break