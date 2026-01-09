"""
Generate feature baseline statistics from training data
Run this after training to create feature_baseline.npz
"""
import pandas as pd
import numpy as np
from pathlib import Path

def generate_baseline():
    """Generate baseline statistics from training data"""
    
    # Load training data
    data_path = Path("data/data.csv")  # Ajusta la ruta
    df = pd.read_csv(data_path)
    
    # Remove target column
    if 'diagnosis' in df.columns:
        df = df.drop('diagnosis', axis=1)
    
    # Remove ID column if exists
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Calculate statistics
    means = df.mean().values
    stds = df.std().values
    
    # Save baseline
    output_path = Path("artifacts/feature_baseline.npz")
    output_path.parent.mkdir(exist_ok=True)
    
    np.savez(
        output_path,
        means=means,
        stds=stds,
        feature_names=df.columns.tolist()
    )
    
    print(f"✅ Baseline saved to {output_path}")
    print(f"   Features: {len(means)}")
    print(f"   Mean range: [{means.min():.2f}, {means.max():.2f}]")
    print(f"   Std range: [{stds.min():.2f}, {stds.max():.2f}]")

if __name__ == "__main__": 
    generate_baseline()