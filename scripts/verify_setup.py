"""
Verify environment setup with CSV dataset
Tests that all packages work and data loads correctly.
"""

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def verify_setup():
    """Verify that the environment is properly set up."""
    
    print("=" * 60)
    print("TESTING SETUP WITH CSV FILE")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading data...")
    try:
        df = pd.read_csv('data/heart_disease/heart_disease_full.csv')
        print(f"   ✅ Data loaded: {df.shape}")
    except FileNotFoundError:
        print("   ❌ CSV file not found! Run scripts/download_dataset.py first")
        return False
    
    # 2. Basic preprocessing
    print("\n2. Basic preprocessing...")
    df['target'] = (df['num'] > 0).astype(int)
    X = df.drop(['num', 'target'], axis=1)
    y = df['target']
    X = X.fillna(X.mean())
    
    print(f"   ✅ Features: {X.shape}")
    print(f"   ✅ Target: {y.shape}")
    print(f"   ✅ Class distribution: {y.value_counts().to_dict()}")
    
    # 3. Train/test split
    print("\n3. Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✅ Train: {X_train.shape[0]} samples")
    print(f"   ✅ Test: {X_test.shape[0]} samples")
    
    # 4. Scale features
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"   ✅ Features scaled")
    
    # 5. Convert to PyTorch tensors
    print("\n5. Converting to PyTorch tensors...")
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values)
    print(f"   ✅ Tensors created")
    
    # 6. Simple model test
    print("\n6. Testing simple PyTorch model...")
    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Model created: {num_params} parameters")
    
    # 7. Quick forward pass test
    print("\n7. Testing forward pass...")
    with torch.no_grad():
        output = model(X_train_tensor[:5])
    print(f"   ✅ Forward pass works! Sample output shape: {output.shape}")
    
    print("\n" + "=" * 60)
    print("✅ ALL CHECKS PASSED - SETUP WORKING!")
    print("=" * 60)
    print("\nYou're ready for:")
    print("  → Week 6: Build centralized baseline model")
    print("  → Week 7: Switch to FL with .data files")
    
    return True


if __name__ == "__main__":
    success = verify_setup()
    exit(0 if success else 1)