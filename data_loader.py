"""
Data Loader for Heart Disease FL - 2 HOSPITALS VERSION
"""

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np


COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]


def load_hospital_data(hospital_name, data_dir='../data/heart_disease/raw'):
    """Load data for one hospital."""
    file_map = {
        'cleveland': 'processed.cleveland.data',
        'hungarian': 'processed.hungarian.data',
        'switzerland': 'processed.switzerland.data',
        'va': 'processed.va.data'
    }
    
    filename = file_map.get(hospital_name.lower())
    if not filename:
        raise ValueError(f"Unknown hospital: {hospital_name}")
    
    filepath = Path(data_dir) / filename
    
    df = pd.read_csv(filepath, names=COLUMNS, na_values='?')
    df['target'] = (df['num'] > 0).astype(int)
    
    X = df.drop(['num', 'target'], axis=1)
    y = df['target']
    X = X.fillna(X.mean())
    
    return X.values, y.values


def load_all_hospitals(data_dir='../data/heart_disease/raw'):
    """Load data from 2 hospitals."""
    hospitals = ['cleveland', 'hungarian']
    hospital_data = []
    
    print("Loading data from hospitals:")
    print("-" * 60)
    
    for hospital in hospitals:
        try:
            X, y = load_hospital_data(hospital, data_dir)
            hospital_data.append((hospital, X, y))
            print(f"{hospital.capitalize():15} : {len(X):3} patients, "
                  f"{y.sum():3} with disease ({y.sum()/len(y)*100:.1f}%)")
        except FileNotFoundError:
            print(f"{hospital.capitalize():15} : FILE NOT FOUND")
    
    total_patients = sum(len(X) for _, X, _ in hospital_data)
    print("-" * 60)
    print(f"{'TOTAL':15} : {total_patients:3} patients")
    
    return hospital_data


def prepare_client_data(X, y, batch_size=32):
    """Prepare data for one client."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, testloader, len(train_dataset)


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING DATA LOADER - 2 HOSPITALS")
    print("=" * 60)
    print()
    
    hospital_data = load_all_hospitals()
    
    print("\n" + "=" * 60)
    print("PREPARING DATA FOR EACH HOSPITAL")
    print("=" * 60)
    
    for hospital_name, X, y in hospital_data:
        print(f"\n{hospital_name.capitalize()} Hospital:")
        print("-" * 40)
        trainloader, testloader, num_train = prepare_client_data(X, y)
        print(f"  Train samples: {num_train}")
        print(f"  Test samples: {len(testloader.dataset)}")
    
    print("\n" + "=" * 60)
    print("✅ Data loader ready for 2 hospitals!")
    print("=" * 60)
