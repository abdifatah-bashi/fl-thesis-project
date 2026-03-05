import io
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from api.config import COLUMNS

def parse_csv(csv_bytes: bytes) -> tuple:
    """Read raw UCI CSV bytes into features (X) and labels (y)."""
    df = pd.read_csv(io.StringIO(csv_bytes.decode()), names=COLUMNS, na_values="?")
    df["target"] = (df["num"] > 0).astype(int)
    X = df.drop(["num", "target"], axis=1)
    X = X.fillna(X.mean())
    y = df["target"]
    return X.values, y.values

def make_loaders(X, y, batch_size: int = 32):
    """Standardize data and create PyTorch DataLoaders."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    train_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr).unsqueeze(1)),
        batch_size=batch_size, shuffle=True,
    )
    test_dl = DataLoader(
        TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(y_te).unsqueeze(1)),
        batch_size=batch_size,
    )
    return train_dl, test_dl, len(X_tr)
