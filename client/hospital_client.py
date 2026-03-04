"""
Hospital FL Client — Standalone Python Script

Data NEVER leaves this machine.

Usage:
  # Using bundled sample data:
  python client/hospital_client.py --name cleveland
  python client/hospital_client.py --name hungarian

  # Using your own CSV:
  python client/hospital_client.py --name oslo --data /path/to/data.csv

  # Custom server URL:
  python client/hospital_client.py --name cleveland --server http://localhost:8000

Flow:
  1. GET  /api/model/global   → fetch global model params
  2. Load local data          → stays on this machine
  3. Train locally            → PyTorch, stays on this machine
  4. POST /api/weights/{name} → send ONLY weight updates (no data)
"""
import sys
import argparse
import io
import time
from pathlib import Path

import requests
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import HeartDiseaseNet, train_model, test_model
from src.utils import get_parameters, set_parameters

# ── Sample data bundled with the project ─────────────────────────────────────
SAMPLE_DATA = {
    "cleveland": ROOT / "data/heart_disease/raw/processed.cleveland.data",
    "hungarian": ROOT / "data/heart_disease/raw/processed.hungarian.data",
}
COLUMNS = ["age","sex","cp","trestbps","chol","fbs","restecg",
           "thalach","exang","oldpeak","slope","ca","thal","num"]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(name: str, data_path: str | None, train_split: float, batch_size: int):
    """Load local data. Supports sample presets and custom CSV files."""
    if data_path:
        path = Path(data_path)
        print(f"  Loading from {path}")
        df = pd.read_csv(path, names=COLUMNS if len(pd.read_csv(path, nrows=0).columns) == 0 else None, na_values="?")
    elif name.lower() in SAMPLE_DATA:
        path = SAMPLE_DATA[name.lower()]
        if not path.exists():
            print(f"  ✗ Sample data not found at {path}")
            print(f"    Run: python scripts/download_data.py")
            sys.exit(1)
        print(f"  Loading sample data for '{name}' ({path.name})")
        df = pd.read_csv(path, names=COLUMNS, na_values="?")
    else:
        print(f"  ✗ No sample data for '{name}'. Provide --data /path/to/file.csv")
        sys.exit(1)

    # Target: 0 = no disease, 1 = disease
    if "num" in df.columns:
        df["target"] = (df["num"] > 0).astype(int)
        df = df.drop("num", axis=1)
    elif "target" not in df.columns:
        df["target"] = df.iloc[:, -1].apply(lambda x: 1 if x > 0 else 0)
        df = df.iloc[:, :-1].assign(target=df["target"])

    X = df.drop("target", axis=1).fillna(df.drop("target", axis=1).mean())
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=1 - train_split, random_state=42, stratify=y.values
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    trainloader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)),
        batch_size=batch_size, shuffle=True,
    )
    testloader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).unsqueeze(1)),
        batch_size=batch_size,
    )
    return trainloader, testloader, len(X_train), len(X)


# ── Main FL client loop ───────────────────────────────────────────────────────

def run(name: str, server: str, data_path: str | None,
        epochs: int, lr: float, batch_size: int, train_split: float):

    print(f"\n{'='*55}")
    print(f"  {name.upper()} HOSPITAL — Federated Learning Client")
    print(f"{'='*55}")
    print(f"  Server : {server}")
    print(f"  Epochs : {epochs}  |  LR: {lr}  |  Batch: {batch_size}")
    print()

    # ── Step 1: Fetch global model ────────────────────────────────────────────
    print("[ Step 1 ] Fetching global model from server...")
    for attempt in range(10):
        try:
            r = requests.get(f"{server}/api/model/global", timeout=10)
            if r.status_code == 200:
                break
            elif r.status_code == 404:
                print(f"  Model not published yet. Retrying in 3s... ({attempt+1}/10)")
                time.sleep(3)
        except requests.ConnectionError:
            print(f"  Cannot reach server at {server}. Retrying in 3s... ({attempt+1}/10)")
            time.sleep(3)
    else:
        print("  ✗ Could not fetch global model after 10 attempts. Is the server running?")
        sys.exit(1)

    params = torch.load(io.BytesIO(r.content), weights_only=True)
    model = HeartDiseaseNet()
    set_parameters(model, params)
    print(f"  ✓ Global model fetched ({len(params)} parameter arrays)")

    # ── Step 2: Load local data ───────────────────────────────────────────────
    print(f"\n[ Step 2 ] Loading local data...")
    trainloader, testloader, num_train, num_total = load_data(
        name, data_path, train_split, batch_size
    )
    print(f"  ✓ {num_total} patients loaded  |  {num_train} train / {num_total - num_train} test")
    print(f"  ✓ Data stays on this machine — never sent to server")

    # ── Step 3: Train locally ─────────────────────────────────────────────────
    print(f"\n[ Step 3 ] Training locally ({epochs} epoch(s))...")
    train_loss = train_model(model, trainloader, epochs=epochs, lr=lr)
    eval_loss, accuracy = test_model(model, testloader)
    print(f"  ✓ Train loss : {train_loss:.4f}")
    print(f"  ✓ Eval loss  : {eval_loss:.4f}")
    print(f"  ✓ Accuracy   : {accuracy*100:.1f}%")

    # ── Step 4: Submit weight update ──────────────────────────────────────────
    print(f"\n[ Step 4 ] Sending weight updates to server...")
    payload = {
        "params": get_parameters(model),
        "num_samples": num_train,
        "metrics": {
            "train_loss": float(train_loss),
            "eval_loss": float(eval_loss),
            "accuracy": float(accuracy),
        },
    }

    buf = io.BytesIO()
    torch.save(payload, buf)
    buf.seek(0)

    r = requests.post(
        f"{server}/api/weights/{name}",
        files={"file": (f"{name}.pt", buf, "application/octet-stream")},
        timeout=30,
    )
    r.raise_for_status()
    print(f"  ✓ Weight update submitted (model weights only — no patient data)")

    print(f"\n{'='*55}")
    print(f"  Done. {name.capitalize()} contributed to the global model.")
    print(f"  Patient data never left this machine.")
    print(f"{'='*55}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hospital FL Client")
    parser.add_argument("--name",    required=True,  help="Hospital name (e.g. cleveland, oslo)")
    parser.add_argument("--server",  default="http://localhost:8000", help="Central server URL")
    parser.add_argument("--data",    default=None,   help="Path to local CSV data file")
    parser.add_argument("--epochs",  type=int,   default=1,    help="Local training epochs")
    parser.add_argument("--lr",      type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch",   type=int,   default=32,   help="Batch size")
    parser.add_argument("--split",   type=float, default=0.8,  help="Train/test split ratio")
    args = parser.parse_args()

    run(
        name=args.name,
        server=args.server.rstrip("/"),
        data_path=args.data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        train_split=args.split,
    )
