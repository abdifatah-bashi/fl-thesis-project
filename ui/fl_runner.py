"""
FL Runner — Bridge between Streamlit UI and Flower-compatible training.

Runs federated learning (FedAvg) in a background thread using direct
PyTorch training — no Ray / Flower simulation server overhead.
Reports per-round progress to the shared state JSON file.
"""

import sys
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.model import HeartDiseaseNet, train_model, test_model
from src.utils import get_parameters, set_parameters
from state_manager import load_state, save_state

# ── constants ────────────────────────────────────────────────────────────────
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
UPLOAD_DIR = ROOT / "data" / "hospital_uploads"


# ── progress persistence helper ──────────────────────────────────────────────

def _persist_progress(
    server_round: int,
    phase: str,
    total_rounds: int,
    train_losses: List[Tuple[int, float]],
    eval_losses: List[Tuple[int, float]],
    accuracies: List[Tuple[int, float]],
) -> None:
    """Write per-round metrics to the shared state file."""
    try:
        state = load_state()
        fed = state["federation"]
        fed["current_round"] = server_round
        fed["status"] = f"round_{server_round}_{phase}"
        hist = fed["history"]
        hist["rounds"] = [r for r, _ in accuracies] or [r for r, _ in train_losses]
        hist["accuracy"] = [round(a, 4) for _, a in accuracies]
        hist["train_loss"] = [round(l, 4) for _, l in train_losses]
        hist["eval_loss"] = [round(l, 4) for _, l in eval_losses]
        if server_round >= total_rounds and phase == "evaluated":
            fed["status"] = "complete"
            fed["completed_at"] = datetime.now().isoformat()
            for h in ("cleveland", "hungarian"):
                if state["hospitals"][h]["registered"]:
                    state["hospitals"][h]["status"] = "done"
        save_state(state)
    except Exception:
        pass


# ── data helpers ─────────────────────────────────────────────────────────────

def _process_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a DataFrame to (X, y) arrays."""
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    if "num" in df.columns:
        df["target"] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)
    elif "target" not in df.columns:
        last = df.columns[-1]
        df["target"] = (pd.to_numeric(df[last], errors="coerce") > 0).astype(int)
    feature_cols = [c for c in df.columns if c not in ("num", "target")]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    y = df["target"].values.astype(int)
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X, y


def prepare_hospital_data(
    hospital_name: str, config: Dict
) -> Tuple[DataLoader, DataLoader, int]:
    """Prepare train/test DataLoaders for one hospital."""
    batch_size = int(config.get("batch_size", 32))
    train_split = float(config.get("train_split", 0.8))

    upload_path = UPLOAD_DIR / f"{hospital_name}.csv"
    if upload_path.exists():
        df = pd.read_csv(upload_path)
        X, y = _process_df(df)
    else:
        from src.data_loader import load_hospital_data
        data_dir = str(ROOT / "data" / "heart_disease" / "raw")
        X, y = load_hospital_data(hospital_name, data_dir=data_dir)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=1.0 - train_split, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr).unsqueeze(1))
    test_ds = TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(y_te).unsqueeze(1))
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=batch_size)
    return trainloader, testloader, len(train_ds)


def load_sample_data_info(hospital_name: str) -> Dict[str, Any]:
    """Return preview info for the built-in sample data."""
    data_dir = ROOT / "data" / "heart_disease" / "raw"
    file_map = {"cleveland": "processed.cleveland.data", "hungarian": "processed.hungarian.data"}
    filename = file_map.get(hospital_name.lower())
    if not filename:
        return {}
    fp = data_dir / filename
    if not fp.exists():
        return {}
    df = pd.read_csv(fp, names=COLUMNS, na_values="?")
    df["target"] = (df["num"].fillna(0) > 0).astype(int)
    preview = df.drop("num", axis=1).head()
    return {
        "num_patients": len(df),
        "num_features": 13,
        "disease_rate": float(df["target"].mean()),
        "preview": preview,
    }


def register_hospital(
    hospital_name: str, df: pd.DataFrame, config: Dict
) -> bool:
    """Save hospital data and update the shared state."""
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(UPLOAD_DIR / f"{hospital_name}.csv", index=False)

        X, y = _process_df(df)
        disease_rate = float(y.mean()) if len(y) > 0 else 0.0

        state = load_state()
        state["hospitals"][hospital_name].update({
            "registered": True,
            "num_patients": len(df),
            "disease_rate": disease_rate,
            "status": "ready",
            "config": config,
            "registered_at": datetime.now().isoformat(),
        })
        save_state(state)
        return True
    except Exception:
        return False


def save_hospital_config(hospital_name: str, config: Dict) -> bool:
    try:
        state = load_state()
        state["hospitals"][hospital_name]["config"] = config
        save_state(state)
        return True
    except Exception:
        return False


# ── simulation runner ─────────────────────────────────────────────────────────

def _run_simulation(state_snapshot: Dict, config: Dict) -> None:
    """Run FedAvg training loop directly (no Ray / Flower simulation server)."""
    num_rounds = int(config.get("num_rounds", 3))
    epochs = int(config.get("epochs_per_round", 1))
    lr = float(config.get("learning_rate", 0.01))
    hospitals_cfg = state_snapshot.get("hospitals", {})

    try:
        # ── prepare one model per hospital ───────────────────────────────
        clients: List[Tuple[HeartDiseaseNet, Any, Any, int]] = []
        hospital_order = ["cleveland", "hungarian"]
        for name in hospital_order:
            h = hospitals_cfg.get(name, {})
            if h.get("registered", False):
                tl, tel, n = prepare_hospital_data(name, h.get("config", {}))
                model = HeartDiseaseNet()
                clients.append((model, tl, tel, n))

        if len(clients) < 2:
            s = load_state()
            s["federation"]["status"] = "error: need at least 2 registered hospitals"
            save_state(s)
            return

        # ── initialise global parameters from a fresh model ──────────────
        global_params = get_parameters(HeartDiseaseNet())

        train_losses: List[Tuple[int, float]] = []
        eval_losses: List[Tuple[int, float]] = []
        accuracies: List[Tuple[int, float]] = []
        accuracy_history: List[Tuple[int, float]] = []
        training_loss_history: List[Tuple[int, float]] = []
        distributed_loss_history: List[Tuple[int, float]] = []

        for rnd in range(1, num_rounds + 1):
            # ── FIT: each client trains on the global parameters ─────────
            fit_results: List[Tuple[list, int, float]] = []
            for model, tl, tel, n in clients:
                set_parameters(model, global_params)
                loss = train_model(model, tl, epochs=epochs, lr=lr)
                fit_results.append((get_parameters(model), n, loss))

            # ── FedAvg aggregation ───────────────────────────────────────
            total_n = sum(n for _, n, _ in fit_results)
            num_layers = len(fit_results[0][0])
            global_params = [
                sum(n * params[i] for params, n, _ in fit_results) / total_n
                for i in range(num_layers)
            ]

            avg_train_loss = sum(n * l for _, n, l in fit_results) / total_n
            train_losses.append((rnd, avg_train_loss))
            training_loss_history.append((rnd, avg_train_loss))

            _persist_progress(rnd, "trained", num_rounds,
                              train_losses, eval_losses, accuracies)

            # ── EVALUATE: each client evaluates the aggregated model ─────
            eval_results: List[Tuple[float, int, float]] = []
            for model, tl, tel, n in clients:
                set_parameters(model, global_params)
                loss, acc = test_model(model, tel)
                eval_results.append((loss, len(tel.dataset), acc))

            total_eval = sum(n for _, n, _ in eval_results)
            avg_eval_loss = sum(n * l for l, n, _ in eval_results) / total_eval
            avg_accuracy = sum(n * a for _, n, a in eval_results) / total_eval

            eval_losses.append((rnd, avg_eval_loss))
            accuracies.append((rnd, avg_accuracy))
            accuracy_history.append((rnd, avg_accuracy))
            distributed_loss_history.append((rnd, avg_eval_loss))

            _persist_progress(rnd, "evaluated", num_rounds,
                              train_losses, eval_losses, accuracies)

        # ── persist final results ────────────────────────────────────────
        results_dir = ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / "simulation_results.json", "w") as f:
            json.dump({
                "accuracy": accuracy_history,
                "training_loss": training_loss_history,
                "distributed_loss": distributed_loss_history,
                "num_rounds": num_rounds,
                "num_clients": len(clients),
            }, f, indent=2)

    except Exception as e:
        try:
            s = load_state()
            s["federation"]["status"] = f"error: {e}"
            save_state(s)
        except Exception:
            pass


def start_simulation_thread(config: Dict) -> threading.Thread:
    """Start the FL simulation in a non-blocking background thread."""
    state_snapshot = load_state()

    # Mark federation active immediately so the UI picks it up
    # without waiting for data preparation inside the thread.
    s = load_state()
    s["federation"].update({
        "active": True,
        "status": "training",
        "total_rounds": int(config.get("num_rounds", 3)),
        "started_at": datetime.now().isoformat(),
    })
    for h in ("cleveland", "hungarian"):
        if s["hospitals"][h].get("registered"):
            s["hospitals"][h]["status"] = "training"
    save_state(s)

    t = threading.Thread(
        target=_run_simulation,
        args=(state_snapshot, config),
        daemon=True,
    )
    t.start()
    return t
