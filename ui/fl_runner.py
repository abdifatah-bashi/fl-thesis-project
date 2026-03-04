"""
FL Runner — True Federated Learning with independent roles.

Architecture:
  - Central Server: creates initial model, publishes global_params.pt,
    aggregates weight files from hospitals. NEVER touches raw data.
  - Hospitals: fetch global_params.pt, train locally on own data,
    submit only weights_{name}.pt. Data never leaves the hospital.

Communication is via shared state JSON + .pt weight files in results/.
"""

import sys
import json
import time
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
from state_manager import load_state, save_state, get_or_create_hospital

RESULTS_DIR = ROOT / "results"

# ── constants ────────────────────────────────────────────────────────────────
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
UPLOAD_DIR = ROOT / "data" / "hospital_uploads"




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
        h = get_or_create_hospital(state, hospital_name)
        h.update({
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
        h = get_or_create_hospital(state, hospital_name)
        h["config"] = config
        save_state(state)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# CENTRAL SERVER — aggregation only, never touches raw data
# ══════════════════════════════════════════════════════════════════════════════

def _server_aggregation_loop(total_rounds: int, registered: List[str]) -> None:
    """Background thread: wait for hospital weight submissions, FedAvg, repeat."""
    train_losses: List[Tuple[int, float]] = []
    eval_losses: List[Tuple[int, float]] = []
    accuracies: List[Tuple[int, float]] = []

    try:
        for rnd in range(1, total_rounds + 1):
            # ── wait for ALL hospitals to submit weights for this round ──
            while True:
                s = load_state()
                all_submitted = all(
                    s["hospitals"][name].get("round_submitted", 0) >= rnd
                    for name in registered
                )
                if all_submitted:
                    break
                time.sleep(0.3)

            # ── load weight files (NEVER raw data) ───────────────────────
            fit_results: List[Tuple[list, int]] = []
            for name in registered:
                params = torch.load(RESULTS_DIR / f"weights_{name}.pt",
                                    weights_only=True)
                n_samples = s["hospitals"][name].get("num_train_samples", 1)
                fit_results.append((params, n_samples))

            # ── FedAvg aggregation ───────────────────────────────────────
            total_n = sum(n for _, n in fit_results)
            num_layers = len(fit_results[0][0])
            global_params = [
                sum(n * params[i] for params, n in fit_results) / total_n
                for i in range(num_layers)
            ]

            # ── publish new global model ─────────────────────────────────
            torch.save(global_params, RESULTS_DIR / "global_params.pt")

            # ── aggregate per-hospital metrics from state ────────────────
            avg_train_loss = sum(
                s["hospitals"][n].get("num_train_samples", 1)
                * s["hospitals"][n].get("local_train_loss", 0)
                for n in registered
            ) / total_n
            avg_eval_loss = sum(
                s["hospitals"][n].get("num_train_samples", 1)
                * s["hospitals"][n].get("local_eval_loss", 0)
                for n in registered
            ) / total_n
            avg_accuracy = sum(
                s["hospitals"][n].get("num_train_samples", 1)
                * s["hospitals"][n].get("local_accuracy", 0)
                for n in registered
            ) / total_n

            train_losses.append((rnd, avg_train_loss))
            eval_losses.append((rnd, avg_eval_loss))
            accuracies.append((rnd, avg_accuracy))

            # ── update federation state ──────────────────────────────────
            s = load_state()
            fed = s["federation"]
            hist = fed["history"]
            hist["rounds"] = [r for r, _ in accuracies]
            hist["accuracy"] = [round(a, 4) for _, a in accuracies]
            hist["train_loss"] = [round(l, 4) for _, l in train_losses]
            hist["eval_loss"] = [round(l, 4) for _, l in eval_losses]

            if rnd >= total_rounds:
                fed["current_round"] = rnd
                fed["status"] = "complete"
                fed["completed_at"] = datetime.now().isoformat()
                for name in registered:
                    s["hospitals"][name]["status"] = "done"
            else:
                fed["current_round"] = rnd + 1
                fed["status"] = f"round_{rnd}_aggregated"

            save_state(s)

        # ── save final results file ──────────────────────────────────────
        with open(RESULTS_DIR / "simulation_results.json", "w") as f:
            json.dump({
                "accuracy": [(r, a) for r, a in accuracies],
                "training_loss": [(r, l) for r, l in train_losses],
                "distributed_loss": [(r, l) for r, l in eval_losses],
                "num_rounds": total_rounds,
                "num_clients": len(registered),
            }, f, indent=2)

    except Exception as e:
        try:
            s = load_state()
            s["federation"]["status"] = f"error: {e}"
            save_state(s)
        except Exception:
            pass


def publish_global_model() -> None:
    """Central Server Step 1: create initial model and publish global_params.pt.

    No dependencies — can be done before any hospital registers.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    model = HeartDiseaseNet()
    torch.save(get_parameters(model), RESULTS_DIR / "global_params.pt")

    s = load_state()
    s["federation"].update({
        "model_published": True,
        "published_at": datetime.now().isoformat(),
    })
    save_state(s)


def start_aggregation(config: Dict) -> threading.Thread:
    """Central Server Step 2: start the aggregation loop.

    Requires at least one hospital to have submitted weights.
    """
    total_rounds = int(config.get("num_rounds", 3))

    s = load_state()
    registered = [n for n, h in s["hospitals"].items() if h.get("registered")]
    s["federation"].update({
        "active": True,
        "status": "training",
        "current_round": 1,
        "total_rounds": total_rounds,
        "started_at": datetime.now().isoformat(),
    })
    save_state(s)

    t = threading.Thread(
        target=_server_aggregation_loop,
        args=(total_rounds, registered),
        daemon=True,
    )
    t.start()
    return t


# ══════════════════════════════════════════════════════════════════════════════
# HOSPITAL — local training only, data never leaves
# ══════════════════════════════════════════════════════════════════════════════

def _hospital_training_loop(hospital_name: str) -> None:
    """Background thread: fetch global params → train locally → submit weights.

    Round 1 runs immediately using the published global_params.pt.
    Rounds 2+ wait for the server to aggregate and publish new params.
    """
    try:
        # Load THIS hospital's own data (never another hospital's)
        s = load_state()
        h_cfg = s["hospitals"][hospital_name].get("config", {})
        trainloader, testloader, num_train = prepare_hospital_data(
            hospital_name, h_cfg
        )

        total_rounds = s["federation"].get("total_rounds", 3)
        epochs = int(h_cfg.get("epochs_per_round", 1))
        lr = float(h_cfg.get("learning_rate", 0.01))
        model = HeartDiseaseNet()

        for rnd in range(1, total_rounds + 1):
            # ── Round 1: use published params. Round 2+: wait for server ─
            if rnd > 1:
                while True:
                    s = load_state()
                    fed = s["federation"]
                    if fed.get("current_round", 0) > rnd - 1:
                        break
                    if fed.get("status", "").startswith("error"):
                        return
                    time.sleep(0.3)

            # ── fetch global model (params only, not data) ───────────────
            global_params = torch.load(
                RESULTS_DIR / "global_params.pt", weights_only=True
            )
            set_parameters(model, global_params)

            # ── train locally on own private data ────────────────────────
            loss = train_model(model, trainloader, epochs=epochs, lr=lr)

            # ── evaluate locally ─────────────────────────────────────────
            eval_loss, accuracy = test_model(model, testloader)

            # ── submit only weight updates (never raw data) ──────────────
            torch.save(
                get_parameters(model),
                RESULTS_DIR / f"weights_{hospital_name}.pt",
            )

            # ── update state: mark round as submitted + local metrics ────
            s = load_state()
            s["hospitals"][hospital_name].update({
                "round_submitted": rnd,
                "local_train_loss": round(loss, 4),
                "local_eval_loss": round(eval_loss, 4),
                "local_accuracy": round(accuracy, 4),
                "num_train_samples": num_train,
                "status": "done" if rnd >= total_rounds else "training",
            })
            save_state(s)

    except Exception as e:
        try:
            s = load_state()
            s["hospitals"][hospital_name]["status"] = f"error: {e}"
            save_state(s)
        except Exception:
            pass


def start_hospital_training(hospital_name: str) -> threading.Thread:
    """Hospital: fetch global model, train locally, send updates."""
    s = load_state()
    h = get_or_create_hospital(s, hospital_name)
    h.update({
        "training_joined": True,
        "status": "training",
        "round_submitted": 0,
    })
    save_state(s)

    t = threading.Thread(
        target=_hospital_training_loop,
        args=(hospital_name,),
        daemon=True,
    )
    t.start()
    return t
