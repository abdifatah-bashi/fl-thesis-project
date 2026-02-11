"""
FL Runner — Bridge between Streamlit UI and Flower simulation.

Runs the federated learning simulation in a background thread
and reports per-round progress to the shared state JSON file.
"""

import sys
import os
import json
import logging
import warnings
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

# ── suppress noisy FL / Ray logs ────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_INIT_LOG_TO_DRIVER"] = "0"
warnings.simplefilter("ignore")
logging.getLogger("flwr").setLevel(logging.CRITICAL)
logging.getLogger("ray").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import flwr as fl

from src.model import HeartDiseaseNet, train_model, test_model
from src.client import HeartDiseaseClient
from src.server import weighted_average, weighted_average_loss
from src.utils import get_parameters, set_parameters
from state_manager import load_state, save_state

# ── constants ────────────────────────────────────────────────────────────────
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
UPLOAD_DIR = ROOT / "data" / "hospital_uploads"


# ── custom FedAvg strategy with progress callbacks ───────────────────────────

class ProgressFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that writes per-round metrics to the shared state file."""

    def __init__(self, total_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.total_rounds = total_rounds
        self._train_losses: List[Tuple[int, float]] = []
        self._eval_losses: List[Tuple[int, float]] = []
        self._accuracies: List[Tuple[int, float]] = []

    def _persist(self, server_round: int, phase: str) -> None:
        try:
            state = load_state()
            fed = state["federation"]
            fed["current_round"] = server_round
            fed["status"] = f"round_{server_round}_{phase}"
            hist = fed["history"]
            hist["rounds"] = [r for r, _ in self._accuracies] or [r for r, _ in self._train_losses]
            hist["accuracy"] = [round(a, 4) for _, a in self._accuracies]
            hist["train_loss"] = [round(l, 4) for _, l in self._train_losses]
            hist["eval_loss"] = [round(l, 4) for _, l in self._eval_losses]
            if server_round >= self.total_rounds and phase == "evaluated":
                fed["status"] = "complete"
                fed["completed_at"] = datetime.now().isoformat()
                for h in ("cleveland", "hungarian"):
                    if state["hospitals"][h]["registered"]:
                        state["hospitals"][h]["status"] = "done"
            save_state(state)
        except Exception:
            pass

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if results:
            total = sum(r.num_examples for _, r in results)
            loss_sum = sum(r.num_examples * r.metrics.get("loss", 0.0) for _, r in results)
            self._train_losses.append((server_round, loss_sum / total if total else 0.0))
        # Mark hospitals as training
        try:
            state = load_state()
            for h in ("cleveland", "hungarian"):
                if state["hospitals"][h]["registered"]:
                    state["hospitals"][h]["status"] = "training"
            save_state(state)
        except Exception:
            pass
        self._persist(server_round, "trained")
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if results:
            total = sum(r.num_examples for _, r in results)
            loss_sum = sum(r.num_examples * r.loss for _, r in results)
            acc_sum = sum(r.num_examples * r.metrics.get("accuracy", 0.0) for _, r in results)
            self._eval_losses.append((server_round, loss_sum / total if total else 0.0))
            self._accuracies.append((server_round, acc_sum / total if total else 0.0))
        self._persist(server_round, "evaluated")
        return aggregated


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
    """Internal: run FL simulation (called from background thread)."""
    num_rounds = int(config.get("num_rounds", 3))
    hospitals_cfg = state_snapshot.get("hospitals", {})

    try:
        client_data: Dict[str, Tuple] = {}
        hospital_order = ["cleveland", "hungarian"]
        idx = 0
        for name in hospital_order:
            h = hospitals_cfg.get(name, {})
            if h.get("registered", False):
                tl, tel, n = prepare_hospital_data(name, h.get("config", {}))
                client_data[str(idx)] = (tl, tel, n)
                idx += 1

        num_clients = len(client_data)
        if num_clients < 2:
            s = load_state()
            s["federation"]["status"] = "error: need at least 2 registered hospitals"
            save_state(s)
            return

        def client_fn(context: fl.common.Context) -> fl.client.Client:
            pid = str(context.node_config["partition-id"])
            tl, tel, n = client_data[pid]
            return HeartDiseaseClient(int(pid), tl, tel, n).to_client()

        strategy = ProgressFedAvg(
            total_rounds=num_rounds,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            fit_metrics_aggregation_fn=weighted_average_loss,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=lambda r: {
                "epochs": int(config.get("epochs_per_round", 1)),
                "lr": float(config.get("learning_rate", 0.01)),
                "round": r,
            },
        )

        # Mark federation as active
        s = load_state()
        s["federation"].update({
            "active": True,
            "status": "training",
            "total_rounds": num_rounds,
            "started_at": datetime.now().isoformat(),
        })
        save_state(s)

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )

        # Persist final results to the standard results file
        results_dir = ROOT / "results"
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / "simulation_results.json", "w") as f:
            json.dump({
                "accuracy": history.metrics_distributed.get("accuracy", []),
                "training_loss": history.metrics_distributed_fit.get("loss", []),
                "distributed_loss": history.losses_distributed,
                "num_rounds": num_rounds,
                "num_clients": num_clients,
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
    t = threading.Thread(
        target=_run_simulation,
        args=(state_snapshot, config),
        daemon=True,
    )
    t.start()
    return t
