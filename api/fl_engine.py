"""
FL Engine — Central Server logic.
Handles model publishing, FedAvg aggregation, and hospital local training.
"""
import io
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from flwr.server.strategy.aggregate import aggregate as flwr_aggregate

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import HeartDiseaseNet, train_model, test_model
from src.utils import get_parameters, set_parameters
from api.state import GLOBAL_PARAMS, WEIGHTS_DIR, load, save

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

SAMPLE_DATA: dict[str, Path] = {
    "cleveland":  ROOT / "data/heart_disease/raw/processed.cleveland.data",
    "hungarian":  ROOT / "data/heart_disease/raw/processed.hungarian.data",
    "va":         ROOT / "data/heart_disease/raw/processed.va.data",
    "switzerland": ROOT / "data/heart_disease/raw/processed.switzerland.data",
}


def _parse_csv(csv_bytes: bytes) -> tuple:
    df = pd.read_csv(io.StringIO(csv_bytes.decode()), names=COLUMNS, na_values="?")
    df["target"] = (df["num"] > 0).astype(int)
    X = df.drop(["num", "target"], axis=1)
    X = X.fillna(X.mean())
    y = df["target"]
    return X.values, y.values


def _make_loaders(X, y, batch_size: int = 32):
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


def train_hospital(
    name: str,
    csv_bytes: bytes | None,
    dataset: str | None,
    epochs: int,
    learning_rate: float,
) -> dict:
    """Train a hospital's local model and submit weight updates.
    Data is processed in isolation per hospital — never shared between hospitals.
    """
    if not GLOBAL_PARAMS.exists():
        raise ValueError("Global model not published yet.")

    # Resolve data source
    if csv_bytes:
        X, y = _parse_csv(csv_bytes)
    elif dataset and dataset.lower() in SAMPLE_DATA:
        path = SAMPLE_DATA[dataset.lower()]
        if not path.exists():
            raise FileNotFoundError(f"Sample data not found: {path}")
        X, y = _parse_csv(path.read_bytes())
    else:
        raise ValueError("Provide a CSV file or a valid dataset name (cleveland, hungarian, va, switzerland).")

    train_dl, test_dl, num_train = _make_loaders(X, y)

    # Load the current global model
    model = HeartDiseaseNet()
    global_params = torch.load(GLOBAL_PARAMS, weights_only=False)
    set_parameters(model, global_params)

    # Train locally
    train_loss = train_model(model, train_dl, epochs=epochs, lr=learning_rate)
    eval_loss, accuracy = test_model(model, test_dl)

    # Save weight update
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": get_parameters(model),
        "num_samples": num_train,
        "metrics": {
            "train_loss": float(train_loss),
            "eval_loss": float(eval_loss),
            "accuracy": float(accuracy),
        },
    }
    torch.save(payload, WEIGHTS_DIR / f"{name}.pt")

    # Self-register and update client state
    state = load()
    if name not in state["clients"]:
        state["clients"][name] = {"rounds_submitted": 0, "num_samples": 0, "last_seen": None, "metrics": {}}
    prev = state["clients"][name]
    state["clients"][name] = {
        "rounds_submitted": prev.get("rounds_submitted", 0) + 1,
        "num_samples": num_train,
        "metrics": payload["metrics"],
        "last_seen": datetime.now().isoformat(),
    }
    save(state)

    return {"trained": True, "num_samples": num_train, "metrics": payload["metrics"]}


def publish_model() -> dict:
    """Create a fresh HeartDiseaseNet and save global_params.pt.
    Any client can now fetch it. No hospital knowledge needed.
    """
    GLOBAL_PARAMS.parent.mkdir(parents=True, exist_ok=True)
    model = HeartDiseaseNet()
    torch.save(get_parameters(model), GLOBAL_PARAMS)

    state = load()
    state.update({
        "status": "ready",
        "model_published": True,
        "published_at": datetime.now().isoformat(),
        "current_round": 0,
    })
    save(state)
    return {"published": True, "path": str(GLOBAL_PARAMS)}


def aggregate(num_rounds: int, epochs_per_round: int, learning_rate: float) -> dict:
    """Run FedAvg over all submitted weight files.
    Discovers clients dynamically from weights/ directory — no hardcoded names.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    state = load()

    state.update({
        "status": "aggregating",
        "total_rounds": num_rounds,
        "started_at": datetime.now().isoformat(),
        "history": {"rounds": [], "accuracy": [], "train_loss": [], "eval_loss": []},
    })
    save(state)

    for rnd in range(1, num_rounds + 1):
        state = load()
        state["current_round"] = rnd
        save(state)

        # Discover submitted weights dynamically
        weight_files = list(WEIGHTS_DIR.glob("*.pt"))
        if not weight_files:
            state["status"] = "error: no weight files found"
            save(state)
            return {"error": "No weight files found for aggregation"}

        # FedAvg — weighted average by num_samples
        all_params: list[list[np.ndarray]] = []
        all_samples: list[int] = []
        round_metrics: dict[str, list] = {"train_loss": [], "eval_loss": [], "accuracy": []}

        for wf in weight_files:
            client_name = wf.stem  # filename without .pt = hospital name
            data = torch.load(wf, weights_only=False)
            params = data["params"]
            n = data.get("num_samples", 1)
            all_params.append(params)
            all_samples.append(n)

            # Collect per-client metrics
            m = data.get("metrics", {})
            if m.get("train_loss") is not None:
                round_metrics["train_loss"].append(m["train_loss"])
            if m.get("eval_loss") is not None:
                round_metrics["eval_loss"].append(m["eval_loss"])
            if m.get("accuracy") is not None:
                round_metrics["accuracy"].append(m["accuracy"])

            # Update client record in state
            state = load()
            if client_name not in state["clients"]:
                state["clients"][client_name] = {"rounds_submitted": 0, "num_samples": n}
            state["clients"][client_name]["rounds_submitted"] = rnd
            state["clients"][client_name]["num_samples"] = n
            state["clients"][client_name]["metrics"] = m
            state["clients"][client_name]["last_seen"] = datetime.now().isoformat()
            save(state)

        # Flower's FedAvg — weighted average by num_samples
        # Convert to Flower's expected format: List[Tuple[List[np.ndarray], int]]
        results = [(params, n) for params, n in zip(all_params, all_samples)]
        aggregated = flwr_aggregate(results)

        # Save new global model
        model = HeartDiseaseNet()
        set_parameters(model, aggregated)
        torch.save(get_parameters(model), GLOBAL_PARAMS)

        # Aggregate round metrics
        def wavg(vals):
            if not vals:
                return None
            return float(sum(w * v for w, v in zip(weights[:len(vals)], vals)))

        state = load()
        h = state["history"]
        h["rounds"].append(rnd)
        h["train_loss"].append(wavg(round_metrics["train_loss"]))
        h["eval_loss"].append(wavg(round_metrics["eval_loss"]))
        h["accuracy"].append(wavg(round_metrics["accuracy"]))
        save(state)

    state = load()
    state.update({
        "status": "complete",
        "completed_at": datetime.now().isoformat(),
    })
    save(state)
    return {"rounds_completed": num_rounds, "clients": len(weight_files)}
