from datetime import datetime
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import HeartDiseaseNet, train_model, test_model
from src.utils import get_parameters, set_parameters
from api.state import GLOBAL_PARAMS, WEIGHTS_DIR, load, save
from api.config import SAMPLE_DATA
from api.data_processing import parse_csv, make_loaders

def train_hospital(
    name: str,
    csv_bytes: bytes | None,
    dataset: str | None,
    epochs: int,
    learning_rate: float,
) -> dict:
    """Train a hospital's local model and submit weight updates via Python.
    Data is processed in isolation per hospital.
    """
    if not GLOBAL_PARAMS.exists():
        raise ValueError("Global model not published yet.")

    # Resolve data source
    if csv_bytes:
        X, y = parse_csv(csv_bytes)
    elif dataset and dataset.lower() in SAMPLE_DATA:
        path = SAMPLE_DATA[dataset.lower()]
        if not path.exists():
            raise FileNotFoundError(f"Sample data not found: {path}")
        X, y = parse_csv(path.read_bytes())
    else:
        raise ValueError("Provide a CSV file or a valid dataset name (cleveland, hungarian, va, switzerland).")

    train_dl, test_dl, num_train = make_loaders(X, y)

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
