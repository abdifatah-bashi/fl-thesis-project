from datetime import datetime
import numpy as np
import torch
import sys
from pathlib import Path
from flwr.server.strategy.aggregate import aggregate as flwr_aggregate

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model import HeartDiseaseNet
from src.utils import get_parameters, set_parameters
from api.state import GLOBAL_PARAMS, WEIGHTS_DIR, load, save

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
    Discovers clients dynamically from weights/ directory using Flower.
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
        results = [(params, n) for params, n in zip(all_params, all_samples)]
        aggregated = flwr_aggregate(results)

        # Save new global model
        model = HeartDiseaseNet()
        set_parameters(model, aggregated)
        torch.save(get_parameters(model), GLOBAL_PARAMS)

        # Aggregate round metrics
        total = sum(all_samples)
        weights = [n / total for n in all_samples]
        
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
