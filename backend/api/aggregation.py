from datetime import datetime
import numpy as np
import torch
import sys
from pathlib import Path
import threading
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

# Global lock to serialize aggregation, preventing parallel race conditions
aggregation_lock = threading.Lock()

def run_aggregation_round() -> dict:
    """Run FedAvg over all currently submitted weight files for a single round.
    Discovers clients dynamically from weights/ directory.
    Secured by a thread lock to prevent parallel background tasks from conflicting.
    """
    with aggregation_lock:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        state = load()

        # Increment round sequentially
        rnd = state.get("current_round", 0) + 1
        state.update({
            "status": "aggregating",
            "current_round": rnd,
        })
        
        # Initialize history if missing
        if "history" not in state:
            state["history"] = {"rounds": [], "accuracy": [], "train_loss": [], "eval_loss": []}
            
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

            # Remove the weight file so stale weights aren't reused next round!
            # (This forces clients to participate in every round if they want to contribute)
            wf.unlink(missing_ok=True)

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
        
        state.update({
            "status": "ready", # Ready for next round
            "completed_at": datetime.now().isoformat(),
        })
        save(state)
        return {"round_completed": rnd, "clients_aggregated": len(weight_files)}
