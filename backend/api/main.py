"""
FL Central Server — FastAPI

Endpoints:
  POST /api/model/publish              → publish initial global model
  GET  /api/model/global               → download global_params.pt (binary)
  GET  /api/model/global/json          → model weights as JSON for TF.js
  POST /api/hospital/{name}/submit     → hospital submits trained weights (JSON from browser)
  GET  /api/data/sample/{name}         → serve demo CSV data to hospital browser
  POST /api/aggregate                  → run FedAvg over all submissions
  GET  /api/status                     → federation state
  GET  /api/results                    → training history
  DELETE /api/reset                    → reset everything
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from api.state import load, save, reset, GLOBAL_PARAMS, WEIGHTS_DIR
from api.aggregation import publish_model, run_aggregation_round
from api.config import SAMPLE_DATA

app = FastAPI(title="FL Central Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────────────────

class AggregateRequest(BaseModel):
    num_rounds: int = 3
    epochs_per_round: int = 1
    learning_rate: float = 0.01


class LayerWeights(BaseModel):
    kernel: list  # shape [in_features, out_features]  ← TF.js format
    bias: list    # shape [out_features]


class WeightSubmission(BaseModel):
    """JSON weight payload sent by the hospital's browser after local TF.js training."""
    layers: list[LayerWeights]
    num_samples: int
    metrics: dict


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/model/publish")
def publish():
    """Step 1: Server publishes the initial global model. No hospital dependency."""
    return publish_model()


@app.get("/api/model/global")
def get_global_model():
    """Download global_params.pt (binary). For standalone Python clients."""
    if not GLOBAL_PARAMS.exists():
        raise HTTPException(status_code=404, detail="Global model not published yet.")
    return FileResponse(str(GLOBAL_PARAMS), media_type="application/octet-stream", filename="global_params.pt")


@app.get("/api/model/global/json")
def get_global_model_json():
    """Return global model weights as JSON in TF.js format (kernel transposed to [in, out]).
    The browser loads this, trains locally with TF.js, then submits only weight updates.
    Patient data NEVER leaves the hospital browser tab.
    """
    if not GLOBAL_PARAMS.exists():
        raise HTTPException(status_code=404, detail="Global model not published yet.")

    params = torch.load(GLOBAL_PARAMS, weights_only=False)
    # params = [fc1.weight[64,13], fc1.bias[64], fc2.weight[32,64], fc2.bias[32], fc3.weight[1,32], fc3.bias[1]]
    # TF.js Dense layer expects kernel shape [in_features, out_features], so we transpose PyTorch's [out, in]
    layers = []
    for i in range(0, len(params), 2):
        w = params[i]    # [out, in] in PyTorch
        b = params[i + 1]  # [out]
        layers.append({"kernel": w.T.tolist(), "bias": b.tolist()})

    return {"layers": layers}


@app.post("/api/hospital/{name}/submit")
def submit_weights_json(name: str, body: WeightSubmission, background_tasks: BackgroundTasks):
    """Hospital browser submits trained weights as JSON after local TF.js training.
    Receives TF.js format (kernel [in, out]), converts back to PyTorch format for FedAvg.
    """
    # Convert TF.js [in, out] kernel back to PyTorch [out, in]
    params = []
    for layer in body.layers:
        kernel = np.array(layer.kernel)  # [in, out]
        bias = np.array(layer.bias)      # [out]
        params.append(kernel.T)          # → [out, in] for PyTorch
        params.append(bias)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"params": params, "num_samples": body.num_samples, "metrics": body.metrics}
    torch.save(payload, WEIGHTS_DIR / f"{name}.pt")

    state = load()
    if name not in state["clients"]:
        state["clients"][name] = {"rounds_submitted": 0, "num_samples": 0, "last_seen": None, "metrics": {}}
    prev = state["clients"][name]
    state["clients"][name] = {
        "rounds_submitted": prev.get("rounds_submitted", 0) + 1,
        "num_samples": body.num_samples,
        "metrics": body.metrics,
        "last_seen": datetime.now().isoformat(),
    }
    save(state)
    
    # Automatically aggregate immediately after a hospital submits!
    background_tasks.add_task(run_aggregation_round)
    
    return {"submitted": True, "client": name, "num_samples": body.num_samples, "metrics": body.metrics}


@app.get("/api/data/sample/{name}")
def get_sample_data(name: str):
    """Serve demo UCI dataset as CSV text. For demonstration only — real hospitals
    use their own local files via the browser FileReader API (data never leaves).
    """
    if name not in SAMPLE_DATA:
        raise HTTPException(404, f"Unknown dataset '{name}'. Available: {list(SAMPLE_DATA.keys())}")
    path = SAMPLE_DATA[name]
    if not path.exists():
        raise HTTPException(404, f"Sample data not found at {path}")
    return PlainTextResponse(path.read_text(), media_type="text/plain")


@app.post("/api/weights/{client_name}")
async def submit_weights_file(
    client_name: str,
    file: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks
):
    """Accept a .pt weight file from a standalone Python hospital client."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / f"{client_name}.pt"
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)

    state = load()
    if client_name not in state["clients"]:
        state["clients"][client_name] = {"rounds_submitted": 0, "num_samples": 0, "last_seen": None, "metrics": {}}
    state["clients"][client_name]["last_seen"] = datetime.now().isoformat()
    save(state)

    # Automatically aggregate
    background_tasks.add_task(run_aggregation_round)

    return {"received": True, "client": client_name}


@app.post("/api/aggregate")
def run_aggregate(req: AggregateRequest, background_tasks: BackgroundTasks):
    """Fallback manual trigger for FedAvg aggregation over all submitted weights."""
    if not GLOBAL_PARAMS.exists():
        raise HTTPException(status_code=400, detail="Publish global model first.")
    if not list(WEIGHTS_DIR.glob("*.pt")):
        raise HTTPException(status_code=400, detail="No weight submissions yet.")
    
    # Execute single round manually via dashboard if needed
    background_tasks.add_task(run_aggregation_round)
    return {"started": True}


@app.get("/api/status")
def get_status():
    state = load()
    return {
        "status": state["status"],
        "model_published": state["model_published"],
        "current_round": state["current_round"],
        "total_rounds": state["total_rounds"],
        "clients": state["clients"],
        "published_at": state.get("published_at"),
        "started_at": state.get("started_at"),
        "completed_at": state.get("completed_at"),
    }


@app.get("/api/results")
def get_results():
    state = load()
    return state["history"]


@app.delete("/api/reset")
def reset_all():
    reset()
    return {"reset": True}


@app.get("/")
def root():
    return {"service": "FL Central Server", "docs": "/docs"}
