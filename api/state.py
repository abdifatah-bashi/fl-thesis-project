"""
Federation state — persisted as JSON.
The server knows nothing about specific hospitals until they submit weights.
"""
import json
import os
import copy
from pathlib import Path
from datetime import datetime
from typing import Any

STATE_FILE = Path(__file__).parent.parent / "results" / "fl_state.json"
WEIGHTS_DIR = Path(__file__).parent.parent / "results" / "weights"
GLOBAL_PARAMS = Path(__file__).parent.parent / "results" / "global_params.pt"

DEFAULT: dict[str, Any] = {
    "status": "idle",          # idle | ready | aggregating | complete
    "model_published": False,
    "published_at": None,
    "current_round": 0,
    "total_rounds": 3,
    "started_at": None,
    "completed_at": None,
    "clients": {},             # name → {rounds_submitted, last_seen, num_samples, metrics}
    "history": {
        "rounds": [],
        "accuracy": [],
        "train_loss": [],
        "eval_loss": [],
    },
}


def load() -> dict:
    if not STATE_FILE.exists():
        return copy.deepcopy(DEFAULT)
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        merged = copy.deepcopy(DEFAULT)
        _merge(merged, data)
        return merged
    except (json.JSONDecodeError, OSError):
        return copy.deepcopy(DEFAULT)


def save(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def reset() -> None:
    save(copy.deepcopy(DEFAULT))
    # Remove all weight files
    if WEIGHTS_DIR.exists():
        for f in WEIGHTS_DIR.glob("*.pt"):
            f.unlink(missing_ok=True)
    GLOBAL_PARAMS.unlink(missing_ok=True)


def _merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _merge(base[k], v)
        else:
            base[k] = v
