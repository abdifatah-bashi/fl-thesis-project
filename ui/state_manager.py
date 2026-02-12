"""
Shared State Manager for FL Multi-Hospital Demo

Uses a JSON file as shared storage so all browser tabs (roles) can
read each other's updates in real-time via polling.
"""

import json
import os
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

STATE_FILE = Path("results/fl_state.json")

DEFAULT_STATE: Dict[str, Any] = {
    "hospitals": {
        "cleveland": {
            "registered": False,
            "num_patients": 0,
            "disease_rate": 0.0,
            "status": "waiting",   # waiting | ready | training | done
            "config": {
                "epochs_per_round": 1,
                "batch_size": 32,
                "learning_rate": 0.01,
                "train_split": 0.8,
            },
            "registered_at": None,
        },
        "hungarian": {
            "registered": False,
            "num_patients": 0,
            "disease_rate": 0.0,
            "status": "waiting",
            "config": {
                "epochs_per_round": 1,
                "batch_size": 32,
                "learning_rate": 0.01,
                "train_split": 0.8,
            },
            "registered_at": None,
        },
    },
    "federation": {
        "active": False,
        "current_round": 0,
        "total_rounds": 3,
        "status": "waiting",      # waiting | training | complete | error
        "started_at": None,
        "completed_at": None,
        "celebrated": False,
        "history": {
            "rounds": [],
            "accuracy": [],
            "train_loss": [],
            "eval_loss": [],
        },
    },
    "last_updated": None,
}


def load_state() -> Dict[str, Any]:
    """Load shared federation state from JSON file."""
    if not STATE_FILE.exists():
        return copy.deepcopy(DEFAULT_STATE)
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        merged = copy.deepcopy(DEFAULT_STATE)
        _deep_merge(merged, data)
        return merged
    except (json.JSONDecodeError, IOError):
        return copy.deepcopy(DEFAULT_STATE)


def save_state(state: Dict[str, Any]) -> None:
    """Save shared federation state to JSON file (atomic write)."""
    STATE_FILE.parent.mkdir(exist_ok=True)
    state["last_updated"] = datetime.now().isoformat()
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def reset_state() -> None:
    """Reset all federation state to defaults."""
    save_state(copy.deepcopy(DEFAULT_STATE))


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override values into base dict in place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
