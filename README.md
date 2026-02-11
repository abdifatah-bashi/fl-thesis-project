# Federated Learning Thesis Project

A Federated Learning framework for heart disease prediction using Flower (flwr) and PyTorch.

## Project Structure

This project follows a standard Python package structure:

```
fl-thesis-project/
├── src/
│   └── fl_core/             # Core FL framework package
│       ├── client.py        # Client implementation
│       ├── server.py        # Server strategy
│       ├── model.py         # PyTorch model definition
│       ├── data_loader.py   # Data loading and preprocessing
│       └── utils.py         # Utility functions
├── scripts/
│   ├── download_data.py     # Script to download UCI Heart Disease dataset
│   └── run_client.py        # Script to run a single client
├── run_simulation.py        # Main simulation entry point
├── pyproject.toml           # Project configuration and dependencies
└── requirements.txt         # Python dependencies
```

## Installation

1.  **Create a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install the package in editable mode**:
    ```bash
    pip install -e .
    ```
    This will install all dependencies listed in `pyproject.toml` and `requirements.txt`.

## Usage

### 1. Download Data
First, download the required dataset:
```bash
python scripts/download_data.py
```

### 2. Run Simulation
Run the full Federated Learning simulation (Server + 2 Clients):
```bash
python run_simulation.py
```

### 3. Run Individual Components (Advanced)
You can also run clients manually if you have a running server:
```bash
python scripts/run_client.py 0  # Run client 0 (Cleveland)
python scripts/run_client.py 1  # Run client 1 (Hungarian)
```

## Features
- **Federated Learning**: Uses Flower framework.
- **Privacy**: Data remains on local clients; only model updates are shared.
- **Model**: Neural Network for heart disease classification.
- **Data**: UCI Heart Disease dataset (Cleveland, Hungarian, etc.).
