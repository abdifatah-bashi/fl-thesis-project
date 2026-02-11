# Server

`src/server.py`

Server initialization, FedAvg strategy configuration, and metric aggregation.

---

## `weighted_average()`

```python
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics
```

Aggregate evaluation metrics from multiple clients using a weighted average. Hospitals with more patients have proportionally more influence on the reported accuracy.

| Parameter | Type | Description |
|-----------|------|-------------|
| `metrics` | `List[Tuple[int, Metrics]]` | List of `(num_examples, {"accuracy": float})` per client |

**Returns**: `Dict` — `{"accuracy": float}` — the weighted global accuracy.

### Example

```python
# Cleveland: 60 test samples, 85% accuracy
# Hungarian: 58 test samples, 80% accuracy
metrics = [
    (60, {"accuracy": 0.85}),
    (58, {"accuracy": 0.80}),
]

result = weighted_average(metrics)
# result = {"accuracy": 0.8254}
# (60*0.85 + 58*0.80) / (60+58) = 0.8254
```

---

## `start_server()`

```python
def start_server(num_rounds=5, num_clients=2, server_address="0.0.0.0:8080")
```

Initialize and launch the FL server with a FedAvg strategy.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_rounds` | `int` | `5` | Number of federated training rounds |
| `num_clients` | `int` | `2` | Minimum clients required before starting |
| `server_address` | `str` | `"0.0.0.0:8080"` | gRPC bind address |

### Strategy configuration

The function internally creates a `FedAvg` strategy with these settings:

| Setting | Value | Effect |
|---------|-------|--------|
| `fraction_fit` | `1.0` | Train on 100% of available clients |
| `fraction_evaluate` | `1.0` | Evaluate on 100% of available clients |
| `min_fit_clients` | `num_clients` | Wait for all clients before starting |
| `min_available_clients` | `num_clients` | Minimum connected clients |
| `evaluate_metrics_aggregation_fn` | `weighted_average` | Custom metric aggregation |

### Client training config

Sent to every client at the start of each round:

```python
{"epochs": 1, "lr": 0.01, "round": current_round}
```

### Example

```python
from fl_core.server import start_server

# Start server — blocks until all rounds complete
start_server(num_rounds=3, num_clients=2)
```

!!! info "Blocking call"
    `start_server()` blocks the current thread. It waits for clients to connect, then runs the training loop for `num_rounds` rounds. Run clients in separate processes or use the simulation engine.
