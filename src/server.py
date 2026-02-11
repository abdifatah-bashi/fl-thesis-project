"""
FL Server - 2 HOSPITALS VERSION
"""

import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics using weighted average."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    weighted_accuracy = sum(accuracies) / total_examples
    return {"accuracy": weighted_accuracy}

def weighted_average_loss(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate training loss metrics using weighted average."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    weighted_loss = sum(losses) / total_examples
    return {"loss": weighted_loss}

def start_server(num_rounds=5, num_clients=2, server_address="0.0.0.0:8080"):
    """Start FL server for 2 hospitals."""
    print("=" * 60)
    print("FEDERATED LEARNING SERVER - 2 HOSPITALS (Cleveland & Hungarian)")
    print("=" * 60)
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Define config for each round
    strategy.on_fit_config_fn = lambda r: {"epochs": 1, "lr": 0.01, "round": r}
    strategy.on_evaluate_config_fn = lambda r: {"round": r}
    
    print("Starting server... Waiting for hospitals...")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server()
