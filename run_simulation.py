"""
Federated Learning Simulation

This script runs the entire FL system (Server + 2 Clients) in a single process.
It is the EASIEST way to understand how FL works!
"""

import sys
import os
import logging
import warnings

# 0. Set environment variables to suppress Tensorflow/Ray logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_INIT_LOG_TO_DRIVER"] = "0"

# 1. Suppress all Python Warnings (including Flower deprecation warnings)
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")

# 2. Aggressively silence loggers
# We set them to CRITICAL to ensure almost nothing gets through
logging.getLogger("flwr").setLevel(logging.CRITICAL)
logging.getLogger("ray").setLevel(logging.CRITICAL)
logging.getLogger("root").setLevel(logging.CRITICAL)

# 3. Filter stderr to suppress Flower deprecation warnings
import io
class FilteredStderr:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.buffer = ""
        
    def write(self, text):
        # Filter out lines containing DEPRECATED
        if "DEPRECATED" not in text:
            self.original_stderr.write(text)
        return len(text)
    
    def flush(self):
        self.original_stderr.flush()
    
    def fileno(self):
        """Return the file descriptor for Ray compatibility."""
        return self.original_stderr.fileno()
    
    def isatty(self):
        """Check if stderr is a terminal."""
        return self.original_stderr.isatty()

# Replace stderr with filtered version
sys.stderr = FilteredStderr(sys.stderr)

from pathlib import Path

import flwr as fl
from src.data_loader import load_all_hospitals, prepare_client_data
from src.client import HeartDiseaseClient
from src.server import weighted_average, weighted_average_loss

def run_simulation():
    print("Starting FL Simulation...")
    
    # 1. Load Data
    # Only loads Cleveland and Hungarian data (Privacy preserved: we simulate separation)
    hospital_data = load_all_hospitals()
    
    # Map client ID to specific hospital data
    # Client 0 -> Cleveland
    # Client 1 -> Hungarian
    client_resources = {}
    for i, (name, X, y) in enumerate(hospital_data):
        trainloader, testloader, num_examples = prepare_client_data(X, y)
        client_resources[str(i)] = (trainloader, testloader, num_examples)
        print(f"Client {i}: {name.capitalize()} prepared with {num_examples} training samples.")

    # 2. Define Client Function
    # Flower calls this to spawn a client when needed
    def client_fn(context: fl.common.Context) -> fl.client.Client:
        # Get client ID
        partition_id = context.node_config["partition-id"]
        
        # Get data for this specific hospital
        try:
            trainloader, testloader, num_examples = client_resources[str(partition_id)]
            return HeartDiseaseClient(partition_id, trainloader, testloader, num_examples).to_client()
        except KeyError:
            # Fallback (should not happen in this simplified simulation)
            print(f"Error: No data for client {partition_id}")
            raise

    # 3. Define Strategy (Server Logic)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,      # Sample 100% of available clients for training
        fraction_evaluate=1.0, # Sample 100% of available clients for evaluation
        min_fit_clients=2,     # Never train unless 2 clients are available
        min_evaluate_clients=2,# Never evaluate unless 2 clients are available
        min_available_clients=2,
        fit_metrics_aggregation_fn=weighted_average_loss,  # Aggregate training loss
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate evaluation accuracy
    )
    
    # Define config for each round (pass learning rate, epochs etc.)
    strategy.on_fit_config_fn = lambda r: {"epochs": 1, "lr": 0.01, "round": r}

    # 4. Start Simulation
    print("\nStarting Simulation Engine...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}, # Use CPU for simplicity
    )
    
    # 5. Save results for visualization
    import json
    from pathlib import Path
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    results = {
        "accuracy": history.metrics_distributed["accuracy"],
        "training_loss": history.metrics_distributed_fit["loss"],
        "distributed_loss": history.losses_distributed,
        "num_rounds": 3,
        "num_clients": 2,
    }
    
    with open(results_dir / "simulation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved to results/simulation_results.json")
    print("📊 Run 'python visualize_results.py' to generate charts!")

if __name__ == "__main__":
    run_simulation()
