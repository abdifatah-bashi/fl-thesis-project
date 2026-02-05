"""
Federated Learning Simulation

This script runs the entire FL system (Server + 2 Clients) in a single process.
It is the EASIEST way to understand how FL works!
"""

import sys
from pathlib import Path

# Add project root to python path to import modules
sys.path.append(str(Path(__file__).parent))

import flwr as fl
from simple_fl.data_loader import load_all_hospitals, prepare_client_data
from simple_fl.client import HeartDiseaseClient
from simple_fl.server import weighted_average

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
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Define config for each round (pass learning rate, epochs etc.)
    strategy.on_fit_config_fn = lambda r: {"epochs": 1, "lr": 0.01, "round": r}

    # 4. Start Simulation
    print("\nStarting Simulation Engine...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=2,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}, # Use CPU for simplicity
    )

if __name__ == "__main__":
    run_simulation()
