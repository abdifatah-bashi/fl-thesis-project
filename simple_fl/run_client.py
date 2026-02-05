"""
Run FL Client - 2 HOSPITALS VERSION
"""

import sys
from data_loader import load_all_hospitals, prepare_client_data
from client import start_client


def main(client_id=0):
    """Start a FL client (0 or 1)."""
    hospital_names = {
        0: "Cleveland Clinic",
        1: "Hungarian Institute"
    }
    
    print("=" * 60)
    print(f"HOSPITAL CLIENT: {hospital_names.get(client_id, 'Unknown')}")
    print("=" * 60)
    
    print("\nLoading hospital data files...")
    hospital_data = load_all_hospitals()
    
    if client_id >= len(hospital_data):
        print(f"\nError: Only {len(hospital_data)} hospitals available")
        print(f"Use client_id 0 or 1")
        sys.exit(1)
    
    hospital_name, X, y = hospital_data[client_id]
    
    print(f"\nPreparing data for {hospital_name.capitalize()}...")
    trainloader, testloader, num_examples = prepare_client_data(X, y)
    
    print(f"Training samples: {num_examples}")
    print(f"Test samples: {len(testloader.dataset)}")
    print()
    
    start_client(
        client_id=client_id,
        trainloader=trainloader,
        testloader=testloader,
        num_examples=num_examples,
        server_address="127.0.0.1:8080"
    )


if __name__ == "__main__":
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    
    if client_id not in [0, 1]:
        print("Error: client_id must be 0 or 1")
        print("Usage: python run_client.py [0|1]")
        print("  0 = Cleveland Clinic")
        print("  1 = Hungarian Institute")
        sys.exit(1)
    
    main(client_id)
