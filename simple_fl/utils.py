"""
Utility Functions for Federated Learning

Helper functions to convert between PyTorch models and NumPy arrays.
Flower needs NumPy arrays for network transmission!
"""

import torch
from collections import OrderedDict


def get_parameters(model):
    """
    Extract model parameters as NumPy arrays.
    
    Why? Flower sends parameters over network as NumPy arrays.
    Can't send PyTorch tensors directly!
    
    Args:
        model: PyTorch model
        
    Returns:
        List of NumPy arrays (model weights)
    """
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model, parameters):
    """
    Load parameters from NumPy arrays into model.
    
    Args:
        model: PyTorch model
        parameters: List of NumPy arrays
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


if __name__ == "__main__":
    from model import HeartDiseaseNet
    
    print("Testing utility functions...")
    print("=" * 60)
    
    # Create model
    model = HeartDiseaseNet()
    
    # Get parameters
    print("\nExtracting parameters...")
    params = get_parameters(model)
    print(f"Number of parameter arrays: {len(params)}")
    print(f"Shapes: {[p.shape for p in params]}")
    
    # Set parameters
    print("\nLoading parameters back...")
    set_parameters(model, params)
    
    print("\n" + "=" * 60)
    print("✅ Utilities working correctly!")
