"""
Federated Learning Client (Hospital)
"""

import flwr as fl
from src.model import HeartDiseaseNet, train_model, test_model
from src.utils import get_parameters, set_parameters


class HeartDiseaseClient(fl.client.NumPyClient):
    """Flower client for one hospital."""
    
    def __init__(self, client_id, trainloader, testloader, num_examples):
        self.client_id = client_id
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples
        self.model = HeartDiseaseNet()
    
    def get_parameters(self, config):
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        # 1. Update local model with global parameters
        set_parameters(self.model, parameters)
        
        # 2. Train on local private data
        epochs = config.get("epochs", 1)
        lr = config.get("lr", 0.01)
        loss = train_model(self.model, self.trainloader, epochs=epochs, lr=lr)
        
        # 3. Return updated weights (Privacy preserved!)
        return get_parameters(self.model), self.num_examples, {"loss": loss}
    
    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = test_model(self.model, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def start_client(client_id, trainloader, testloader, num_examples, server_address="127.0.0.1:8080"):
    """Start a Flower client/hospital."""
    client = HeartDiseaseClient(client_id, trainloader, testloader, num_examples)
    fl.client.start_client(server_address=server_address, client=client.to_client())
