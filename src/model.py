"""
Neural Network Model for Heart Disease Prediction

Simple feedforward neural network:
- Input: 13 features (age, blood pressure, cholesterol, etc.)
- Hidden layers: 64 → 32 neurons
- Output: 1 (probability of heart disease)

Architecture is intentionally simple for learning FL!
"""

import torch
import torch.nn as nn


class HeartDiseaseNet(nn.Module):
    """
    Neural network for heart disease prediction.
    
    Architecture:
        Input (13) → Dense(64) → ReLU → Dense(32) → ReLU → Dense(1) → Sigmoid
        
    Why this architecture?
    - 13 inputs: Our 13 medical features
    - 64 neurons: First hidden layer (learns feature combinations)
    - 32 neurons: Second hidden layer (refines patterns)
    - 1 output: Single probability (0-1, disease or not)
    - Sigmoid: Converts output to probability
    """
    
    def __init__(self, input_size=13):
        super(HeartDiseaseNet, self).__init__()
        
        # Layer 1: Input → 64 neurons
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        
        # Layer 2: 64 → 32 neurons
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        
        # Layer 3: 32 → 1 output
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, 13)
            
        Returns:
            Probability of heart disease (batch_size, 1)
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def train_model(model, trainloader, epochs=1, lr=0.01):
    """
    Train the model on local hospital data.
    
    This happens at each hospital independently!
    
    Args:
        model: Neural network
        trainloader: Training data for this hospital
        epochs: Number of passes through the data
        lr: Learning rate
        
    Returns:
        Average loss during training
    """
    criterion = nn.BCELoss()  # Binary Cross Entropy for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    epoch_loss = 0.0
    
    for epoch in range(epochs):
        batch_loss = 0.0
        for features, labels in trainloader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.item()
        
        epoch_loss = batch_loss / len(trainloader)
    
    return epoch_loss


def test_model(model, testloader):
    """
    Evaluate model accuracy on test data.
    
    Args:
        model: Neural network
        testloader: Test data
        
    Returns:
        loss: Average test loss
        accuracy: Classification accuracy (0-1)
    """
    criterion = nn.BCELoss()
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in testloader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    avg_loss = test_loss / len(testloader)
    
    return avg_loss, accuracy


if __name__ == "__main__":
    # Test the model
    print("Testing Heart Disease Model...")
    print("=" * 60)
    
    # Create model
    model = HeartDiseaseNet(input_size=13)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created!")
    print(f"Total parameters: {num_params:,}")
    print(f"Model size: ~{num_params * 4 / 1024:.2f} KB")
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(5, 13)  # 5 patients, 13 features
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output[:3].squeeze().tolist()}")
    
    print("\n" + "=" * 60)
    print("✅ Model working correctly!")
