# HeartDiseaseNet

`src/model.py`

The neural network model for heart disease prediction. Provides the model architecture, local training function, and evaluation function.

---

## `HeartDiseaseNet`

```python
class HeartDiseaseNet(nn.Module)
```

A feedforward neural network for binary classification of heart disease.

### Architecture

```
Input (13 features)
  → Linear(13, 64) → ReLU
  → Linear(64, 32) → ReLU
  → Linear(32, 1)  → Sigmoid
  → Output (probability 0-1)
```

| Layer | Shape | Parameters | Activation |
|-------|-------|:----------:|------------|
| `fc1` | 13 → 64 | 896 | ReLU |
| `fc2` | 64 → 32 | 2,080 | ReLU |
| `fc3` | 32 → 1 | 33 | Sigmoid |
| **Total** | | **3,169** | |

### Constructor

```python
HeartDiseaseNet(input_size=13)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | `int` | `13` | Number of input features |

### Methods

#### `forward(x)`

Forward pass through the network.

```python
model = HeartDiseaseNet()
output = model(torch.randn(32, 13))  # batch of 32 patients
# output.shape → torch.Size([32, 1])
# output values → probabilities between 0 and 1
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Input tensor of shape `(batch_size, 13)` |

**Returns**: `torch.Tensor` of shape `(batch_size, 1)` — probability of heart disease.

---

## `train_model()`

```python
def train_model(model, trainloader, epochs=1, lr=0.01) -> float
```

Train the model on local hospital data using Binary Cross Entropy loss and Adam optimizer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `HeartDiseaseNet` | — | The neural network to train |
| `trainloader` | `DataLoader` | — | Training data for this hospital |
| `epochs` | `int` | `1` | Number of passes through the data |
| `lr` | `float` | `0.01` | Learning rate for Adam optimizer |

**Returns**: `float` — Average loss over the final epoch.

### Training loop

```python
for epoch in range(epochs):
    for features, labels in trainloader:
        optimizer.zero_grad()           # Reset gradients
        outputs = model(features)       # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                 # Compute gradients
        optimizer.step()                # Update weights
```

---

## `test_model()`

```python
def test_model(model, testloader) -> Tuple[float, float]
```

Evaluate model accuracy on test data. Runs in `torch.no_grad()` mode (no gradient computation).

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `HeartDiseaseNet` | The neural network to evaluate |
| `testloader` | `DataLoader` | Test data |

**Returns**: `Tuple[float, float]` — `(average_loss, accuracy)` where accuracy is between 0 and 1.

### Example

```python
model = HeartDiseaseNet()
loss, accuracy = test_model(model, testloader)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")
# Loss: 0.6832, Accuracy: 65.00%
```
