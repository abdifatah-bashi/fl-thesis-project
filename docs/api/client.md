# HeartDiseaseClient

`src/client.py`

The Flower client implementation representing a single hospital in the federation.

---

## `HeartDiseaseClient`

```python
class HeartDiseaseClient(fl.client.NumPyClient)
```

A Flower NumPyClient that holds private hospital data and participates in federated training rounds.

### Constructor

```python
HeartDiseaseClient(client_id, trainloader, testloader, num_examples)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `client_id` | `int` | Unique identifier for this hospital |
| `trainloader` | `DataLoader` | Private training data (PyTorch DataLoader) |
| `testloader` | `DataLoader` | Private test data (PyTorch DataLoader) |
| `num_examples` | `int` | Number of training samples (used for FedAvg weighting) |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `self.model` | `HeartDiseaseNet` | Local model instance (created fresh) |
| `self.trainloader` | `DataLoader` | Private training data — never transmitted |
| `self.testloader` | `DataLoader` | Private test data — never transmitted |

---

### Methods

#### `get_parameters(config)`

Extract the local model's weights as a list of NumPy arrays.

```python
params = client.get_parameters(config={})
# Returns: [ndarray(13, 64), ndarray(64), ndarray(64, 32), ndarray(32), ndarray(32, 1), ndarray(1)]
```

**Returns**: `List[np.ndarray]` — Model parameters as NumPy arrays.

---

#### `fit(parameters, config)`

Receive global weights, train locally, and return updated weights.

```python
updated_params, num_examples, metrics = client.fit(global_params, config)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `parameters` | `List[np.ndarray]` | Global model weights from server |
| `config` | `Dict` | Training config (`epochs`, `lr`, `round`) |

**Returns**: `Tuple[List[np.ndarray], int, Dict]`

| Return value | Type | Description |
|-------------|------|-------------|
| Updated weights | `List[np.ndarray]` | Model weights after local training |
| Sample count | `int` | Number of training examples (for FedAvg) |
| Metrics | `Dict` | `{"loss": float}` — training loss |

---

#### `evaluate(parameters, config)`

Receive global weights and evaluate on local test data.

```python
loss, num_examples, metrics = client.evaluate(global_params, config)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `parameters` | `List[np.ndarray]` | Global model weights from server |
| `config` | `Dict` | Evaluation config (`round`) |

**Returns**: `Tuple[float, int, Dict]`

| Return value | Type | Description |
|-------------|------|-------------|
| Loss | `float` | Average test loss |
| Sample count | `int` | Number of test examples |
| Metrics | `Dict` | `{"accuracy": float}` — classification accuracy |

---

## `start_client()`

```python
def start_client(client_id, trainloader, testloader, num_examples,
                 server_address="127.0.0.1:8080")
```

Create a `HeartDiseaseClient` and connect to the FL server.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client_id` | `int` | — | Hospital identifier |
| `trainloader` | `DataLoader` | — | Private training data |
| `testloader` | `DataLoader` | — | Private test data |
| `num_examples` | `int` | — | Training sample count |
| `server_address` | `str` | `"127.0.0.1:8080"` | Server gRPC address |

### Example

```python
from fl_core.data_loader import load_hospital_data, prepare_client_data
from fl_core.client import start_client

X, y = load_hospital_data('cleveland')
trainloader, testloader, num_examples = prepare_client_data(X, y)
start_client(0, trainloader, testloader, num_examples)
```
