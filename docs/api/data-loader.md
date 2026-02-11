# Data Loader

`src/data_loader.py`

Functions for loading, cleaning, and preparing hospital data for federated training.

---

## Constants

### `COLUMNS`

```python
COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]
```

Column names for the UCI Heart Disease CSV files (14 columns: 13 features + 1 target).

---

## `load_hospital_data()`

```python
def load_hospital_data(hospital_name, data_dir='data/heart_disease/raw')
    -> Tuple[np.ndarray, np.ndarray]
```

Load and clean data for a single hospital.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hospital_name` | `str` | — | Hospital name: `"cleveland"` or `"hungarian"` |
| `data_dir` | `str` | `"data/heart_disease/raw"` | Path to raw data directory |

**Returns**: `Tuple[np.ndarray, np.ndarray]` — `(X, y)` where X is features and y is binary labels.

**Raises**: `ValueError` if hospital name is not recognized.

### Processing steps

1. Read CSV with pandas (no header, columns named by `COLUMNS`)
2. Replace `?` with `NaN`
3. Create binary target: `1` if `num > 0`, else `0`
4. Drop the original `num` and new `target` from features
5. Fill remaining `NaN` with column mean

### File mapping

| Hospital name | CSV file |
|--------------|----------|
| `"cleveland"` | `processed.cleveland.data` |
| `"hungarian"` | `processed.hungarian.data` |

### Example

```python
X, y = load_hospital_data('cleveland')
print(X.shape)  # (303, 13)
print(y.shape)  # (303,)
print(y.mean())  # ~0.54 (54% have heart disease)
```

---

## `load_all_hospitals()`

```python
def load_all_hospitals(data_dir='data/heart_disease/raw')
    -> List[Tuple[str, np.ndarray, np.ndarray]]
```

Load data from all supported hospitals.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | `str` | `"data/heart_disease/raw"` | Path to raw data directory |

**Returns**: `List[Tuple[str, np.ndarray, np.ndarray]]` — List of `(hospital_name, X, y)` tuples.

### Example

```python
hospital_data = load_all_hospitals()
for name, X, y in hospital_data:
    print(f"{name}: {len(X)} patients, {y.sum():.0f} with disease")
# cleveland: 303 patients, 165 with disease
# hungarian: 294 patients, 139 with disease
```

---

## `prepare_client_data()`

```python
def prepare_client_data(X, y, batch_size=32)
    -> Tuple[DataLoader, DataLoader, int]
```

Transform raw NumPy arrays into PyTorch DataLoaders ready for training.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | — | Feature matrix of shape `(n_samples, 13)` |
| `y` | `np.ndarray` | — | Label array of shape `(n_samples,)` |
| `batch_size` | `int` | `32` | Number of samples per training batch |

**Returns**: `Tuple[DataLoader, DataLoader, int]` — `(trainloader, testloader, num_train_examples)`.

### Pipeline

| Step | Operation | Details |
|------|-----------|---------|
| 1 | Train/test split | 80/20 split, stratified by label |
| 2 | Normalization | `StandardScaler` fit on train, applied to both |
| 3 | Tensor conversion | `torch.FloatTensor` for features, labels unsqueezed |
| 4 | DataLoader creation | Batched, train shuffled |

### Example

```python
X, y = load_hospital_data('cleveland')
trainloader, testloader, num_examples = prepare_client_data(X, y, batch_size=32)

print(num_examples)                # 242
print(len(trainloader))            # 8 batches
print(len(testloader.dataset))     # 61 test samples
```
