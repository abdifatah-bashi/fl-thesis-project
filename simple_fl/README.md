# Simple Federated Learning with Heart Disease Data

## The Problem We're Solving

Three hospitals want to predict heart disease but can't share patient data (GDPR/HIPAA).

**Traditional ML:** Pool all data → Privacy violation ❌  
**Federated Learning:** Train together without sharing data ✅

## The Setup

**Hospital A (Cleveland):** ~300 patients  
**Hospital B (Hungary):** ~290 patients  
**Hospital C (Switzerland):** ~120 patients  

Each hospital trains locally, only model updates are shared!

## How to Run

### Terminal 1: Start Server
```bash
cd ~/fl-thesis-project/simple_fl
source ../venv/bin/activate
python server.py
```

### Terminal 2: Start Client 0 (Cleveland)
```bash
cd ~/fl-thesis-project/simple_fl
source ../venv/bin/activate
python run_client.py 0
```

### Terminal 3: Start Client 1 (Hungary)
```bash
cd ~/fl-thesis-project/simple_fl
source ../venv/bin/activate
python run_client.py 1
```

### Terminal 4: Start Client 2 (Switzerland)
```bash
cd ~/fl-thesis-project/simple_fl
source ../venv/bin/activate
python run_client.py 2
```

## What You'll See

**Server:**
- Round 1: Aggregate updates from 3 hospitals → Accuracy ~60%
- Round 2: Aggregate updates → Accuracy ~70%
- Round 3: Aggregate updates → Accuracy ~75%
- Round 4: Aggregate updates → Accuracy ~78%
- Round 5: Aggregate updates → Accuracy ~80%

**Clients:**
- Receive model from server
- Train on local data
- Send updates back
- Repeat

## Key Concepts

**Server:** Coordinator (no patient data!)  
**Client:** Hospital (keeps data private)  
**Round:** One cycle of training + aggregation  
**FedAvg:** Weighted average of hospital updates  
**Model Update:** Numbers sent (NOT patient data)  

## Files

- `data_loader.py` - Split data into 3 hospitals
- `model.py` - Neural network definition
- `utils.py` - Helper functions
- `server.py` - FL server
- `client.py` - FL client
- `run_client.py` - Easy client starter
