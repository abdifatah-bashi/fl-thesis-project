# Federated Learning — Hospital Network Demo

A Federated Learning system for heart disease prediction across independent hospitals.
Each hospital trains locally on its own data — patient records never leave the machine.
Built with FastAPI, Next.js, TensorFlow.js, and PyTorch.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Hospital)                    │
│                                                          │
│  1. Fetches global model weights (JSON)                  │
│  2. Reads local CSV with FileReader API  ← stays here   │
│  3. TF.js trains the model in-browser                    │
│  4. Sends ONLY weight deltas → server                    │
└─────────────────────────────────────────────────────────┘
                          ↕ weight deltas only
┌─────────────────────────────────────────────────────────┐
│               FastAPI Backend (Central Server)           │
│                                                          │
│  • Publishes global HeartDiseaseNet parameters          │
│  • Receives weight updates from hospitals               │
│  • Runs FedAvg aggregation across all submissions       │
│  • Discovers hospitals dynamically (no hardcoding)      │
└─────────────────────────────────────────────────────────┘
```

**Key FL properties enforced:**
- Server never sees raw patient data — only weight arrays
- Server doesn't know hospital names upfront — discovered on first submission
- Each hospital is a fully independent browser tab
- Adding a new hospital = open a new tab and type a name (zero code changes)

---

## Project Structure

```
fl-thesis-project/
├── api/
│   ├── main.py          # FastAPI routes
│   ├── fl_engine.py     # FedAvg aggregation, model publish
│   └── state.py         # JSON state persistence
├── frontend/
│   ├── app/
│   │   ├── page.tsx               # Landing — role selection
│   │   ├── server/page.tsx        # Central Server dashboard
│   │   └── hospital/[name]/page.tsx  # Hospital monitor + TF.js training
│   └── lib/api.ts       # Typed API client
├── src/
│   ├── model.py         # HeartDiseaseNet (PyTorch, 13→64→32→1)
│   └── utils.py         # get_parameters / set_parameters
├── data/
│   └── heart_disease/raw/   # UCI Heart Disease datasets
└── results/
    ├── fl_state.json        # Live federation state
    ├── global_params.pt     # Current global model
    └── weights/             # Per-hospital weight submissions
```

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- The project's virtual environment (`venv/`)

---

## Installation

### 1. Python dependencies

```bash
cd fl-thesis-project
source venv/bin/activate
pip install fastapi uvicorn python-multipart
```

> All other dependencies (torch, sklearn, pandas, numpy) are already in the venv.

### 2. Frontend dependencies

```bash
cd frontend
npm install
```

---

## Running the Application

You need **2 terminals**.

### Terminal 1 — Backend (FastAPI)

```bash
cd fl-thesis-project
source venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

### Terminal 2 — Frontend (Next.js)

```bash
cd fl-thesis-project/frontend
npm run dev
```

Expected output:
```
▲ Next.js
  Local: http://localhost:3000
```

Open **http://localhost:3000** in your browser.

---

## Running a Full Federated Learning Round

### Step 1 — Central Server: Publish Global Model

1. Go to **http://localhost:3000**
2. Click **"Enter as Central Server"** → opens `/server`
3. Click **"Publish Global Model"**
   - Status badge changes to `ready`
   - Step 1 shows ✓

### Step 2 — Hospital A: Train Locally

1. Open a **new browser tab** → **http://localhost:3000**
2. Click **"Cleveland"** (or type any hospital name) → opens `/hospital/cleveland`
3. Step 1 shows ✓ (model is published)
4. Under **Patient Data**, choose:
   - **Sample dataset** → select `Cleveland (UCI)` from dropdown, or
   - **Upload CSV** → pick your own file (read by FileReader, never sent to server)
5. Set **Epochs** to `5`, **Learning Rate** to `0.01`
6. Click **"Train & Submit Updates"**
   - TensorFlow.js loads (~3 MB, first time only)
   - Progress bar shows training epoch by epoch
   - Results appear: Accuracy, Train Loss, Samples used
   - Step 3 shows ✓

### Step 3 — Hospital B: Train Locally (independent)

1. Open **another new tab** → **http://localhost:3000**
2. Click **"Hungarian"** → opens `/hospital/hungarian`
3. Select **Hungarian (UCI)** from the sample dataset dropdown
4. Click **"Train & Submit Updates"**

### Step 4 — Central Server: Aggregate

1. Switch back to the **`/server`** tab
2. See **Connected Hospitals (2)** — Cleveland and Hungarian discovered automatically
3. Configure: **Rounds** = `3`, **Epochs/round** = `1`
4. Click **"Start Aggregation"**
   - Progress bar fills across rounds
   - Results table shows accuracy + loss per round

---

## Adding More Hospitals

No code changes needed. Just open a new browser tab and type any name:

- Go to **http://localhost:3000**
- Type `Budapest` in the custom name field → click **Go**
- Train with any UCI dataset or upload a CSV
- The Central Server discovers it automatically

---

## API Reference

The backend auto-generates interactive docs at **http://localhost:8000/docs**.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/model/publish` | Create and publish initial global model |
| `GET` | `/api/model/global/json` | Download model weights as JSON (for TF.js) |
| `POST` | `/api/hospital/{name}/submit` | Submit weight updates from browser |
| `GET` | `/api/data/sample/{name}` | Fetch demo UCI CSV (`cleveland`, `hungarian`, `va`, `switzerland`) |
| `POST` | `/api/aggregate` | Start FedAvg aggregation |
| `GET` | `/api/status` | Live federation state |
| `GET` | `/api/results` | Per-round metrics history |
| `DELETE` | `/api/reset` | Reset all state and weight files |

---

## What to Expect

| Action | Result |
|--------|--------|
| Publish global model | Status → `ready`, hospital pages show Step 1 ✓ |
| Train Cleveland, 5 epochs | ~75–82% accuracy |
| Train Hungarian, 5 epochs | ~72–78% accuracy |
| Aggregate 3 rounds | Results table with per-round accuracy and loss |
| Open a 3rd hospital tab | Appears in "Connected Hospitals" after training |
| Reset | Clears all state, weights, and history |

---

## Data Privacy

When a hospital uploads their own CSV:
- The file is read by the browser's **FileReader API** entirely in memory
- It is **never included in any HTTP request**
- Only the trained weight arrays (floating-point numbers) are sent to `/api/hospital/{name}/submit`
- The server stores one `.pt` file per hospital containing weights and metrics — no patient records

For the **sample datasets** (Cleveland, Hungarian, VA, Switzerland): these are public UCI Heart Disease datasets served by the backend for demonstration purposes. In a real deployment, each hospital would use only their own local files.

---

## Model

**HeartDiseaseNet** — a small feedforward neural network:

```
Input (13 features) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
```

Features: age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar,
resting ECG, max heart rate, exercise angina, ST depression, ST slope, major vessels, thalassemia.

Output: probability of heart disease (> 0.5 → positive).

The same architecture is implemented twice — in PyTorch (server-side aggregation) and TensorFlow.js (in-browser training) — with weight conversion handling the transposition between the two frameworks' conventions.
