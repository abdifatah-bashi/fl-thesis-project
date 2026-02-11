# FL Multi-Hospital Simulation UI

A Streamlit application that lets you simulate federated learning across two hospitals
(Cleveland and Hungarian) from a single browser or multiple browser tabs.

## Quick Start

```bash
# From the project root
cd ~/fl-thesis-project
source venv/bin/activate

# Install UI dependencies (only needed once)
pip install streamlit plotly

# Launch the app
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser.

---

## Demo Workflows

### Single-Window Demo

1. Click **Cleveland Hospital** → load sample data → Register
2. Click **Switch Role** → **Hungarian Hospital** → load sample data → Register
3. Click **Switch Role** → **FL Coordinator** → Start Federated Training
4. Watch live progress, then view charts in the Results tab

### Multi-Tab Demo *(recommended for thesis presentations)*

Open `http://localhost:8501` in **3 browser tabs**:

| Tab | Role | What to do |
|-----|------|-----------|
| 1 | FL Coordinator | Monitor dashboard, start training |
| 2 | Cleveland Hospital | Upload / load data, register |
| 3 | Hungarian Hospital | Upload / load data, register |

All tabs share state via `results/fl_state.json` and refresh automatically.

---

## File Structure

```
ui/
├── app.py                  # Main entry point (streamlit run ui/app.py)
├── state_manager.py        # Shared JSON state for multi-tab coordination
├── fl_runner.py            # FL simulation bridge (background thread)
├── components/
│   └── charts.py           # Reusable Plotly chart functions
└── pages/
    ├── coordinator.py      # FL Coordinator dashboard
    └── hospital.py         # Hospital interface (Cleveland & Hungarian)
```

---

## Architecture

```
Browser Tab 1 (Coordinator)
   └── reads/writes  ──┐
Browser Tab 2 (Cleveland) │   results/fl_state.json   ←── Background FL thread
   └── reads/writes  ──┤        (shared state)              (Flower simulation)
Browser Tab 3 (Hungarian) │
   └── reads/writes  ──┘
```

- **State** is persisted in `results/fl_state.json` (atomic writes, safe for concurrent access)
- **FL simulation** runs in a background thread spawned by the Coordinator tab
- **Hospital data** is saved to `data/hospital_uploads/{name}.csv` before training
- **Results** are also written to `results/simulation_results.json` after completion

---

## Resetting State

Click the **Reset All** button in the Coordinator dashboard, or delete the state file:

```bash
rm results/fl_state.json
```
