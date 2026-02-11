"""
Coordinator Dashboard

Allows the FL Coordinator to:
  - Monitor which hospitals are connected and ready
  - Configure and launch federation training
  - Watch live per-round progress
  - View final accuracy / loss charts and download results
  - Understand the privacy model
"""

import sys
import csv
import io
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from state_manager import load_state, save_state, reset_state
from fl_runner import start_simulation_thread
from components.charts import (
    create_accuracy_chart,
    create_loss_chart,
    create_hospital_distribution_chart,
    create_round_progress,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    badges = {
        "waiting":  "⚪ Waiting",
        "ready":    "🟢 Ready",
        "training": "🔵 Training",
        "done":     "✅ Done",
    }
    return badges.get(status, f"❓ {status}")


def _fed_status_label(status: str) -> str:
    if status == "waiting":
        return "⏳ Waiting for hospitals"
    if status == "training" or status.startswith("round_"):
        return "🔵 Training in progress"
    if status == "complete":
        return "✅ Training complete"
    if status.startswith("error"):
        return f"❌ {status}"
    return status


# ── main render ───────────────────────────────────────────────────────────────

def render_coordinator():
    st.title("🌐 FL Coordinator Dashboard")
    st.caption("Manage the federation, launch training, and view aggregated results.")

    state = load_state()
    hospitals = state["hospitals"]
    fed = state["federation"]
    hist = fed["history"]

    registered = [n for n, h in hospitals.items() if h.get("registered")]
    training_active = fed.get("active", False)
    status = fed.get("status", "waiting")
    is_complete = status == "complete"
    is_training = training_active and not is_complete

    # ── tabs ─────────────────────────────────────────────────────────────────
    tab_status, tab_monitor, tab_results, tab_privacy = st.tabs([
        "🏥 Hospital Status",
        "📡 Training Monitor",
        "📊 Results Dashboard",
        "🔒 Privacy Explanation",
    ])

    # ── TAB 1: Hospital Status ────────────────────────────────────────────────
    with tab_status:
        st.subheader("Connected Hospitals")

        col1, col2, col3 = st.columns(3)
        col1.metric("Registered", f"{len(registered)}/2")
        col2.metric("Federation Status", _fed_status_label(status).split(" ", 1)[1] if " " in status else status)
        col3.metric(
            "Current Round",
            f"{fed.get('current_round', 0)}/{fed.get('total_rounds', 3)}",
        )

        st.divider()

        for name in ("cleveland", "hungarian"):
            h = hospitals[name]
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
                c1.markdown(f"### 🏥 {name.capitalize()}")
                c2.markdown(f"**Status:** {_status_badge(h['status'])}")
                c3.markdown(f"**Patients:** {h.get('num_patients', 0)}")
                c4.markdown(
                    f"**Disease rate:** {h.get('disease_rate', 0)*100:.1f}%"
                    if h.get("registered") else "**Disease rate:** —"
                )
                if not h.get("registered"):
                    st.info(f"{name.capitalize()} has not registered yet. "
                            "Open the hospital tab in another browser window.")

        st.divider()

        # Training config
        with st.expander("⚙️ Training Configuration", expanded=not training_active):
            c1, c2, c3 = st.columns(3)
            num_rounds = c1.slider("Number of Rounds", 1, 10, 3, disabled=training_active)
            epochs = c2.slider("Epochs per Round", 1, 10, 1, disabled=training_active)
            lr_options = [0.001, 0.01, 0.05, 0.1]
            lr = c3.selectbox("Learning Rate", lr_options, index=1, disabled=training_active)

        # Start / Reset buttons
        bc1, bc2 = st.columns([3, 1])
        can_start = len(registered) >= 2 and not training_active
        if bc1.button(
            "🚀 Start Federated Training",
            disabled=not can_start,
            type="primary",
            use_container_width=True,
        ):
            config = {
                "num_rounds": num_rounds,
                "epochs_per_round": epochs,
                "learning_rate": lr,
            }
            if "sim_thread" not in st.session_state or not st.session_state.sim_thread.is_alive():
                st.session_state.sim_thread = start_simulation_thread(config)
            st.rerun()

        if not can_start and len(registered) < 2:
            st.warning(f"Waiting for hospitals to register ({len(registered)}/2 connected).")

        if bc2.button("🔄 Reset All", use_container_width=True):
            reset_state()
            for key in list(st.session_state.keys()):
                if key not in ("role",):
                    del st.session_state[key]
            st.rerun()

    # ── TAB 2: Training Monitor ───────────────────────────────────────────────
    with tab_monitor:
        st.subheader("Live Training Progress")

        current_round = fed.get("current_round", 0)
        total_rounds = fed.get("total_rounds", 3)

        if not training_active and not is_complete:
            st.info("Training has not started yet. Register at least 2 hospitals and click 'Start Federated Training'.")
        else:
            # Overall progress
            pct = int(current_round / total_rounds * 100) if total_rounds > 0 else 0
            st.markdown(f"**Round {current_round} / {total_rounds}**")
            st.progress(pct)

            if is_complete:
                st.success("Training complete! View results in the Results Dashboard tab.")
            elif is_training:
                st.info(f"Federation status: {_fed_status_label(status)}")

            st.divider()

            # Per-hospital status
            st.markdown("**Hospital Training Status**")
            for name in ("cleveland", "hungarian"):
                h = hospitals[name]
                if h.get("registered"):
                    hc1, hc2, hc3 = st.columns([2, 3, 2])
                    hc1.markdown(f"🏥 **{name.capitalize()}**")
                    hc2.progress(pct if h["status"] == "training" else (100 if h["status"] == "done" else 0))
                    hc3.markdown(_status_badge(h["status"]))

            # Live metrics from history
            if hist["train_loss"] or hist["accuracy"]:
                st.divider()
                st.markdown("**Metrics so far**")
                mc1, mc2 = st.columns(2)
                if hist["train_loss"]:
                    mc1.metric("Latest Training Loss", f"{hist['train_loss'][-1]:.4f}")
                if hist["accuracy"]:
                    mc2.metric("Latest Accuracy", f"{hist['accuracy'][-1]*100:.1f}%")

            # Auto-refresh while training
            if is_training:
                time.sleep(2)
                st.rerun()

    # ── TAB 3: Results Dashboard ──────────────────────────────────────────────
    with tab_results:
        st.subheader("Federated Learning Results")

        if not hist["accuracy"] and not hist["train_loss"]:
            st.info("Results will appear here once training is underway.")
        else:
            rounds = hist["rounds"] or list(range(1, len(hist["accuracy"]) + 1))

            # Summary metrics row
            if hist["accuracy"]:
                m1, m2, m3, m4 = st.columns(4)
                final_acc = hist["accuracy"][-1]
                final_loss = hist["eval_loss"][-1] if hist["eval_loss"] else (hist["train_loss"][-1] if hist["train_loss"] else 0)
                total_patients = sum(h.get("num_patients", 0) for h in hospitals.values() if h.get("registered"))
                m1.metric("Final Accuracy", f"{final_acc*100:.1f}%")
                m2.metric("Final Loss", f"{final_loss:.4f}")
                m3.metric("Total Patients", total_patients)
                m4.metric("Rounds Completed", len(rounds))

            st.divider()

            # Charts
            ch1, ch2 = st.columns(2)
            if hist["accuracy"] and rounds:
                ch1.plotly_chart(
                    create_accuracy_chart(rounds, hist["accuracy"]),
                    use_container_width=True,
                )
            if rounds and (hist["train_loss"] or hist["eval_loss"]):
                ch2.plotly_chart(
                    create_loss_chart(rounds, hist["train_loss"], hist["eval_loss"]),
                    use_container_width=True,
                )

            # Hospital distribution
            st.plotly_chart(
                create_hospital_distribution_chart(hospitals),
                use_container_width=True,
            )

            # Download CSV
            if is_complete or hist["accuracy"]:
                st.divider()
                rows = []
                for i, r in enumerate(rounds):
                    rows.append({
                        "round": r,
                        "accuracy": hist["accuracy"][i] if i < len(hist["accuracy"]) else "",
                        "train_loss": hist["train_loss"][i] if i < len(hist["train_loss"]) else "",
                        "eval_loss": hist["eval_loss"][i] if i < len(hist["eval_loss"]) else "",
                    })
                buf = io.StringIO()
                writer = csv.DictWriter(buf, fieldnames=["round", "accuracy", "train_loss", "eval_loss"])
                writer.writeheader()
                writer.writerows(rows)
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=buf.getvalue(),
                    file_name="fl_results.csv",
                    mime="text/csv",
                )

    # ── TAB 4: Privacy Explanation ────────────────────────────────────────────
    with tab_privacy:
        st.subheader("Privacy in Federated Learning")
        st.markdown(
            "Federated Learning trains a shared model **without sharing raw patient data**. "
            "This demo shows how hospitals can collaborate while keeping data local."
        )

        priv_col, share_col = st.columns(2)

        with priv_col:
            st.markdown("### 🔒 What Stays Private")
            st.success(
                "- Raw patient records\n"
                "- Patient names and IDs\n"
                "- Individual diagnoses\n"
                "- Hospital-specific statistics\n"
                "- Any personally identifiable information"
            )

        with share_col:
            st.markdown("### 📤 What Gets Shared")
            st.info(
                "- Model weight *updates* (not data)\n"
                "- Aggregated loss values\n"
                "- Aggregated accuracy metrics\n"
                "- Number of training samples (count only)\n"
                "- Final global model parameters"
            )

        st.divider()
        st.markdown("### How Aggregation Works")

        steps = [
            ("1️⃣ Global Model Sent", "Server sends the current global model to all hospitals."),
            ("2️⃣ Local Training", "Each hospital trains on their own private data. Data never leaves the hospital."),
            ("3️⃣ Gradient Upload", "Hospitals send only the model weight *updates* back to the server."),
            ("4️⃣ FedAvg Aggregation", "Server averages all updates (weighted by patient count) into a new global model."),
            ("5️⃣ Repeat", "Process repeats for the configured number of rounds."),
        ]
        for title, desc in steps:
            with st.container(border=True):
                st.markdown(f"**{title}** — {desc}")
