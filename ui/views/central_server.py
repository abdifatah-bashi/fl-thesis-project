"""
Central Server Dashboard

Flow:
  Step 1 → Publish Global Model  (no dependencies)
  Step 2 → Configure & Start Aggregation  (once hospitals submit weights)
  Step 3 → View Results
"""

import streamlit as st
import pandas as pd
import time
import io
import csv
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from state_manager import load_state, save_state, reset_state
from fl_runner import publish_global_model, start_aggregation
from components.charts import (
    create_accuracy_chart,
    create_loss_chart,
    create_hospital_distribution_chart,
)

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"


def _label(text: str):
    st.markdown(f'<p class="sec-label">{text}</p>', unsafe_allow_html=True)


def _pill(status: str) -> str:
    mapping = {
        "waiting":  '<span class="pill pill-waiting">Waiting</span>',
        "ready":    '<span class="pill pill-ready">Ready</span>',
        "training": '<span class="pill pill-training">Training</span>',
        "done":     '<span class="pill pill-done">Complete</span>',
    }
    return mapping.get(status, f'<span class="pill pill-waiting">{status}</span>')


def render_central_server():
    from components.nav import render_page_header

    state = load_state()
    hospitals = state["hospitals"]
    fed = state["federation"]
    hist = fed["history"]

    registered = [n for n, h in hospitals.items() if h.get("registered")]
    n_reg = len(registered)
    model_published = (RESULTS_DIR / "global_params.pt").exists()
    fed_active = fed.get("active", False)
    fed_status = fed.get("status", "waiting")
    is_complete = fed_status == "complete"
    is_training = fed_active and not is_complete
    current_round = fed.get("current_round", 0)
    total_rounds = fed.get("total_rounds", 3)

    # Celebrate once
    if is_complete and not fed.get("celebrated", False) and hist.get("accuracy"):
        fed["celebrated"] = True
        save_state(state)
        st.balloons()

    render_page_header("🖥️", "Central Server", "Orchestrate the federated learning session")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Publish Global Model
    # ══════════════════════════════════════════════════════════════════════════
    _label("Step 1 — Publish Global Model")

    if model_published:
        st.success("Global model published — hospitals can now fetch it.")
    else:
        st.info("Publish the initial model so hospitals can fetch it and start training.")
        if st.button("Publish Global Model", type="primary", use_container_width=True):
            publish_global_model()
            st.toast("Global model published!", icon="✅")
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Configure & Start Aggregation
    # ══════════════════════════════════════════════════════════════════════════
    _label("Step 2 — Configure & Start Aggregation")

    if not model_published:
        st.caption("Publish the global model first.")

    elif is_complete:
        st.success(f"Aggregation complete — {total_rounds} rounds finished.")

    elif is_training:
        # Show live progress
        pct = int(current_round / total_rounds * 100) if total_rounds > 0 else 0
        st.progress(pct)
        st.caption(f"Round {current_round} / {total_rounds}")

        # Per-hospital status (dynamic)
        for name, h in hospitals.items():
            if h.get("registered"):
                h_round = h.get("round_submitted", 0)
                st.markdown(
                    f"**{name.capitalize()}** — submitted {h_round}/{total_rounds} rounds "
                    f"{_pill(h.get('status', 'waiting'))}",
                    unsafe_allow_html=True,
                )

        # Live metrics
        if hist.get("accuracy"):
            st.markdown("")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{hist['accuracy'][-1]*100:.1f}%")
            if hist.get("train_loss"):
                m2.metric("Train Loss", f"{hist['train_loss'][-1]:.4f}")
            if hist.get("eval_loss"):
                m3.metric("Eval Loss", f"{hist['eval_loss'][-1]:.4f}")

        time.sleep(1)
        st.rerun()

    else:
        # Configuration
        c1, c2, c3 = st.columns(3)
        num_rounds = c1.slider("Rounds", 1, 10, 3, help="Number of federated rounds.")
        epochs = c2.slider("Epochs/round", 1, 10, 1, help="Local training epochs per round.")
        lr_opts = [0.001, 0.01, 0.05, 0.1]
        lr = c3.selectbox("Learning rate", lr_opts, index=1)

        st.markdown("")

        # Hospital status (dynamic)
        if hospitals:
            for name, h in hospitals.items():
                if h.get("registered"):
                    st.markdown(
                        f"**{name.capitalize()}** — {h['num_patients']} patients {_pill('ready')}",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"**{name.capitalize()}** — {_pill('waiting')}",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No hospitals have connected yet.")

        st.markdown("")

        can_start = n_reg >= 1
        if can_start:
            if st.button("Start Aggregation", type="primary", use_container_width=True):
                config = {"num_rounds": num_rounds, "epochs_per_round": epochs, "learning_rate": lr}
                st.session_state.sim_thread = start_aggregation(config)
                st.rerun()
        else:
            st.caption("Waiting for at least one hospital to load data and train.")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Results
    # ══════════════════════════════════════════════════════════════════════════
    if hist.get("accuracy") or hist.get("train_loss"):
        _label("Step 3 — Results")

        rounds = hist["rounds"] or list(range(1, len(hist.get("accuracy", [])) + 1))

        # Summary
        final_acc = hist["accuracy"][-1] if hist.get("accuracy") else 0
        final_loss = (hist["eval_loss"][-1] if hist.get("eval_loss")
                      else hist["train_loss"][-1] if hist.get("train_loss") else 0)
        total_pts = sum(h.get("num_patients", 0) for h in hospitals.values() if h.get("registered"))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final Accuracy", f"{final_acc*100:.1f}%",
                  delta=f"+{(final_acc - hist['accuracy'][0])*100:.1f}pp" if len(hist["accuracy"]) > 1 else None)
        m2.metric("Final Loss", f"{final_loss:.4f}")
        m3.metric("Total Patients", total_pts)
        m4.metric("Rounds", len(rounds))

        st.markdown("")

        # Charts
        ch1, ch2 = st.columns(2)
        if hist.get("accuracy") and rounds:
            ch1.plotly_chart(create_accuracy_chart(rounds, hist["accuracy"]),
                            use_container_width=True)
        if rounds and (hist.get("train_loss") or hist.get("eval_loss")):
            ch2.plotly_chart(create_loss_chart(rounds, hist.get("train_loss", []), hist.get("eval_loss", [])),
                            use_container_width=True)

        st.plotly_chart(create_hospital_distribution_chart(hospitals), use_container_width=True)

        # Download
        rows = []
        for i, r in enumerate(rounds):
            rows.append({
                "round": r,
                "accuracy": hist["accuracy"][i] if i < len(hist.get("accuracy", [])) else "",
                "train_loss": hist["train_loss"][i] if i < len(hist.get("train_loss", [])) else "",
                "eval_loss": hist["eval_loss"][i] if i < len(hist.get("eval_loss", [])) else "",
            })
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["round", "accuracy", "train_loss", "eval_loss"])
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("Download Results CSV", data=buf.getvalue(),
                           file_name="fl_results.csv", mime="text/csv")

    # ── Reset button ─────────────────────────────────────────────────────────
    st.markdown("")
    st.divider()
    if st.button("Reset Everything", help="Clear all state and start over"):
        reset_state()
        # Clean up weight files
        for f in RESULTS_DIR.glob("*.pt"):
            f.unlink(missing_ok=True)
        for k in [k for k in st.session_state if k != "role"]:
            del st.session_state[k]
        st.rerun()
