"""
Hospital Interface — Clean FL client view.

Flow: Fetch Global Model → Load Data → Train Locally → Send Updates
No waiting, no bottlenecks. Hospital acts independently.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import streamlit as st

from state_manager import load_state, save_state
from fl_runner import load_sample_data_info, start_hospital_training

ROOT_PATH = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT_PATH / "results"
EXPECTED_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def _label(text: str):
    st.markdown(f'<p class="sec-label">{text}</p>', unsafe_allow_html=True)


def _step_indicator(number: int, title: str, status: str):
    """Render a step row: number circle, title, status icon."""
    styles = {
        "done":    ("#059669", "#d1fae5", "&#10003;"),
        "active":  ("#4f46e5", "#e0e7ff", "&#9679;"),
        "waiting": ("#94a3b8", "#f1f5f9", "&#8212;"),
    }
    color, bg, icon = styles.get(status, styles["waiting"])
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;padding:10px 0;'
        f'border-bottom:1px solid #f1f5f9">'
        f'<div style="width:32px;height:32px;border-radius:8px;background:{bg};'
        f'display:flex;align-items:center;justify-content:center;'
        f'font-weight:700;font-size:.85rem;color:{color};flex-shrink:0">{number}</div>'
        f'<span style="font-weight:600;color:#1e293b;flex:1">{title}</span>'
        f'<span style="font-size:.8rem;color:{color};font-weight:600">{icon}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_hospital(hospital_name: str):
    from components.nav import render_page_header

    display = hospital_name.capitalize()
    state = load_state()
    h = state["hospitals"].get(hospital_name, {})
    fed = state["federation"]
    is_complete = h.get("status") == "done"
    is_training = h.get("status") == "training"
    global_params_exist = (RESULTS_DIR / "global_params.pt").exists()

    df_key = f"{hospital_name}_df"
    src_key = f"{hospital_name}_data_source"
    data_loaded = df_key in st.session_state or h.get("registered", False)

    render_page_header("🏥", f"{display} Hospital", "Federated learning client")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Fetch Global Model
    # ══════════════════════════════════════════════════════════════════════════
    _label("Step 1 — Fetch Global Model")

    if global_params_exist:
        st.success("Global model parameters fetched.")
    else:
        st.info(
            "Global model not yet available. "
            "Switch to **Central Server** and click **Publish Global Model**.",
            icon="📡",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Load Data
    # ══════════════════════════════════════════════════════════════════════════
    _label("Step 2 — Load Local Data")

    if h.get("registered", False):
        st.success(
            f"Data loaded · **{h['num_patients']}** patients · "
            f"**{h.get('disease_rate', 0)*100:.1f}%** disease rate"
        )
    elif df_key in st.session_state:
        df_prev = st.session_state[df_key]
        src = st.session_state.get(src_key, "upload")
        if src == "sample":
            inf = st.session_state.get(f"{hospital_name}_info", {})
            n_pts = inf.get("num_patients", len(df_prev))
            dr = inf.get("disease_rate", 0) * 100
        else:
            tgt = "num" if "num" in df_prev.columns else df_prev.columns[-1]
            dr = (pd.to_numeric(df_prev[tgt], errors="coerce").fillna(0) > 0).mean() * 100
            n_pts = len(df_prev)

        st.success(f"Data loaded · **{n_pts}** patients · **{dr:.1f}%** disease rate")

        # Auto-register (saves data locally, no sharing)
        if not h.get("registered"):
            if src == "sample":
                from src.data_loader import load_hospital_data
                X, y = load_hospital_data(
                    hospital_name,
                    data_dir=str(ROOT_PATH / "data" / "heart_disease" / "raw"),
                )
                full_df = pd.DataFrame(
                    np.column_stack([X, y]),
                    columns=EXPECTED_COLS + ["num"],
                )
            else:
                full_df = df_prev
            from fl_runner import register_hospital
            register_hospital(hospital_name, full_df, h.get("config", {}))
            st.rerun()
    else:
        col_sample, col_or, col_upload = st.columns([2, 0.5, 2])

        with col_sample:
            info = load_sample_data_info(hospital_name)
            if info:
                st.caption(f"UCI Heart Disease · {info['num_patients']} patients")
            if st.button("Load Sample Data", key=f"{hospital_name}_sample",
                         type="primary", use_container_width=True):
                info2 = load_sample_data_info(hospital_name)
                if info2:
                    st.session_state[df_key] = info2["preview"].copy()
                    st.session_state[src_key] = "sample"
                    st.session_state[f"{hospital_name}_info"] = info2
                    st.rerun()
                else:
                    st.error("Sample file not found.")

        with col_or:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:80px;color:#94a3b8;font-size:.8rem">or</div>',
                unsafe_allow_html=True,
            )

        with col_upload:
            uploaded = st.file_uploader(
                "Upload CSV", type=["csv"], key=f"{hospital_name}_uploader",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                try:
                    df_up = pd.read_csv(uploaded)
                    if len(df_up.columns) >= 13 and len(df_up) >= 10:
                        st.session_state[df_key] = df_up
                        st.session_state[src_key] = "upload"
                        st.rerun()
                    else:
                        st.error("Need ≥ 13 columns and ≥ 10 rows.")
                except Exception as e:
                    st.error(f"Could not parse file: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Train Locally & Send Updates
    # ══════════════════════════════════════════════════════════════════════════
    _label("Step 3 — Train & Send Updates")

    ready = global_params_exist and h.get("registered", False)

    if is_complete:
        # Training finished
        _step_indicator(1, "Fetched global model parameters", "done")
        _step_indicator(2, "Trained on local data", "done")
        _step_indicator(3, "Sent weight updates to server", "done")

        st.markdown("")
        st.success(
            "Training complete — weight updates sent. "
            "Your data never left this hospital."
        )

        h_acc = h.get("local_accuracy", 0)
        h_loss = h.get("local_train_loss", 0)
        h_eval = h.get("local_eval_loss", 0)
        if h_acc or h_loss:
            _label("Local Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy", f"{h_acc*100:.1f}%")
            m2.metric("Train Loss", f"{h_loss:.4f}")
            m3.metric("Eval Loss", f"{h_eval:.4f}")

        hist = fed.get("history", {})
        if hist.get("accuracy"):
            _label("Global Model Metrics")
            m1, m2 = st.columns(2)
            m1.metric("Global Accuracy", f"{hist['accuracy'][-1]*100:.1f}%")
            if hist.get("train_loss"):
                m2.metric("Global Loss", f"{hist['train_loss'][-1]:.4f}")

        if not fed.get("celebrated", False):
            fed["celebrated"] = True
            save_state(state)
            st.balloons()

    elif is_training:
        # Training in progress
        total_rounds = fed.get("total_rounds", 3)
        h_round = h.get("round_submitted", 0)

        _step_indicator(1, "Fetch global model", "done")
        _step_indicator(2, f"Training locally (round {h_round + 1}/{total_rounds})", "active")
        _step_indicator(3, "Send weight updates", "done" if h_round > 0 else "waiting")

        st.markdown("")
        pct = int(h_round / total_rounds * 100) if total_rounds > 0 else 0
        st.progress(pct)
        st.caption(f"Completed {h_round} of {total_rounds} rounds")

        h_acc = h.get("local_accuracy", 0)
        h_loss = h.get("local_train_loss", 0)
        if h_acc or h_loss:
            m1, m2 = st.columns(2)
            m1.metric("Train Loss", f"{h_loss:.4f}")
            m2.metric("Accuracy", f"{h_acc*100:.1f}%")

        time.sleep(1)
        st.rerun()

    elif ready:
        # Both global model and data ready — start training
        st.info("Global model fetched and data loaded. Click below to start local training.")
        if st.button("Train & Send Updates", type="primary", use_container_width=True):
            st.session_state[f"{hospital_name}_training_thread"] = \
                start_hospital_training(hospital_name)
            st.rerun()

    else:
        # Not ready yet
        missing = []
        if not global_params_exist:
            missing.append("global model (central server must publish it)")
        if not h.get("registered"):
            missing.append("local data (load above)")
        st.caption(f"Waiting for: {', '.join(missing)}")
