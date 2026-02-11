"""
Hospital Interface (reused for both Cleveland and Hungarian)

Allows a hospital to:
  - Upload CSV data or load built-in sample data
  - Configure local training hyperparameters
  - Register with the federation
  - Watch its own local training progress during federation rounds
  - Understand what data stays private
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
from fl_runner import load_sample_data_info, register_hospital, save_hospital_config


# ── helpers ───────────────────────────────────────────────────────────────────

EXPECTED_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

def _status_badge(status: str) -> str:
    return {
        "waiting":  "⚪ Waiting",
        "ready":    "🟢 Ready",
        "training": "🔵 Training",
        "done":     "✅ Done",
    }.get(status, f"❓ {status}")


def _validate_df(df: pd.DataFrame) -> tuple[bool, str]:
    """Return (ok, message) for an uploaded DataFrame."""
    if len(df.columns) < 13:
        return False, f"Expected at least 13 feature columns, got {len(df.columns)}."
    if len(df) < 10:
        return False, "Dataset is too small (need at least 10 rows)."
    return True, "Dataset looks good."


# ── main render ───────────────────────────────────────────────────────────────

def render_hospital(hospital_name: str):
    display = hospital_name.capitalize()
    st.title(f"🏥 {display} Hospital Interface")
    st.caption("Upload your patient data, configure training, and join the federated learning session.")

    state = load_state()
    h = state["hospitals"].get(hospital_name, {})
    fed = state["federation"]
    is_registered = h.get("registered", False)
    status = h.get("status", "waiting")
    fed_status = fed.get("status", "waiting")
    is_training = fed.get("active", False) and fed_status not in ("complete", "waiting")

    # ── session state keys scoped to this hospital ────────────────────────────
    df_key = f"{hospital_name}_df"
    src_key = f"{hospital_name}_data_source"

    # ── tabs ──────────────────────────────────────────────────────────────────
    tab_upload, tab_config, tab_train, tab_privacy = st.tabs([
        "📁 Upload Data",
        "⚙️ Configuration",
        "🔄 Local Training",
        "🔒 Privacy Dashboard",
    ])

    # ── TAB 1: Upload Data ────────────────────────────────────────────────────
    with tab_upload:
        st.subheader(f"Patient Data — {display} Hospital")

        if is_registered:
            st.success(
                f"✅ **Registered** — {h['num_patients']} patients "
                f"({h.get('disease_rate', 0)*100:.1f}% disease rate). "
                "Data is ready for federated training."
            )

        # Choose data source
        st.markdown("**Select a data source:**")
        src_col1, src_col2 = st.columns(2)

        with src_col1:
            if st.button("📂 Use Sample Data", use_container_width=True,
                         help="Load the built-in UCI heart disease dataset for this hospital."):
                info = load_sample_data_info(hospital_name)
                if info:
                    st.session_state[df_key] = info["preview"].copy()
                    st.session_state[src_key] = "sample"
                    st.session_state[f"{hospital_name}_full_info"] = info
                else:
                    st.error("Sample data file not found. Check that data/heart_disease/raw/ exists.")

        with src_col2:
            uploaded = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                key=f"{hospital_name}_uploader",
                help="CSV with 13 feature columns + a target/num column.",
            )
            if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded)
                    ok, msg = _validate_df(df)
                    if ok:
                        st.session_state[df_key] = df
                        st.session_state[src_key] = "upload"
                        st.success(msg)
                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")

        # Preview & register
        if df_key in st.session_state:
            df_preview = st.session_state[df_key]
            src = st.session_state.get(src_key, "upload")

            if src == "sample":
                full_info = st.session_state.get(f"{hospital_name}_full_info", {})
                st.markdown(f"**Source:** Built-in UCI sample data  |  "
                            f"**Patients:** {full_info.get('num_patients', '?')}  |  "
                            f"**Features:** {full_info.get('num_features', 13)}  |  "
                            f"**Disease rate:** {full_info.get('disease_rate', 0)*100:.1f}%")
            else:
                target_col = "num" if "num" in df_preview.columns else df_preview.columns[-1]
                disease_rate = (pd.to_numeric(df_preview[target_col], errors="coerce").fillna(0) > 0).mean()
                st.markdown(f"**Source:** Uploaded CSV  |  "
                            f"**Patients:** {len(df_preview)}  |  "
                            f"**Features:** {len(df_preview.columns) - 1}  |  "
                            f"**Disease rate:** {disease_rate*100:.1f}%")

            st.markdown("**Preview (first 5 rows):**")
            st.dataframe(df_preview.head(), use_container_width=True)

            st.warning("⚠️ Ensure this file contains no patient names, IDs, or other personally identifiable information.")

            if st.button(
                "📡 Register with Federation",
                type="primary",
                disabled=is_registered,
                use_container_width=True,
            ):
                # Determine which full dataset to use
                if src == "sample":
                    info = st.session_state.get(f"{hospital_name}_full_info", {})
                    full_info = load_sample_data_info(hospital_name)
                    from src.data_loader import load_hospital_data
                    from pathlib import Path as _Path
                    ROOT = _Path(__file__).parent.parent.parent
                    data_dir = str(ROOT / "data" / "heart_disease" / "raw")
                    X, y = load_hospital_data(hospital_name, data_dir=data_dir)
                    cols = EXPECTED_COLS + ["num"]
                    full_df = pd.DataFrame(
                        np.column_stack([X, y]), columns=cols
                    )
                else:
                    full_df = df_preview

                cfg = state["hospitals"][hospital_name].get("config", {})
                ok = register_hospital(hospital_name, full_df, cfg)
                if ok:
                    st.success(f"✅ {display} successfully registered with the federation!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Registration failed. Please try again.")

    # ── TAB 2: Configuration ──────────────────────────────────────────────────
    with tab_config:
        st.subheader(f"Training Configuration — {display}")
        st.caption("These settings are applied when the coordinator starts training.")

        cfg = h.get("config", {})

        c1, c2 = st.columns(2)
        epochs = c1.slider(
            "Local epochs per round", 1, 10,
            int(cfg.get("epochs_per_round", 1)),
            disabled=is_training,
        )
        batch_options = [8, 16, 32, 64, 128]
        current_batch = int(cfg.get("batch_size", 32))
        batch_idx = batch_options.index(current_batch) if current_batch in batch_options else 2
        batch_size = c2.selectbox(
            "Batch size", batch_options, index=batch_idx,
            disabled=is_training,
        )

        c3, c4 = st.columns(2)
        lr_options = [0.001, 0.01, 0.05, 0.1]
        current_lr = float(cfg.get("learning_rate", 0.01))
        lr_idx = lr_options.index(current_lr) if current_lr in lr_options else 1
        lr = c3.selectbox(
            "Learning rate", lr_options, index=lr_idx,
            disabled=is_training,
        )
        train_split_pct = c4.slider(
            "Train / test split (%)", 60, 90,
            int(float(cfg.get("train_split", 0.8)) * 100),
            disabled=is_training,
        )
        train_split = train_split_pct / 100

        # Show calculated counts
        num_patients = h.get("num_patients", 0)
        if num_patients > 0:
            n_train = int(num_patients * train_split)
            n_test = num_patients - n_train
            st.caption(f"Estimated: **{n_train}** training samples, **{n_test}** test samples")

        if st.button("💾 Save Configuration", disabled=is_training):
            new_cfg = {
                "epochs_per_round": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "train_split": train_split,
            }
            ok = save_hospital_config(hospital_name, new_cfg)
            if ok:
                st.success("Configuration saved.")
                st.rerun()
            else:
                st.error("Failed to save configuration.")

    # ── TAB 3: Local Training ─────────────────────────────────────────────────
    with tab_train:
        st.subheader(f"Training Status — {display}")

        current_round = fed.get("current_round", 0)
        total_rounds = fed.get("total_rounds", 3)
        pct = int(current_round / total_rounds * 100) if total_rounds > 0 else 0

        if not is_registered:
            st.warning("Register your data first (Upload Data tab).")
        elif not fed.get("active", False) and not (fed_status == "complete"):
            st.info("⏳ Waiting for the coordinator to start training.")
        else:
            # Status line
            st.markdown(f"**Hospital status:** {_status_badge(status)}")
            st.markdown(f"**Federation round:** {current_round} / {total_rounds}")
            st.progress(pct)

            if fed_status == "complete" or status == "done":
                st.success("Training complete! The global model has been updated with contributions from all hospitals.")
            elif is_training:
                st.info("Training in progress. Your local model is being trained and updates are being sent to the coordinator.")

            # Show history metrics relevant to this hospital
            hist = fed.get("history", {})
            if hist.get("train_loss") or hist.get("accuracy"):
                st.divider()
                st.markdown("**Federation Metrics**")
                mc1, mc2 = st.columns(2)
                if hist.get("train_loss"):
                    mc1.metric("Latest Training Loss", f"{hist['train_loss'][-1]:.4f}")
                if hist.get("accuracy"):
                    mc2.metric("Latest Global Accuracy", f"{hist['accuracy'][-1]*100:.1f}%")

            # Auto-refresh while training
            if is_training:
                time.sleep(2)
                st.rerun()

    # ── TAB 4: Privacy Dashboard ──────────────────────────────────────────────
    with tab_privacy:
        st.subheader("Your Data Stays Here")
        st.markdown(
            f"**{display} Hospital** participates in federated learning. "
            "Your patient records **never leave this hospital**."
        )

        st.success(
            f"### 🔒 {display}'s Private Data\n\n"
            "- Individual patient records\n"
            "- Diagnoses and test results\n"
            "- Patient demographics\n"
            "- All raw medical features"
        )

        st.info(
            "### 📤 What You Share with the Federation\n\n"
            "- Model weight *updates* (mathematical gradients)\n"
            "- Aggregated loss metric (single number per round)\n"
            "- Aggregated accuracy metric (single number per round)\n"
            "- Number of training examples (count only)"
        )

        st.divider()
        st.markdown("### Why This is Secure")
        with st.container(border=True):
            st.markdown(
                "**Gradient sharing != data sharing.**  \n"
                "The weight updates sent to the server are high-dimensional mathematical vectors "
                "that encode *how the model changed* during training — not the training data itself.  \n\n"
                "In this simulation, an adversary with access to only the weight updates cannot "
                "reconstruct any individual patient's record."
            )

        with st.container(border=True):
            st.markdown(
                "**FedAvg aggregation adds noise.**  \n"
                "The server averages updates from *all* hospitals together. "
                "This aggregation further obscures any single hospital's contribution, "
                "making it even harder to infer private data."
            )
