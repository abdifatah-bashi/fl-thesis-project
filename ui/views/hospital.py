"""
Hospital Interface — Cleveland & Hungarian.

Four-step workflow displayed as a progress wizard:
  1 → Load Data    2 → Configure    3 → Register    4 → Training
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

ROOT_PATH = Path(__file__).parent.parent.parent
EXPECTED_COLS = ["age","sex","cp","trestbps","chol","fbs","restecg",
                 "thalach","exang","oldpeak","slope","ca","thal"]


# ── tiny helpers ─────────────────────────────────────────────────────────────

def _pill(status: str) -> str:
    m = {
        "waiting":  '<span class="pill pill-waiting">⏸ Waiting</span>',
        "ready":    '<span class="pill pill-ready">✓ Ready</span>',
        "training": '<span class="pill pill-training">⟳ Training</span>',
        "done":     '<span class="pill pill-done">✓ Complete</span>',
    }
    return m.get(status, f'<span class="pill pill-waiting">{status}</span>')


def _validate_df(df):
    if len(df.columns) < 13:
        return False, f"Need ≥ 13 feature columns — got {len(df.columns)}."
    if len(df) < 10:
        return False, "Dataset too small (need ≥ 10 rows)."
    return True, "Dataset looks good."


def _progress_pills(data_loaded: bool, is_registered: bool,
                    is_training: bool, is_complete: bool) -> None:
    """Step progress tracker — numbered circles + connecting lines.
    Classic stepper pattern: visually distinct from tabs/buttons.
    Not clickable. Rendered outside st.columns() — no clipping risk.
    """
    step_defs = [
        ("Load Data", data_loaded or is_registered),
        ("Register",  is_registered),
        ("Training",  is_complete),
    ]
    # Active = first step that isn't done yet
    active_idx = next((i for i, (_, done) in enumerate(step_defs) if not done), None)

    circles, lines = [], []
    for i, (label, done) in enumerate(step_defs):
        active = (i == active_idx)
        if done:
            # filled green circle, ✓
            c_bg, c_brd, c_txt, c_col = "#d1fae5", "2px solid #6ee7b7",  "✓",       "#065f46"
            l_col, l_wt = "#1e293b", "600"
        elif active:
            # solid blue circle, white number — clearly "current", not a button
            c_bg, c_brd, c_txt, c_col = "#3b82f6", "2px solid #3b82f6",  str(i+1),  "#ffffff"
            l_col, l_wt = "#1d4ed8", "700"
        else:
            # hollow gray circle, gray number
            c_bg, c_brd, c_txt, c_col = "#f8fafc", "2px solid #e2e8f0",  str(i+1),  "#94a3b8"
            l_col, l_wt = "#94a3b8", "400"

        circles.append(
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'gap:8px;min-width:90px;flex:0 0 auto">'
            f'<div style="width:40px;height:40px;border-radius:50%;background:{c_bg};'
            f'border:{c_brd};display:flex;align-items:center;justify-content:center;'
            f'font-size:1rem;font-weight:700;color:{c_col}">{c_txt}</div>'
            f'<span style="font-size:.9rem;font-weight:{l_wt};color:{l_col};'
            f'white-space:nowrap;letter-spacing:.01em">{label}</span>'
            f'</div>'
        )
        if i < len(step_defs) - 1:
            line_col = "#6ee7b7" if done else "#e2e8f0"
            lines.append(
                f'<div style="flex:1;height:2px;background:{line_col};'
                f'margin:0 6px;align-self:flex-start;margin-top:19px"></div>'
            )

    interleaved = []
    for i, circle in enumerate(circles):
        interleaved.append(circle)
        if i < len(lines):
            interleaved.append(lines[i])

    st.markdown(
        '<p style="font-size:.7rem;font-weight:700;letter-spacing:.1em;'
        'color:#94a3b8;text-transform:uppercase;margin:16px 0 10px">Progress</p>'
        '<div style="display:flex;align-items:flex-start;padding:0 0 24px">'
        + "".join(interleaved)
        + "</div>",
        unsafe_allow_html=True,
    )


def _next_step(title: str, body: str):
    st.markdown(
        f'<div class="next-step"><div class="next-step-title">→ Next: {title}</div>'
        f'<div style="color:#374151;margin-top:4px;font-size:.875rem">{body}</div></div>',
        unsafe_allow_html=True,
    )


def _label(text: str):
    """Small all-caps section label."""
    st.markdown(
        f'<p style="font-size:.7rem;font-weight:700;letter-spacing:.08em;'
        f'text-transform:uppercase;color:#94a3b8;margin:20px 0 8px">{text}</p>',
        unsafe_allow_html=True,
    )


# ── main ─────────────────────────────────────────────────────────────────────

def render_hospital(hospital_name: str):
    from components.nav import render_page_header

    display = hospital_name.capitalize()

    state         = load_state()
    h             = state["hospitals"].get(hospital_name, {})
    fed           = state["federation"]
    is_registered = h.get("registered", False)
    status        = h.get("status", "waiting")
    fed_status    = fed.get("status", "waiting")
    is_training   = fed.get("active", False) and fed_status not in ("complete", "waiting")
    is_complete   = fed_status == "complete" or status == "done"

    df_key      = f"{hospital_name}_df"
    src_key     = f"{hospital_name}_data_source"
    data_loaded = df_key in st.session_state

    # ── header ────────────────────────────────────────────────────────────────
    render_page_header("🏥", f"{display} Hospital", "Join the federated learning session")

    # ── progress status (informational pills, not navigation) ─────────────────
    _progress_pills(data_loaded, is_registered, is_training, is_complete)

    # ── top-of-page status message (only when meaningful) ─────────────────────
    if is_complete:
        st.success("🎉 Training complete! Your data contributed to the global model.")
    elif is_registered and not is_training:
        st.success(f"✅ Registered · {h['num_patients']} patients · waiting for training.")
        _next_step(
            "Open the Coordinator",
            "Switch to <b>FL Coordinator</b> and click <b>Start Federated Training</b> "
            "once both hospitals are registered.",
        )

    # ── custom tab navigation — session-state driven, supports auto-switching ──
    _TAB_KEY = f"tab_{hospital_name}"
    if _TAB_KEY not in st.session_state:
        st.session_state[_TAB_KEY] = 0   # 0=Data  1=Setup  2=Training  3=Privacy

    _tab_labels = [
        "Data"     + (" ✓" if data_loaded or is_registered else ""),
        "Setup"    + (" ✓" if is_registered else ""),
        "Training" + (" ●" if is_training else (" ✓" if is_complete else "")),
        "Privacy",
    ]
    _active = st.session_state[_TAB_KEY]

    tc0, tc1, tc2, tc3 = st.columns(4)
    for _idx, (_col, _lbl) in enumerate(zip([tc0, tc1, tc2, tc3], _tab_labels)):
        with _col:
            if st.button(_lbl, key=f"{hospital_name}_tab_{_idx}",
                         use_container_width=True,
                         type="primary" if _active == _idx else "secondary"):
                st.session_state[_TAB_KEY] = _idx
                st.rerun()
    st.markdown(
        '<hr style="margin:2px 0 20px;border:none;border-top:2px solid #e2e8f0">',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 0 — Data
    # ══════════════════════════════════════════════════════════════════════════
    if _active == 0:

        # ── PRIMARY ACTION: sample data ───────────────────────────────────────
        _label("Recommended")

        with st.container(border=True):
            col_info, col_btn = st.columns([3, 1], gap="large")

            with col_info:
                st.markdown("#### 🏥 Use Built-in Sample Data")
                st.markdown(
                    f"Pre-validated **UCI Heart Disease dataset** "
                    f"for **{display} Hospital** — bundled with this project."
                )
                info_preview = load_sample_data_info(hospital_name)
                if info_preview:
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Patients",     info_preview["num_patients"])
                    s2.metric("Features",     13)
                    s3.metric("Disease rate", f"{info_preview['disease_rate']*100:.1f}%")

            with col_btn:
                st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
                if st.button(
                    "Load Sample Data",
                    key=f"{hospital_name}_sample",
                    use_container_width=True,
                    type="primary",
                ):
                    with st.spinner("Loading…"):
                        info2 = load_sample_data_info(hospital_name)
                    if info2:
                        st.session_state[df_key]  = info2["preview"].copy()
                        st.session_state[src_key] = "sample"
                        st.session_state[f"{hospital_name}_info"] = info2
                        st.session_state[_TAB_KEY] = 1   # ← auto-navigate to Setup
                        st.toast(f"Loaded {info2['num_patients']} patients", icon="✅")
                        st.rerun()
                    else:
                        st.error("Sample file not found — check `data/heart_disease/raw/`.")

        # ── SECONDARY: upload ─────────────────────────────────────────────────
        st.markdown(
            '<div style="display:flex;align-items:center;gap:12px;margin:20px 0 14px">'
            '<hr style="flex:1;border:none;border-top:1px solid #e2e8f0">'
            '<span style="color:#94a3b8;font-size:.82rem;white-space:nowrap">'
            'or upload your own CSV</span>'
            '<hr style="flex:1;border:none;border-top:1px solid #e2e8f0">'
            '</div>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "CSV with 13 feature columns + a target or num column",
            type=["csv"],
            key=f"{hospital_name}_uploader",
        )
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                ok, msg = _validate_df(df_up)
                if ok:
                    st.session_state[df_key]  = df_up
                    st.session_state[src_key] = "upload"
                    st.session_state[_TAB_KEY] = 1   # ← auto-navigate to Setup
                    st.toast(msg, icon="✅")
                    st.rerun()
                else:
                    st.error(msg)
            except Exception as e:
                st.error(f"Could not parse file: {e}")

        # ── DATA PREVIEW ──────────────────────────────────────────────────────
        if data_loaded:
            st.divider()
            df_prev = st.session_state[df_key]
            src     = st.session_state.get(src_key, "upload")

            if src == "sample":
                inf     = st.session_state.get(f"{hospital_name}_info", {})
                n       = inf.get("num_patients", "?")
                dr      = inf.get("disease_rate", 0) * 100
                src_lbl = "Built-in UCI sample"
            else:
                tgt     = "num" if "num" in df_prev.columns else df_prev.columns[-1]
                dr      = (pd.to_numeric(df_prev[tgt], errors="coerce").fillna(0) > 0).mean() * 100
                n       = len(df_prev)
                src_lbl = "Uploaded CSV"

            _label("data loaded")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Source",       src_lbl)
            c2.metric("Patients",     n)
            c3.metric("Features",     13)
            c4.metric("Disease rate", f"{dr:.1f}%")

            if not is_registered:
                _, _clr = st.columns([5, 1])
                with _clr:
                    if st.button(
                        "✕ Change data",
                        key=f"{hospital_name}_clear",
                        use_container_width=True,
                        help="Remove this dataset and choose a different one",
                    ):
                        for k in [df_key, src_key, f"{hospital_name}_info"]:
                            st.session_state.pop(k, None)
                        st.toast("Dataset cleared — choose a new one above", icon="🗑️")
                        st.rerun()

            st.markdown("**Preview — first 5 rows**")
            st.dataframe(df_prev.head(), use_container_width=True, hide_index=True)

            st.warning(
                "Confirm this file contains **no patient names, IDs, or personally "
                "identifiable information** before registering.",
                icon="⚠️",
            )

        else:
            st.markdown(
                '<div style="text-align:center;padding:36px 0 12px;color:#94a3b8">'
                '<div style="font-size:2rem;margin-bottom:10px">📋</div>'
                '<div style="font-size:.95rem;font-weight:600;color:#64748b;margin-bottom:4px">'
                'No data loaded yet</div>'
                '<div style="font-size:.85rem">'
                'Click <b>Load Sample Data</b> above or upload a CSV.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Setup  (Configure + Register combined)
    # ══════════════════════════════════════════════════════════════════════════
    elif _active == 1:

        if not data_loaded and not is_registered:
            st.write("")
            st.info(
                "**Load your data first.**\n\n"
                "Go to the **Data** tab, load the sample dataset or upload a CSV, "
                "then return here to configure and register.",
                icon="📋",
            )
        else:
            # ── Part A: Hyperparameters ───────────────────────────────────────
            st.markdown("#### ⚙️ Training Hyperparameters")
            st.caption("Applied locally at this hospital during each FL round.")

            if is_training:
                st.info("Training is in progress — settings are locked.", icon="🔒")

            cfg = h.get("config", {})
            c1, c2 = st.columns(2)
            epochs = c1.slider(
                "Epochs per round", 1, 10,
                int(cfg.get("epochs_per_round", 1)),
                disabled=is_training,
                help="Full passes over local data per FL round.",
            )
            batch_opts = [8, 16, 32, 64, 128]
            cur_b = int(cfg.get("batch_size", 32))
            batch = c2.selectbox(
                "Batch size", batch_opts,
                index=batch_opts.index(cur_b) if cur_b in batch_opts else 2,
                disabled=is_training,
                help="Samples per gradient update step.",
            )

            c3, c4 = st.columns(2)
            lr_opts = [0.001, 0.01, 0.05, 0.1]
            cur_lr = float(cfg.get("learning_rate", 0.01))
            lr = c3.selectbox(
                "Learning rate", lr_opts,
                index=lr_opts.index(cur_lr) if cur_lr in lr_opts else 1,
                disabled=is_training,
                help="Adam optimiser step size.",
            )
            split_pct = c4.slider(
                "Train / test split", 60, 90,
                int(float(cfg.get("train_split", 0.8)) * 100),
                format="%d%%",
                disabled=is_training,
                help="% of data used for training; rest for evaluation.",
            )

            n_pts = h.get("num_patients", 0)
            if n_pts > 0:
                n_tr = int(n_pts * split_pct / 100)
                st.caption(f"**{n_tr}** training · **{n_pts - n_tr}** test (from {n_pts} total)")

            if not is_training:
                if st.button("💾 Save Configuration", type="primary", disabled=is_training):
                    new_cfg = {
                        "epochs_per_round": epochs,
                        "batch_size":       batch,
                        "learning_rate":    lr,
                        "train_split":      split_pct / 100,
                    }
                    if save_hospital_config(hospital_name, new_cfg):
                        st.toast("Configuration saved!", icon="✅")
                        st.rerun()
                    else:
                        st.error("Save failed — please try again.")

            # ── Part B: Register ──────────────────────────────────────────────
            st.divider()
            st.markdown("#### 📡 Register with the Federation")

            if is_registered:
                st.success(
                    f"**{display} is registered.** "
                    f"{h['num_patients']} patients · "
                    f"{h.get('disease_rate', 0)*100:.1f}% disease rate"
                )
                st.caption(f"Registered at: {h.get('registered_at', '—')}")
                _next_step(
                    "Wait for training to start",
                    "The coordinator will launch training once all hospitals have registered. "
                    "Check the <b>Training</b> tab for live progress.",
                )
            else:
                df_prev = st.session_state[df_key]
                src     = st.session_state.get(src_key, "upload")

                r1, r2, r3 = st.columns(3)
                if src == "sample":
                    inf = st.session_state.get(f"{hospital_name}_info", {})
                    r1.metric("Patients",     inf.get("num_patients", "?"))
                    r2.metric("Disease rate", f"{inf.get('disease_rate', 0)*100:.1f}%")
                else:
                    tgt = "num" if "num" in df_prev.columns else df_prev.columns[-1]
                    dr  = (pd.to_numeric(df_prev[tgt], errors="coerce").fillna(0) > 0).mean() * 100
                    r1.metric("Patients",     len(df_prev))
                    r2.metric("Disease rate", f"{dr:.1f}%")
                cfg2 = h.get("config", {})
                r3.metric("Config", f"lr={cfg2.get('learning_rate', 0.01)} · {cfg2.get('epochs_per_round', 1)} ep")

                st.info(
                    "By registering you confirm data has been de-identified and you consent "
                    "to sharing **model weight updates only** with the federation.",
                    icon="🔒",
                )

                if st.button("📡 Register with Federation", type="primary", use_container_width=True):
                    with st.spinner("Registering…"):
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
                        ok = register_hospital(hospital_name, full_df, h.get("config", {}))
                    if ok:
                        st.toast(f"✅ {display} registered!", icon="🏥")
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error("Registration failed — please try again.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Training
    # ══════════════════════════════════════════════════════════════════════════
    elif _active == 2:
        current_round = fed.get("current_round", 0)
        total_rounds  = fed.get("total_rounds", 3)
        pct  = int(current_round / total_rounds * 100) if total_rounds > 0 else 0
        hist = fed.get("history", {})

        if not is_registered:
            st.write("")
            st.info(
                "**This section activates once you've registered.**\n\n"
                "Steps to complete first:\n"
                "1. **Data** tab — load your patient dataset\n"
                "2. **Setup** tab — configure and register with the federation\n\n"
                "Training starts automatically when the coordinator launches a session.",
                icon="🔒",
            )

        elif not fed.get("active") and not is_complete:
            st.write("")
            st.success(
                f"**{display} is registered and ready!** "
                f"Waiting for the coordinator to launch training.",
                icon="⏳",
            )
            _next_step(
                "Switch to FL Coordinator",
                "Use the sidebar <b>← Change Role</b> to open the Coordinator dashboard "
                "and click <b>Start Federated Training</b> once both hospitals are registered.",
            )

        else:
            if is_complete:
                st.success("🎉 Training finished! Your updates shaped the global model.")
                st.balloons()
            else:
                st.markdown(
                    f'**Round {current_round} / {total_rounds}** &nbsp; {_pill(status)}',
                    unsafe_allow_html=True,
                )
                st.progress(pct)
                st.caption("Auto-refreshes every 2 s while training is active.")

            if hist.get("train_loss") or hist.get("accuracy"):
                st.divider()
                m1, m2, m3 = st.columns(3)
                if hist.get("train_loss"):
                    delta = (
                        f"{hist['train_loss'][-1] - hist['train_loss'][-2]:+.4f}"
                        if len(hist["train_loss"]) > 1 else None
                    )
                    m1.metric("Training Loss", f"{hist['train_loss'][-1]:.4f}",
                              delta=delta, delta_color="inverse")
                if hist.get("accuracy"):
                    m2.metric("Global Accuracy", f"{hist['accuracy'][-1]*100:.1f}%")
                if hist.get("eval_loss"):
                    m3.metric("Eval Loss", f"{hist['eval_loss'][-1]:.4f}")

                if hist.get("rounds") and len(hist["rounds"]) > 1:
                    st.divider()
                    st.caption("Per-round summary")
                    import pandas as _pd
                    st.dataframe(_pd.DataFrame({
                        "Round":        hist["rounds"],
                        "Accuracy (%)": [f"{a*100:.1f}" for a in hist.get("accuracy", [])],
                        "Train Loss":   [f"{l:.4f}" for l in hist.get("train_loss", [])],
                        "Eval Loss":    [f"{l:.4f}" for l in hist.get("eval_loss", [])],
                    }), use_container_width=True, hide_index=True)

            if is_training:
                time.sleep(2)
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Privacy
    # ══════════════════════════════════════════════════════════════════════════
    elif _active == 3:
        st.caption("Informational — explains how FL keeps your patient data private.")
        st.markdown("#### Privacy Guarantees")
        st.markdown(
            f"**{display}**'s patient records **never leave this hospital.** "
            "Only mathematical model updates are shared."
        )

        col_priv, col_share = st.columns(2, gap="large")
        with col_priv:
            with st.container(border=True):
                st.markdown("##### 🔒 Stays Private")
                for item in [
                    "Individual patient records",
                    "Diagnoses & test results",
                    "Patient demographics",
                    "All raw feature values",
                    "Any personal identifiers",
                ]:
                    st.markdown(f"✓ &nbsp; {item}")

        with col_share:
            with st.container(border=True):
                st.markdown("##### 📤 Shared with Federation")
                for item in [
                    "Model **weight updates** (not data)",
                    "Aggregated loss — 1 number/round",
                    "Aggregated accuracy — 1 number/round",
                    "Training sample **count** only",
                    "Final global model weights",
                ]:
                    st.markdown(f"→ &nbsp; {item}")

        st.divider()
        with st.container(border=True):
            st.markdown("##### Why weight updates ≠ patient data")
            st.markdown(
                "Weight updates are high-dimensional vectors that encode *how the model changed* — "
                "not the data it trained on. FedAvg further blends updates from **all hospitals** "
                "before the coordinator ever sees them, making reverse-engineering practically impossible."
            )
