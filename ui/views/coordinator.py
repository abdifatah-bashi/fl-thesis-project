"""
Coordinator Dashboard

Workflow:
  Step 1 → Wait for Hospitals  (both register)
  Step 2 → Configure Training  (rounds, epochs, lr)
  Step 3 → Start Training      (launch FL simulation)
  Step 4 → View Results        (charts, download CSV)
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

import streamlit as st

from state_manager import load_state, save_state, reset_state
from fl_runner import start_simulation_thread
from components.charts import (
    create_accuracy_chart,
    create_loss_chart,
    create_hospital_distribution_chart,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _pill(status: str) -> str:
    mapping = {
        "waiting":  '<span class="pill pill-waiting">Waiting</span>',
        "ready":    '<span class="pill pill-ready">Ready</span>',
        "training": '<span class="pill pill-training">Training</span>',
        "done":     '<span class="pill pill-done">Complete</span>',
        "error":    '<span class="pill pill-error">Error</span>',
    }
    return mapping.get(status, f'<span class="pill pill-waiting">{status}</span>')


def _coord_step(registered: int, training_active: bool, is_complete: bool) -> int:
    if is_complete:     return 4
    if training_active: return 3
    if registered >= 2: return 2
    return min(registered, 1)   # 0 or 1


def _render_coord_steps(step: int):
    STEPS = ["Hospitals Join", "Configure", "Training", "Results"]
    parts = []
    for i, label in enumerate(STEPS):
        if i < step:
            parts.append(f'<span class="sw-item sw-done">&#10003; {label}</span>')
        elif i == step:
            parts.append(f'<span class="sw-item sw-active">{label}</span>')
        else:
            parts.append(f'<span class="sw-item sw-pending">{label}</span>')
    html = ' <span class="sw-arrow">&#8250;</span> '.join(parts)
    st.markdown(f'<div class="step-wizard">{html}</div>', unsafe_allow_html=True)


def _hosp_card(name: str, h: dict):
    """Rich hospital status card with colour-coded border."""
    status    = h.get("status", "waiting")
    css_class = {
        "ready":    "hosp-card registered",
        "training": "hosp-card training",
        "done":     "hosp-card done",
    }.get(status, "hosp-card")

    n_pts      = h.get("num_patients", 0)
    d_rate     = h.get("disease_rate", 0) * 100
    registered = h.get("registered", False)

    st.markdown(
        f'<div class="{css_class}">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">'
        f'  <span style="font-size:1.05rem;font-weight:700;color:#0f172a">{name.capitalize()} Hospital</span>'
        f'  {_pill(status)}'
        f'</div>',
        unsafe_allow_html=True,
    )
    if registered:
        c1, c2, c3 = st.columns(3)
        c1.metric("Patients",     n_pts)
        c2.metric("Disease rate", f"{d_rate:.1f}%")
        c3.metric("Learning rate", h.get("config", {}).get("learning_rate", "—"))
    else:
        st.caption(f"Waiting for {name.capitalize()} to register.")
    st.markdown("</div>", unsafe_allow_html=True)


# ── main render ───────────────────────────────────────────────────────────────

def render_coordinator():
    from components.nav import render_page_header

    # Load state
    state     = load_state()
    hospitals = state["hospitals"]
    fed       = state["federation"]
    hist      = fed["history"]

    registered    = [n for n, h in hospitals.items() if h.get("registered")]
    n_reg         = len(registered)
    training_active = fed.get("active", False)
    fed_status    = fed.get("status", "waiting")
    is_complete   = fed_status == "complete"
    is_training   = training_active and not is_complete
    current_round = fed.get("current_round", 0)
    total_rounds  = fed.get("total_rounds", 3)
    pct           = int(current_round / total_rounds * 100) if total_rounds > 0 else 0

    # Celebrate training completion once per session
    just_completed = (
        is_complete
        and not st.session_state.get("_coord_celebrated", False)
        and hist.get("accuracy")
    )
    if just_completed:
        st.session_state["_coord_celebrated"] = True
        st.balloons()

    # Determine current state
    training_starting = st.session_state.get("training_started", False)
    
    # Clear training_started flag once actual training begins
    if training_active and training_starting:
        del st.session_state["training_started"]
        training_starting = False  # Update local variable too
    
    # Reset celebration flag if state was reset
    if not is_complete and not training_active and not training_starting:
        st.session_state.pop("_coord_celebrated", None)
    
    # Auto-refresh while waiting for training to start
    if training_starting and not training_active:
        time.sleep(0.5)
        st.rerun()

    # ── page header ───────────────────────────────────────────────────────────
    render_page_header("🌐", "FL Coordinator", "Orchestrate the federated learning session")
    _render_coord_steps(_coord_step(n_reg, is_training, is_complete))

    # ── top status bar ────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Hospitals",  f"{n_reg} / 2")
    s2.metric("Status",     fed_status.replace("_", " ").title())
    s3.metric("Round",      f"{current_round} / {total_rounds}")
    s4.metric("Accuracy",   f"{hist['accuracy'][-1]*100:.1f}%" if hist.get("accuracy") else "---")

    st.markdown("")

    # ── tabs ──────────────────────────────────────────────────────────────────
    t1 = "Hospitals" + (f" ({n_reg}/2)" if n_reg < 2 else " ✓")
    t2 = "Results" + (" ✓" if hist.get("accuracy") else "")
    t3 = "Monitor" + (" ●" if is_training else (" ✓" if is_complete else ""))
    t4 = "Privacy"

    tab_hosp, tab_results, tab_monitor, tab_privacy = st.tabs([t1, t2, t3, t4])

    # ── TAB 1: Hospitals ─────────────────────────────────────────────────────
    with tab_hosp:
        # Hospital cards
        for name in ("cleveland", "hungarian"):
            _hosp_card(name, hospitals[name])

        st.markdown("")

        # Training config
        st.markdown(
            '<p class="sec-label">Training Configuration</p>',
            unsafe_allow_html=True,
        )
        if is_training:
            st.info("Training in progress — configuration locked.", icon="🔒")

        cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
        num_rounds = cfg_c1.slider("Number of rounds",   1, 10, 3, disabled=is_training,
                                   help="How many federated rounds to run.")
        epochs     = cfg_c2.slider("Epochs per round",   1, 10, 1, disabled=is_training,
                                   help="Local epochs at each hospital per round.")
        lr_opts    = [0.001, 0.01, 0.05, 0.1]
        lr         = cfg_c3.selectbox("Learning rate", lr_opts, index=1, disabled=is_training,
                                      help="Adam optimiser learning rate.")

        st.markdown("")

        # CTA row
        cta_col, reset_col = st.columns([4, 1])
        can_start = n_reg >= 2 and not training_active and not training_starting

        with cta_col:
            if can_start:
                if st.button("Start Federated Training", type="primary", use_container_width=True):
                    st.session_state["training_started"] = True
                    config = {"num_rounds": num_rounds, "epochs_per_round": epochs, "learning_rate": lr}
                    st.session_state.sim_thread = start_simulation_thread(config)
                    st.rerun()
            elif training_starting or is_training:
                st.markdown(
                    '<div class="status-banner status-running">'
                    '<div style="width:8px;height:8px;border-radius:50%;background:#6366f1;'
                    'animation:pulse-glow 2s infinite"></div>'
                    'Federated training is running&hellip;</div>',
                    unsafe_allow_html=True,
                )
            elif is_complete:
                st.markdown(
                    '<div class="status-banner status-complete">'
                    'Training complete &mdash; see the Results tab for charts and download.</div>',
                    unsafe_allow_html=True,
                )
            else:
                missing = [n for n in ("cleveland", "hungarian") if not hospitals[n].get("registered")]
                missing_str = " and ".join(m.capitalize() for m in missing)
                st.markdown(
                    f'<div class="status-banner status-waiting">'
                    f'Waiting for <b>{missing_str}</b> to register ({n_reg}/2 connected).</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
                st.markdown(
                    f'<div class="next-step">'
                    f'<div class="next-step-title">How to register hospitals</div>'
                    f'Open a new browser tab at <code>localhost:8501</code>, select '
                    f'<b>{missing[0].capitalize() if missing else "a hospital"}</b>, '
                    f'load data, and click Register.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with reset_col:
            if st.button("Reset", use_container_width=True, help="Clear all state and start over"):
                reset_state()
                for k in [k for k in st.session_state if k != "role"]:
                    del st.session_state[k]
                st.rerun()

    # ── TAB 2: Monitor ───────────────────────────────────────────────────────
    with tab_monitor:
        if not training_active and not is_complete:
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-state-icon">📡</div>'
                '<div class="empty-state-title">Training not started yet</div>'
                '<div class="empty-state-desc">'
                'Register both hospitals and click <b>Start Federated Training</b> '
                'in the Hospitals tab.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            # Overall progress
            if is_complete:
                st.success(f"✅ **All {total_rounds} rounds complete.** Head to the Results tab for charts.")
            else:
                st.markdown(f"**Round {current_round} / {total_rounds}** &nbsp; — &nbsp; {fed_status.replace('_',' ').title()}")

            st.progress(pct)

            # Per-hospital progress
            st.markdown(
                '<p class="sec-label">Hospital Progress</p>',
                unsafe_allow_html=True,
            )
            for name in ("cleveland", "hungarian"):
                h = hospitals[name]
                if h.get("registered"):
                    h_status = h.get("status", "waiting")
                    h_pct    = pct if h_status == "training" else (100 if h_status == "done" else 0)
                    hc1, hc2 = st.columns([3, 1])
                    hc1.markdown(f"**{name.capitalize()}**")
                    hc1.progress(h_pct)
                    hc2.markdown(_pill(h_status), unsafe_allow_html=True)

            # Live round metrics
            if hist.get("train_loss") or hist.get("accuracy"):
                st.markdown("")
                st.markdown(
                    '<p class="sec-label">Metrics</p>',
                    unsafe_allow_html=True,
                )
                mc1, mc2, mc3 = st.columns(3)
                if hist.get("train_loss"):
                    delta = None
                    if len(hist["train_loss"]) > 1:
                        delta = f"{hist['train_loss'][-1] - hist['train_loss'][-2]:+.4f}"
                    mc1.metric("Training Loss", f"{hist['train_loss'][-1]:.4f}", delta=delta, delta_color="inverse")
                if hist.get("accuracy"):
                    mc2.metric("Global Accuracy", f"{hist['accuracy'][-1]*100:.1f}%")
                if hist.get("eval_loss"):
                    mc3.metric("Eval Loss", f"{hist['eval_loss'][-1]:.4f}")

                # Per-round table (live)
                if hist.get("rounds"):
                    import pandas as _pd
                    st.divider()
                    tbl = {
                        "Round":        hist["rounds"],
                        "Accuracy (%)": [f"{a*100:.1f}" for a in hist.get("accuracy", [])],
                        "Train Loss":   [f"{l:.4f}" for l in hist.get("train_loss", [])],
                        "Eval Loss":    [f"{l:.4f}" for l in hist.get("eval_loss", [])],
                    }
                    st.dataframe(_pd.DataFrame(tbl), use_container_width=True, hide_index=True)

            # Auto-refresh while training
            if is_training:
                time.sleep(2)
                st.rerun()

    # ── TAB 3: Results ───────────────────────────────────────────────────────
    with tab_results:
        if not hist["accuracy"] and not hist["train_loss"]:
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-state-icon">📊</div>'
                '<div class="empty-state-title">No results yet</div>'
                '<div class="empty-state-desc">'
                'Results will appear here as rounds complete.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            rounds = hist["rounds"] or list(range(1, len(hist.get("accuracy", [])) + 1))

            # Summary metrics
            final_acc  = hist["accuracy"][-1] if hist.get("accuracy") else 0
            final_loss = (hist["eval_loss"][-1] if hist.get("eval_loss")
                          else hist["train_loss"][-1] if hist.get("train_loss") else 0)
            total_pts  = sum(h.get("num_patients", 0) for h in hospitals.values() if h.get("registered"))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final Accuracy",   f"{final_acc*100:.1f}%",
                      delta=f"+{(final_acc - hist['accuracy'][0])*100:.1f}pp vs round 1" if len(hist["accuracy"]) > 1 else None)
            m2.metric("Final Loss",        f"{final_loss:.4f}")
            m3.metric("Total Patients",    total_pts)
            m4.metric("Rounds Completed",  len(rounds))

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
            st.markdown("")
            rows = []
            for i, r in enumerate(rounds):
                rows.append({
                    "round":      r,
                    "accuracy":   hist["accuracy"][i]   if i < len(hist.get("accuracy",   [])) else "",
                    "train_loss": hist["train_loss"][i]  if i < len(hist.get("train_loss", [])) else "",
                    "eval_loss":  hist["eval_loss"][i]   if i < len(hist.get("eval_loss",  [])) else "",
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

    # ── TAB 4: Privacy ───────────────────────────────────────────────────────
    with tab_privacy:
        st.markdown(
            '<p style="font-size:.95rem;color:#64748b;line-height:1.7;max-width:680px;margin:8px 0 24px">'
            'The coordinator <b>never sees raw patient data</b>. '
            'Only aggregated model weight updates travel over the network.</p>',
            unsafe_allow_html=True,
        )

        pc1, pc2 = st.columns(2, gap="large")
        with pc1:
            with st.container(border=True):
                st.markdown("**Stays at Each Hospital**")
                for item in ["Raw patient records", "Individual diagnoses",
                             "Patient names / IDs", "Feature values", "Hospital statistics"]:
                    st.markdown(f"&check; &nbsp; {item}")

        with pc2:
            with st.container(border=True):
                st.markdown("**Shared with Coordinator**")
                for item in ["Model weight **updates** only", "Weighted-avg training loss",
                             "Weighted-avg accuracy", "Sample count per hospital",
                             "Final aggregated model"]:
                    st.markdown(f"&rarr; &nbsp; {item}")

        st.markdown("")
        st.markdown(
            '<p class="sec-label">Round-by-round aggregation</p>',
            unsafe_allow_html=True,
        )
        steps = [
            ("1. Distribute",  "Coordinator broadcasts current global model to all hospitals."),
            ("2. Local train", "Each hospital trains privately on local data for N epochs."),
            ("3. Upload",      "Hospitals send back **only** weight deltas — no data."),
            ("4. FedAvg",      "Coordinator averages deltas (weighted by patient count)."),
            ("5. Repeat",      "Process repeats for the configured number of rounds."),
        ]
        for title, desc in steps:
            with st.container(border=True):
                st.markdown(f"**{title}** — {desc}")
