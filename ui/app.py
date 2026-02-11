"""
Federated Learning Multi-Hospital Simulation — Main Entry Point

Run from the project root:
    streamlit run ui/app.py
"""

import sys
from pathlib import Path

# ── path setup (must happen before any local imports) ──────────────────────
ROOT = Path(__file__).parent.parent          # fl-thesis-project/
UI_DIR = Path(__file__).parent              # fl-thesis-project/ui/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UI_DIR))

import streamlit as st

from state_manager import load_state, reset_state

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FL Multi-Hospital Simulation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state init ─────────────────────────────────────────────────────
if "role" not in st.session_state:
    st.session_state.role = None


# ── sidebar ────────────────────────────────────────────────────────────────
def render_sidebar():
    state = load_state()
    hospitals = state["hospitals"]
    fed = state["federation"]

    with st.sidebar:
        st.markdown("## 🏥 FL System")
        st.divider()

        # Current role
        role = st.session_state.role
        if role:
            role_labels = {
                "coordinator": "🌐 FL Coordinator",
                "cleveland":   "🏥 Cleveland Hospital",
                "hungarian":   "🏥 Hungarian Hospital",
            }
            st.markdown(f"**Current Role:**  \n{role_labels.get(role, role.capitalize())}")
            if st.button("↩️ Switch Role", use_container_width=True):
                st.session_state.role = None
                st.rerun()
        else:
            st.markdown("**No role selected**")

        st.divider()

        # Federation status
        st.markdown("**Federation Status**")
        registered = [n for n, h in hospitals.items() if h.get("registered")]
        st.markdown(f"Hospitals connected: **{len(registered)}/2**")

        fed_status = fed.get("status", "waiting")
        status_emoji = {
            "waiting":  "⏳",
            "training": "🔵",
            "complete": "✅",
        }
        emoji = status_emoji.get(
            "complete" if fed_status == "complete"
            else ("training" if "round" in fed_status or fed_status == "training"
                  else "waiting"),
            "⏳",
        )
        st.markdown(f"Training: {emoji} **{fed_status.replace('_', ' ').title()}**")

        current = fed.get("current_round", 0)
        total = fed.get("total_rounds", 3)
        if total > 0:
            st.progress(int(current / total * 100))
            st.caption(f"Round {current} / {total}")

        st.divider()

        # Help
        with st.expander("❓ Help"):
            st.markdown(
                "**Quick Start (single window)**\n"
                "1. Select **Cleveland Hospital** — load sample data & register\n"
                "2. Select **Hungarian Hospital** — load sample data & register\n"
                "3. Select **FL Coordinator** — click Start Training\n\n"
                "**Multi-Tab Demo** _(recommended)_\n"
                "Open three browser tabs to `localhost:8501`, then:\n"
                "- Tab 1 → FL Coordinator\n"
                "- Tab 2 → Cleveland Hospital\n"
                "- Tab 3 → Hungarian Hospital"
            )


# ── landing page ────────────────────────────────────────────────────────────
def render_landing():
    st.markdown(
        "<h1 style='text-align:center'>Federated Learning<br>Multi-Hospital Simulation</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:gray'>Simulate collaborative AI training across hospitals — without sharing patient data.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### Select your role to get started")
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("## 🌐")
            st.markdown("### FL Coordinator")
            st.markdown(
                "Manage the federation, monitor hospital connections, "
                "launch training rounds, and view aggregated results."
            )
            if st.button("Select Coordinator", key="sel_coord", type="primary", use_container_width=True):
                st.session_state.role = "coordinator"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("## 🏥")
            st.markdown("### Cleveland Hospital")
            st.markdown(
                "Upload or load the Cleveland patient dataset, "
                "configure local training parameters, and join the federation."
            )
            if st.button("Select Cleveland", key="sel_clev", use_container_width=True):
                st.session_state.role = "cleveland"
                st.rerun()

    with col3:
        with st.container(border=True):
            st.markdown("## 🏥")
            st.markdown("### Hungarian Hospital")
            st.markdown(
                "Upload or load the Hungarian patient dataset, "
                "configure local training parameters, and join the federation."
            )
            if st.button("Select Hungarian", key="sel_hung", use_container_width=True):
                st.session_state.role = "hungarian"
                st.rerun()

    st.divider()
    st.info(
        "**Tip — Multi-Tab Demo:** Open this app in **3 browser tabs** "
        "(`localhost:8501`) and select a different role in each tab "
        "to fully simulate the multi-hospital federated learning experience."
    )

    # Show current federation state summary
    state = load_state()
    hospitals = state["hospitals"]
    registered = [n for n, h in hospitals.items() if h.get("registered")]
    if registered:
        st.markdown("---")
        st.markdown("**Current Federation State**")
        cols = st.columns(len(hospitals))
        for i, (name, h) in enumerate(hospitals.items()):
            with cols[i]:
                icon = "✅" if h.get("registered") else "⚪"
                st.metric(
                    f"{icon} {name.capitalize()}",
                    f"{h.get('num_patients', 0)} patients" if h.get("registered") else "Not registered",
                )


# ── main router ────────────────────────────────────────────────────────────
render_sidebar()

role = st.session_state.role

if role is None:
    render_landing()

elif role == "coordinator":
    from pages.coordinator import render_coordinator
    render_coordinator()

elif role in ("cleveland", "hungarian"):
    from pages.hospital import render_hospital
    render_hospital(role)

else:
    st.error(f"Unknown role: {role}")
    st.session_state.role = None
    st.rerun()
