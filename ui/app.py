"""
Federated Learning Multi-Hospital Simulation — Main Entry Point

Run from the project root:
    streamlit run ui/app.py
"""

import sys
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent
UI_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UI_DIR))

import streamlit as st
from state_manager import load_state, reset_state

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FL Multi-Hospital Simulation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state ────────────────────────────────────────────────────────────
if "role" not in st.session_state:
    st.session_state.role = None


# ── CSS ──────────────────────────────────────────────────────────────────────
def _inject_css():
    st.markdown("""
    <style>
    /* ── Global ── */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 2.5rem; max-width: 1100px; }

    /* ── Hero ── */
    .fl-hero { text-align: center; padding: 32px 0 8px; }
    .fl-hero-badge {
        display: inline-block;
        background: #eff6ff; color: #1d4ed8;
        border: 1px solid #bfdbfe; border-radius: 20px;
        padding: 4px 16px; font-size: 12px; font-weight: 600;
        margin-bottom: 16px; letter-spacing: .04em; text-transform: uppercase;
    }
    .fl-hero-title {
        font-size: 2.6rem; font-weight: 800; color: #0f172a;
        line-height: 1.15; letter-spacing: -.03em; margin-bottom: 12px;
    }
    .fl-hero-sub {
        font-size: 1.05rem; color: #64748b; line-height: 1.65; max-width: 580px;
        margin: 0 auto 32px;
    }

    /* ── Role cards ── */
    .role-card {
        background: #fff; border: 2px solid #e2e8f0; border-radius: 16px;
        padding: 28px 22px; height: 100%;
        transition: border-color .18s, box-shadow .18s, transform .18s;
    }
    .role-card:hover {
        border-color: #3b82f6; box-shadow: 0 8px 28px rgba(59,130,246,.14);
        transform: translateY(-3px);
    }
    .role-card-icon { font-size: 2.4rem; margin-bottom: 10px; }
    .role-card-title { font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 6px; }
    .role-card-desc  { font-size: .88rem; color: #64748b; line-height: 1.55; }
    .role-card-tag {
        display: inline-block; margin-top: 12px;
        font-size: 11px; font-weight: 600; text-transform: uppercase;
        letter-spacing: .06em; padding: 3px 10px; border-radius: 20px;
    }
    .tag-coord   { background: #eff6ff; color: #2563eb; }
    .tag-hosp    { background: #f0fdf4; color: #16a34a; }

    /* ── How-it-works row ── */
    .hiw-row { display: flex; gap: 0; align-items: flex-start; justify-content: center; margin: 16px 0 8px; }
    .hiw-step { flex: 1; text-align: center; padding: 0 12px; }
    .hiw-num {
        width: 36px; height: 36px; border-radius: 50%;
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: #fff; font-weight: 700; font-size: .95rem;
        display: flex; align-items: center; justify-content: center; margin: 0 auto 8px;
    }
    .hiw-label { font-size: .82rem; font-weight: 600; color: #374151; margin-bottom: 4px; }
    .hiw-desc  { font-size: .78rem; color: #6b7280; line-height: 1.4; }
    .hiw-arrow { padding-top: 18px; color: #cbd5e1; font-size: 1.3rem; }

    /* ── Step wizard ── */
    .step-wizard {
        display: flex; align-items: center; flex-wrap: wrap; gap: 2px;
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 10px 16px; margin-bottom: 22px;
    }
    .sw-item { display: flex; align-items: center; gap: 6px; padding: 4px 10px;
               border-radius: 7px; font-size: 13px; font-weight: 500; }
    .sw-active  { background: #3b82f6; color: #fff; }
    .sw-done    { color: #16a34a; }
    .sw-pending { color: #94a3b8; }
    .sw-arrow   { color: #cbd5e1; font-size: 12px; margin: 0 2px; }

    /* ── Page header ── */
    .page-hdr {
        display: flex; align-items: center; gap: 16px;
        border-bottom: 2px solid #f1f5f9; padding-bottom: 16px; margin-bottom: 20px;
    }
    .page-hdr-title { font-size: 1.6rem; font-weight: 700; color: #0f172a; margin: 0; }
    .page-hdr-sub   { font-size: .85rem; color: #64748b; margin: 2px 0 0; }

    /* ── Next-step callout ── */
    .next-step {
        background: linear-gradient(135deg,#eff6ff,#f0fdf4);
        border: 1.5px solid #bfdbfe; border-radius: 12px;
        padding: 14px 18px; margin-top: 12px; font-size: .9rem;
    }
    .next-step-title { font-weight: 700; color: #1e40af; margin-bottom: 4px; }

    /* ── Hospital cards (coordinator) ── */
    .hosp-card {
        background: #fff; border: 2px solid #e2e8f0; border-radius: 14px;
        padding: 20px; margin-bottom: 12px;
    }
    .hosp-card.registered { border-color: #86efac; background: #f0fdf4; }
    .hosp-card.training   { border-color: #93c5fd; background: #eff6ff; }
    .hosp-card.done       { border-color: #6ee7b7; background: #ecfdf5; }

    /* ── Status pills ── */
    .pill {
        display: inline-block; padding: 3px 11px;
        border-radius: 20px; font-size: 11px; font-weight: 700;
        letter-spacing: .04em; text-transform: uppercase;
    }
    .pill-waiting  { background: #f1f5f9; color: #64748b; }
    .pill-ready    { background: #d1fae5; color: #065f46; }
    .pill-training { background: #dbeafe; color: #1e40af; }
    .pill-done     { background: #a7f3d0; color: #064e3b; }
    .pill-error    { background: #fee2e2; color: #991b1b; }

    /* ── Custom Tab Buttons ── */
    .custom-tab-button {
        padding: 10px 18px;
        border: none;
        border-bottom: 3px solid transparent;
        background: transparent;
        color: #64748b;
        font-size: .95rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
    }
    .custom-tab-button:hover {
        color: #1d4ed8;
        background: #f8fafc;
    }
    .custom-tab-button.active {
        color: #1d4ed8;
        font-weight: 600;
        border-bottom-color: #3b82f6;
    }
    
    /* Override Streamlit button styles for tabs */
    div[data-testid="column"] > div > div > button[kind="secondary"] {
        border: none !important;
        border-bottom: 3px solid transparent !important;
        border-radius: 0 !important;
        background: transparent !important;
        color: #64748b !important;
        font-size: .95rem !important;
        font-weight: 500 !important;
        padding: 10px 18px !important;
    }
    div[data-testid="column"] > div > div > button[kind="secondary"]:hover {
        background: #f8fafc !important;
        color: #1d4ed8 !important;
        border-bottom-color: transparent !important;
    }
    div[data-testid="column"] > div > div > button[kind="primary"] {
        border: none !important;
        border-bottom: 3px solid #3b82f6 !important;
        border-radius: 0 !important;
        background: transparent !important;
        color: #1d4ed8 !important;
        font-size: .95rem !important;
        font-weight: 600 !important;
        padding: 10px 18px !important;
    }
    div[data-testid="column"] > div > div > button[kind="primary"]:hover {
        background: #f8fafc !important;
        color: #1d4ed8 !important;
    }

    /* ── Sidebar polish ── */
    [data-testid="stSidebar"] .block-container { padding-top: 1rem; }

    /* ── Primary buttons → blue (overrides Streamlit's default red/orange) ── */
    button[kind="primary"], .stButton > button[data-testid*="primary"] {
        background: #3b82f6 !important;
        border-color: #2563eb !important;
        color: #fff !important;
    }
    button[kind="primary"]:hover, .stButton > button[data-testid*="primary"]:hover {
        background: #2563eb !important;
        border-color: #1d4ed8 !important;
    }
    /* Streamlit 1.x uses this data-testid */
    [data-testid="stBaseButton-primary"] {
        background: #3b82f6 !important;
        border-color: #2563eb !important;
        color: #fff !important;
    }
    [data-testid="stBaseButton-primary"]:hover {
        background: #2563eb !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ── sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    state     = load_state()
    hospitals = state["hospitals"]
    fed       = state["federation"]
    role      = st.session_state.role

    with st.sidebar:
        # Branding
        st.markdown("### 🏥 FL Simulation")
        st.caption("Multi-Hospital Federated Learning")
        st.divider()

        # Role chip
        if role:
            LABELS = {
                "coordinator": ("🌐", "FL Coordinator"),
                "cleveland":   ("🏥", "Cleveland Hospital"),
                "hungarian":   ("🏥", "Hungarian Hospital"),
            }
            icon, label = LABELS.get(role, ("", role.capitalize()))
            st.markdown(f"**Active role**")
            st.markdown(f"### {icon} {label}")
            if st.button("← Change Role", use_container_width=True, type="primary"):
                st.session_state.role = None
                st.rerun()
        else:
            st.markdown("*No role selected — choose one below.*")

        st.divider()

        # Federation status
        st.markdown("**Federation**")
        registered = [n for n, h in hospitals.items() if h.get("registered")]
        n_reg = len(registered)

        # Hospital pills
        for name in ("cleveland", "hungarian"):
            h = hospitals[name]
            if h.get("registered"):
                st.markdown(f"✅ **{name.capitalize()}** — {h['num_patients']} pts")
            else:
                st.markdown(f"⚪ *{name.capitalize()} — not joined*")

        st.caption(f"Hospitals ready: **{n_reg}/2**")

        # Round progress
        fed_status = fed.get("status", "waiting")
        current    = fed.get("current_round", 0)
        total      = fed.get("total_rounds", 3)

        STATUS_ICON = {
            "waiting":  "⏳", "training": "🔵", "complete": "✅",
        }
        s_key = ("complete" if fed_status == "complete"
                 else "training" if ("round" in fed_status or fed_status == "training")
                 else "waiting")
        emoji = STATUS_ICON.get(s_key, "⏳")

        st.markdown(f"{emoji} **{fed_status.replace('_',' ').title()}**")
        if total > 0 and (current > 0 or fed_status not in ("waiting",)):
            st.progress(int(current / total * 100))
            st.caption(f"Round {current} / {total}")

        st.divider()

        # Quick guide
        with st.expander("❓ Quick Guide"):
            st.markdown(
                "**Single window:**\n"
                "1. Select Cleveland → load data → register\n"
                "2. Select Hungarian → load data → register\n"
                "3. Select Coordinator → Start Training\n\n"
                "**Recommended:** Open 3 tabs at `localhost:8501` and select a different role in each."
            )


# ── landing page ─────────────────────────────────────────────────────────────
def render_landing():
    # Hero
    st.markdown("""
    <div class="fl-hero">
      <div class="fl-hero-badge">🔬 Research Demo</div>
      <div class="fl-hero-title">Federated Learning<br>Multi-Hospital Simulation</div>
      <div class="fl-hero-sub">
        Train a shared heart disease prediction model across two hospitals —
        without any patient data ever leaving its source.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="hiw-row">
      <div class="hiw-step">
        <div class="hiw-num">1</div>
        <div class="hiw-label">Hospitals Join</div>
        <div class="hiw-desc">Each hospital loads its local patient data and registers with the federation.</div>
      </div>
      <div class="hiw-arrow">→</div>
      <div class="hiw-step">
        <div class="hiw-num">2</div>
        <div class="hiw-label">Local Training</div>
        <div class="hiw-desc">The coordinator starts training. Each hospital trains on its own private data.</div>
      </div>
      <div class="hiw-arrow">→</div>
      <div class="hiw-step">
        <div class="hiw-num">3</div>
        <div class="hiw-label">Secure Aggregation</div>
        <div class="hiw-desc">Only model weight updates — never raw data — are shared with the coordinator.</div>
      </div>
      <div class="hiw-arrow">→</div>
      <div class="hiw-step">
        <div class="hiw-num">4</div>
        <div class="hiw-label">Global Model</div>
        <div class="hiw-desc">The coordinator aggregates updates into a stronger global model. Repeat.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Role selection
    st.markdown("#### Choose your role to get started")
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("""
        <div class="role-card">
          <div class="role-card-icon">🌐</div>
          <div class="role-card-title">FL Coordinator</div>
          <div class="role-card-desc">Oversee the federation — monitor which hospitals are connected,
          configure training, launch rounds, and analyse aggregated results.</div>
          <span class="role-card-tag tag-coord">Orchestrator</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Enter as Coordinator", key="sel_coord", type="primary", use_container_width=True):
            st.session_state.role = "coordinator"
            st.rerun()

    with c2:
        st.markdown("""
        <div class="role-card">
          <div class="role-card-icon">🏥</div>
          <div class="role-card-title">Cleveland Hospital</div>
          <div class="role-card-desc">Load the Cleveland UCI heart disease dataset, configure
          local training hyperparameters, and join the federated session.</div>
          <span class="role-card-tag tag-hosp">Participant</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Enter as Cleveland", key="sel_clev", use_container_width=True):
            st.session_state.role = "cleveland"
            st.rerun()

    with c3:
        st.markdown("""
        <div class="role-card">
          <div class="role-card-icon">🏥</div>
          <div class="role-card-title">Hungarian Hospital</div>
          <div class="role-card-desc">Load the Hungarian UCI heart disease dataset, configure
          local training hyperparameters, and join the federated session.</div>
          <span class="role-card-tag tag-hosp">Participant</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Enter as Hungarian", key="sel_hung", use_container_width=True):
            st.session_state.role = "hungarian"
            st.rerun()

    st.divider()

    # Current federation state (if any hospitals registered)
    state     = load_state()
    hospitals = state["hospitals"]
    fed       = state["federation"]
    registered = [n for n, h in hospitals.items() if h.get("registered")]

    if registered or fed.get("active"):
        st.markdown("#### Current Federation State")
        sc1, sc2, sc3, sc4 = st.columns(4)
        clev = hospitals["cleveland"]
        hung = hospitals["hungarian"]
        sc1.metric("Cleveland", f"{clev['num_patients']} pts" if clev["registered"] else "Not joined",
                   delta="✓ Ready" if clev["status"] == "ready" else None)
        sc2.metric("Hungarian", f"{hung['num_patients']} pts" if hung["registered"] else "Not joined",
                   delta="✓ Ready" if hung["status"] == "ready" else None)
        sc3.metric("Hospitals Ready", f"{len(registered)} / 2")
        sc4.metric("Training", fed.get("status", "waiting").replace("_", " ").title())

    else:
        st.info("**Getting started:** Open this app in **3 browser tabs** and select a different role in each for the full multi-hospital experience.", icon="💡")


# ── main ──────────────────────────────────────────────────────────────────────
_inject_css()
render_sidebar()

role = st.session_state.role

if role is None:
    render_landing()

elif role == "coordinator":
    from views.coordinator import render_coordinator
    render_coordinator()

elif role in ("cleveland", "hungarian"):
    from views.hospital import render_hospital
    render_hospital(role)

else:
    st.error(f"Unknown role: {role}")
    st.session_state.role = None
    st.rerun()
