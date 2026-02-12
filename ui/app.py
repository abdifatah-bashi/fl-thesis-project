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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ── Global Reset ── */
    #MainMenu, footer { visibility: hidden; }
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    .block-container { padding-top: 2rem; max-width: 1120px; }
    [data-testid="stAppViewContainer"] { background: #fafbfe; }

    /* ── Animations ── */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,.15); }
        50%      { box-shadow: 0 0 0 8px rgba(99,102,241,0); }
    }
    @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }

    /* ── Hero ── */
    .fl-hero {
        text-align: center; padding: 56px 20px 24px;
        animation: fadeUp .6s ease-out;
    }
    .fl-hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: linear-gradient(135deg, #eef2ff, #f5f3ff);
        color: #6366f1; border: 1px solid #c7d2fe; border-radius: 100px;
        padding: 6px 18px; font-size: 11px; font-weight: 700;
        margin-bottom: 20px; letter-spacing: .06em; text-transform: uppercase;
    }
    .fl-hero-title {
        font-size: 3rem; font-weight: 900; line-height: 1.1; letter-spacing: -.04em;
        margin-bottom: 16px;
        background: linear-gradient(135deg, #0f172a 0%, #4f46e5 50%, #7c3aed 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .fl-hero-sub {
        font-size: 1.1rem; color: #64748b; line-height: 1.7; max-width: 540px;
        margin: 0 auto 40px; font-weight: 400;
    }

    /* ── How-it-works flow ── */
    .hiw-row {
        display: flex; gap: 0; align-items: flex-start; justify-content: center;
        margin: 24px 0 16px; animation: fadeUp .6s ease-out .15s both;
    }
    .hiw-step { flex: 1; text-align: center; padding: 0 16px; }
    .hiw-num {
        width: 44px; height: 44px; border-radius: 14px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: #fff; font-weight: 800; font-size: .95rem;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 10px; box-shadow: 0 4px 12px rgba(99,102,241,.25);
    }
    .hiw-label { font-size: .82rem; font-weight: 700; color: #1e293b; margin-bottom: 4px; }
    .hiw-desc  { font-size: .76rem; color: #94a3b8; line-height: 1.5; }
    .hiw-arrow { padding-top: 18px; color: #c7d2fe; font-size: 1.1rem; font-weight: 300; }

    /* ── Role cards ── */
    .role-card {
        background: #fff; border: 1px solid #e8ecf4; border-radius: 20px;
        padding: 32px 24px; height: 100%; position: relative; overflow: hidden;
        transition: all .3s cubic-bezier(.4,0,.2,1);
        box-shadow: 0 1px 3px rgba(0,0,0,.04);
    }
    .role-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa);
        opacity: 0; transition: opacity .3s;
    }
    .role-card:hover {
        border-color: #c7d2fe;
        box-shadow: 0 20px 40px rgba(99,102,241,.1), 0 4px 12px rgba(0,0,0,.04);
        transform: translateY(-4px);
    }
    .role-card:hover::before { opacity: 1; }
    .role-card-icon {
        width: 52px; height: 52px; border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.6rem; margin-bottom: 16px;
    }
    .role-card-icon.coord { background: linear-gradient(135deg, #eef2ff, #e0e7ff); }
    .role-card-icon.hosp  { background: linear-gradient(135deg, #ecfdf5, #d1fae5); }
    .role-card-title { font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 8px; }
    .role-card-desc  { font-size: .85rem; color: #64748b; line-height: 1.6; margin-bottom: 16px; }
    .role-card-tag {
        display: inline-block; font-size: 10px; font-weight: 700; text-transform: uppercase;
        letter-spacing: .08em; padding: 4px 12px; border-radius: 100px;
    }
    .tag-coord { background: #eef2ff; color: #6366f1; }
    .tag-hosp  { background: #ecfdf5; color: #059669; }

    /* ── Step wizard ── */
    .step-wizard {
        display: flex; align-items: center; flex-wrap: wrap; gap: 4px;
        background: #fff; border: 1px solid #e8ecf4;
        border-radius: 14px; padding: 12px 18px; margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,.03);
    }
    .sw-item {
        display: flex; align-items: center; gap: 6px; padding: 5px 14px;
        border-radius: 8px; font-size: 13px; font-weight: 600;
        transition: all .2s;
    }
    .sw-active {
        background: linear-gradient(135deg, #6366f1, #7c3aed);
        color: #fff; box-shadow: 0 2px 8px rgba(99,102,241,.3);
    }
    .sw-done    { color: #059669; }
    .sw-pending { color: #c0c5d0; }
    .sw-arrow   { color: #dde0e9; font-size: 12px; margin: 0 2px; }

    /* ── Page header ── */
    .page-hdr {
        display: flex; align-items: center; gap: 16px;
        border-bottom: 1px solid #edf0f7; padding-bottom: 16px; margin-bottom: 24px;
    }
    .page-hdr-title { font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0; }
    .page-hdr-sub   { font-size: .85rem; color: #94a3b8; margin: 2px 0 0; font-weight: 400; }

    /* ── Next-step callout ── */
    .next-step {
        background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
        border: 1px solid #c7d2fe; border-radius: 16px;
        padding: 16px 20px; margin-top: 14px; font-size: .9rem;
    }
    .next-step-title { font-weight: 700; color: #4f46e5; margin-bottom: 4px; }

    /* ── Hospital cards (coordinator) ── */
    .hosp-card {
        background: #fff; border: 1px solid #e8ecf4; border-radius: 16px;
        padding: 22px; margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,.03);
        transition: all .2s ease;
    }
    .hosp-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,.06); }
    .hosp-card.registered {
        border-color: #a7f3d0;
        background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    }
    .hosp-card.training {
        border-color: #c7d2fe;
        background: linear-gradient(135deg, #eef2ff, #f5f3ff);
    }
    .hosp-card.done {
        border-color: #6ee7b7;
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    }

    /* ── Status pills ── */
    .pill {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 4px 14px; border-radius: 100px;
        font-size: 11px; font-weight: 700;
        letter-spacing: .04em; text-transform: uppercase;
    }
    .pill-waiting  { background: #f1f5f9; color: #94a3b8; }
    .pill-ready    { background: #d1fae5; color: #059669; }
    .pill-training { background: #e0e7ff; color: #4f46e5; }
    .pill-done     { background: #a7f3d0; color: #047857; }
    .pill-error    { background: #fee2e2; color: #dc2626; }

    /* ── Custom tab bar ── */
    .tab-bar {
        display: flex; gap: 2px; background: #f1f3f9; border-radius: 12px;
        padding: 4px; margin-bottom: 24px;
    }
    .tab-bar-item {
        flex: 1; text-align: center; padding: 10px 16px; border-radius: 10px;
        font-size: .88rem; font-weight: 600; color: #64748b;
        cursor: pointer; transition: all .2s;
    }
    .tab-bar-item:hover { color: #4f46e5; background: rgba(255,255,255,.5); }
    .tab-bar-item.active {
        background: #fff; color: #4f46e5;
        box-shadow: 0 1px 3px rgba(0,0,0,.08);
    }

    /* Override Streamlit button styles for tab buttons */
    div[data-testid="column"] > div > div > button[kind="secondary"] {
        border: none !important; border-radius: 10px !important;
        background: transparent !important; color: #64748b !important;
        font-size: .88rem !important; font-weight: 600 !important;
        padding: 10px 16px !important; transition: all .2s !important;
    }
    div[data-testid="column"] > div > div > button[kind="secondary"]:hover {
        background: rgba(99,102,241,.06) !important; color: #4f46e5 !important;
    }
    div[data-testid="column"] > div > div > button[kind="primary"] {
        border: none !important; border-radius: 10px !important;
        background: #fff !important; color: #4f46e5 !important;
        font-size: .88rem !important; font-weight: 700 !important;
        padding: 10px 16px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,.08) !important;
    }
    div[data-testid="column"] > div > div > button[kind="primary"]:hover {
        background: #fff !important; color: #4f46e5 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fc 0%, #f1f3f9 100%) !important;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1rem; }

    /* ── Primary buttons → indigo gradient ── */
    button[kind="primary"], .stButton > button[data-testid*="primary"],
    [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
        border: none !important; color: #fff !important;
        border-radius: 12px !important; font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(99,102,241,.25) !important;
        transition: all .25s cubic-bezier(.4,0,.2,1) !important;
        letter-spacing: .01em !important;
    }
    button[kind="primary"]:hover, .stButton > button[data-testid*="primary"]:hover,
    [data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, #4f46e5, #6d28d9) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,.35) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Secondary buttons ── */
    [data-testid="stBaseButton-secondary"] {
        border: 1px solid #e2e5ef !important; border-radius: 12px !important;
        background: #fff !important; color: #475569 !important;
        font-weight: 600 !important;
        transition: all .2s !important;
    }
    [data-testid="stBaseButton-secondary"]:hover {
        border-color: #c7d2fe !important; color: #4f46e5 !important;
        background: #fafbff !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #e8ecf4; border-radius: 14px;
        padding: 16px 18px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,.03);
        transition: all .2s;
    }
    [data-testid="stMetric"]:hover { box-shadow: 0 4px 12px rgba(0,0,0,.06); }
    [data-testid="stMetricLabel"] {
        font-size: .75rem !important; font-weight: 700 !important;
        text-transform: uppercase !important; letter-spacing: .06em !important;
        color: #94a3b8 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important; font-weight: 800 !important;
        color: #1e293b !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px; background: #f1f3f9; border-radius: 12px; padding: 4px;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important; padding: 10px 20px !important;
        font-weight: 600 !important; font-size: .88rem !important;
        color: #64748b !important; border-bottom: none !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: #fff !important; color: #4f46e5 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,.08) !important;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ── Info/Warning/Success ── */
    [data-testid="stAlert"] {
        border-radius: 14px !important; border-left: 4px solid !important;
        font-size: .9rem !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        border: 1px solid #e8ecf4 !important; border-radius: 14px !important;
        background: #fff !important;
    }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #e8ecf4 !important; border-radius: 14px !important;
        overflow: hidden;
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        border-radius: 100px !important; height: 8px !important;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        border-radius: 100px !important;
    }

    /* ── Dividers ── */
    hr { border-color: #edf0f7 !important; opacity: .6; }

    /* ── Container borders ── */
    [data-testid="stVerticalBlock"] > div[data-testid="element-container"] > div > div[style*="border"] {
        border-radius: 16px !important;
        border-color: #e8ecf4 !important;
    }

    /* ── Download button ── */
    [data-testid="stDownloadButton"] button {
        border: 1px solid #e2e5ef !important; border-radius: 12px !important;
        background: #fff !important; color: #475569 !important;
        font-weight: 600 !important; transition: all .2s !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        border-color: #c7d2fe !important; color: #4f46e5 !important;
        background: #fafbff !important;
    }

    /* ── Section label ── */
    .sec-label {
        font-size: .7rem; font-weight: 700; letter-spacing: .1em;
        text-transform: uppercase; color: #b0b8c9; margin: 20px 0 10px;
    }

    /* ── Glass card ── */
    .glass-card {
        background: rgba(255,255,255,.85); backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,.5); border-radius: 20px;
        padding: 28px; box-shadow: 0 4px 24px rgba(0,0,0,.04);
        transition: all .3s cubic-bezier(.4,0,.2,1);
    }
    .glass-card:hover {
        box-shadow: 0 8px 32px rgba(99,102,241,.08);
        transform: translateY(-2px);
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center; padding: 56px 24px;
        animation: fadeUp .5s ease-out;
    }
    .empty-state-icon {
        width: 72px; height: 72px; border-radius: 20px;
        background: linear-gradient(135deg, #eef2ff, #f5f3ff);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.8rem; margin: 0 auto 18px;
    }
    .empty-state-title {
        font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 6px;
    }
    .empty-state-desc {
        font-size: .9rem; color: #94a3b8; max-width: 380px; margin: 0 auto; line-height: 1.6;
    }

    /* ── Sidebar role chip ── */
    .role-chip {
        display: flex; align-items: center; gap: 10px;
        background: #fff; border: 1px solid #e8ecf4; border-radius: 12px;
        padding: 10px 14px; margin: 4px 0 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,.03);
    }
    .role-chip-icon {
        width: 36px; height: 36px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
    }
    .role-chip-icon.coord { background: linear-gradient(135deg, #eef2ff, #e0e7ff); }
    .role-chip-icon.hosp  { background: linear-gradient(135deg, #ecfdf5, #d1fae5); }
    .role-chip-name { font-size: .88rem; font-weight: 700; color: #1e293b; }

    /* ── Sidebar federation ── */
    .fed-hosp-row {
        display: flex; align-items: center; gap: 8px;
        padding: 6px 0; font-size: .85rem;
    }
    .fed-dot {
        width: 8px; height: 8px; border-radius: 50%;
    }
    .fed-dot.active { background: #10b981; box-shadow: 0 0 6px rgba(16,185,129,.4); }
    .fed-dot.inactive { background: #d1d5db; }

    /* ── Status banner ── */
    .status-banner {
        border-radius: 14px; padding: 14px 18px;
        font-weight: 600; font-size: .9rem;
        display: flex; align-items: center; gap: 10px;
    }
    .status-running {
        background: linear-gradient(135deg, #eef2ff, #f5f3ff);
        border: 1px solid #c7d2fe; color: #4f46e5;
    }
    .status-waiting {
        background: linear-gradient(135deg, #fff7ed, #fffbeb);
        border: 1px solid #fed7aa; color: #c2410c;
    }
    .status-complete {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border: 1px solid #86efac; color: #047857;
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
        st.markdown(
            '<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">'
            '<div style="width:36px;height:36px;border-radius:10px;'
            'background:linear-gradient(135deg,#6366f1,#8b5cf6);'
            'display:flex;align-items:center;justify-content:center;font-size:1rem;color:#fff">FL</div>'
            '<div><div style="font-size:.95rem;font-weight:800;color:#0f172a;letter-spacing:-.02em">'
            'FL Simulation</div>'
            '<div style="font-size:.7rem;color:#94a3b8;font-weight:500">Multi-Hospital</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr style="margin:12px 0;border:none;border-top:1px solid #e8ecf4">', unsafe_allow_html=True)

        # Role chip
        if role:
            LABELS = {
                "coordinator": ("🌐", "FL Coordinator", "coord"),
                "cleveland":   ("🏥", "Cleveland Hospital", "hosp"),
                "hungarian":   ("🏥", "Hungarian Hospital", "hosp"),
            }
            icon, label, chip_cls = LABELS.get(role, ("", role.capitalize(), "hosp"))
            st.markdown(
                '<p style="font-size:.68rem;font-weight:700;letter-spacing:.08em;'
                'text-transform:uppercase;color:#b0b8c9;margin:0 0 6px">Active Role</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="role-chip">'
                f'<div class="role-chip-icon {chip_cls}">{icon}</div>'
                f'<div class="role-chip-name">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Switch Role", use_container_width=True):
                st.session_state.role = None
                st.rerun()
        else:
            st.markdown(
                '<p style="font-size:.85rem;color:#94a3b8;font-style:italic;margin:4px 0 8px">'
                'No role selected</p>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr style="margin:12px 0;border:none;border-top:1px solid #e8ecf4">', unsafe_allow_html=True)

        # Federation status
        st.markdown(
            '<p style="font-size:.68rem;font-weight:700;letter-spacing:.08em;'
            'text-transform:uppercase;color:#b0b8c9;margin:0 0 10px">Federation</p>',
            unsafe_allow_html=True,
        )
        registered = [n for n, h in hospitals.items() if h.get("registered")]
        n_reg = len(registered)

        for name in ("cleveland", "hungarian"):
            h = hospitals[name]
            is_active = h.get("registered")
            dot_cls = "active" if is_active else "inactive"
            pts_str = f" · {h['num_patients']} pts" if is_active else ""
            st.markdown(
                f'<div class="fed-hosp-row">'
                f'<div class="fed-dot {dot_cls}"></div>'
                f'<span style="font-weight:600;color:{"#1e293b" if is_active else "#94a3b8"}">'
                f'{name.capitalize()}{pts_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.caption(f"**{n_reg}** of 2 connected")

        # Round progress
        fed_status = fed.get("status", "waiting")
        current    = fed.get("current_round", 0)
        total      = fed.get("total_rounds", 3)

        s_key = ("complete" if fed_status == "complete"
                 else "training" if ("round" in fed_status or fed_status == "training")
                 else "waiting")
        status_colors = {"complete": "#059669", "training": "#6366f1", "waiting": "#94a3b8"}
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:6px;margin:10px 0 4px">'
            f'<div style="width:6px;height:6px;border-radius:50%;background:{status_colors.get(s_key,"#94a3b8")}"></div>'
            f'<span style="font-size:.85rem;font-weight:600;color:#475569">'
            f'{fed_status.replace("_"," ").title()}</span></div>',
            unsafe_allow_html=True,
        )
        if total > 0 and (current > 0 or fed_status not in ("waiting",)):
            st.progress(int(current / total * 100))
            st.caption(f"Round {current} / {total}")

        st.markdown('<hr style="margin:12px 0;border:none;border-top:1px solid #e8ecf4">', unsafe_allow_html=True)

        with st.expander("Quick Guide"):
            st.markdown(
                "1. Select **Cleveland** → load data → register\n"
                "2. Select **Hungarian** → load data → register\n"
                "3. Select **Coordinator** → start training\n\n"
                "Open 3 tabs at `localhost:8501` for the best experience."
            )


# ── landing page ─────────────────────────────────────────────────────────────
def render_landing():
    # Hero
    st.markdown("""
    <div class="fl-hero">
      <div class="fl-hero-badge">Research Demo</div>
      <div class="fl-hero-title">Federated Learning<br>Hospital Simulation</div>
      <div class="fl-hero-sub">
        Train a shared heart-disease prediction model across two hospitals —
        without patient data ever leaving its source.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="hiw-row">
      <div class="hiw-step">
        <div class="hiw-num">1</div>
        <div class="hiw-label">Hospitals Join</div>
        <div class="hiw-desc">Each hospital loads local patient data and registers with the federation.</div>
      </div>
      <div class="hiw-arrow">&rarr;</div>
      <div class="hiw-step">
        <div class="hiw-num">2</div>
        <div class="hiw-label">Local Training</div>
        <div class="hiw-desc">Each hospital trains privately on its own data.</div>
      </div>
      <div class="hiw-arrow">&rarr;</div>
      <div class="hiw-step">
        <div class="hiw-num">3</div>
        <div class="hiw-label">Aggregation</div>
        <div class="hiw-desc">Only weight updates — never raw data — reach the coordinator.</div>
      </div>
      <div class="hiw-arrow">&rarr;</div>
      <div class="hiw-step">
        <div class="hiw-num">4</div>
        <div class="hiw-label">Global Model</div>
        <div class="hiw-desc">Updates are merged into a stronger global model. Repeat.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")

    # Role selection
    st.markdown(
        '<p class="sec-label" style="text-align:center;margin-bottom:18px">Choose your role</p>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("""
        <div class="role-card">
          <div class="role-card-icon coord">🌐</div>
          <div class="role-card-title">FL Coordinator</div>
          <div class="role-card-desc">Oversee the federation — monitor connections,
          configure training, launch rounds, and view aggregated results.</div>
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
          <div class="role-card-icon hosp">🏥</div>
          <div class="role-card-title">Cleveland Hospital</div>
          <div class="role-card-desc">Load the Cleveland UCI heart-disease dataset,
          configure local hyperparameters, and join the session.</div>
          <span class="role-card-tag tag-hosp">Participant</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Enter as Cleveland", key="sel_clev", type="primary", use_container_width=True):
            st.session_state.role = "cleveland"
            st.rerun()

    with c3:
        st.markdown("""
        <div class="role-card">
          <div class="role-card-icon hosp">🏥</div>
          <div class="role-card-title">Hungarian Hospital</div>
          <div class="role-card-desc">Load the Hungarian UCI heart-disease dataset,
          configure local hyperparameters, and join the session.</div>
          <span class="role-card-tag tag-hosp">Participant</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Enter as Hungarian", key="sel_hung", type="primary", use_container_width=True):
            st.session_state.role = "hungarian"
            st.rerun()

    st.markdown("")

    # Current federation state (if any hospitals registered)
    state     = load_state()
    hospitals = state["hospitals"]
    fed       = state["federation"]
    registered = [n for n, h in hospitals.items() if h.get("registered")]

    if registered or fed.get("active"):
        st.markdown(
            '<p class="sec-label" style="text-align:center;margin:24px 0 14px">Current Federation</p>',
            unsafe_allow_html=True,
        )
        sc1, sc2, sc3, sc4 = st.columns(4)
        clev = hospitals["cleveland"]
        hung = hospitals["hungarian"]
        sc1.metric("Cleveland", f"{clev['num_patients']} pts" if clev["registered"] else "Not joined",
                   delta="Ready" if clev["status"] == "ready" else None)
        sc2.metric("Hungarian", f"{hung['num_patients']} pts" if hung["registered"] else "Not joined",
                   delta="Ready" if hung["status"] == "ready" else None)
        sc3.metric("Connected", f"{len(registered)} / 2")
        sc4.metric("Training", fed.get("status", "waiting").replace("_", " ").title())

    else:
        st.markdown(
            '<div style="text-align:center;margin-top:12px">'
            '<p style="font-size:.88rem;color:#94a3b8">'
            'Open this app in <b>3 browser tabs</b> and select a different role in each '
            'for the full experience.</p></div>',
            unsafe_allow_html=True,
        )


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
