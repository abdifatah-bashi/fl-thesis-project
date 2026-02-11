"""
Shared navigation component.
Uses only native Streamlit calls to avoid HTML overflow/clipping inside columns.
"""

import streamlit as st


def render_page_header(icon: str, title: str, subtitle: str = "") -> None:
    """← Roles back-button + page title. No custom HTML — avoids column clipping."""
    st.write("")          # top breathing room
    bc, tc = st.columns([1, 6])
    with bc:
        st.write("")      # vertically align button with heading
        if st.button("← Roles", help="Return to role selection", use_container_width=True):
            st.session_state.role = None
            st.rerun()
    with tc:
        st.markdown(f"## {icon} {title}")
        if subtitle:
            st.caption(subtitle)
    st.markdown(
        '<hr style="margin:8px 0 0;border:none;border-top:2px solid #f1f5f9">',
        unsafe_allow_html=True,
    )
