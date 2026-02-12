"""
Shared navigation component.
Uses only native Streamlit calls to avoid HTML overflow/clipping inside columns.
"""

import streamlit as st


def render_page_header(icon: str, title: str, subtitle: str = "") -> None:
    """Back-button + page title. Clean, minimal header."""
    st.write("")
    bc, tc = st.columns([1, 6])
    with bc:
        st.write("")
        if st.button("Switch Role", help="Return to role selection", use_container_width=True):
            st.session_state.role = None
            st.rerun()
    with tc:
        st.markdown(f"## {icon} {title}")
        if subtitle:
            st.caption(subtitle)
    st.markdown(
        '<hr style="margin:8px 0 4px;border:none;border-top:1px solid #e8ecf4">',
        unsafe_allow_html=True,
    )
