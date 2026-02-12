"""
Shared navigation component.
Page title only — role switching is handled by the sidebar.
"""

import streamlit as st


def render_page_header(icon: str, title: str, subtitle: str = "") -> None:
    """Page title with optional subtitle. No back button — sidebar handles nav."""
    st.write("")
    st.markdown(f"## {icon} {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown(
        '<hr style="margin:8px 0 4px;border:none;border-top:1px solid #e8ecf4">',
        unsafe_allow_html=True,
    )
