"""Insight / anomaly callout component."""
from __future__ import annotations

import streamlit as st


def render_insight(body: str, level: str = "info", title: str | None = None) -> None:
    """
    Parameters
    ----------
    body  : HTML-allowed body text
    level : one of {"info", "warning", "danger"}
    title : optional bold title shown first
    """
    label_map = {"info": "INSIGHT", "warning": "ATTENTION", "danger": "CRITICAL"}
    label = label_map.get(level, "INSIGHT")
    title_html = f"<strong>{title}</strong> " if title else ""

    st.markdown(
        f"""
        <div class="insight-box {level}">
          <span class="insight-title"><span class="insight-icon"></span>{label}</span>
          {title_html}{body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_label(label: str) -> None:
    """Render a small uppercase pill label — used above charts."""
    st.markdown(
        f'<span class="section-pill">{label}</span>',
        unsafe_allow_html=True,
    )
