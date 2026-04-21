"""Shared page header component — badge pill + title + subtitle."""
from __future__ import annotations

import streamlit as st

from utils.data_loader import COLORS


def render_page_header(
    title: str,
    subtitle: str = "",
    badge: str = "ANALYSIS",
    badge_color: str = "primary",
) -> None:
    """Render a consistent page header across all pages."""
    color_map = {
        "primary": COLORS["primary"],
        "warning": COLORS["warning"],
        "danger":  COLORS["danger"],
        "info":    COLORS["info"],
    }
    pill_color = color_map.get(badge_color, COLORS["primary"])

    st.markdown(
        f"""
        <div style="padding: 8px 0 20px 0; border-bottom: 1px solid {COLORS['border']};
                    margin-bottom: 24px">
          <span class="section-pill" style="color:{pill_color};
                border-color:{pill_color}55; background:rgba(0,0,0,0.0)">
            {badge}
          </span>
          <h1 class="page-title" style="margin: 10px 0 8px 0; font-size: 36px;
                font-family: 'Outfit', sans-serif !important; font-weight: 800 !important;
                letter-spacing: -0.03em; line-height: 1.15;">
            {title}
          </h1>
          <p style="font-family: 'Inter', sans-serif; color: {COLORS['text_med']};
                font-size: 15px; line-height: 1.65; margin: 0; max-width: 780px;
                font-weight: 400;">
            {subtitle}
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
