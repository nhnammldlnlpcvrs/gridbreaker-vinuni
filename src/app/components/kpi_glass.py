"""Glassmorphism KPI card component."""
from __future__ import annotations

from typing import Optional

import streamlit as st

from utils.data_loader import COLORS


def render_kpi(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_kind: str = "flat",
    caption: str = "",
) -> None:
    """
    Parameters
    ----------
    label      : short label on top (e.g. "REVENUE")
    value      : the main number already formatted (e.g. "14.4 B₫")
    delta      : delta string (e.g. "▼ -46%") or None
    delta_kind : one of {"up", "down", "flat"} — drives colour
    caption    : small subtext under the delta (e.g. "vs 2016 peak")
    """
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta {delta_kind}">{delta}</div>'

    caption_html = ""
    if caption:
        caption_html = (
            f'<div style="font-family:\'Inter\',sans-serif;font-size:11px;'
            f'font-weight:500;color:{COLORS["text_dim"]};margin-top:3px;'
            f'letter-spacing:0.01em">{caption}</div>'
        )

    st.markdown(
        f"""
        <div class="kpi-card" role="figure" aria-label="{label}: {value}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="font-variant-numeric:tabular-nums">{value}</div>
          {delta_html}
          {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
