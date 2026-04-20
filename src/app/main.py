"""
Gridbreaker — Fashion E-Commerce Diagnosis Dashboard
====================================================
Entry point for the Streamlit multi-page app.

Run:
    streamlit run src/app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sibling modules importable when running via `streamlit run`
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from utils.data_loader import COLORS, inject_global_css


st.set_page_config(
    page_title="Gridbreaker · Fashion E-Commerce Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()


# ---------------------------------------------------------------------------
# Sidebar branding
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 16px 8px 20px 8px; border-bottom: 1px solid {COLORS['border']}; margin-bottom: 16px">
          <div style="font-size: 11px; letter-spacing: 0.2em; color: {COLORS['primary']}; font-weight: 700">
            DATATHON 2026
          </div>
          <h2 style="font-family: 'Outfit', sans-serif; color: {COLORS['text_hi']};
                     margin: 10px 0 6px 0; font-size: 26px; font-weight: 800; line-height: 1.2;">
            🌿 Gridbreaker
          </h2>
          <div style="font-size: 13px; color: {COLORS['text_med']}">
            Fashion E-Commerce Diagnosis
          </div>
          <div style="font-size: 11px; color: {COLORS['text_dim']}; margin-top: 6px">
            Breaking Business Boundaries
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
pages = {
    "DASHBOARD": [
        st.Page("app_pages/00_overview.py",          title="Tổng quan",          icon=":material/dashboard:"),
    ],
    "DIAGNOSIS": [
        st.Page("app_pages/01_revenue_collapse.py",  title="Revenue Collapse",   icon=":material/trending_down:"),
        st.Page("app_pages/02_funnel_customer.py",   title="Funnel & Customer",  icon=":material/filter_alt:"),
        st.Page("app_pages/03_inventory.py",         title="Inventory Paradox",  icon=":material/inventory_2:"),
        st.Page("app_pages/05_patterns.py",          title="Patterns & Timing",  icon=":material/calendar_month:"),
        st.Page("app_pages/06_promo.py",             title="Promo ROI",          icon=":material/local_offer:"),
        st.Page("app_pages/07_geo.py",               title="Geographic",         icon=":material/location_on:"),
    ],
    "STRATEGY": [
        st.Page("app_pages/04_prescriptive.py",      title="Recovery Simulator", icon=":material/auto_fix_high:"),
    ],
}

page = st.navigation(pages, position="sidebar")
page.run()


# ---------------------------------------------------------------------------
# Sidebar footer
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<br>" * 4, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="padding: 12px 8px; border-top: 1px solid {COLORS['border']};
                    font-size: 11px; color: {COLORS['text_dim']}">
          <div style="color: {COLORS['text_med']}; font-weight: 600">The Gridbreakers</div>
          <div style="margin-top: 2px">VinTelligence · VinUni DS&AI Club</div>
          <div style="margin-top: 6px; font-size: 10px">Built with Streamlit · Plotly</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
