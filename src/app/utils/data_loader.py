"""
Centralised data loaders, colour palette and constants for the Gridbreaker
dashboard. All pages should import from this module so styling stays in sync.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve the project root (three levels up from this file: utils → app → src).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"
DATASET = PROJECT_ROOT / "dataset"

TRAIN_CUTOFF = pd.Timestamp("2022-12-31")


# ---------------------------------------------------------------------------
# Colour palette — dark cinematic theme with green brand accents
# ---------------------------------------------------------------------------
COLORS = {
    # Backgrounds
    "bg_deep":    "#0A1F1A",
    "bg_card":    "#132A22",
    "bg_glass":   "rgba(82,183,136,0.08)",
    "border":     "rgba(82,183,136,0.22)",
    # Brand
    "primary":    "#52B788",
    "primary_dim": "#2D6A4F",
    "glow":       "#95D5B2",
    # Status
    "danger":     "#F25F5C",
    "warning":    "#FFA94D",
    "info":       "#74C0FC",
    "success":    "#52B788",
    # Text
    "text_hi":    "#F0F4EF",
    "text_med":   "#B7C9B7",
    "text_dim":   "#6C8476",
    # Categories (fashion)
    "cat_streetwear": "#52B788",
    "cat_casual":     "#74C0FC",
    "cat_outdoor":    "#FFA94D",
    "cat_genz":       "#E599F7",
}

CATEGORY_COLORS = {
    "streetwear": COLORS["cat_streetwear"],
    "casual":     COLORS["cat_casual"],
    "outdoor":    COLORS["cat_outdoor"],
    "genz":       COLORS["cat_genz"],
}

CHANNEL_COLORS = {
    "paid_search":    "#52B788",
    "organic_search": "#95D5B2",
    "social_media":   "#E599F7",
    "email_campaign": "#74C0FC",
    "referral":       "#FFA94D",
    "direct":         "#F0F4EF",
}


# ---------------------------------------------------------------------------
# Plotly shared layout
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    font=dict(family="Plus Jakarta Sans, system-ui, sans-serif", size=13, color=COLORS["text_hi"]),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=[
        COLORS["primary"], COLORS["info"], COLORS["warning"],
        COLORS["cat_genz"], COLORS["glow"], COLORS["danger"],
    ],
    hoverlabel=dict(
        bgcolor=COLORS["bg_card"],
        font=dict(family="Plus Jakarta Sans", size=13, color=COLORS["text_hi"]),
        bordercolor=COLORS["primary"],
    ),
    margin=dict(l=50, r=30, t=60, b=50),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.08)"),
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading daily ABT…")
def load_abt_daily() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "abt_daily.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    # Restrict to training window (leakage-safe)
    return df[df["date"] <= TRAIN_CUTOFF].copy()


@st.cache_data(show_spinner="Loading orders enriched…")
def load_orders_enriched() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "abt_orders_enriched.parquet")
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"])
    return df


@st.cache_data(show_spinner="Loading customers…")
def load_customers() -> pd.DataFrame:
    df = pd.read_parquet(INTERIM / "customers.parquet")
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    return df


@st.cache_data(show_spinner="Loading orders…")
def load_orders() -> pd.DataFrame:
    df = pd.read_parquet(INTERIM / "orders.parquet")
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df[df["order_date"] <= TRAIN_CUTOFF].copy()


@st.cache_data(show_spinner="Loading products…")
def load_products() -> pd.DataFrame:
    return pd.read_parquet(INTERIM / "products.parquet")


@st.cache_data(show_spinner="Loading order items…")
def load_order_items() -> pd.DataFrame:
    return pd.read_parquet(INTERIM / "order_items.parquet")


@st.cache_data(show_spinner="Loading returns…")
def load_returns() -> pd.DataFrame:
    df = pd.read_parquet(INTERIM / "returns.parquet")
    df["return_date"] = pd.to_datetime(df["return_date"])
    return df


@st.cache_data(show_spinner="Loading inventory…")
def load_inventory() -> pd.DataFrame:
    """Load inventory from raw CSV (no interim parquet built yet)."""
    df = pd.read_csv(DATASET / "Operational" / "inventory.csv")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


@st.cache_data(show_spinner="Loading cohort ABT…")
def load_cohort() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "abt_customer_cohort.parquet")


# ---------------------------------------------------------------------------
# Global CSS injection
# ---------------------------------------------------------------------------
def inject_global_css() -> None:
    """Inject dark cinematic CSS used across every page."""
    css = f"""
    <style>
      /* ——— Google Fonts: Plus Jakarta Sans (body) + Fraunces (display) ——— */

      /* ——— Design tokens ——— */
      :root {{
        --space-1: 4px;  --space-2: 8px;  --space-3: 12px;
        --space-4: 16px; --space-6: 24px; --space-8: 32px;
        --radius-sm: 8px; --radius-md: 14px; --radius-lg: 20px;
        --transition-fast: 150ms ease-out;
        --transition-base: 220ms ease-out;
      }}

      /* ——— Base typography ——— */
      html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
        font-size: 15px;
        line-height: 1.6;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
      }}
      h1, h2, h3, .page-title {{
        font-family: 'Fraunces', Georgia, serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        line-height: 1.15 !important;
        color: {COLORS['text_hi']} !important;
        font-style: italic !important;
      }}
      p, li, td, th {{
        line-height: 1.65;
      }}

      /* ——— Section pill above charts ——— */
      .section-pill {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 10.5px;
        font-weight: 600;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: {COLORS['primary']};
        background: rgba(82,183,136,0.09);
        border: 1px solid rgba(82,183,136,0.28);
        border-radius: 20px;
        padding: 4px 12px;
        margin-bottom: 8px;
      }}
      .section-pill::before {{
        content: '';
        display: inline-block;
        width: 5px;
        height: 5px;
        background: {COLORS['primary']};
        border-radius: 50%;
        opacity: 0.8;
      }}

      /* ——— Glass KPI cards ——— */
      .kpi-card {{
        background: linear-gradient(140deg,
          rgba(82,183,136,0.09) 0%,
          rgba(19,42,34,0.60) 100%);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(82,183,136,0.20);
        border-radius: var(--radius-md);
        padding: var(--space-4) 18px;
        min-height: 110px;
        box-shadow:
          0 1px 0 rgba(255,255,255,0.04) inset,
          0 4px 24px rgba(0,0,0,0.20);
        transition: transform var(--transition-fast),
                    box-shadow var(--transition-fast),
                    border-color var(--transition-fast);
      }}
      .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow:
          0 1px 0 rgba(255,255,255,0.06) inset,
          0 8px 32px rgba(0,0,0,0.28);
        border-color: rgba(82,183,136,0.36);
      }}
      .kpi-label {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 10.5px;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: {COLORS['text_dim']};
        margin-bottom: var(--space-2);
      }}
      .kpi-value {{
        font-family: 'Fraunces', Georgia, serif;
        font-size: 28px;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.02em;
        color: {COLORS['text_hi']};
        line-height: 1.1;
      }}
      .kpi-delta {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 12px;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        margin-top: var(--space-1);
      }}
      .kpi-delta.up   {{ color: {COLORS['success']}; }}
      .kpi-delta.down {{ color: {COLORS['danger']};  }}
      .kpi-delta.flat {{ color: {COLORS['text_dim']}; }}

      /* ——— Insight callout ——— */
      .insight-box {{
        background: rgba(82,183,136,0.06);
        border-left: 3px solid {COLORS['primary']};
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        padding: 14px 18px;
        margin: 14px 0;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 14px;
        line-height: 1.65;
        color: {COLORS['text_med']};
      }}
      .insight-box .insight-icon {{
        display: inline-block;
        width: 18px; height: 18px;
        border-radius: 50%;
        background: {COLORS['primary']};
        opacity: 0.85;
        margin-right: 6px;
        vertical-align: middle;
      }}
      .insight-box.warning {{
        border-left-color: {COLORS['warning']};
        background: rgba(255,169,77,0.05);
      }}
      .insight-box.warning .insight-icon {{ background: {COLORS['warning']}; }}
      .insight-box.danger  {{
        border-left-color: {COLORS['danger']};
        background: rgba(242,95,92,0.05);
      }}
      .insight-box.danger  .insight-icon {{ background: {COLORS['danger']}; }}
      .insight-box strong  {{ color: {COLORS['text_hi']}; font-weight: 600; }}
      .insight-title {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        display: block;
        margin-bottom: 6px;
      }}

      /* ——— Sidebar polish ——— */
      [data-testid="stSidebar"] {{
        background: {COLORS['bg_deep']};
        border-right: 1px solid rgba(82,183,136,0.14);
      }}

      /* ——— Sidebar nav typography ——— */
      [data-testid="stSidebarNav"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
      }}
      [data-testid="stSidebarNav"] [data-testid="stPageLink-NavLink"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
      }}
      [data-testid="stSidebarNav"] [data-testid="stPageLink-NavLink"] > span:not([data-testid="stIconMaterial"]) {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 14px;
        font-weight: 500;
      }}
      /* Streamlit 1.56 renders :material/...: icons via stIconMaterial spans */
      [data-testid="stSidebarNav"] [data-testid="stIconMaterial"],
      [data-testid="stSidebar"] [data-testid="stIconMaterial"] {{
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal;
        font-style: normal;
        font-size: 20px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
      }}

      /* ——— Streamlit metric overrides ——— */
      [data-testid="stMetricValue"] {{
        font-family: 'Fraunces', Georgia, serif !important;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.02em;
      }}
      [data-testid="stMetricDelta"] {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-variant-numeric: tabular-nums;
      }}

      /* ——— Dataframe / table ——— */
      [data-testid="stDataFrame"] {{
        border-radius: var(--radius-sm) !important;
        overflow: hidden;
      }}

      /* ——— Slider ——— */
      [data-testid="stSlider"] label {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 13px;
        font-weight: 500;
      }}

      /* Hide default Streamlit chrome */
      #MainMenu {{ visibility: hidden; }}
      footer    {{ visibility: hidden; }}
    </style>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,400;1,9..144,700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block" rel="stylesheet">
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_vnd(x: float, precision: int = 1) -> str:
    """Format VND in human-readable form (tỷ / triệu)."""
    if pd.isna(x):
        return "—"
    if abs(x) >= 1e9:
        return f"{x/1e9:.{precision}f} B VND"
    if abs(x) >= 1e6:
        return f"{x/1e6:.{precision}f} M VND"
    if abs(x) >= 1e3:
        return f"{x/1e3:.0f} K VND"
    return f"{x:.0f} VND"


def fmt_num(x: float, precision: int = 0) -> str:
    if pd.isna(x):
        return "—"
    if abs(x) >= 1e6:
        return f"{x/1e6:.{precision}f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.{precision}f}K"
    return f"{x:.0f}"


def fmt_pct(x: float, precision: int = 1) -> str:
    if pd.isna(x):
        return "—"
    return f"{x*100:.{precision}f}%"
