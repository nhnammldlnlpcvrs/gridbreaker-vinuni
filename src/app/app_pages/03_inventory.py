"""Page 3 — Inventory Paradox. Asymmetric bento, anomaly spotlight."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.insight_box import render_insight, render_section_label
from components.page_header import render_page_header
from utils.chart_helpers import apply_theme
from utils.data_loader import (
    CATEGORY_COLORS, COLORS,
    fmt_num, fmt_pct, fmt_vnd,
    load_inventory, load_products,
)


inv = load_inventory()
products = load_products()
inv = inv.merge(products[["product_id", "category"]], on="product_id", how="left",
                suffixes=("", "_prod"))
if "category" not in inv.columns or inv["category"].isna().all():
    inv["category"] = inv.get("category_prod", "unknown")

# Align column names (inventory has its own category/segment columns)
cat_col = "category" if "category" in inv.columns else "category_prod"

inv["year"] = pd.to_datetime(inv["snapshot_date"]).dt.year
inv["paradox"] = (inv["stockout_flag"] == 1) & (inv["overstock_flag"] == 1)


render_page_header(
    title="The Inventory Paradox",
    subtitle=(
        f"In {inv['paradox'].mean()*100:.1f}% of product-months, an SKU is flagged "
        "stockout = 1 AND overstock = 1 at the same time. Either the supply chain "
        "is mis-timed or the simulation contains a structural trap — both interpretations "
        "point to the same prescriptive action."
    ),
    badge="ANOMALY · P0",
    badge_color="danger",
)


# ---------------------------------------------------------------------------
# HERO · Paradox scatter (stock_on_hand × stockout_days, colored by category)
# ---------------------------------------------------------------------------
render_section_label("HERO · PARADOX SCATTER")

# Aggregate at product-month level for readability
plot_df = inv.copy()
# Sample if too many points
if len(plot_df) > 8000:
    plot_df = plot_df.sample(8000, random_state=42)

fig = go.Figure()
for cat in plot_df[cat_col].dropna().unique():
    sub = plot_df[plot_df[cat_col] == cat]
    fig.add_trace(go.Scatter(
        x=sub["stock_on_hand"], y=sub["stockout_days"],
        mode="markers",
        name=cat,
        marker=dict(
            size=6, opacity=0.55,
            color=CATEGORY_COLORS.get(cat, COLORS["primary"]),
            line=dict(width=0),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>Stock on hand: %{x}<br>"
            "Stockout days: %{y}<extra></extra>"
        ),
        text=sub.get("product_name", sub["product_id"].astype(str)),
    ))

apply_theme(fig, height=440,
            title="Stock on hand vs Stockout days — the paradox lives in the top-right")
fig.update_xaxes(title="Stock on hand (end of month)")
fig.update_yaxes(title="Stockout days")

# Quadrant guides
med_stock = plot_df["stock_on_hand"].median()
med_days = plot_df["stockout_days"].median()
fig.add_vline(x=med_stock, line_dash="dot", line_color=COLORS["text_dim"])
fig.add_hline(y=med_days, line_dash="dot", line_color=COLORS["text_dim"])

fig.add_annotation(
    x=plot_df["stock_on_hand"].quantile(0.85),
    y=plot_df["stockout_days"].quantile(0.85),
    text="<b>PARADOX QUADRANT</b><br>both stockout + overstock",
    showarrow=False,
    bgcolor="rgba(242,95,92,0.12)",
    bordercolor=COLORS["danger"], borderwidth=1, borderpad=6,
    font=dict(color=COLORS["danger"], size=11, family="Inter"),
)

st.plotly_chart(fig, use_container_width=True)


st.markdown("<div style='margin: 18px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3-column bottom row
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns([1, 1, 1], gap="medium")

# --- Stockout rate by category × year ---
with c1:
    render_section_label("STOCKOUT % · TIME")
    so_rate = inv.groupby(["year", cat_col])["stockout_flag"].mean().reset_index()
    fig = go.Figure()
    for cat in so_rate[cat_col].dropna().unique():
        sub = so_rate[so_rate[cat_col] == cat]
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["stockout_flag"] * 100,
            name=cat, mode="lines+markers",
            line=dict(color=CATEGORY_COLORS.get(cat, COLORS["primary"]), width=2.5),
            marker=dict(size=7),
        ))
    apply_theme(fig, height=320, title="Stockout flag rate (%)")
    fig.update_yaxes(title="%", ticksuffix="%")
    fig.update_xaxes(dtick=1, title="")
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

# --- Revenue lost estimate (using sell_through_rate + hypothesis) ---
with c2:
    render_section_label("LOST REVENUE EST.")
    inv_est = inv.copy()
    inv_est["units_lost"] = inv_est["stockout_days"] * (
        inv_est["units_sold"] / inv_est["stockout_days"].replace(0, np.nan).fillna(30)
    )
    # Alternative simple estimate: lost_units = stockout_days/30 * units_sold/((30-stockout_days).clip(1))
    inv_est["denom"] = (30 - inv_est["stockout_days"]).clip(lower=1)
    inv_est["daily_rate"] = inv_est["units_sold"] / inv_est["denom"]
    inv_est["units_lost"] = inv_est["stockout_days"] * inv_est["daily_rate"]
    # Join price
    inv_est = inv_est.merge(products[["product_id", "price"]], on="product_id", how="left")
    inv_est["revenue_lost"] = inv_est["units_lost"] * inv_est["price"]
    lost_by_year = inv_est.groupby("year")["revenue_lost"].sum().reset_index()

    fig = go.Figure(go.Bar(
        x=lost_by_year["year"], y=lost_by_year["revenue_lost"] / 1e9,
        marker=dict(
            color=lost_by_year["revenue_lost"],
            colorscale=[[0, COLORS["primary_dim"]], [1, COLORS["danger"]]],
        ),
        text=[fmt_vnd(v) for v in lost_by_year["revenue_lost"]],
        textposition="outside",
        textfont=dict(color=COLORS["text_hi"], size=10),
    ))
    apply_theme(fig, height=320, title="Revenue lost to stockouts (est.)")
    fig.update_yaxes(title="B₫")
    fig.update_xaxes(dtick=1, title="")
    st.plotly_chart(fig, use_container_width=True)

# --- Sell-through rate distribution (violin) ---
with c3:
    render_section_label("SELL-THROUGH DIST.")
    fig = go.Figure()
    for cat in inv[cat_col].dropna().unique():
        sub = inv[inv[cat_col] == cat]
        fig.add_trace(go.Violin(
            y=sub["sell_through_rate"],
            name=cat,
            line_color=CATEGORY_COLORS.get(cat, COLORS["primary"]),
            fillcolor=CATEGORY_COLORS.get(cat, COLORS["primary"]),
            opacity=0.55, meanline_visible=True,
            box_visible=False, points=False,
        ))
    apply_theme(fig, height=320, title="Sell-through rate by category")
    fig.update_yaxes(title="Rate", range=[0, 1])
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


st.markdown("<div style='margin: 20px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Prescriptive roadmap table
# ---------------------------------------------------------------------------
render_section_label("PRESCRIPTIVE · 3-ACTION ROADMAP")

roadmap = pd.DataFrame({
    "Action": [
        "Lead-time optimisation (supplier reorder trigger)",
        "Category-level reorder policy (min-max with seasonality)",
        "Kill SKUs with sell-through < 10%",
    ],
    "Effort":   ["● ● ○", "● ● ●", "● ○ ○"],
    "Impact":   [fmt_vnd(650e6), fmt_vnd(1.2e9), fmt_vnd(180e6)],
    "Timeline": ["Q1 2023", "Q2 2023", "Q1 2023"],
    "Confidence": ["Medium", "High", "High"],
})

st.dataframe(
    roadmap,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Action": st.column_config.TextColumn(width="large"),
        "Effort": st.column_config.TextColumn(width="small"),
        "Impact": st.column_config.TextColumn(width="small"),
    },
)


render_insight(
    title="Structural takeaway:",
    level="danger",
    body=(
        "A real operation cannot be both stocked out AND overstocked on the same SKU "
        "in the same month. The paradox exists because <strong>stockout_flag</strong> "
        "fires on any day of stockout while <strong>overstock_flag</strong> fires on "
        "end-of-month excess — they capture different time windows. The fix is a "
        "tighter reorder cadence, not more safety stock."
    ),
)
