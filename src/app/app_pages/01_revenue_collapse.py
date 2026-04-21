"""Page 1 — Revenue Collapse. Descriptive → Diagnostic → Predictive."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from components.insight_box import render_insight, render_section_label
from components.page_header import render_page_header
from utils.chart_helpers import annotate, apply_theme
from utils.data_loader import (
    CATEGORY_COLORS, COLORS,
    fmt_num, fmt_pct, fmt_vnd,
    load_abt_daily, load_orders, load_order_items, load_products,
)


abt = load_abt_daily()
orders = load_orders()
items = load_order_items()
products = load_products()

yearly = abt.groupby("year", as_index=False).agg(
    revenue=("Revenue", "sum"),
    orders=("n_orders", "sum"),
    sessions=("sessions_total", "sum"),
)
yearly["conv_rate"] = yearly["orders"] / yearly["sessions"]
yearly["aov"] = yearly["revenue"] / yearly["orders"]

peak_year = int(yearly.loc[yearly["revenue"].idxmax(), "year"])
crash_year = 2019


render_page_header(
    title="Revenue Collapse",
    subtitle=(
        f"Why did revenue fall {(yearly[yearly.year==crash_year].revenue.iloc[0]/yearly[yearly.year==peak_year].revenue.iloc[0]-1)*100:+.0f}% "
        f"from {peak_year} to {crash_year} while web traffic kept growing? "
        "The answer is conversion, not demand."
    ),
    badge="DIAGNOSTIC · WHY",
    badge_color="danger",
)


# ---------------------------------------------------------------------------
# HERO — annotated dual-axis timeline
# ---------------------------------------------------------------------------
render_section_label("HERO · ANNOTATED TIMELINE")

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(
    x=yearly["year"], y=yearly["revenue"] / 1e9,
    name="Revenue (B₫)", mode="lines+markers",
    line=dict(color=COLORS["primary"], width=3),
    marker=dict(size=9), fill="tozeroy",
    fillcolor="rgba(82,183,136,0.12)",
), secondary_y=False)

fig.add_trace(go.Scatter(
    x=yearly["year"], y=yearly["sessions"] / 1e6,
    name="Sessions (M)", mode="lines+markers",
    line=dict(color=COLORS["info"], width=2, dash="dot"),
    marker=dict(size=7),
), secondary_y=True)

# Shaded crash zone
fig.add_vrect(
    x0=2018.5, x1=2019.5,
    fillcolor=COLORS["danger"], opacity=0.08,
    line_width=0,
)

apply_theme(fig, height=440, title="Revenue ▼ -46% | Sessions ▲ +19% — 2016→2019")
fig.update_yaxes(title_text="Revenue (B₫)", secondary_y=False)
fig.update_yaxes(title_text="Sessions (M)", secondary_y=True)
fig.update_xaxes(dtick=1, title="")

annotate(fig, x=peak_year, y=yearly.loc[yearly.year==peak_year,"revenue"].iloc[0]/1e9,
         text=f"{peak_year} PEAK<br>{fmt_vnd(yearly.loc[yearly.year==peak_year,'revenue'].iloc[0])}",
         ax=-50, ay=-60)
annotate(fig, x=2019, y=yearly.loc[yearly.year==2019,"revenue"].iloc[0]/1e9,
         text="2019 CRASH<br>funnel broken",
         color=COLORS["danger"], ax=40, ay=40)
annotate(fig, x=2022, y=yearly.loc[yearly.year==2022,"sessions"].iloc[0]/1e6,
         text="Traffic keeps growing", color=COLORS["info"],
         ax=-40, ay=-20, arrow=True)

st.plotly_chart(fig, use_container_width=True)


st.markdown("<div style='margin: 20px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Waterfall + Category heatmap (2-col)
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1.6], gap="medium")

with left:
    render_section_label("WATERFALL · WHERE THE REVENUE WENT")

    peak = yearly[yearly.year == peak_year].iloc[0]
    crash = yearly[yearly.year == crash_year].iloc[0]

    # Attribute revenue change to: Δ Sessions, Δ Conv rate, Δ AOV
    # R = S × C × A. Use additive decomposition of log-change.
    log_d = np.log(crash["revenue"] / peak["revenue"])
    log_s = np.log(crash["sessions"] / peak["sessions"])
    log_c = np.log(crash["conv_rate"] / peak["conv_rate"])
    log_a = np.log(crash["aov"] / peak["aov"])
    # Share of each to total log-change
    shares = {"Sessions ∆": log_s, "Conv rate ∆": log_c, "AOV ∆": log_a}
    total_d = crash["revenue"] - peak["revenue"]
    # Allocate VND by log-share
    alloc = {k: v / log_d * total_d if log_d != 0 else 0 for k, v in shares.items()}

    fig = go.Figure(go.Waterfall(
        x=[f"{peak_year} Revenue", "Sessions ∆", "Conv rate ∆", "AOV ∆", f"{crash_year} Revenue"],
        measure=["absolute", "relative", "relative", "relative", "total"],
        y=[peak["revenue"]/1e9,
           alloc["Sessions ∆"]/1e9,
           alloc["Conv rate ∆"]/1e9,
           alloc["AOV ∆"]/1e9,
           crash["revenue"]/1e9],
        text=[fmt_vnd(peak["revenue"]),
              fmt_vnd(alloc["Sessions ∆"]),
              fmt_vnd(alloc["Conv rate ∆"]),
              fmt_vnd(alloc["AOV ∆"]),
              fmt_vnd(crash["revenue"])],
        textposition="outside",
        increasing=dict(marker=dict(color=COLORS["primary"])),
        decreasing=dict(marker=dict(color=COLORS["danger"])),
        totals=dict(marker=dict(color=COLORS["info"])),
        connector=dict(line=dict(color=COLORS["text_dim"], dash="dot")),
    ))
    apply_theme(fig, height=400, title=f"Revenue change {peak_year}→{crash_year}")
    fig.update_yaxes(title="B₫")
    st.plotly_chart(fig, use_container_width=True)

with right:
    render_section_label("HEATMAP · CATEGORY × YEAR")

    items_p = items.merge(products[["product_id", "category"]], on="product_id", how="left")
    items_p = items_p.merge(
        orders[["order_id", "order_date"]], on="order_id", how="inner",
    )
    items_p["year"] = items_p["order_date"].dt.year
    cat_year = (
        items_p.groupby(["category", "year"])["net_revenue"].sum().reset_index()
    )
    pivot = cat_year.pivot(index="category", columns="year", values="net_revenue").fillna(0)
    # Index each row to its own max = 100 for relative view
    pivot_idx = pivot.div(pivot.max(axis=1), axis=0) * 100

    fig = go.Figure(go.Heatmap(
        z=pivot_idx.values,
        x=pivot_idx.columns, y=pivot_idx.index,
        colorscale=[[0, COLORS["bg_deep"]],
                    [0.5, COLORS["primary_dim"]],
                    [1, COLORS["glow"]]],
        text=pivot.map(lambda v: fmt_vnd(v, 1)).values,
        texttemplate="%{text}",
        textfont=dict(size=10, color=COLORS["text_hi"]),
        colorbar=dict(title=dict(text="% of cat peak", side="right")),
    ))
    apply_theme(fig, height=400, title="Category revenue (indexed to each category's peak)")
    fig.update_xaxes(dtick=1, title="")
    fig.update_yaxes(title="")
    st.plotly_chart(fig, use_container_width=True)


st.markdown("<div style='margin: 20px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Predictive scenario
# ---------------------------------------------------------------------------
render_section_label("PREDICTIVE · RECOVERY SCENARIO")

st.markdown(
    f"<p style='color:{COLORS['text_med']};margin-top:0'>"
    "If the conversion funnel could be restored, how much revenue would come back? "
    "Drag the slider:</p>",
    unsafe_allow_html=True,
)

target_conv = st.slider(
    "Target conversion rate (%)",
    min_value=0.40, max_value=1.20,
    value=0.70, step=0.05,
    format="%.2f%%",
)

last_sessions = yearly.iloc[-1]["sessions"]
last_aov = yearly.iloc[-1]["aov"]
baseline_rev = yearly.iloc[-1]["revenue"]
projected_rev = last_sessions * (target_conv / 100) * last_aov
uplift = projected_rev - baseline_rev

c1, c2, c3 = st.columns(3, gap="medium")
with c1:
    st.metric("Baseline revenue (2022)", fmt_vnd(baseline_rev))
with c2:
    st.metric(
        f"Projected @ {target_conv:.2f}%",
        fmt_vnd(projected_rev),
        delta=fmt_vnd(uplift),
    )
with c3:
    st.metric(
        "Uplift vs baseline",
        f"{uplift/baseline_rev*100:+.0f}%",
        delta=f"{(projected_rev/yearly[yearly.year==peak_year].revenue.iloc[0]-1)*100:+.0f}% vs {peak_year} peak",
    )


render_insight(
    title="Key diagnostic finding:",
    level="danger",
    body=(
        f"Between {peak_year} and 2019, sessions grew but conversion rate halved "
        "(0.98% → 0.42%). Isolating the three drivers shows the funnel — not demand "
        f"— accounts for <strong>{abs(alloc['Conv rate ∆']/total_d)*100:.0f}%</strong> "
        "of the revenue drop. Fixing it brings back the lion's share."
    ),
)
