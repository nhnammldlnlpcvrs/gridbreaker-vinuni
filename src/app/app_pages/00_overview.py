"""Page 0 — Overview. Z-pattern scan, hero story, 5 KPI cards, bento grid."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from components.insight_box import render_insight, render_section_label
from components.kpi_glass import render_kpi
from components.page_header import render_page_header
from utils.chart_helpers import annotate, apply_theme
from utils.data_loader import (
    CATEGORY_COLORS, COLORS,
    fmt_num, fmt_pct, fmt_vnd,
    load_abt_daily, load_orders, load_order_items, load_products, load_customers,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
abt = load_abt_daily()
orders = load_orders()
items = load_order_items()
products = load_products()
customers = load_customers()

yearly = abt.groupby("year", as_index=False).agg(
    revenue=("Revenue", "sum"),
    cogs=("COGS", "sum"),
    orders=("n_orders", "sum"),
    cancelled=("n_cancelled", "sum"),
    sessions=("sessions_total", "sum"),
)
yearly["margin_pct"] = (yearly["revenue"] - yearly["cogs"]) / yearly["revenue"]
yearly["conv_rate"] = yearly["orders"] / yearly["sessions"]
yearly["aov"] = yearly["revenue"] / yearly["orders"]

peak = yearly.loc[yearly["revenue"].idxmax()]
last = yearly.iloc[-1]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
render_page_header(
    title="The Great Divergence",
    subtitle=(
        "Fashion e-commerce Vietnam · 2012 → 2022. Revenue collapsed "
        f"{(last['revenue']/peak['revenue']-1)*100:+.0f}% vs the {int(peak['year'])} peak "
        "while web traffic kept rising — a supply-side and conversion failure, "
        "not a demand problem."
    ),
    badge="DESCRIPTIVE · OVERVIEW",
)


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
cols = st.columns(5, gap="medium")

rev_2022 = float(yearly.loc[yearly["year"] == 2022, "revenue"].iloc[0])
rev_2021 = float(yearly.loc[yearly["year"] == 2021, "revenue"].iloc[0])
yoy_delta = rev_2022 / rev_2021 - 1
total_orders = yearly["orders"].sum()
total_cust = customers["customer_id"].nunique()
avg_aov = last["aov"]
avg_margin = yearly["margin_pct"].mean()

orders_delta = (last["orders"] - peak["orders"]) / peak["orders"]
cust_delta = (customers["signup_date"].dt.year == int(last["year"])).sum() \
             / max((customers["signup_date"].dt.year == int(peak["year"])).sum(), 1) - 1

with cols[0]:
    render_kpi(
        "2022 REVENUE", fmt_vnd(rev_2022),
        delta=f"{'▲' if yoy_delta > 0 else '▼'} {abs(yoy_delta)*100:.0f}% YoY",
        delta_kind="up" if yoy_delta > 0 else "down",
        caption=f"peak was {fmt_vnd(float(peak['revenue']))} ({int(peak['year'])})",
    )
with cols[1]:
    render_kpi("TOTAL ORDERS", fmt_num(total_orders, 1),
               delta=f"▼ {orders_delta*100:.0f}%", delta_kind="down",
               caption="year-on-year vs peak")
with cols[2]:
    render_kpi("CUSTOMERS", fmt_num(total_cust, 0),
               delta=f"▲ {cust_delta*100:+.0f}%" if cust_delta else "flat",
               delta_kind="up" if cust_delta > 0 else "flat",
               caption="signups still rising")
with cols[3]:
    render_kpi("AVG ORDER VALUE", fmt_vnd(avg_aov),
               delta="▲ stable", delta_kind="flat",
               caption="price not the issue")
with cols[4]:
    render_kpi("GROSS MARGIN", fmt_pct(avg_margin, 0),
               delta="▲ stable", delta_kind="flat",
               caption="profitability intact")


st.markdown("<div style='margin: 24px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero row: 2/3 big chart + 1/3 anomaly callout
# ---------------------------------------------------------------------------
hero_left, hero_right = st.columns([2, 1], gap="medium")

with hero_left:
    render_section_label("HERO · THE DIVERGENCE")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=yearly["year"], y=yearly["revenue"] / 1e9,
            name="Revenue (B VND)", mode="lines+markers",
            line=dict(color=COLORS["primary"], width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(82,183,136,0.15)",
        ), secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=yearly["year"], y=yearly["sessions"] / 1e6,
            name="Sessions (M)", mode="lines+markers",
            line=dict(color=COLORS["info"], width=2, dash="dot"),
            marker=dict(size=6),
        ), secondary_y=True,
    )
    apply_theme(fig, height=380, title="Revenue vs Traffic — the contradiction")
    fig.update_yaxes(title_text="Revenue (B VND)", secondary_y=False)
    fig.update_yaxes(title_text="Sessions (M)", secondary_y=True)
    fig.update_xaxes(title_text="Year", dtick=1)
    annotate(fig, x=int(peak["year"]), y=peak["revenue"]/1e9,
             text=f"{int(peak['year'])} PEAK<br>{fmt_vnd(peak['revenue'])}",
             ax=-40, ay=-60)
    annotate(fig, x=2019, y=yearly.loc[yearly.year==2019,"revenue"].iloc[0]/1e9,
             text="2019 CRASH<br>-46%",
             color=COLORS["danger"], ax=40, ay=40)
    st.plotly_chart(fig, use_container_width=True)

with hero_right:
    render_section_label("ANOMALY · P0")
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left: 3px solid {COLORS['danger']}; min-height: 380px">
          <span class="material-symbols-rounded"
                style="font-size:42px; color:{COLORS['danger']}; display:block; margin-bottom:8px">
            warning
          </span>
          <div class="kpi-label" style="color: {COLORS['danger']}">INVENTORY PARADOX</div>
          <div class="kpi-value" style="font-size: 44px; color: {COLORS['danger']}">50.6%</div>
          <div style="color: {COLORS['text_med']}; font-size: 13px; margin-top: 10px; line-height: 1.5">
            of product-months are flagged <strong>stockout = 1</strong>
            <u>and</u> <strong>overstock = 1</strong> simultaneously.
          </div>
          <div style="color: {COLORS['text_dim']}; font-size: 12px; margin-top: 12px">
            Economic reality or simulation artefact?
            Either way — supply-chain timing is broken.
          </div>
          <div style="margin-top: 16px; font-size: 12px; color: {COLORS['primary']}; font-weight: 600">
            → See Inventory Paradox page
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("<div style='margin: 24px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Bento bottom row: 3 smaller panels
# ---------------------------------------------------------------------------
b1, b2, b3 = st.columns(3, gap="medium")

# --- Bento 1: Category mix stacked area ---
with b1:
    render_section_label("CATEGORY MIX · OVER TIME")
    items_p = items.merge(products[["product_id", "category"]], on="product_id", how="left")
    items_p = items_p.merge(
        orders[["order_id", "order_date"]].assign(year=orders["order_date"].dt.year),
        on="order_id", how="inner",
    )
    cat_year = (
        items_p.groupby(["year", "category"])["net_revenue"].sum().reset_index()
    )
    cat_year = cat_year.pivot(index="year", columns="category", values="net_revenue").fillna(0)
    cat_year_pct = cat_year.div(cat_year.sum(axis=1), axis=0) * 100

    fig = go.Figure()
    for cat in cat_year_pct.columns:
        fig.add_trace(go.Scatter(
            x=cat_year_pct.index, y=cat_year_pct[cat],
            name=cat, stackgroup="one",
            line=dict(width=0.5, color=CATEGORY_COLORS.get(cat, COLORS["primary"])),
            fillcolor=CATEGORY_COLORS.get(cat, COLORS["primary"]),
        ))
    apply_theme(fig, height=300, title=None)
    fig.update_layout(
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        yaxis_ticksuffix="%",
    )
    fig.update_xaxes(dtick=2)
    st.plotly_chart(fig, use_container_width=True)

# --- Bento 2: Cancel rate by payment method ---
with b2:
    render_section_label("CANCEL RATE · PAYMENT")
    cancel_by_pay = (
        orders.groupby("payment_method")
        .agg(total=("order_id", "count"),
             cancelled=("order_status", lambda s: (s == "cancelled").sum()))
        .assign(cancel_rate=lambda d: d["cancelled"] / d["total"])
        .sort_values("cancel_rate", ascending=True)
        .reset_index()
    )
    fig = go.Figure(go.Bar(
        x=cancel_by_pay["cancel_rate"] * 100,
        y=cancel_by_pay["payment_method"],
        orientation="h",
        marker=dict(
            color=cancel_by_pay["cancel_rate"],
            colorscale=[[0, COLORS["primary_dim"]], [1, COLORS["danger"]]],
        ),
        text=[f"{v*100:.1f}%" for v in cancel_by_pay["cancel_rate"]],
        textposition="outside",
        textfont=dict(color=COLORS["text_hi"]),
    ))
    apply_theme(fig, height=300, title=None)
    fig.update_xaxes(ticksuffix="%", title="")
    fig.update_yaxes(title="")
    st.plotly_chart(fig, use_container_width=True)

# --- Bento 3: Wed>Sat day-of-week bar ---
with b3:
    render_section_label("REVENUE · DAY OF WEEK")
    DOW_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    abt_tmp = abt.copy()
    abt_tmp["dow_name"] = abt_tmp["date"].dt.day_name() if hasattr(abt_tmp["date"], "dt") else \
        pd.to_datetime(abt_tmp["date"]).dt.day_name() if "date" in abt_tmp.columns else None
    if "dow_name" not in abt_tmp.columns or abt_tmp["dow_name"].isna().all():
        import pandas as _pd
        abt_tmp["dow_name"] = _pd.to_datetime(abt_tmp.index).day_name()
    dow_avg = (
        abt_tmp.groupby("dow_name")["Revenue"].mean()
        .reindex(DOW_ORDER)
        .reset_index()
    )
    dow_avg.columns = ["dow", "avg_rev"]
    dow_avg["color"] = dow_avg["dow"].apply(
        lambda d: COLORS["primary"] if d == "Wednesday" else
                  COLORS["warning"] if d in ("Saturday", "Sunday") else
                  COLORS["info"]
    )
    fig = go.Figure(go.Bar(
        x=dow_avg["dow"].str[:3],
        y=dow_avg["avg_rev"] / 1e6,
        marker=dict(color=dow_avg["color"].tolist(), line=dict(width=0)),
        text=[f"{v/1e6:.2f}M" for v in dow_avg["avg_rev"]],
        textposition="outside",
        textfont=dict(color=COLORS["text_hi"], size=9),
    ))
    apply_theme(fig, height=300, title=None)
    fig.update_yaxes(title="M VND/day")
    fig.update_xaxes(title="")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Narrative insights
# ---------------------------------------------------------------------------
st.markdown("<div style='margin: 20px 0'></div>", unsafe_allow_html=True)

render_insight(
    title="Total economic cost of conversion failure (2017–2022):",
    level="danger",
    body=(
        "If conversion had stayed at the 2016 rate (0.98%) on actual sessions, "
        "the business would have generated an estimated <strong>~4.6 billion VND</strong> "
        "more in revenue over 6 years. "
        "That is the cumulative cost of letting the funnel break without intervention."
    ),
)

render_insight(
    title="The contradiction:",
    level="warning",
    body=(
        f"Web sessions grew <strong>{((yearly.loc[yearly.year==2019,'sessions'].iloc[0] / yearly.loc[yearly.year==2016,'sessions'].iloc[0]) - 1)*100:+.0f}%</strong> "
        f"from 2016 → 2019, yet revenue dropped "
        f"<strong>{((yearly.loc[yearly.year==2019,'revenue'].iloc[0] / yearly.loc[yearly.year==2016,'revenue'].iloc[0]) - 1)*100:+.0f}%</strong>. "
        "Conversion rate halved from <strong>0.98%</strong> to <strong>0.42%</strong> — "
        "customers were visiting but not buying."
    ),
)

render_insight(
    title="Next step:",
    level="info",
    body=(
        "Move to the <strong>Revenue Collapse</strong> page for the diagnostic deep-dive, "
        "then <strong>Funnel & Customer</strong> for the retention analysis, and finally "
        "the <strong>Recovery Simulator</strong> for prescriptive scenarios."
    ),
)
