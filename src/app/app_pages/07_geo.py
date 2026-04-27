"""Page 7 — Geographic Analysis. Top cities, channel by city."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.insight_box import render_insight, render_section_label
from components.page_header import render_page_header
from utils.chart_helpers import apply_theme
from utils.data_loader import CHANNEL_COLORS, COLORS, fmt_num, fmt_vnd, load_customers, load_orders, load_order_items

TRAIN_CUTOFF = pd.Timestamp("2022-12-31")

customers  = load_customers()
orders     = load_orders()
items      = load_order_items()

# Join orders → customers for city info
ord_cust = orders.merge(customers[["customer_id", "city", "acquisition_channel"]], on="customer_id", how="left")
ord_items = ord_cust.merge(items.groupby("order_id")["net_revenue"].sum().reset_index(), on="order_id", how="left")


render_page_header(
    title="Geographic Concentration",
    subtitle=(
        "Revenue is heavily concentrated in a handful of cities. "
        "Understanding which cities drive volume — and which channels acquired them — "
        "guides where to deploy conversion and retention investment."
    ),
    badge="DESCRIPTIVE · WHERE",
    badge_color="info",
)


# ---------------------------------------------------------------------------
# Row 1: Top 20 cities by revenue + orders
# ---------------------------------------------------------------------------
render_section_label("TOP CITIES · REVENUE & ORDER VOLUME")

city_stats = (
    ord_items.groupby("city")
    .agg(
        revenue=("net_revenue", "sum"),
        orders=("order_id", "count"),
        customers=("customer_id", "nunique"),
    )
    .reset_index()
    .sort_values("revenue", ascending=False)
    .head(20)
)
city_stats["aov"] = city_stats["revenue"] / city_stats["orders"]

left, right = st.columns([1.6, 1], gap="medium")

with left:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=city_stats["city"],
        x=city_stats["revenue"] / 1e9,
        orientation="h",
        name="Revenue",
        marker=dict(
            color=city_stats["revenue"],
            colorscale=[[0, COLORS["primary_dim"]], [1, COLORS["glow"]]],
            line=dict(width=0),
        ),
        text=[f"{v/1e9:.2f} B VND" for v in city_stats["revenue"]],
        textposition="outside",
        textfont=dict(color=COLORS["text_hi"], size=10),
    ))
    apply_theme(fig, height=560, title="Top 20 cities by total revenue (2012–2022)")
    fig.update_xaxes(title="Revenue (B VND)")
    fig.update_yaxes(title="", autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

with right:
    top3_rev = city_stats["revenue"].sum()
    total_rev = ord_items["net_revenue"].sum()
    top3_share = city_stats.head(3)["revenue"].sum() / total_rev * 100
    top1 = city_stats.iloc[0]

    st.markdown(
        f"""
        <div class="kpi-card" style="border-left:3px solid {COLORS['primary']}; margin-bottom:12px">
          <div class="kpi-label" style="color:{COLORS['primary']}">TOP CITY</div>
          <div class="kpi-value" style="font-size:22px">{top1['city']}</div>
          <div style="color:{COLORS['text_med']};font-size:13px;margin-top:4px">
            {fmt_vnd(top1['revenue'])} · {fmt_num(top1['orders'])} orders
          </div>
        </div>
        <div class="kpi-card" style="border-left:3px solid {COLORS['info']}; margin-bottom:12px">
          <div class="kpi-label" style="color:{COLORS['info']}">TOP-3 REVENUE SHARE</div>
          <div class="kpi-value" style="font-size:36px">{top3_share:.1f}%</div>
          <div style="color:{COLORS['text_med']};font-size:13px;margin-top:4px">
            of total 10-year revenue
          </div>
        </div>
        <div class="kpi-card" style="border-left:3px solid {COLORS['warning']}">
          <div class="kpi-label" style="color:{COLORS['warning']}">CITIES IN TOP 20</div>
          <div class="kpi-value" style="font-size:36px">20</div>
          <div style="color:{COLORS['text_med']};font-size:13px;margin-top:4px">
            cover {city_stats['revenue'].sum()/total_rev*100:.0f}% of revenue
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin: 24px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Row 2: Channel mix by top cities (stacked bar)
# ---------------------------------------------------------------------------
render_section_label("ACQUISITION CHANNEL · TOP 15 CITIES")

top15_cities = city_stats.head(15)["city"].tolist()
ch_city = (
    ord_cust[ord_cust["city"].isin(top15_cities)]
    .groupby(["city", "acquisition_channel"])["order_id"]
    .count().reset_index(name="orders")
)
pivot = ch_city.pivot(index="city", columns="acquisition_channel", values="orders").fillna(0)
# Sort cities by total orders
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

fig2 = go.Figure()
for ch in pivot.columns:
    fig2.add_trace(go.Bar(
        name=ch,
        x=pivot.index,
        y=pivot[ch],
        marker=dict(color=CHANNEL_COLORS.get(ch, COLORS["primary"])),
    ))
apply_theme(fig2, height=400, title="Order volume by acquisition channel — top 15 cities")
fig2.update_layout(barmode="stack",
                   legend=dict(orientation="h", y=-0.2, yanchor="top"))
fig2.update_xaxes(title="")
fig2.update_yaxes(title="Orders")
st.plotly_chart(fig2, use_container_width=True)


st.markdown("<div style='margin: 24px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Row 3: AOV bubble by city
# ---------------------------------------------------------------------------
render_section_label("AOV · BUBBLE MAP BY CITY")

top_aov = city_stats.copy()
fig3 = go.Figure(go.Scatter(
    x=top_aov["orders"],
    y=top_aov["aov"],
    mode="markers+text",
    marker=dict(
        size=top_aov["customers"] / top_aov["customers"].max() * 60 + 8,
        color=top_aov["revenue"],
        colorscale=[[0, COLORS["primary_dim"]], [1, COLORS["glow"]]],
        line=dict(color=COLORS["text_hi"], width=0.8),
        opacity=0.80,
        showscale=True,
        colorbar=dict(title="Revenue (VND)"),
    ),
    text=top_aov["city"],
    textposition="top center",
    textfont=dict(size=10, color=COLORS["text_med"]),
    hovertemplate=(
        "<b>%{text}</b><br>Orders: %{x:,.0f}<br>"
        "AOV: %{y:,.0f}VND<extra></extra>"
    ),
))
apply_theme(fig3, height=420,
            title="Orders vs AOV per city — bubble size = unique customers")
fig3.update_xaxes(title="Total orders", tickformat=",")
fig3.update_yaxes(title="Avg order value (VND)")
st.plotly_chart(fig3, use_container_width=True)


render_insight(
    title="Data note — city distribution reflects simulation geography:",
    level="warning",
    body=(
        "Real Vietnamese e-commerce skews heavily toward HCMC and Hanoi (typically "
        "60%+ of revenue). This dataset's flat distribution across smaller cities "
        "(Son Tay, Nam Dinh, Thai Nguyen) reflects the synthetic zip→city mapping "
        "in the simulation. Geographic <em>relative</em> comparisons (channel mix "
        "by city, AOV by city) remain valid; absolute city rankings should not be "
        "used to draw market-entry conclusions."
    ),
)

render_insight(
    title="Geographic concentration risk:",
    level="warning",
    body=(
        f"The top 3 cities account for <strong>{top3_share:.0f}%</strong> of revenue. "
        "Any demand shock in these markets (competitor entry, platform shift) has "
        "outsized impact. Geographic diversification — paired with channel mix "
        "analysis — should inform where to deploy acquisition budget in 2023."
    ),
)
