"""Page 2 — Funnel & Customer. Descriptive → Diagnostic → Prescriptive."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.insight_box import render_insight, render_section_label
from components.page_header import render_page_header
from utils.chart_helpers import apply_theme
from utils.data_loader import (
    CHANNEL_COLORS, COLORS,
    fmt_num, fmt_pct, fmt_vnd,
    load_abt_daily, load_customers, load_orders, load_order_items,
    load_products, load_returns,
)


abt = load_abt_daily()
customers = load_customers()
orders = load_orders()
items = load_order_items()
products = load_products()
returns = load_returns()


render_page_header(
    title="The Leaky Bucket",
    subtitle=(
        "Customers visit, order, then quietly disappear. This page dissects "
        "the funnel, the cohorts, and the reasons behind it — and quantifies "
        "how much each leak is costing."
    ),
    badge="DIAGNOSTIC · WHO",
    badge_color="warning",
)


# ---------------------------------------------------------------------------
# Top row: Funnel + Cohort retention heatmap
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1.5], gap="medium")

with left:
    render_section_label("FUNNEL · SESSION → REVIEW")

    total_sessions = int(abt["sessions_total"].sum())
    total_orders = int(orders["order_id"].nunique())
    delivered = int((orders["order_status"] == "delivered").sum())
    # Reviews are ~20% of delivered per spec
    reviewed_est = int(delivered * 0.20)

    fig = go.Figure(go.Funnel(
        y=["Sessions", "Orders", "Delivered", "Reviewed (~20%)"],
        x=[total_sessions, total_orders, delivered, reviewed_est],
        textinfo="value+percent previous",
        marker=dict(color=[COLORS["info"], COLORS["primary"],
                           COLORS["glow"], COLORS["warning"]]),
        connector=dict(line=dict(color=COLORS["text_dim"], dash="dot")),
    ))
    apply_theme(fig, height=420, title=None)
    st.plotly_chart(fig, use_container_width=True)

    # Drop stats
    cancel_rate = (orders["order_status"] == "cancelled").mean()
    return_rate = (orders["order_status"] == "returned").mean()
    st.markdown(
        f"""
        <div style="display:flex; gap:10px; margin-top:6px">
          <div class="kpi-card" style="flex:1; min-height:70px; padding:10px 12px">
            <div class="kpi-label">CANCEL RATE</div>
            <div class="kpi-value" style="font-size:22px;color:{COLORS['danger']}">
              {cancel_rate*100:.1f}%
            </div>
          </div>
          <div class="kpi-card" style="flex:1; min-height:70px; padding:10px 12px">
            <div class="kpi-label">RETURN RATE</div>
            <div class="kpi-value" style="font-size:22px;color:{COLORS['warning']}">
              {return_rate*100:.1f}%
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    render_section_label("COHORT · RETENTION HEATMAP")

    # Build cohort: customers grouped by signup month × months since signup
    orders_with_customer = orders.merge(
        customers[["customer_id", "signup_date"]], on="customer_id", how="left",
    )
    orders_with_customer["cohort"] = (
        orders_with_customer["signup_date"].dt.to_period("Q").astype(str)
    )
    orders_with_customer["order_period"] = (
        orders_with_customer["order_date"].dt.to_period("Q").astype(str)
    )
    orders_with_customer["periods_since"] = (
        orders_with_customer["order_date"].dt.to_period("Q").astype("int64")
        - orders_with_customer["signup_date"].dt.to_period("Q").astype("int64")
    )
    orders_with_customer = orders_with_customer[orders_with_customer["periods_since"] >= 0]

    cohort_sizes = (
        orders_with_customer.groupby("cohort")["customer_id"].nunique()
    )
    cohort_matrix = (
        orders_with_customer.groupby(["cohort", "periods_since"])["customer_id"]
        .nunique().unstack(fill_value=0)
    )
    retention = cohort_matrix.div(cohort_sizes, axis=0) * 100
    # Keep last 20 cohorts × first 10 periods for readability
    retention = retention.tail(20).iloc[:, :10]

    fig = go.Figure(go.Heatmap(
        z=retention.values,
        x=[f"+{int(q)}Q" for q in retention.columns],
        y=retention.index,
        colorscale=[[0, COLORS["bg_deep"]],
                    [0.4, COLORS["primary_dim"]],
                    [1, COLORS["glow"]]],
        text=retention.map(lambda v: f"{v:.0f}").values,
        texttemplate="%{text}",
        textfont=dict(size=10, color=COLORS["text_hi"]),
        colorbar=dict(title=dict(text="% retained", side="right")),
        zmin=0, zmax=100,
    ))
    apply_theme(fig, height=420, title="Cohort retention (% returning in quarter N)")
    fig.update_xaxes(title="Quarters since signup", side="top")
    fig.update_yaxes(title="Signup cohort")
    st.plotly_chart(fig, use_container_width=True)


st.markdown("<div style='margin: 20px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Middle: Channel efficiency bubble
# ---------------------------------------------------------------------------
render_section_label("CHANNEL · EFFICIENCY BUBBLE")

channel_stats = (
    customers.merge(orders, on="customer_id", how="left")
    .groupby("acquisition_channel")
    .agg(
        orders=("order_id", "count"),
        customers=("customer_id", "nunique"),
        cancelled=("order_status", lambda s: (s == "cancelled").sum()),
    )
    .reset_index()
)
channel_stats["orders_per_cust"] = channel_stats["orders"] / channel_stats["customers"]
channel_stats["cancel_rate"] = channel_stats["cancelled"] / channel_stats["orders"]
channel_stats = channel_stats.dropna(subset=["acquisition_channel"])

fig = go.Figure()
for _, row in channel_stats.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["orders_per_cust"]],
        y=[row["cancel_rate"] * 100],
        mode="markers+text",
        marker=dict(
            size=np.sqrt(row["customers"]) / 3,
            color=CHANNEL_COLORS.get(row["acquisition_channel"], COLORS["primary"]),
            line=dict(color=COLORS["text_hi"], width=1),
            opacity=0.85,
        ),
        text=[row["acquisition_channel"]],
        textposition="top center",
        textfont=dict(size=11, color=COLORS["text_med"]),
        name=row["acquisition_channel"],
        showlegend=False,
    ))

apply_theme(fig, height=380,
            title="Orders per customer vs Cancel rate — bubble size = customer volume")
fig.update_xaxes(title="Orders per customer")
fig.update_yaxes(title="Cancel rate (%)", ticksuffix="%")
st.plotly_chart(fig, use_container_width=True)


st.markdown("<div style='margin: 20px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Bottom: Return reasons + prescriptive actions
# ---------------------------------------------------------------------------
b1, b2 = st.columns([1.3, 1], gap="medium")

with b1:
    render_section_label("RETURNS · REASON × CATEGORY")

    ret_with_prod = returns.merge(
        products[["product_id", "category"]], on="product_id", how="left",
    )
    reason_cat = (
        ret_with_prod.groupby(["category", "return_reason"])["return_id"]
        .count().reset_index(name="count")
    )
    top_reasons = ret_with_prod["return_reason"].value_counts().head(5).index
    reason_cat = reason_cat[reason_cat["return_reason"].isin(top_reasons)]

    fig = go.Figure()
    for reason in top_reasons:
        sub = reason_cat[reason_cat["return_reason"] == reason]
        fig.add_trace(go.Bar(
            name=reason, x=sub["category"], y=sub["count"],
        ))
    apply_theme(fig, height=380, title="Top-5 return reasons by category")
    fig.update_layout(barmode="stack",
                      legend=dict(orientation="h", y=-0.2, yanchor="top"))
    fig.update_yaxes(title="Returns")
    fig.update_xaxes(title="")
    st.plotly_chart(fig, use_container_width=True)

with b2:
    render_section_label("PRESCRIPTIVE · TOP ACTIONS")

    wrong_size_n = int((returns["return_reason"] == "wrong_size").sum())
    wrong_size_refund = float(
        returns.loc[returns["return_reason"] == "wrong_size", "refund_amount"].sum()
    )
    cod_orders = int((orders["payment_method"] == "cod").sum())
    cod_cancel = int(((orders["payment_method"] == "cod") &
                      (orders["order_status"] == "cancelled")).sum())
    cod_cancel_rate = cod_cancel / max(cod_orders, 1)

    avg_order_value = float(
        orders.assign(year=orders["order_date"].dt.year)
        .groupby("order_id").size().reindex(orders["order_id"].unique(), fill_value=1)
        .size
    )
    # Simple prescriptive estimate for COD: if we convert 20% COD to prepay
    # and cut that subset cancel rate by 80%, revenue uplift ≈
    # 20% * COD_orders * (cancel_rate_reduction) * AOV
    # Use abt for AOV
    aov = float(abt["Revenue"].sum() / abt["n_orders"].sum())
    cod_uplift = 0.20 * cod_orders * cod_cancel_rate * 0.80 * aov

    st.markdown(
        f"""
        <div class="kpi-card" style="margin-bottom:12px; border-left:3px solid {COLORS['primary']}">
          <div class="kpi-label" style="color:{COLORS['primary']}">ACTION 1 · SIZE GUIDE FIX</div>
          <div class="kpi-value" style="font-size:24px">
            -30% wrong-size returns
          </div>
          <div style="color:{COLORS['text_med']};font-size:13px;margin-top:6px">
            Current: <strong>{fmt_num(wrong_size_n)}</strong> returns = <strong>{fmt_vnd(wrong_size_refund)}</strong> refunded.
            Target uplift: <strong>{fmt_vnd(wrong_size_refund*0.3)}</strong>/yr.
          </div>
          <div style="color:{COLORS['text_dim']};font-size:11px;margin-top:4px">
            Effort: low · Timeline: Q1
          </div>
        </div>

        <div class="kpi-card" style="margin-bottom:12px; border-left:3px solid {COLORS['warning']}">
          <div class="kpi-label" style="color:{COLORS['warning']}">ACTION 2 · COD → PREPAY</div>
          <div class="kpi-value" style="font-size:24px">
            Reduce COD cancellations
          </div>
          <div style="color:{COLORS['text_med']};font-size:13px;margin-top:6px">
            COD cancel rate: <strong>{cod_cancel_rate*100:.1f}%</strong>.
            Switching 20% of COD to prepay ≈ <strong>{fmt_vnd(cod_uplift)}</strong>/yr uplift.
          </div>
          <div style="color:{COLORS['text_dim']};font-size:11px;margin-top:4px">
            Effort: medium · Timeline: Q2
          </div>
        </div>

        <div class="kpi-card" style="border-left:3px solid {COLORS['info']}">
          <div class="kpi-label" style="color:{COLORS['info']}">ACTION 3 · COHORT WIN-BACK</div>
          <div class="kpi-value" style="font-size:24px">
            Email Q3+ churned cohorts
          </div>
          <div style="color:{COLORS['text_med']};font-size:13px;margin-top:6px">
            Retention cliff at Q3+. Segmented win-back flow + voucher
            = cheap re-activation.
          </div>
          <div style="color:{COLORS['text_dim']};font-size:11px;margin-top:4px">
            Effort: low · Timeline: Q1
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_insight(
    title="Prescriptive summary:",
    level="info",
    body=(
        "The funnel bleeds in two very specific places — size-mismatch returns and "
        "COD cancellations. Both are <strong>operationally fixable</strong> (not "
        "marketing spend), making them the highest-ROI levers before touching "
        "acquisition channels."
    ),
)
