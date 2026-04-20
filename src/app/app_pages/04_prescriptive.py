"""Page 4 — Recovery Simulator. Prescriptive with interactive sliders."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.insight_box import render_insight, render_section_label
from components.page_header import render_page_header
from utils.chart_helpers import apply_theme
from utils.data_loader import (
    COLORS, fmt_num, fmt_pct, fmt_vnd,
    load_abt_daily,
)


abt = load_abt_daily()

yearly = abt.groupby("year", as_index=False).agg(
    revenue=("Revenue", "sum"),
    orders=("n_orders", "sum"),
    sessions=("sessions_total", "sum"),
    cancelled=("n_cancelled", "sum"),
)
yearly["conv_rate"] = yearly["orders"] / yearly["sessions"]
yearly["aov"] = yearly["revenue"] / yearly["orders"]
yearly["cancel_rate"] = yearly["cancelled"] / yearly["orders"]

peak = yearly.loc[yearly["revenue"].idxmax()]
last = yearly.iloc[-1]


render_page_header(
    title="Recovery Simulator",
    subtitle=(
        "Drag the levers below to simulate a 2023 recovery scenario. "
        "The baseline is the 2022 trajectory; each slider quantifies "
        "an operational improvement and compounds the uplift."
    ),
    badge="PRESCRIPTIVE · WHAT TO DO",
    badge_color="primary",
)


# ---------------------------------------------------------------------------
# Controls + live projection (two-column hero)
# ---------------------------------------------------------------------------
left, right = st.columns([1, 1.5], gap="large")

with left:
    render_section_label("CONTROLS · LEVERS")

    conv_target = st.slider(
        "Conversion rate target (%)",
        min_value=float(last["conv_rate"] * 100),
        max_value=1.50,
        value=0.70, step=0.05, format="%.2f%%",
        help=f"Current (2022): {last['conv_rate']*100:.2f}%. Peak (2016): 0.98%.",
    )
    stockout_reduction = st.slider(
        "Stockout reduction (%)",
        min_value=0, max_value=60, value=30, step=5,
        help="Lead-time + reorder fixes cut lost revenue proportionally.",
    )
    cancel_cut = st.slider(
        "Cancel-rate reduction (%)",
        min_value=0, max_value=50, value=20, step=5,
        help="COD → prepay shift. Current cancel rate: "
             f"{last['cancel_rate']*100:.1f}%.",
    )
    aov_uplift = st.slider(
        "AOV uplift (%)",
        min_value=0, max_value=25, value=8, step=1,
        help="Bundling / cross-sell. Current AOV: " + fmt_vnd(last["aov"]),
    )
    session_growth = st.slider(
        "Session growth (%)",
        min_value=-10, max_value=25, value=5, step=1,
        help="Organic traffic growth (minus paid acquisition).",
    )

with right:
    render_section_label("LIVE PROJECTION · 2023")

    baseline_sessions = last["sessions"] * (1 + session_growth / 100)
    projected_conv = conv_target / 100
    projected_aov = last["aov"] * (1 + aov_uplift / 100)
    projected_cancel_factor = 1 - last["cancel_rate"] * (cancel_cut / 100)
    # Stockout uplift ~ proportional recovery of currently-lost revenue (assume 10% lost)
    lost_share = 0.10
    stockout_uplift_factor = 1 + lost_share * (stockout_reduction / 100)

    baseline_rev = last["revenue"]
    projected_rev = (
        baseline_sessions * projected_conv * projected_aov
        * projected_cancel_factor * stockout_uplift_factor
    )
    uplift = projected_rev - baseline_rev
    vs_peak = projected_rev / peak["revenue"] - 1

    # Timeline with projection appended
    sim_rows = yearly.copy()
    sim_rows = pd.concat([
        sim_rows,
        pd.DataFrame([{
            "year": 2023, "revenue": projected_rev,
            "orders": np.nan, "sessions": baseline_sessions,
        }])
    ], ignore_index=True)

    # Baseline line ("do nothing" scenario)
    baseline_line = yearly["revenue"].tolist() + [baseline_rev * 1.00]

    fig = go.Figure()
    # Baseline (flat)
    fig.add_trace(go.Scatter(
        x=sim_rows["year"], y=baseline_line,
        name="Baseline (do nothing)",
        mode="lines", line=dict(color=COLORS["text_dim"], width=2, dash="dot"),
    ))
    # Actual history
    fig.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["revenue"] / 1e9,
        name="History",
        mode="lines+markers",
        line=dict(color=COLORS["info"], width=2),
        marker=dict(size=7),
    ))
    # Scenario point
    fig.add_trace(go.Scatter(
        x=[2022, 2023], y=[last["revenue"] / 1e9, projected_rev / 1e9],
        name="Your scenario",
        mode="lines+markers",
        line=dict(color=COLORS["primary"], width=4),
        marker=dict(size=12, color=COLORS["glow"],
                    line=dict(color=COLORS["primary"], width=2)),
    ))
    apply_theme(fig, height=400, title="Revenue trajectory")
    fig.update_yaxes(title="Revenue (B₫)")
    fig.update_xaxes(dtick=1, title="")
    fig.add_annotation(
        x=2023, y=projected_rev / 1e9,
        text=f"<b>{fmt_vnd(projected_rev)}</b><br>{uplift/baseline_rev*100:+.0f}% vs 2022",
        showarrow=True, arrowhead=2,
        arrowcolor=COLORS["primary"], ax=-60, ay=-40,
        bgcolor="rgba(10,31,26,0.85)", bordercolor=COLORS["primary"],
        borderwidth=1, borderpad=6,
        font=dict(color=COLORS["glow"], size=12),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("2023 projected", fmt_vnd(projected_rev),
                  delta=fmt_vnd(uplift))
    with m2:
        st.metric("Uplift vs 2022",
                  f"{uplift/baseline_rev*100:+.0f}%")
    with m3:
        st.metric(f"vs {int(peak['year'])} peak",
                  f"{vs_peak*100:+.0f}%")


st.markdown("<div style='margin: 24px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Priority matrix (effort × impact)
# ---------------------------------------------------------------------------
render_section_label("PRIORITY MATRIX · EFFORT × IMPACT")

actions = pd.DataFrame([
    {"action": "Size guide fix",            "effort": 1, "impact": 0.42, "confidence": 0.90},
    {"action": "COD → prepay nudge",        "effort": 2, "impact": 1.40, "confidence": 0.75},
    {"action": "Lead-time optimisation",    "effort": 3, "impact": 0.65, "confidence": 0.70},
    {"action": "Reorder policy rebuild",    "effort": 4, "impact": 1.20, "confidence": 0.65},
    {"action": "Cohort win-back flow",      "effort": 1, "impact": 0.35, "confidence": 0.80},
    {"action": "Kill dead-stock SKUs",      "effort": 1, "impact": 0.18, "confidence": 0.95},
    {"action": "Paid-channel re-allocation","effort": 2, "impact": 0.50, "confidence": 0.60},
])

fig = go.Figure(go.Scatter(
    x=actions["effort"], y=actions["impact"],
    mode="markers+text",
    marker=dict(
        size=actions["confidence"] * 60,
        color=actions["impact"],
        colorscale=[[0, COLORS["primary_dim"]], [1, COLORS["glow"]]],
        line=dict(color=COLORS["text_hi"], width=1.5),
        opacity=0.85,
        showscale=True,
        colorbar=dict(title="Impact (B₫)", ticksuffix=" B"),
    ),
    text=actions["action"],
    textposition="top center",
    textfont=dict(size=11, color=COLORS["text_med"]),
))
apply_theme(fig, height=440,
            title="Quick wins in the bottom-left · transformational in the top-right")
fig.update_xaxes(title="Effort (1 = low, 5 = high)", range=[0.3, 4.7], dtick=1)
fig.update_yaxes(title="Estimated revenue impact (B₫/yr)", range=[0, 1.8])
fig.add_vline(x=2.5, line_dash="dot", line_color=COLORS["text_dim"])
fig.add_hline(y=0.8, line_dash="dot", line_color=COLORS["text_dim"])
st.plotly_chart(fig, use_container_width=True)


st.markdown("<div style='margin: 20px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Gantt roadmap
# ---------------------------------------------------------------------------
render_section_label("IMPLEMENTATION · 2023 GANTT")

gantt_df = pd.DataFrame([
    {"action": "Size guide fix",             "start": "2023-01-01", "end": "2023-02-28", "owner": "Product"},
    {"action": "Cohort win-back flow",       "start": "2023-01-15", "end": "2023-03-31", "owner": "CRM"},
    {"action": "Kill dead-stock SKUs",       "start": "2023-01-01", "end": "2023-02-15", "owner": "Merchandising"},
    {"action": "COD → prepay nudge",         "start": "2023-03-01", "end": "2023-06-30", "owner": "Checkout"},
    {"action": "Lead-time optimisation",     "start": "2023-04-01", "end": "2023-08-31", "owner": "Supply chain"},
    {"action": "Reorder policy rebuild",     "start": "2023-02-01", "end": "2023-09-30", "owner": "Planning"},
    {"action": "Paid-channel reallocation",  "start": "2023-05-01", "end": "2023-07-31", "owner": "Marketing"},
])
gantt_df["start"] = pd.to_datetime(gantt_df["start"])
gantt_df["end"]   = pd.to_datetime(gantt_df["end"])
gantt_df["duration_days"] = (gantt_df["end"] - gantt_df["start"]).dt.days

owner_colors = {
    "Product":       COLORS["primary"],
    "CRM":           COLORS["info"],
    "Merchandising": COLORS["warning"],
    "Checkout":      COLORS["glow"],
    "Supply chain":  COLORS["cat_genz"],
    "Planning":      COLORS["danger"],
    "Marketing":     COLORS["cat_outdoor"],
}

fig = go.Figure()
for _, row in gantt_df.iterrows():
    fig.add_trace(go.Bar(
        x=[row["duration_days"]],
        y=[row["action"]],
        base=[row["start"]],
        orientation="h",
        marker=dict(color=owner_colors.get(row["owner"], COLORS["primary"]),
                    line=dict(color=COLORS["bg_deep"], width=1)),
        hovertemplate=(
            f"<b>{row['action']}</b><br>"
            f"Owner: {row['owner']}<br>"
            f"{row['start'].date()} → {row['end'].date()}<extra></extra>"
        ),
        name=row["owner"],
        showlegend=False,
    ))
apply_theme(fig, height=360,
            title="Initiative schedule · colour = owning team")
fig.update_layout(barmode="stack")
fig.update_xaxes(type="date", title="")
fig.update_yaxes(autorange="reversed", title="")
st.plotly_chart(fig, use_container_width=True)


render_insight(
    title="Executive takeaway:",
    level="info",
    body=(
        "A balanced portfolio of <strong>quick wins</strong> (size guide, win-back, "
        "dead-stock cull) finishing inside Q1 funds the <strong>transformational "
        "initiatives</strong> (reorder policy, lead-time optimisation) that pay off "
        f"by Q3. Combined, the simulator projects up to <strong>{uplift/baseline_rev*100:+.0f}%</strong> "
        "revenue uplift vs the 2022 baseline."
    ),
)
