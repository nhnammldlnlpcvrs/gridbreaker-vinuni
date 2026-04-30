"""Page 5 — Seasonal & Day-of-Week Patterns. Anti-Tết + Wed>Sat anomalies."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from components.insight_box import render_insight, render_section_label
from components.page_header import render_page_header
from utils.chart_helpers import apply_theme
from utils.data_loader import COLORS, fmt_vnd, load_abt_daily

TRAIN_CUTOFF = pd.Timestamp("2022-12-31")

abt = load_abt_daily()
abt = abt[abt["date"] <= TRAIN_CUTOFF].copy()
abt["date"] = pd.to_datetime(abt["date"])
abt["year"] = abt["date"].dt.year
abt["month"] = abt["date"].dt.month
abt["dow"] = abt["date"].dt.dayofweek          # 0=Mon … 6=Sun
abt["dow_name"] = abt["date"].dt.day_name()

MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
DOW_ORDER   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

render_page_header(
    title="Counter-Intuitive Patterns",
    subtitle=(
        "Two findings that contradict Vietnamese retail intuition: "
        "peak sales fall in April–June (not Tết), and Wednesday out-earns Saturday. "
        "Both reshape how forecasting features should be engineered."
    ),
    badge="DIAGNOSTIC · WHEN",
    badge_color="info",
)


# ---------------------------------------------------------------------------
# Section 1 — Monthly seasonality
# ---------------------------------------------------------------------------
render_section_label("SEASONALITY · MONTHLY REVENUE PROFILE")

monthly = (
    abt.groupby("month")["Revenue"]
    .agg(["mean", "sum", "std"])
    .reset_index()
)
monthly.columns = ["month", "avg_daily_rev", "total_rev", "std_rev"]
monthly["month_name"] = [MONTH_NAMES[m - 1] for m in monthly["month"]]

# Highlight Apr-Jun
monthly["color"] = monthly["month"].apply(
    lambda m: COLORS["primary"] if m in (4, 5, 6) else
              COLORS["danger"] if m in (11, 12, 1) else
              COLORS["info"]
)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=monthly["month_name"],
    y=monthly["avg_daily_rev"] / 1e6,
    marker=dict(color=monthly["color"], line=dict(width=0)),
    error_y=dict(type="data", array=monthly["std_rev"] / 1e6,
                 color=COLORS["text_dim"], thickness=1.5, width=4),
    text=[f"{v/1e6:.1f}M" for v in monthly["avg_daily_rev"]],
    textposition="outside",
    textfont=dict(color=COLORS["text_hi"], size=10),
    hovertemplate="<b>%{x}</b><br>Avg daily: %{y:.2f}M VND<extra></extra>",
))
apply_theme(fig, height=400, title="Average daily revenue by month — April–June peak (green), Nov–Jan trough (red)")
fig.update_yaxes(title="Avg daily revenue (M VND)")
fig.update_xaxes(title="")
fig.add_annotation(
    x="Apr", y=monthly.loc[monthly.month == 4, "avg_daily_rev"].iloc[0] / 1e6,
    text="<b>SUMMER PEAK</b><br>Apr–Jun highest",
    showarrow=True, arrowhead=2,
    arrowcolor=COLORS["primary"], ax=0, ay=-50,
    bgcolor="rgba(10,31,26,0.85)", bordercolor=COLORS["primary"],
    borderwidth=1, borderpad=6,
    font=dict(color=COLORS["glow"], size=11),
)
fig.add_annotation(
    x="Dec", y=monthly.loc[monthly.month == 12, "avg_daily_rev"].iloc[0] / 1e6,
    text="<b>TẾT WINDOW</b><br>Dec–Jan lowest",
    showarrow=True, arrowhead=2,
    arrowcolor=COLORS["danger"], ax=40, ay=-40,
    bgcolor="rgba(10,31,26,0.85)", bordercolor=COLORS["danger"],
    borderwidth=1, borderpad=6,
    font=dict(color=COLORS["danger"], size=11),
)
st.plotly_chart(fig, use_container_width=True)

render_insight(
    title="Anti-Tết anomaly:",
    level="warning",
    body=(
        "Standard VN retail peaks in <strong>Nov–Feb</strong> (Tết + year-end bonuses). "
        "This business peaks in <strong>Apr–Jun</strong> — consistent with a summer "
        "fashion cycle or an office-worker demographic that does not gift apparel. "
        "Any forecasting model using generic VN holiday features will "
        "<strong>over-forecast Q1 and under-forecast Q2</strong>."
    ),
)

st.markdown("<div style='margin: 24px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 2 — Year × Month heatmap
# ---------------------------------------------------------------------------
render_section_label("HEATMAP · YEAR × MONTH REVENUE")

ym = (
    abt.groupby(["year", "month"])["Revenue"]
    .sum().reset_index()
)
ym_pivot = ym.pivot(index="year", columns="month", values="Revenue").fillna(0)
ym_pivot.columns = [MONTH_NAMES[m - 1] for m in ym_pivot.columns]

fig2 = go.Figure(go.Heatmap(
    z=ym_pivot.values,
    x=ym_pivot.columns,
    y=ym_pivot.index,
    colorscale=[[0, COLORS["bg_deep"]], [0.5, COLORS["primary_dim"]], [1, COLORS["glow"]]],
    text=[[f"{v/1e6:.0f}M" for v in row] for row in ym_pivot.values],
    texttemplate="%{text}",
    textfont=dict(size=9, color=COLORS["text_hi"]),
    colorbar=dict(title=dict(text="Revenue (VND)", side="right")),
))
apply_theme(fig2, height=380, title="Monthly revenue by year — consistent Apr–Jun brightness across all years")
fig2.update_xaxes(title="")
fig2.update_yaxes(title="Year", dtick=1)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("<div style='margin: 24px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section 3 — Day-of-week pattern
# ---------------------------------------------------------------------------
render_section_label("DAY-OF-WEEK · WEDNESDAY > SATURDAY")

dow_stats = (
    abt.groupby("dow_name")["Revenue"]
    .agg(["mean", "std"])
    .reset_index()
)
dow_stats.columns = ["dow_name", "avg_rev", "std_rev"]
dow_stats["dow_name"] = pd.Categorical(dow_stats["dow_name"], categories=DOW_ORDER, ordered=True)
dow_stats = dow_stats.sort_values("dow_name")

dow_stats["color"] = dow_stats["dow_name"].apply(
    lambda d: COLORS["primary"] if d == "Wednesday" else
              COLORS["warning"] if d in ("Saturday", "Sunday") else
              COLORS["info"]
)

left, right = st.columns([1.6, 1], gap="medium")

with left:
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=dow_stats["dow_name"],
        y=dow_stats["avg_rev"] / 1e6,
        marker=dict(color=dow_stats["color"].tolist(), line=dict(width=0)),
        error_y=dict(type="data", array=dow_stats["std_rev"] / 1e6,
                     color=COLORS["text_dim"], thickness=1.5, width=4),
        text=[f"{v/1e6:.2f}M" for v in dow_stats["avg_rev"]],
        textposition="outside",
        textfont=dict(color=COLORS["text_hi"], size=10),
    ))
    apply_theme(fig3, height=380, title="Average daily revenue by day of week")
    fig3.update_yaxes(title="Avg daily revenue (M VND)")
    fig3.update_xaxes(title="")
    wed_val = dow_stats.loc[dow_stats.dow_name == "Wednesday", "avg_rev"].iloc[0]
    sat_val = dow_stats.loc[dow_stats.dow_name == "Saturday", "avg_rev"].iloc[0]
    fig3.add_annotation(
        x="Wednesday", y=wed_val / 1e6,
        text=f"<b>WED</b><br>{fmt_vnd(wed_val)}",
        showarrow=True, arrowhead=2,
        arrowcolor=COLORS["primary"], ax=0, ay=-45,
        bgcolor="rgba(10,31,26,0.85)", bordercolor=COLORS["primary"],
        borderwidth=1, borderpad=5,
        font=dict(color=COLORS["glow"], size=11),
    )
    st.plotly_chart(fig3, use_container_width=True)

with right:
    wed_lift = (wed_val / sat_val - 1) * 100
    dow_rank = dow_stats.sort_values("avg_rev", ascending=False).reset_index(drop=True)
    wed_rank = int(dow_rank[dow_rank["dow_name"] == "Wednesday"].index[0]) + 1
    sat_rank = int(dow_rank[dow_rank["dow_name"] == "Saturday"].index[0]) + 1

    st.markdown(
        f"""
        <div class="kpi-card" style="border-left:3px solid {COLORS['primary']}; margin-bottom:14px">
          <div class="kpi-label" style="color:{COLORS['primary']}">WEDNESDAY RANK</div>
          <div class="kpi-value" style="font-size:48px">#{wed_rank}</div>
          <div style="color:{COLORS['text_med']};font-size:13px;margin-top:6px">
            of 7 days by avg daily revenue
          </div>
        </div>
        <div class="kpi-card" style="border-left:3px solid {COLORS['warning']}; margin-bottom:14px">
          <div class="kpi-label" style="color:{COLORS['warning']}">SATURDAY RANK</div>
          <div class="kpi-value" style="font-size:48px">#{sat_rank}</div>
          <div style="color:{COLORS['text_med']};font-size:13px;margin-top:6px">
            Retail expects #1–2
          </div>
        </div>
        <div class="kpi-card" style="border-left:3px solid {COLORS['glow']}">
          <div class="kpi-label" style="color:{COLORS['glow']}">WED vs SAT LIFT</div>
          <div class="kpi-value" style="font-size:36px; color:{COLORS['primary']}">
            +{wed_lift:.1f}%
          </div>
          <div style="color:{COLORS['text_med']};font-size:13px;margin-top:6px">
            Wednesday out-earns Saturday
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

render_insight(
    title="Office-worker hypothesis:",
    level="info",
    body=(
        "Pure consumer retail peaks on <strong>Friday–Saturday</strong> (pre-weekend purchases). "
        "A mid-week peak consistent with <strong>corporate procurement</strong> or "
        "<strong>lunch-break mobile shopping</strong> by office workers. "
        "This validates building <strong>day-of-week features</strong> in the forecast model "
        "rather than relying on a generic 'weekend effect'."
    ),
)
