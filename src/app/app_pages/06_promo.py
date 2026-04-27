"""Page 6 — Promo ROI Analysis. Calendar uniformity, discount anomaly, lift."""
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from components.insight_box import render_insight, render_section_label
from components.page_header import render_page_header
from utils.chart_helpers import apply_theme
from utils.data_loader import COLORS, fmt_vnd, fmt_pct, load_abt_daily, load_order_items

DATASET = __import__("pathlib").Path(__file__).resolve().parents[3] / "dataset"
TRAIN_CUTOFF = pd.Timestamp("2022-12-31")

abt = load_abt_daily()
items = load_order_items()
promos = pd.read_csv(DATASET / "Master" / "promotions.csv")
promos["start_date"] = pd.to_datetime(promos["start_date"])
promos["end_date"]   = pd.to_datetime(promos["end_date"])
promos["duration_days"] = (promos["end_date"] - promos["start_date"]).dt.days + 1


render_page_header(
    title="Promo Calendar & ROI",
    subtitle=(
        "50 promotions over 10 years reveal a 6-4-6-4 week cadence and only two "
        "discount levels. The calendar is synthetic by design — yet promo days "
        "generate measurable revenue lift on percentage-type promos."
    ),
    badge="DIAGNOSTIC · PROMO",
    badge_color="warning",
)


# ---------------------------------------------------------------------------
# Row 1: Discount value distribution + Calendar uniformity
# ---------------------------------------------------------------------------
top_left, top_right = st.columns([1, 1.5], gap="medium")

with top_left:
    render_section_label("DISCOUNT VALUE DISTRIBUTION")

    disc_counts = (
        promos.groupby(["promo_type", "discount_value"])
        .size().reset_index(name="count")
    )
    colors_map = {"percentage": COLORS["primary"], "fixed": COLORS["warning"]}

    fig = go.Figure()
    for ptype in disc_counts["promo_type"].unique():
        sub = disc_counts[disc_counts["promo_type"] == ptype]
        fig.add_trace(go.Bar(
            name=ptype,
            x=sub["discount_value"].astype(str),
            y=sub["count"],
            marker=dict(color=colors_map.get(ptype, COLORS["info"])),
            text=sub["count"],
            textposition="outside",
            textfont=dict(color=COLORS["text_hi"]),
        ))
    apply_theme(fig, height=340, title="Promos by type & discount value")
    fig.update_layout(barmode="group",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(title="Discount value (% or VND)")
    fig.update_yaxes(title="Number of promos")
    st.plotly_chart(fig, use_container_width=True)

    # Compute fixed vs total discount share for the actionable insight
    _fixed_promo_ids = set(promos.loc[promos["promo_type"] == "fixed", "promo_id"])
    _fixed_orders = items[items["promo_id"].isin(_fixed_promo_ids)]
    _fixed_discount_total = _fixed_orders["discount_amount"].sum()
    _total_discount_all = items["discount_amount"].sum()
    _fixed_share = _fixed_discount_total / max(_total_discount_all, 1) * 100

    st.markdown(
        f"""
        <div class="insight-box warning" style="margin-top:8px">
          <span class="insight-title" style="color:{COLORS['warning']}">DISCOUNT ANOMALY</span>
          Fixed discount = <strong>50 VND</strong> on items priced 1,000–100,000 VND.
          That's <strong>0.05%–5%</strong> effective discount — economically negligible.
          Across all order items, fixed promos account for only
          <strong>{_fixed_share:.1f}%</strong> of total discount value.
          Recommendation: convert fixed-VND promos to percentage promos or eliminate them
          to reduce catalogue complexity with <em>zero</em> revenue impact.
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    render_section_label("PROMO CALENDAR · GANTT UNIFORMITY")

    promos_sorted = promos.sort_values("start_date").reset_index(drop=True)
    promos_sorted["label"] = promos_sorted["promo_name"].str[:28]

    years_avail = sorted(promos_sorted["start_date"].dt.year.unique())
    year_range = st.select_slider(
        "Filter promo calendar by year range",
        options=years_avail,
        value=(years_avail[0], years_avail[-1]),
        key="promo_year_range",
    )
    filtered_promos = promos_sorted[
        (promos_sorted["start_date"].dt.year >= year_range[0]) &
        (promos_sorted["start_date"].dt.year <= year_range[1])
    ]
    gantt_height = max(300, len(filtered_promos) * 24)

    fig2 = go.Figure()
    for _, row in filtered_promos.iterrows():
        color = COLORS["primary"] if row["promo_type"] == "percentage" else COLORS["warning"]
        fig2.add_trace(go.Bar(
            x=[row["duration_days"]],
            y=[row["label"]],
            base=[row["start_date"]],
            orientation="h",
            marker=dict(color=color, opacity=0.75,
                        line=dict(color=COLORS["bg_deep"], width=0.5)),
            hovertemplate=(
                f"<b>{row['promo_name']}</b><br>"
                f"Type: {row['promo_type']} — {row['discount_value']}<br>"
                f"{row['start_date'].date()} → {row['end_date'].date()}<br>"
                f"Duration: {row['duration_days']} days<extra></extra>"
            ),
            showlegend=False,
        ))
    apply_theme(fig2, height=gantt_height, title="All promotions — green=%, yellow=fixed | 6-4-6-4 cadence visible")
    fig2.update_layout(barmode="stack", margin=dict(l=220, r=30))
    fig2.update_xaxes(type="date", title="")
    fig2.update_yaxes(autorange="reversed", title="", tickfont=dict(size=11))
    st.plotly_chart(fig2, use_container_width=True)


st.markdown("<div style='margin: 24px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Row 2: Revenue lift on promo days vs non-promo days
# ---------------------------------------------------------------------------
render_section_label("PROMO LIFT · REVENUE ON vs OFF PROMO DAYS")

abt_train = abt[abt["date"] <= TRAIN_CUTOFF].copy()
abt_train["promo_day"] = abt_train["any_pct_promo"] | abt_train["any_fixed_promo"]
abt_train["promo_type_label"] = abt_train.apply(
    lambda r: "Percentage promo"
    if r["any_pct_promo"] and not r["any_fixed_promo"]
    else "Fixed promo"
    if r["any_fixed_promo"] and not r["any_pct_promo"]
    else "Both active"
    if r["any_pct_promo"] and r["any_fixed_promo"]
    else "No promo",
    axis=1,
)

lift_stats = (
    abt_train.groupby("promo_type_label")["Revenue"]
    .agg(["mean", "median", "count"])
    .reset_index()
)
lift_stats.columns = ["label", "avg_rev", "med_rev", "days"]
lift_stats = lift_stats.sort_values("avg_rev", ascending=False)

no_promo_avg = float(
    lift_stats.loc[lift_stats.label == "No promo", "avg_rev"].iloc[0]
    if "No promo" in lift_stats["label"].values else lift_stats["avg_rev"].min()
)
lift_stats["lift_pct"] = (lift_stats["avg_rev"] / no_promo_avg - 1) * 100

color_map = {
    "Percentage promo": COLORS["primary"],
    "Fixed promo": COLORS["warning"],
    "Both active": COLORS["glow"],
    "No promo": COLORS["text_dim"],
}

c1, c2, c3 = st.columns([1.4, 1.4, 1], gap="medium")

with c1:
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=lift_stats["label"],
        y=lift_stats["avg_rev"] / 1e6,
        marker=dict(color=[color_map.get(l, COLORS["info"]) for l in lift_stats["label"]],
                    line=dict(width=0)),
        text=[f"{v/1e6:.2f}M VND" for v in lift_stats["avg_rev"]],
        textposition="outside",
        textfont=dict(color=COLORS["text_hi"], size=11),
    ))
    apply_theme(fig3, height=340, title="Avg daily revenue by promo status")
    fig3.update_yaxes(title="Avg daily rev (M VND)")
    fig3.update_xaxes(title="")
    st.plotly_chart(fig3, use_container_width=True)

with c2:
    fig4 = go.Figure(go.Bar(
        x=lift_stats["label"],
        y=lift_stats["lift_pct"],
        marker=dict(
            color=[COLORS["primary"] if v >= 0 else COLORS["danger"]
                   for v in lift_stats["lift_pct"]],
            line=dict(width=0),
        ),
        text=[f"{v:+.1f}%" for v in lift_stats["lift_pct"]],
        textposition="outside",
        textfont=dict(color=COLORS["text_hi"], size=11),
    ))
    apply_theme(fig4, height=340, title="Revenue lift vs no-promo baseline")
    fig4.update_yaxes(title="Lift (%)", ticksuffix="%")
    fig4.update_xaxes(title="")
    fig4.add_hline(y=0, line_dash="dot", line_color=COLORS["text_dim"])
    st.plotly_chart(fig4, use_container_width=True)

with c3:
    for _, row in lift_stats.iterrows():
        border = color_map.get(row["label"], COLORS["info"])
        st.markdown(
            f"""
            <div class="kpi-card" style="border-left:3px solid {border}; margin-bottom:10px; min-height:0; padding:10px 12px">
              <div class="kpi-label" style="color:{border}">{row['label'].upper()}</div>
              <div class="kpi-value" style="font-size:20px">{fmt_vnd(row['avg_rev'])}/day</div>
              <div style="color:{COLORS['text_dim']};font-size:11px;margin-top:3px">
                {int(row['days'])} days · lift {row['lift_pct']:+.1f}%
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<div style='margin: 20px 0'></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Row 3: applicable_category gap
# ---------------------------------------------------------------------------
render_section_label("TARGETING GAP · applicable_category")

null_pct = promos["applicable_category"].isna().mean() * 100
fig5 = go.Figure(go.Pie(
    labels=["Not targeted (null)", "Category-targeted"],
    values=[promos["applicable_category"].isna().sum(),
            promos["applicable_category"].notna().sum()],
    hole=0.55,
    marker=dict(colors=[COLORS["warning"], COLORS["primary"]]),
    textinfo="label+percent",
    textfont=dict(color=COLORS["text_hi"], size=13),
))
apply_theme(fig5, height=280, title=None)
fig5.update_layout(
    showlegend=False,
    annotations=[dict(
        text=f"<b>{null_pct:.0f}%</b><br><span style='font-size:11px'>untargeted</span>",
        x=0.5, y=0.5, font_size=22, showarrow=False,
        font=dict(color=COLORS["warning"]),
    )],
)

col_pie, col_ins = st.columns([1, 2], gap="medium")
with col_pie:
    st.plotly_chart(fig5, use_container_width=True)
with col_ins:
    render_insight(
        title="Promo targeting nearly absent:",
        level="warning",
        body=(
            f"<strong>{null_pct:.0f}%</strong> of promotions have "
            "<strong>applicable_category = null</strong> — meaning every promo applies "
            "site-wide regardless of category. Combined with negligible fixed-promo "
            "discounts (50 VND), the promo calendar operates as a <em>traffic signal</em> "
            "rather than a margin-management tool. "
            "Category-targeted promotions for Casual/GenZ during off-peak months "
            "represent an <strong>untapped conversion lever</strong>."
        ),
    )


render_insight(
    title="Key promo finding:",
    level="info",
    body=(
        "Percentage promos generate measurable lift vs the no-promo baseline. "
        "The 6-4-6-4 week cadence (6 weeks on, 4 weeks off) is synthetic and uniform — "
        "aligning promo windows with the Apr–Jun seasonal peak would amplify ROI. "
        "promo_id_2 (stackable second promo) appears in only <strong>0.03%</strong> of "
        "order items — stackability is effectively unused and should not drive strategy."
    ),
)
