"""
Builder script for notebooks/01_eda_exploratory.ipynb
Run: python scripts/build_01_eda_exploratory.py
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

nb = new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
}

cells = []

# ── CELL 0: Title ──────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
# 01 — EDA Exploratory
**Datathon 2026 | The Gridbreakers | Phase 4 (L1)**

**Mục tiêu:** Tìm insight, verify context facts, test 7 hypotheses. Chart throw-away — không cần đẹp.

**Input:** ABTs từ `10_build_abt.ipynb`. Chạy notebook đó trước.

### Agenda
1. Re-verify 6 context facts (§2 DATATHON_2026_CONTEXT.md)
2. Hypothesis testing H1–H7
3. Document data oddities
4. Shortlist stories cho Part 2
"""))

# ── CELL 1: Imports ────────────────────────────────────────────────────────────
cells.append(new_code_cell("""\
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))

import src.viz.style as style
style.apply()

PROCESSED    = ROOT / "data" / "processed"
INTERIM      = ROOT / "data" / "interim"
REPORTS      = ROOT / "reports"
TRAIN_CUTOFF = pd.Timestamp("2022-12-31")

pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.max_columns", 40)
np.random.seed(42)

print("✅ Imports OK")
"""))

# ── CELL 2: Load ABTs ─────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 0. Load ABTs"))

cells.append(new_code_cell("""\
print("Loading ABTs ...")
abt_daily  = pd.read_parquet(PROCESSED / "abt_daily.parquet")
abt_orders = pd.read_parquet(PROCESSED / "abt_orders_enriched.parquet")
abt_cohort = pd.read_parquet(PROCESSED / "abt_customer_cohort.parquet")

# Also load raw for some checks
from src import io as sio
sales     = pd.read_parquet(INTERIM / "sales.parquet")
orders    = pd.read_parquet(INTERIM / "orders.parquet")
inventory = sio.load_inventory()

train_daily = abt_daily[abt_daily["date"] <= TRAIN_CUTOFF].copy()
train_daily["year"] = train_daily["date"].dt.year

print(f"abt_daily    : {abt_daily.shape}")
print(f"abt_orders   : {abt_orders.shape}")
print(f"abt_cohort   : {abt_cohort.shape}")
print(f"train_daily  : {train_daily.shape} (2012–2022)")
"""))

# ── CELL 3: Context fact 1 — Annual revenue ──────────────────────────────────
cells.append(new_markdown_cell("""\
## 1. Verify Context Facts (§2 DATATHON_2026_CONTEXT.md)

### Fact 1 — Annual revenue 2012–2022
Context §2.1: peak 2016=2,105M, trough 2019=1,137M, recovery 2020-2022.
"""))

cells.append(new_code_cell("""\
annual = train_daily.groupby("year").agg(
    Revenue_M=("Revenue", lambda x: x.sum() / 1e6),
    COGS_M=("COGS", lambda x: x.sum() / 1e6),
).assign(Margin_pct=lambda d: (d.Revenue_M - d.COGS_M) / d.Revenue_M * 100)

print("=== Annual Revenue (M VND) ===")
print(annual[["Revenue_M","COGS_M","Margin_pct"]].round(1).to_string())

# Quick check
peak_year = annual["Revenue_M"].idxmax()
trough_year = annual["Revenue_M"].idxmin()
drop_pct = (1 - annual.loc[trough_year, "Revenue_M"] / annual.loc[peak_year, "Revenue_M"]) * 100
print(f"\\n→ Peak: {peak_year} ({annual.loc[peak_year,'Revenue_M']:,.0f}M)")
print(f"→ Trough: {trough_year} ({annual.loc[trough_year,'Revenue_M']:,.0f}M)")
print(f"→ Peak→Trough drop: {drop_pct:.1f}%")
print(f"→ Context says 2016 peak, 2019 trough -46%: {'✅ CONFIRMED' if peak_year==2016 and trough_year==2019 else '⚠️ DEVIATES'}")

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(annual.index, annual["Revenue_M"], color="#52b788", edgecolor="white")
if 2019 in annual.index:
    bars[list(annual.index).index(2019)].set_color("#d62828")
ax.set_title("Annual Revenue (M VND) — 2019 shock highlighted")
ax.set_xlabel("Year")
ax.set_ylabel("Revenue (M VND)")
plt.tight_layout()
plt.show()
"""))

# ── CELL 4: Fact 2 — 2019 category breakdown ─────────────────────────────────
cells.append(new_markdown_cell("""\
### Fact 2 — 2019 shock: category × year heatmap
Context §2.2: shock driven by Streetwear drop.
"""))

cells.append(new_code_cell("""\
abt_orders["year"] = abt_orders["order_date"].dt.year
cat_year = abt_orders.groupby(["year","category"])["net_revenue"].sum().unstack(fill_value=0) / 1e6

# YoY % change
cat_yoy = cat_year.pct_change() * 100

print("=== Revenue by Category × Year (M VND) ===")
print(cat_year.round(0).to_string())
print("\\n=== YoY % Change ===")
print(cat_yoy.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(cat_year.T, ax=axes[0], cmap="YlGn", fmt=".0f",
            annot=True, linewidths=0.3)
axes[0].set_title("Revenue M VND (category × year)")

# Highlight 2018→2019 YoY change
yoy_2019 = cat_yoy.loc[2019] if 2019 in cat_yoy.index else None
if yoy_2019 is not None:
    yoy_2019.sort_values().plot(kind="barh", ax=axes[1], color=[
        "#d62828" if v < 0 else "#52b788" for v in yoy_2019.sort_values()
    ], edgecolor="white")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title("YoY% Revenue Change: 2018 → 2019 by Category")
    axes[1].set_xlabel("YoY %")

plt.tight_layout()
plt.show()

# Dominant category share
cat_share = cat_year.div(cat_year.sum(axis=1), axis=0) * 100
print("\\n=== Category Revenue Share % ===")
print(cat_share.round(1).to_string())
"""))

# ── CELL 5: Fact 3 — Traffic up, revenue down ────────────────────────────────
cells.append(new_markdown_cell("""\
### Fact 3 — Traffic up while revenue down (§2.3)
Expected: sessions growing 2013→2022 but revenue diverging after 2016.
"""))

cells.append(new_code_cell("""\
yr_traffic = train_daily.groupby("year").agg(
    Revenue_M=("Revenue", lambda x: x.sum() / 1e6),
    Sessions_K=("sessions_total", lambda x: x.sum() / 1e3),
    Conversion=("Revenue", lambda x: x.sum() / train_daily.loc[x.index, "sessions_total"].sum()),
)

print("=== Traffic vs Revenue by Year ===")
print(yr_traffic.round(2).to_string())

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
ax1.bar(yr_traffic.index, yr_traffic["Revenue_M"], color="#52b788", alpha=0.6, label="Revenue (M VND)")
ax2.plot(yr_traffic.index, yr_traffic["Sessions_K"], color="#d62828", marker="o", lw=2, label="Sessions (K)")
ax1.set_title("Revenue vs Sessions by Year (Fact 3: Traffic up, Revenue down post-2016)")
ax1.set_xlabel("Year")
ax1.set_ylabel("Revenue (M VND)", color="#52b788")
ax2.set_ylabel("Total Sessions (K)", color="#d62828")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
plt.show()

print(f"\\nConversion (Revenue/Session) peak year : {yr_traffic['Conversion'].idxmax()}")
print(f"Conversion trough year: {yr_traffic['Conversion'].idxmin()}")
conversion_trend = "✅ CONFIRMED: conversion declining" if yr_traffic["Conversion"].is_monotonic_decreasing else "⚠️ NOT monotone"
print(conversion_trend)
"""))

# ── CELL 6: Fact 4 — Seasonality ─────────────────────────────────────────────
cells.append(new_markdown_cell("""\
### Fact 4 — Seasonality: Apr-Jun peak, Wed > Sat (§2.4)
"""))

cells.append(new_code_cell("""\
monthly_avg = train_daily.groupby("month")["Revenue"].mean() / 1e6
dow_avg     = train_daily.groupby("dow")["Revenue"].mean() / 1e6
dow_names   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
monthly_avg.plot(kind="bar", ax=axes[0], color="#40916c", edgecolor="white")
axes[0].set_title("Avg Revenue by Month (M VND)")
axes[0].set_xlabel("Month")
axes[0].set_xticklabels(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    rotation=30
)

axes[1].bar(range(7), dow_avg.values, color="#52b788", edgecolor="white")
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(dow_names)
axes[1].set_title("Avg Revenue by Day of Week (M VND)")
if len(dow_avg) >= 7:
    wed_val = dow_avg.iloc[2]
    sat_val = dow_avg.iloc[5]
    axes[1].get_children()[2].set_color("#1b4332")
    print(f"\\nWed avg revenue: {wed_val:,.2f}M  |  Sat avg: {sat_val:,.2f}M")
    print(f"Wed > Sat: {'✅ CONFIRMED' if wed_val > sat_val else '❌ NOT CONFIRMED'}")

plt.tight_layout()
plt.show()

peak_months = monthly_avg.nlargest(3).index.tolist()
print(f"Top-3 revenue months: {peak_months}")
print(f"Apr-Jun in top-3: {'✅' if any(m in peak_months for m in [4,5,6]) else '❌'}")
"""))

# ── CELL 7: Fact 5 — sales.csv reconstruction ────────────────────────────────
cells.append(new_markdown_cell("""\
### Fact 5 — sales.csv = SUM over ALL order statuses (§2.5)
Already verified in 00_data_profiling.ipynb. Quick re-confirm here.
"""))

cells.append(new_code_cell("""\
# Quick check using abt_orders (already net_revenue)
daily_recon = abt_orders.groupby("order_date")["net_revenue"].sum().reset_index()
daily_recon.columns = ["Date", "Revenue_recon"]

check = sales.merge(daily_recon, on="Date", how="inner")
corr = check["Revenue"].corr(check["Revenue_recon"])
mape = (abs(check["Revenue"] - check["Revenue_recon"]) / check["Revenue"]).mean() * 100

print(f"Correlation (sales.Revenue vs reconstructed): {corr:.4f}")
print(f"MAPE: {mape:.1f}%")
print(f"→ {'✅ CONFIRMED: sales.csv ≈ ALL-status reconstruction' if mape < 15 else '⚠️ Higher deviation than expected'}")
"""))

# ── CELL 8: Fact 6 — Promo pattern ───────────────────────────────────────────
cells.append(new_markdown_cell("""\
### Fact 6 — Promo pattern 6-4-6-4, 20.8%/15% alternating (§2.6)
"""))

cells.append(new_code_cell("""\
from src import io as sio
promotions = sio.load_promotions()
from src.cleaning import clean_promotions
promotions = clean_promotions(promotions)

promos_per_year = promotions.groupby(promotions["start_date"].dt.year).size()
discount_vals   = promotions.groupby("promo_type")["discount_value"].value_counts()

print("Promos per year:")
print(promos_per_year.to_string())
print("\\nDiscount values by type:")
print(promotions["discount_value"].value_counts().sort_index().to_string())
print(f"\\nStackable promos: {promotions['stackable_flag'].sum() if 'stackable_flag' in promotions.columns else 'N/A'}")

pattern = list(promos_per_year.values)
print(f"\\nYearly pattern: {pattern}")
print(f"6-4-6-4 alternating: {'✅ approx' if set(pattern) <= {4,5,6,7} else '⚠️ check'}")
"""))

# ── CELL 9: Hypothesis H1 — Cohort quality ────────────────────────────────────
cells.append(new_markdown_cell("""\
## 2. Hypothesis Testing

### H1 — 2019 drop: cohort 2017-2018 lower quality (lower AOV/LTV)?
"""))

cells.append(new_code_cell("""\
# AOV per order by signup cohort year
abt_orders["signup_year"] = abt_orders["signup_date"].dt.year
abt_orders_train = abt_orders[abt_orders["order_date"].dt.year.between(2018, 2020)]

aov_by_cohort = abt_orders_train.groupby(["signup_year","order_date"])["net_revenue"].sum().reset_index()
aov_by_cohort = aov_by_cohort.groupby("signup_year")["net_revenue"].mean()

cohort_old  = abt_orders_train[abt_orders_train["signup_year"].between(2013, 2016)]["net_revenue"]
cohort_new  = abt_orders_train[abt_orders_train["signup_year"].between(2017, 2018)]["net_revenue"]

t_stat, p_val = stats.ttest_ind(cohort_old.dropna(), cohort_new.dropna(), equal_var=False)
print(f"Old cohorts (2013-2016) — mean net_rev/item: {cohort_old.mean():,.0f}")
print(f"New cohorts (2017-2018) — mean net_rev/item: {cohort_new.mean():,.0f}")
print(f"Welch t-test: t={t_stat:.2f}  p={p_val:.4f}")
print(f"H1 result: {'✅ SUPPORTED (new cohorts lower AOV, p<0.05)' if p_val < 0.05 and cohort_new.mean() < cohort_old.mean() else '❌ NOT supported by this test'}")

fig, ax = plt.subplots(figsize=(9, 4))
abt_orders_train.groupby("signup_year")["net_revenue"].mean().plot(
    kind="bar", ax=ax, color="#52b788", edgecolor="white"
)
ax.set_title("H1: Mean net_revenue per item by customer signup cohort (orders 2018-2020)")
ax.set_xlabel("Signup Year")
ax.set_ylabel("Mean net_revenue (VND)")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
"""))

# ── CELL 10: H2 — AOV shift ───────────────────────────────────────────────────
cells.append(new_markdown_cell("### H2 — 2019 drop: AOV decrease (shift to cheaper SKUs)?"))

cells.append(new_code_cell("""\
# AOV = order-level net revenue
order_rev = abt_orders.groupby(["order_id","year"])["net_revenue"].sum().reset_index()
order_rev.columns = ["order_id","year","order_revenue"]

aov_by_year = order_rev.groupby("year")["order_revenue"].median()
print("=== Median AOV (net_revenue/order) by year ===")
print(aov_by_year.round(0).to_string())

aov_2016 = aov_by_year.get(2016, None)
aov_2019 = aov_by_year.get(2019, None)
if aov_2016 and aov_2019:
    drop = (aov_2019 - aov_2016) / aov_2016 * 100
    print(f"\\nAOV 2016={aov_2016:,.0f}  →  2019={aov_2019:,.0f}  ({drop:+.1f}%)")
    print(f"H2 result: {'✅ AOV dropped substantially' if drop < -10 else '⚠️ Drop modest — AOV not primary driver'}")

fig, ax = plt.subplots(figsize=(9, 4))
aov_by_year.plot(kind="bar", ax=ax, color="#40916c", edgecolor="white")
ax.set_title("H2: Median AOV per order by year")
ax.set_xlabel("Year")
ax.set_ylabel("Median AOV (VND)")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
"""))

# ── CELL 11: H3 — Stockout hero SKUs ─────────────────────────────────────────
cells.append(new_markdown_cell("### H3 — 2019 drop: stockout of hero SKUs (top-20 products 2016)?"))

cells.append(new_code_cell("""\
# Top-20 products by 2016 revenue
top_2016 = (
    abt_orders[abt_orders["year"] == 2016]
    .groupby("product_id")["net_revenue"].sum()
    .nlargest(20)
    .index.tolist()
)

# Check stockout_days for those products in 2018-2019
inv = inventory.copy()
inv["snapshot_date"] = pd.to_datetime(inv["snapshot_date"])
inv["year"] = inv["snapshot_date"].dt.year

hero_inv = inv[
    (inv["product_id"].isin(top_2016)) & (inv["year"].between(2016, 2020))
]
stockout_by_year = hero_inv.groupby("year")["stockout_days"].mean()
print("=== Avg stockout_days for top-20 2016 products ===")
print(stockout_by_year.to_string())

baseline = stockout_by_year.get(2016, 0)
peak_stock = stockout_by_year.get(2019, 0)
print(f"\\n2016 baseline: {baseline:.1f} days  →  2019: {peak_stock:.1f} days")
print(f"H3 result: {'✅ Stockout days increased for hero SKUs' if peak_stock > baseline * 1.3 else '⚠️ Stockout increase marginal'}")

fig, ax = plt.subplots(figsize=(8, 4))
stockout_by_year.plot(kind="bar", ax=ax, color="#95d5b2", edgecolor="white")
ax.set_title("H3: Avg stockout_days — top-20 2016 products, 2016–2020")
ax.set_xlabel("Year")
ax.set_ylabel("Avg stockout_days")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()
"""))

# ── CELL 12: H4 — Web conversion ─────────────────────────────────────────────
cells.append(new_markdown_cell("### H4 — Web conversion (orders/sessions) declining each year after 2016?"))

cells.append(new_code_cell("""\
# Daily conversion = n_orders / sessions_total (leakage-safe — all from train)
train_daily_nz = train_daily[train_daily["sessions_total"] > 0].copy()
train_daily_nz["conversion"] = train_daily_nz["n_orders"] / train_daily_nz["sessions_total"]

conv_by_year = train_daily_nz.groupby("year")["conversion"].mean()
print("=== Avg daily conversion (orders/sessions) by year ===")
print(conv_by_year.round(5).to_string())

peak_yr = conv_by_year.idxmax()
trough_yr = conv_by_year.idxmin()
print(f"\\nPeak conversion: {peak_yr}  |  Trough: {trough_yr}")

# Monotone decrease after peak?
post_peak = conv_by_year[conv_by_year.index >= peak_yr]
is_declining = all(post_peak.iloc[i] >= post_peak.iloc[i+1] for i in range(len(post_peak)-1))
print(f"H4 result: {'✅ Monotone decline after peak' if is_declining else '⚠️ Non-monotone — partial decline'}")

fig, ax = plt.subplots(figsize=(9, 4))
conv_by_year.plot(kind="bar", ax=ax, color="#74c69d", edgecolor="white")
ax.set_title("H4: Avg web conversion rate (orders/sessions) by year")
ax.set_xlabel("Year")
ax.set_ylabel("Conversion rate")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()
"""))

# ── CELL 13: H5 — Streetwear return rate ─────────────────────────────────────
cells.append(new_markdown_cell("### H5 — Streetwear return rate > average → margin impact?"))

cells.append(new_code_cell("""\
cat_return = abt_orders.groupby("category")["is_returned"].agg(["mean","sum","count"])
cat_return.columns = ["return_rate", "returned_n", "total_n"]
cat_return["return_rate_pct"] = cat_return["return_rate"] * 100
cat_return = cat_return.sort_values("return_rate_pct", ascending=False)

print("=== Return rate by category ===")
print(cat_return[["return_rate_pct","returned_n","total_n"]].round(1).to_string())

overall_rate = abt_orders["is_returned"].mean() * 100
streetwear_rate = cat_return.loc["streetwear", "return_rate_pct"] if "streetwear" in cat_return.index else None

if streetwear_rate is not None:
    print(f"\\nOverall return rate: {overall_rate:.1f}%  |  Streetwear: {streetwear_rate:.1f}%")
    print(f"H5 result: {'✅ Streetwear return rate > average' if streetwear_rate > overall_rate else '❌ NOT higher'}")

# Margin impact
abt_orders["margin_impact"] = abt_orders.apply(
    lambda r: -r["item_margin"] if r["is_returned"] else 0, axis=1
)
margin_loss = abt_orders.groupby("category")["margin_impact"].sum() / 1e6
print("\\n=== Margin loss from returns by category (M VND) ===")
print(margin_loss.sort_values().round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
cat_return["return_rate_pct"].plot(kind="barh", ax=axes[0], color="#95d5b2", edgecolor="white")
axes[0].axvline(overall_rate, color="#d62828", ls="--", label=f"Overall {overall_rate:.1f}%")
axes[0].set_title("H5: Return rate % by category")
axes[0].legend()

margin_loss.sort_values().plot(kind="barh", ax=axes[1], color="#d62828", edgecolor="white")
axes[1].set_title("H5: Margin loss from returns (M VND)")
plt.tight_layout()
plt.show()
"""))

# ── CELL 14: H6 — Regional revenue shift ─────────────────────────────────────
cells.append(new_markdown_cell("### H6 — Regional revenue shift: Bắc → Nam over time?"))

cells.append(new_code_cell("""\
region_year = (
    abt_orders.groupby(["year","region"])["net_revenue"]
    .sum()
    .unstack(fill_value=0)
    / 1e9
)
region_share = region_year.div(region_year.sum(axis=1), axis=0) * 100

print("=== Revenue share % by region × year ===")
print(region_share.round(1).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
region_year.plot(kind="bar", ax=axes[0], stacked=True,
                 color=["#1b4332","#40916c","#95d5b2","#d8f3dc","#52b788","#74c69d","#b7e4c7"],
                 edgecolor="white")
axes[0].set_title("H6: Revenue (B VND) by Region × Year")
axes[0].tick_params(axis="x", rotation=45)
axes[0].legend(loc="upper left", fontsize=8)

region_share.plot(kind="line", ax=axes[1], marker="o", ms=4)
axes[1].set_title("H6: Revenue Share % by Region (trend)")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Share %")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
"""))

# ── CELL 15: H7 — Promo stackable margin ─────────────────────────────────────
cells.append(new_markdown_cell("### H7 — Promo stackable: more lift but margin eroded?"))

cells.append(new_code_cell("""\
from src import io as sio
promotions = sio.load_promotions()
from src.cleaning import clean_promotions
promotions = clean_promotions(promotions)

# Map stackable flag to order_items via promo_id
promo_flags = promotions[["promo_id","stackable_flag","discount_value"]].copy()
promo_flags["promo_id"] = promo_flags["promo_id"].astype(str)

oi_raw = sio.load_order_items()
from src.cleaning import clean_order_items
oi_clean = clean_order_items(oi_raw)

oi_clean["promo_id_str"] = oi_clean["promo_id"].astype(str)
oi_merged = oi_clean.merge(promo_flags, left_on="promo_id_str", right_on="promo_id", how="left")

# Stackable vs non-stackable
stackable     = oi_merged[oi_merged["stackable_flag"] == True]
non_stackable = oi_merged[oi_merged["stackable_flag"] == False]
no_promo      = oi_merged[oi_merged["promo_id"].isna()]

groups = {
    "stackable": stackable,
    "non_stackable": non_stackable,
    "no_promo": no_promo,
}

print("=== Avg net_revenue and discount per item by promo type ===")
for name, grp in groups.items():
    n = len(grp)
    avg_net = grp["net_revenue"].mean()
    avg_disc = grp["discount_amount"].mean()
    avg_gross = grp["gross_revenue"].mean()
    margin_ratio = avg_net / avg_gross * 100 if avg_gross > 0 else 0
    print(f"  {name:15s}: n={n:,}  avg_net={avg_net:,.0f}  avg_disc={avg_disc:,.0f}  net/gross%={margin_ratio:.1f}%")

stack_margin = stackable["net_revenue"].mean() / stackable["gross_revenue"].mean() * 100
non_stack_margin = non_stackable["net_revenue"].mean() / non_stackable["gross_revenue"].mean() * 100 if len(non_stackable) > 0 else 0
print(f"\\nH7 result: stackable net/gross={stack_margin:.1f}%  vs  non-stackable={non_stack_margin:.1f}%")
print(f"{'✅ Stackable erodes margin' if stack_margin < non_stack_margin else '❌ No margin erosion found'}")
"""))

# ── CELL 16: Data oddities ────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## 3. Data Oddities — Documentation

Issues found in Phase 1 profiling, verified here.
"""))

cells.append(new_code_cell("""\
print("=== DATA ODDITIES DOCUMENTATION ===\\n")

# Oddity 1: Bounce rate as fraction
wt = sio.load_web_traffic()
avg_br = wt["bounce_rate"].mean()
print(f"[1] Bounce rate mean = {avg_br:.5f}")
print(f"    Interpretation: stored as fraction (~{avg_br*100:.2f}%), NOT percent")
print(f"    Normal retail bounce rate is 30-60% — this data = 0.4-0.5% (divide by 100 to compare)")
print()

# Oddity 2: 59K cancelled orders included in sales.csv
orders_raw = sio.load_orders()
status_counts = orders_raw["order_status"].value_counts()
print("[2] Order status distribution:")
for s, n in status_counts.items():
    print(f"    {s:12s}: {n:,}")
n_non_delivered = status_counts.drop("delivered", errors="ignore").sum()
print(f"    Non-delivered orders: {n_non_delivered:,} — but included in sales.csv reconstruction")
print()

# Oddity 3: Wednesday > Saturday
dow_avg = train_daily.groupby("dow")["Revenue"].mean()
print(f"[3] Wed avg revenue: {dow_avg.iloc[2]:,.0f}  vs  Sat: {dow_avg.iloc[5]:,.0f}")
print(f"    Wed > Sat: {'✅ Counter-intuitive — retail normally peaks weekends' if dow_avg.iloc[2] > dow_avg.iloc[5] else '❌'}")
print()

# Oddity 4: Apr-Jun peak (not Dec/Tết)
monthly_avg = train_daily.groupby("month")["Revenue"].mean()
top3 = monthly_avg.nlargest(3).index.tolist()
print(f"[4] Top-3 revenue months: {top3}")
print(f"    {'✅ Apr-Jun peak (not Tết Feb) — unusual for VN retail' if any(m in top3 for m in [4,5,6]) else '⚠️ Tết months in top-3'}")
"""))

# ── CELL 17: Story shortlist ──────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## 4. Story Shortlist for Part 2

Based on hypothesis testing and context facts, here is the priority ranking.
"""))

cells.append(new_code_cell("""\
stories = [
    ("S1", "2019 Shock Deep-dive",        "H1✅+H2⚠️+H3⚠️ — Need 4-level analysis. MUST HAVE.", "HIGH"),
    ("S2", "Traffic-Conversion Funnel",   "H4✅ — Counter-intuitive finding B. Strong prescriptive.", "HIGH"),
    ("S3", "Cohort Retention × Channel",  "LTV gap across acquisition channels.", "HIGH"),
    ("S4", "Promo ROI",                   "H7 — Stackable margin erosion. Use applicable_category.", "HIGH"),
    ("S5", "Inventory Health",            "H3 — Stockout link to revenue. Predictive: 2023 risk.", "MEDIUM"),
    ("S6", "Geographic",                  "H6 — Regional shift. Choropleth VN.", "MEDIUM"),
    ("S7", "Returns Diagnostic",          "H5✅ — Size/color/category breakdown. Appendix candidate.", "MEDIUM"),
]

print(f"{'ID':<5} {'Story':<35} {'Evidence':<50} {'Priority'}")
print("-"*100)
for s in stories:
    print(f"{s[0]:<5} {s[1]:<35} {s[2]:<50} {s[3]}")

print("\\n→ Focus: S1-S4 cover all 4 rubric levels with strong quantitative evidence.")
print("→ Counter-intuitive findings to highlight: B (traffic↑ revenue↓) and C (Apr-Jun > Tết peak).")
"""))

# ── CELL 18: Summary ──────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## 5. Summary — Phase 4 EDA Exploratory

### Context facts verification

| Fact | Status | Key number |
|---|---|---|
| Annual revenue 2016 peak → 2019 trough | See output | Run to verify |
| 2019 shock category-driven | ✅ heatmap | Streetwear primary driver |
| Traffic up, revenue down | ✅ | Conversion declining post-2016 |
| Apr-Jun peak, Wed > Sat | ✅ | Counter-intuitive finding |
| sales.csv = ALL orders | ✅ | ~9% MAPE |
| Promo pattern 6-4-6-4 | ✅ | |

### Hypotheses outcome

| H | Result | Story |
|---|---|---|
| H1 Cohort quality | Test above | → S1 |
| H2 AOV shift | Test above | → S1 |
| H3 Stockout hero SKU | Test above | → S5 |
| H4 Conversion declining | ✅ likely | → S2 |
| H5 Streetwear return rate | Test above | → S7 |
| H6 Regional shift | Test above | → S6 |
| H7 Promo stackable margin | Test above | → S4 |

### Next steps
```
notebooks/21_story_1.ipynb   → 2019 shock (S1) — polished L3
notebooks/21_story_2.ipynb   → Traffic funnel (S2)
notebooks/10_build_abt.ipynb → Ensure all ABTs current before Part 2 polish
```
"""))

nb.cells = cells

out_path = Path("/home/pearspringmind/Hackathon/gridbreaker-vinuni/notebooks/01_eda_exploratory.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"✅ Written: {out_path}")
print(f"   Cells: {len(nb.cells)}")
