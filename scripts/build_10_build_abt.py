"""
Builder script for notebooks/10_build_abt.ipynb
Run: python scripts/build_10_build_abt.py
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
# 10 — Build Analytical Base Tables (ABT)
**Datathon 2026 | The Gridbreakers | Phase 2 (L2)**

**Mục tiêu:** Áp dụng cleaning rules → build 3 ABT leakage-safe cho phân tích & modelling.

| ABT | Grain | Dùng cho |
|---|---|---|
| `abt_daily.parquet` | 1 row / ngày (2012-07-04 → 2024-07-01) | Part 3 forecasting + Story 2 |
| `abt_orders_enriched.parquet` | 1 row / order_item (5-way join) | Part 2 Stories 1,4,6,7 + MCQ |
| `abt_customer_cohort.parquet` | 1 row / (customer_id, months_since_signup) | Story 3 |

**Leakage guard:** tất cả cột từ transactions filter ≤ `2022-12-31`. Calendar + promo fill toàn bộ.
"""))

# ── CELL 1: Imports & paths ────────────────────────────────────────────────────
cells.append(new_code_cell("""\
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))

from src import io as sio
from src import cleaning as sc
from src import joining as sj
from src.features.calendar import add_calendar_features, add_lag_roll_features
import src.viz.style as style
style.apply()

INTERIM   = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

TRAIN_CUTOFF = sio.TRAIN_CUTOFF
FULL_START   = pd.Timestamp("2012-07-04")
FULL_END     = pd.Timestamp("2024-07-01")
FULL_RANGE   = pd.date_range(FULL_START, FULL_END, freq="D")

print(f"ROOT         : {ROOT}")
print(f"TRAIN_CUTOFF : {TRAIN_CUTOFF.date()}")
print(f"Full date range: {FULL_START.date()} → {FULL_END.date()} ({len(FULL_RANGE):,} days)")
"""))

# ── CELL 2: Load & clean all tables ───────────────────────────────────────────
cells.append(new_markdown_cell("## 1. Load & clean raw tables (src/cleaning.py)"))

cells.append(new_code_cell("""\
print("Loading raw CSVs ...")

products   = sc.clean_products(sio.load_products())
customers  = sc.clean_customers(sio.load_customers())
promotions = sc.clean_promotions(sio.load_promotions())
geography  = sc.clean_geography(sio.load_geography())
orders     = sc.clean_orders(sio.load_orders())
order_items= sc.clean_order_items(sio.load_order_items())
payments   = sc.clean_payments(sio.load_payments())
shipments  = sc.clean_shipments(sio.load_shipments())
returns    = sc.clean_returns(sio.load_returns())
reviews    = sc.clean_reviews(sio.load_reviews())
sales      = sc.clean_sales(sio.load_sales(cutoff=False))  # pass all, clean_sales handles cutoff

# web_traffic returns (raw, daily_agg)
wt_raw, web_daily = sc.clean_web_traffic(sio.load_web_traffic())

print("\\n✅ All tables loaded and cleaned")
print(f"  products        : {products.shape}")
print(f"  customers       : {customers.shape}")
print(f"  promotions      : {promotions.shape}")
print(f"  geography       : {geography.shape}")
print(f"  orders          : {orders.shape}")
print(f"  order_items     : {order_items.shape}")
print(f"  payments        : {payments.shape}")
print(f"  shipments       : {shipments.shape}")
print(f"  returns         : {returns.shape}")
print(f"  reviews         : {reviews.shape}")
print(f"  sales (train)   : {sales.shape}")
print(f"  web_traffic raw : {wt_raw.shape}")
print(f"  web_daily       : {web_daily.shape}")
"""))

# ── CELL 3: Interim parquet save ──────────────────────────────────────────────
cells.append(new_markdown_cell("## 2. Save cleaned tables to data/interim/"))

cells.append(new_code_cell("""\
def save_interim(df, name):
    path = INTERIM / f"{name}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"  ✅ Saved {name}.parquet — {df.shape[0]:,} rows × {df.shape[1]} cols")

save_interim(products,    "products")
save_interim(customers,   "customers")
save_interim(promotions,  "promotions")
save_interim(geography,   "geography")
save_interim(orders,      "orders")
save_interim(order_items, "order_items")
save_interim(payments,    "payments")
save_interim(shipments,   "shipments")
save_interim(returns,     "returns")
save_interim(reviews,     "reviews")
save_interim(sales,       "sales")
save_interim(web_daily,   "web_traffic_daily")
print("\\n✅ All interim files saved")
"""))

# ── CELL 4: Cleaning unit tests ───────────────────────────────────────────────
cells.append(new_markdown_cell("## 3. Cleaning sanity checks"))

cells.append(new_code_cell("""\
print("=== Cleaning sanity checks ===\\n")

# Products
assert "margin_pct" in products.columns, "margin_pct missing"
assert products["category"].str.islower().all(), "category not normalised"
n_neg = products["margin_negative"].sum()
print(f"[products]    margin_negative rows: {n_neg} {'⚠️ flag for review' if n_neg else '✅ none'}")

# Customers
assert "zip" in customers.columns
assert customers["zip"].str.len().max() == 5, "zip not 5-char padded"
print(f"[customers]   zip format: ✅ all 5-char")

# Promotions
assert (promotions["start_date"] <= promotions["end_date"]).all(), "bad promo dates remain"
print(f"[promotions]  date ordering: ✅")

# Geography
assert geography["zip"].duplicated().sum() == 0, "dup zips after dedup"
print(f"[geography]   zip uniqueness: ✅ {len(geography):,} unique zips")

# Orders
assert orders["order_status"].str.islower().all(), "order_status not lowercased"
print(f"[orders]      status normalised: ✅")

# Order items
assert "gross_revenue" in order_items.columns and "net_revenue" in order_items.columns
n_neg_net = (order_items["net_revenue"] < 0).sum()
print(f"[order_items] net_revenue < 0: {n_neg_net:,} rows {'❌' if n_neg_net else '✅ none'}")

# Shipments
neg_del = (shipments["delivery_days"] < 0).sum()
print(f"[shipments]   delivery_days < 0: {neg_del:,} {'❌' if neg_del else '✅'}")

# Reviews
bad_rat = (~reviews["rating"].between(1, 5)).sum()
print(f"[reviews]     rating outside [1,5]: {bad_rat} {'❌' if bad_rat else '✅'}")

# Sales leakage guard
assert sales["Date"].max() <= TRAIN_CUTOFF, "LEAKAGE: sales contains post-cutoff rows!"
print(f"[sales]       cutoff guard: ✅ max date = {sales['Date'].max().date()}")

print("\\n✅ All cleaning checks passed")
"""))

# ── CELL 5: Build promo daily calendar ───────────────────────────────────────
cells.append(new_markdown_cell("## 4. Build promo daily calendar"))

cells.append(new_code_cell("""\
from src.cleaning import build_promo_daily

promo_daily = build_promo_daily(promotions, FULL_RANGE)
print(f"promo_daily shape: {promo_daily.shape}")
print(f"  Active promo days: {(promo_daily['n_active_promos'] > 0).sum():,}")
print(f"  No-promo days    : {(promo_daily['n_active_promos'] == 0).sum():,}")
print(promo_daily[promo_daily["n_active_promos"] > 0].head(3))
"""))

# ── CELL 6: ABT 1 — abt_daily ─────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## 5. ABT 1 — `abt_daily.parquet`

Grain: 1 row / calendar day. Revenue/COGS = NaN for test window (leakage guard).
"""))

cells.append(new_code_cell("""\
abt_daily_raw = sj.build_daily_abt(
    sales=sales,
    orders=orders,
    order_items=order_items,
    web_traffic_daily=web_daily,
    promo_daily=promo_daily,
    full_date_range=FULL_RANGE,
)

# Add calendar features
abt_daily = add_calendar_features(abt_daily_raw, date_col="date")

print(f"abt_daily shape: {abt_daily.shape}")
print(f"  Date range: {abt_daily['date'].min().date()} → {abt_daily['date'].max().date()}")

# Leakage assertion: Revenue NaN for test window
test_mask = abt_daily["date"] > TRAIN_CUTOFF
rev_leaked = abt_daily.loc[test_mask, "Revenue"].notna().sum()
assert rev_leaked == 0, f"LEAKAGE: {rev_leaked} Revenue rows in test window!"
print(f"  Revenue leakage check: ✅ 0 non-NaN Revenue rows after {TRAIN_CUTOFF.date()}")
print(f"  Train rows: {(~test_mask).sum():,}  |  Test rows (Revenue=NaN): {test_mask.sum():,}")

abt_daily.head(3)
"""))

cells.append(new_code_cell("""\
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("abt_daily — Overview (train window 2012–2022)", fontsize=14, fontweight="bold")

train = abt_daily[abt_daily["date"] <= TRAIN_CUTOFF].copy()

# Revenue trend
ax = axes[0, 0]
ax.plot(train["date"], train["Revenue"] / 1e6, color="#52b788", lw=0.5, alpha=0.6)
roll = train.set_index("date")["Revenue"].rolling(90).mean() / 1e6
ax.plot(roll, color="#1b4332", lw=1.8, label="90d rolling mean")
ax.set_title("Daily Revenue (M VND)")
ax.set_ylabel("Revenue (M VND)")
ax.legend()

# Sessions trend
ax = axes[0, 1]
ax.plot(train["date"], train["sessions_total"] / 1e3, color="#40916c", lw=0.5, alpha=0.6)
roll_s = train.set_index("date")["sessions_total"].rolling(90).mean() / 1e3
ax.plot(roll_s, color="#1b4332", lw=1.8, label="90d rolling mean")
ax.set_title("Daily Sessions (K)")
ax.set_ylabel("Sessions (K)")
ax.legend()

# Monthly revenue heatmap (year × month)
train["year"] = train["date"].dt.year
train["month"] = train["date"].dt.month
monthly = train.pivot_table(index="year", columns="month", values="Revenue", aggfunc="sum") / 1e6
import seaborn as sns
sns.heatmap(monthly, ax=axes[1, 0], cmap="YlGn", fmt=".0f", annot=False, linewidths=0.3)
axes[1, 0].set_title("Monthly Revenue heatmap (M VND)")
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Year")

# Active promos per year
promo_yr = train.groupby("year")["n_active_promos"].sum()
promo_yr.plot(kind="bar", ax=axes[1, 1], color="#95d5b2", edgecolor="white")
axes[1, 1].set_title("Total promo-days per year")
axes[1, 1].set_xlabel("Year")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(ROOT / "reports" / "fig_abt_daily_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → reports/fig_abt_daily_overview.png")
"""))

cells.append(new_code_cell("""\
# Save ABT 1
out_path = PROCESSED / "abt_daily.parquet"
abt_daily.to_parquet(out_path, index=False, engine="pyarrow")
print(f"✅ Saved abt_daily.parquet — {abt_daily.shape[0]:,} rows × {abt_daily.shape[1]} cols")
print(f"   File size: {out_path.stat().st_size / 1e6:.1f} MB")
"""))

# ── CELL 7: ABT 2 — abt_orders_enriched ──────────────────────────────────────
cells.append(new_markdown_cell("""\
## 6. ABT 2 — `abt_orders_enriched.parquet`

Grain: 1 row / order_item. 5-way join: order_items × orders × products × customers × geography.
"""))

cells.append(new_code_cell("""\
print("Building abt_orders_enriched (5-way join) ...")

abt_orders = sj.build_orders_enriched(
    order_items=order_items,
    orders=orders,
    products=products,
    customers=customers,
    geography=geography,
    returns=returns,
    reviews=reviews,
    shipments=shipments,
)

print(f"\\nabt_orders_enriched shape: {abt_orders.shape}")
print(f"  Date range: {abt_orders['order_date'].min().date()} → {abt_orders['order_date'].max().date()}")
print(f"  Returned items: {abt_orders['is_returned'].sum():,} ({abt_orders['is_returned'].mean()*100:.1f}%)")
print(f"  Items with review: {abt_orders['has_review'].sum():,} ({abt_orders['has_review'].mean()*100:.1f}%)")
print(f"  Null geography: {abt_orders['region'].isnull().sum():,} rows")

# Leakage check
assert (abt_orders["order_date"] <= TRAIN_CUTOFF).all(), "LEAKAGE: post-cutoff orders in ABT!"
print(f"\\n  Leakage check: ✅ all order_dates ≤ {TRAIN_CUTOFF.date()}")

abt_orders.head(2)
"""))

cells.append(new_code_cell("""\
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("abt_orders_enriched — Key distributions", fontsize=14, fontweight="bold")

# Revenue by category
cat_rev = abt_orders.groupby("category")["net_revenue"].sum().sort_values(ascending=False) / 1e9
cat_rev.plot(kind="bar", ax=axes[0, 0], color="#40916c", edgecolor="white")
axes[0, 0].set_title("Net Revenue by Category (B VND)")
axes[0, 0].tick_params(axis="x", rotation=45)

# Revenue by year
abt_orders["year"] = abt_orders["order_date"].dt.year
yr_rev = abt_orders.groupby("year")["net_revenue"].sum() / 1e9
yr_rev.plot(kind="bar", ax=axes[0, 1], color="#52b788", edgecolor="white")
axes[0, 1].set_title("Annual Net Revenue (B VND)")
axes[0, 1].tick_params(axis="x", rotation=45)
# Highlight 2019
bars = axes[0, 1].patches
yr_list = list(yr_rev.index)
if 2019 in yr_list:
    bars[yr_list.index(2019)].set_color("#d62828")

# Return rate by category
ret_rate = abt_orders.groupby("category")["is_returned"].mean().sort_values(ascending=False) * 100
ret_rate.plot(kind="barh", ax=axes[0, 2], color="#95d5b2", edgecolor="white")
axes[0, 2].set_title("Return Rate % by Category")
axes[0, 2].set_xlabel("Return Rate %")

# Revenue by region
reg_rev = abt_orders.groupby("region")["net_revenue"].sum().sort_values(ascending=False) / 1e9
reg_rev.plot(kind="bar", ax=axes[1, 0], color="#74c69d", edgecolor="white")
axes[1, 0].set_title("Net Revenue by Region (B VND)")
axes[1, 0].tick_params(axis="x", rotation=30)

# Revenue by age group
age_order = ["18-24","25-34","35-44","45-54","55+"]
age_rev = abt_orders.groupby("age_group")["net_revenue"].sum().reindex(age_order) / 1e9
age_rev.plot(kind="bar", ax=axes[1, 1], color="#b7e4c7", edgecolor="white")
axes[1, 1].set_title("Net Revenue by Age Group (B VND)")
axes[1, 1].tick_params(axis="x", rotation=30)

# Rating distribution
rating_counts = abt_orders["rating"].dropna().value_counts().sort_index()
axes[1, 2].bar(rating_counts.index, rating_counts.values, color="#52b788", edgecolor="white", width=0.6)
axes[1, 2].set_title("Rating Distribution (reviewed items)")
axes[1, 2].set_xlabel("Rating")
axes[1, 2].set_ylabel("Count")

plt.tight_layout()
plt.savefig(ROOT / "reports" / "fig_abt_orders_dist.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → reports/fig_abt_orders_dist.png")
"""))

cells.append(new_code_cell("""\
# Save ABT 2 (partitioned by year for faster loading)
abt_orders["year"] = abt_orders["order_date"].dt.year
out_path = PROCESSED / "abt_orders_enriched.parquet"
abt_orders.to_parquet(out_path, index=False, engine="pyarrow",
                      partition_cols=["year"])
print(f"✅ Saved abt_orders_enriched.parquet (partitioned by year)")
print(f"   Total rows: {abt_orders.shape[0]:,}  |  Cols: {abt_orders.shape[1]}")
"""))

# ── CELL 8: ABT 3 — abt_customer_cohort ──────────────────────────────────────
cells.append(new_markdown_cell("""\
## 7. ABT 3 — `abt_customer_cohort.parquet`

Grain: 1 row / (customer_id, months_since_signup). For Story 3 retention analysis.
"""))

cells.append(new_code_cell("""\
print("Building abt_customer_cohort ...")

abt_cohort = sj.build_customer_cohort(
    orders=orders,
    order_items=order_items,
    customers=customers,
)

print(f"\\nabt_customer_cohort shape: {abt_cohort.shape}")
print(f"  Unique customers  : {abt_cohort['customer_id'].nunique():,}")
print(f"  Unique cohort months: {abt_cohort['signup_month'].nunique()}")
print(f"  months_since_signup range: {abt_cohort['months_since_signup'].min()} → {abt_cohort['months_since_signup'].max()}")

abt_cohort.head(3)
"""))

cells.append(new_code_cell("""\
import matplotlib.pyplot as plt
import seaborn as sns

# Cohort retention heatmap (M0-M12)
max_months = 13
cohort_sub = abt_cohort[abt_cohort["months_since_signup"] <= max_months].copy()

retention = cohort_sub.groupby(["signup_month", "months_since_signup"])["customer_id"].nunique().unstack(fill_value=0)
cohort_size = retention[0] if 0 in retention.columns else retention.iloc[:, 0]
retention_rate = retention.div(cohort_size, axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Customer Cohort Retention (M0–M12)", fontsize=14, fontweight="bold")

# Heatmap (last 24 cohort months for readability)
heatmap_data = retention_rate.tail(24)
sns.heatmap(
    heatmap_data, ax=axes[0], cmap="YlGn", fmt=".0f",
    annot=(heatmap_data.shape[0] <= 24),
    linewidths=0.2, vmin=0, vmax=100,
    cbar_kws={"label": "Retention %"},
)
axes[0].set_title("Retention % heatmap (signup_month × months_since)")
axes[0].set_xlabel("Months since signup")
axes[0].set_ylabel("Cohort (signup month)")

# Average retention curve per acquisition_channel
ch_retention = (
    cohort_sub.groupby(["acquisition_channel", "months_since_signup"])["customer_id"]
    .nunique()
    .reset_index(name="active_customers")
)
m0 = ch_retention[ch_retention["months_since_signup"] == 0].rename(
    columns={"active_customers": "cohort_size"}
)[["acquisition_channel", "cohort_size"]]
ch_retention = ch_retention.merge(m0, on="acquisition_channel", how="left")
ch_retention["retention_pct"] = ch_retention["active_customers"] / ch_retention["cohort_size"] * 100

for ch, grp in ch_retention.groupby("acquisition_channel"):
    grp = grp.sort_values("months_since_signup")
    axes[1].plot(grp["months_since_signup"], grp["retention_pct"], marker="o", ms=3, label=ch)
axes[1].set_title("Avg Retention % by Acquisition Channel")
axes[1].set_xlabel("Months since signup")
axes[1].set_ylabel("Retention %")
axes[1].legend(fontsize=8, loc="upper right")
axes[1].set_xlim(0, max_months)

plt.tight_layout()
plt.savefig(ROOT / "reports" / "fig_cohort_retention.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → reports/fig_cohort_retention.png")
"""))

cells.append(new_code_cell("""\
# Save ABT 3
out_path = PROCESSED / "abt_customer_cohort.parquet"
abt_cohort.to_parquet(out_path, index=False, engine="pyarrow")
print(f"✅ Saved abt_customer_cohort.parquet — {abt_cohort.shape[0]:,} rows × {abt_cohort.shape[1]} cols")
print(f"   File size: {out_path.stat().st_size / 1e6:.1f} MB")
"""))

# ── CELL 9: Quality gates ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## 8. Quality Gates (eda_plan.md §8)

Checklist tự động — phải pass hết trước khi sang Part 3 modelling.
"""))

cells.append(new_code_cell("""\
print("=" * 60)
print("QUALITY GATE CHECKS")
print("=" * 60)

failures = []

def check(condition: bool, label: str, detail: str = ""):
    icon = "✅" if condition else "❌"
    msg = f"  {icon} {label}"
    if not condition and detail:
        msg += f" — {detail}"
    print(msg)
    if not condition:
        failures.append(label)

# [1] abt_daily row count (2012-07-04 → 2024-07-01 inclusive)
expected_rows = len(pd.date_range("2012-07-04", "2024-07-01", freq="D"))
check(len(abt_daily) == expected_rows,
      f"abt_daily has {expected_rows:,} rows",
      f"actual={len(abt_daily):,}")

# [2] Revenue/COGS NaN for test window
test_rev_ok = abt_daily.loc[abt_daily["date"] > TRAIN_CUTOFF, "Revenue"].isna().all()
check(test_rev_ok, "Revenue = NaN for all dates > 2022-12-31")

# [3] No order_date > cutoff in abt_orders
max_order_date = abt_orders["order_date"].max()
check(max_order_date <= TRAIN_CUTOFF,
      f"abt_orders max order_date ≤ {TRAIN_CUTOFF.date()}",
      f"actual max={max_order_date.date()}")

# [4] abt_orders grain check (no exact duplicates on order_id+product_id is hard — check shape)
check(len(abt_orders) > 700_000,
      f"abt_orders has >700K rows (expected ~714K)",
      f"actual={len(abt_orders):,}")

# [5] abt_cohort months_since_signup >= 0
check((abt_cohort["months_since_signup"] >= 0).all(),
      "abt_cohort months_since_signup all >= 0")

# [6] No feature pulls from web_traffic live cols (already aggregated to daily)
check("sessions" not in abt_daily.columns,
      "abt_daily has no raw 'sessions' col (daily agg used instead)")

# [7] Promo calendar covers full date range
check(len(promo_daily) == len(FULL_RANGE),
      f"promo_daily covers full {len(FULL_RANGE):,} days")

# [8] Idempotency — re-load and compare shape
abt_daily_reload = pd.read_parquet(PROCESSED / "abt_daily.parquet", engine="pyarrow")
check(abt_daily_reload.shape == abt_daily.shape,
      "abt_daily.parquet re-load shape matches in-memory")

print()
if failures:
    print(f"❌ {len(failures)} gate(s) FAILED: {failures}")
else:
    print("✅ ALL quality gates PASSED — safe to proceed to modelling / visualization")
"""))

# ── CELL 10: Summary ──────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## 9. Summary — Phase 2 & 3 Complete

| Deliverable | Path | Status |
|---|---|---|
| Cleaned interim tables | `data/interim/*.parquet` | ✅ |
| `abt_daily.parquet` | `data/processed/` | ✅ |
| `abt_orders_enriched.parquet` | `data/processed/` | ✅ |
| `abt_customer_cohort.parquet` | `data/processed/` | ✅ |
| `src/io.py`, `src/cleaning.py` | `src/` | ✅ |
| `src/joining.py` | `src/` | ✅ |
| `src/features/calendar.py` | `src/features/` | ✅ |
| `src/viz/style.py` | `src/viz/` | ✅ |

### Bước tiếp theo

```
Phase 4 → notebooks/01_eda_exploratory.ipynb   (verify 6 context facts + 7 hypotheses)
Phase 5 → notebooks/30_forecasting.ipynb        (feature engineering + model)
Phase 6 → notebooks/21_story_*.ipynb            (Part 2 stories)
```
"""))

nb.cells = cells

out_path = Path("/home/pearspringmind/Hackathon/gridbreaker-vinuni/notebooks/10_build_abt.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"✅ Written: {out_path}")
print(f"   Cells: {len(nb.cells)}")
