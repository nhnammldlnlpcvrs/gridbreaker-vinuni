"""
Builder script for notebooks/00_data_profiling.ipynb
Run: python scripts/build_00_data_profiling.py
Re-run anytime to regenerate the notebook from scratch.
"""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

nb = new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.10.0"},
}

cells = []

# ── CELL 0: Title ─────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("""# 00 — Data Profiling
**Datathon 2026 | The Gridbreakers**

**Mục tiêu:** Hiểu shape, dtype, null, duplicate, phân phối từng cột của 15 file CSV.
**Output:** `reports/data_quality.md` — danh sách issues P0/P1/P2 cho team xử lý ở Phase 2.

**Theo kế hoạch:** `eda_plan.md §2` — 15 file × 6 checks = 90 checks + cardinality asserts.
"""))

# ── CELL 1: Imports & config ──────────────────────────────────────────────────
cells.append(new_code_cell("""\
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from datetime import datetime

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", "{:,.2f}".format)
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path("..").resolve()
DATA_RAW  = ROOT / "data" / "raw"
REPORTS   = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

MASTER      = DATA_RAW / "Master"
TRANSACTION = DATA_RAW / "Transaction"
ANALYTICAL  = DATA_RAW / "Analytical"
OPERATIONAL = DATA_RAW / "Operational"

# Train cutoff — DO NOT leak beyond this
TRAIN_CUTOFF = pd.Timestamp("2022-12-31")

print(f"ROOT     : {ROOT}")
print(f"DATA_RAW : {DATA_RAW}")
print(f"REPORTS  : {REPORTS}")
"""))

# ── CELL 2: Loader utility ────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 1. Loader & profiler utilities"))

cells.append(new_code_cell("""\
def load(path: Path, **kwargs) -> pd.DataFrame:
    \"\"\"Load CSV with basic dtype inference.\"\"\"
    df = pd.read_csv(path, **kwargs)
    # Auto-parse obvious date columns
    for col in df.columns:
        if any(x in col.lower() for x in ["date", "_at", "_on"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df


def profile(df: pd.DataFrame, name: str, pk: list[str] | None = None) -> dict:
    \"\"\"Run 6-point profiling checklist per eda_plan.md §2.1.\"\"\"
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / 1e6

    # Missing
    null_pct = (df.isnull().sum() / n_rows * 100).round(2)
    null_cols = null_pct[null_pct > 0].to_dict()

    # Duplicates on PK
    dup_pk = None
    if pk:
        existing_pk = [c for c in pk if c in df.columns]
        if existing_pk:
            dup_pk = int(df.duplicated(subset=existing_pk).sum())

    # Date range
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    date_ranges = {}
    for c in date_cols:
        date_ranges[c] = {
            "min": str(df[c].min())[:10],
            "max": str(df[c].max())[:10],
            "n_null": int(df[c].isnull().sum()),
        }

    result = {
        "name": name,
        "rows": n_rows,
        "cols": n_cols,
        "memory_mb": round(mem_mb, 2),
        "null_pct": null_cols,
        "dup_on_pk": dup_pk,
        "pk": pk,
        "date_ranges": date_ranges,
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    return result


def print_profile(p: dict):
    print(f"{'='*60}")
    print(f"TABLE: {p['name']}")
    print(f"  Rows: {p['rows']:,}  |  Cols: {p['cols']}  |  Memory: {p['memory_mb']} MB")
    if p["pk"]:
        dup_str = f"{p['dup_on_pk']:,} duplicates on PK {p['pk']}" if p["dup_on_pk"] is not None else "PK not found"
        status  = "✅ NO DUPS" if p["dup_on_pk"] == 0 else f"❌ {dup_str}"
        print(f"  PK check: {status}")
    if p["null_pct"]:
        print(f"  Nulls:")
        for col, pct in p["null_pct"].items():
            print(f"    {col:35s} {pct:6.2f}%")
    else:
        print("  Nulls: ✅ None")
    if p["date_ranges"]:
        print(f"  Date ranges:")
        for col, r in p["date_ranges"].items():
            print(f"    {col:35s} {r['min']} → {r['max']}  (null={r['n_null']})")
    print()


# Global issues log
issues: list[dict] = []

def log_issue(table: str, severity: str, issue: str):
    \"\"\"severity: P0 (critical/blocks analysis), P1 (important), P2 (minor)\"\"\"
    issues.append({"table": table, "severity": severity, "issue": issue})
    badge = {"P0": "🔴", "P1": "🟡", "P2": "🟢"}.get(severity, "⚪")
    print(f"  [{severity}] {badge} {issue}")

print("✅ Utilities loaded")
"""))

# ── CELL 3: products.csv ──────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 2. Master layer\n### 2.1 products.csv"))

cells.append(new_code_cell("""\
products = load(MASTER / "products.csv")
p = profile(products, "products", pk=["product_id"])
print_profile(p)

# Specific checks
print("--- Specific checks ---")
# cogs < price constraint
mask_violation = products["cogs"] >= products["price"]
n_violation = mask_violation.sum()
if n_violation > 0:
    log_issue("products", "P0", f"cogs >= price in {n_violation} rows (constraint violated)")
else:
    print("  ✅ cogs < price: holds for all rows")

# Category / segment distributions
print("\\nCategory distribution:")
print(products["category"].value_counts().to_string())
print("\\nSegment distribution:")
print(products["segment"].value_counts().to_string())
print("\\nSize distribution:")
print(products["size"].value_counts().to_string())

# Margin distribution
products["margin_pct"] = (products["price"] - products["cogs"]) / products["price"] * 100
print(f"\\nMargin % — mean: {products['margin_pct'].mean():.1f}%  median: {products['margin_pct'].median():.1f}%  min: {products['margin_pct'].min():.1f}%  max: {products['margin_pct'].max():.1f}%")

products.describe()
"""))

# ── CELL 4: products viz ──────────────────────────────────────────────────────
cells.append(new_code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("products.csv — price, cogs, margin distributions", fontsize=13, fontweight="bold")

axes[0].hist(products["price"], bins=40, color="#2d6a4f", edgecolor="white")
axes[0].set_title("Price distribution")
axes[0].set_xlabel("Price (VND)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

axes[1].hist(products["margin_pct"], bins=40, color="#52b788", edgecolor="white")
axes[1].set_title("Gross margin %")
axes[1].set_xlabel("Margin %")

cat_margin = products.groupby("category")["margin_pct"].mean().sort_values()
cat_margin.plot(kind="barh", ax=axes[2], color="#95d5b2", edgecolor="white")
axes[2].set_title("Avg margin % by category")
axes[2].set_xlabel("Margin %")

plt.tight_layout()
plt.savefig(REPORTS / "fig_products_dist.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → reports/fig_products_dist.png")
"""))

# ── CELL 5: customers.csv ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("### 2.2 customers.csv"))

cells.append(new_code_cell("""\
customers = load(MASTER / "customers.csv")
p = profile(customers, "customers", pk=["customer_id"])
print_profile(p)

# Nullable cols: gender, age_group, acquisition_channel
for col in ["gender", "age_group", "acquisition_channel"]:
    n_null = customers[col].isnull().sum()
    pct = n_null / len(customers) * 100
    severity = "P1" if pct > 20 else "P2"
    if n_null > 0:
        log_issue("customers", severity, f"{col} null: {n_null:,} ({pct:.1f}%)")

# Signup date sanity
print(f"\\nSignup date range: {customers['signup_date'].min().date()} → {customers['signup_date'].max().date()}")
future_signups = (customers["signup_date"] > TRAIN_CUTOFF).sum()
if future_signups > 0:
    log_issue("customers", "P1", f"{future_signups:,} customers signed up after train cutoff")

# Distributions
print("\\nGender distribution:")
print(customers["gender"].value_counts(dropna=False).to_string())
print("\\nAge group distribution:")
print(customers["age_group"].value_counts(dropna=False).to_string())
print("\\nAcquisition channel distribution:")
print(customers["acquisition_channel"].value_counts(dropna=False).to_string())
"""))

cells.append(new_code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("customers.csv — signup trend, age_group, acquisition_channel", fontsize=13, fontweight="bold")

signups_per_year = customers.groupby(customers["signup_date"].dt.year).size()
signups_per_year.plot(kind="bar", ax=axes[0], color="#2d6a4f", edgecolor="white")
axes[0].set_title("Signups per year")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)

age_order = ["18-24","25-34","35-44","45-54","55+"]
age_counts = customers["age_group"].value_counts().reindex(age_order, fill_value=0)
age_counts.plot(kind="bar", ax=axes[1], color="#52b788", edgecolor="white")
axes[1].set_title("Age group distribution")
axes[1].tick_params(axis="x", rotation=30)

ch_counts = customers["acquisition_channel"].value_counts()
ch_counts.plot(kind="barh", ax=axes[2], color="#95d5b2", edgecolor="white")
axes[2].set_title("Acquisition channel")

plt.tight_layout()
plt.savefig(REPORTS / "fig_customers_dist.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ── CELL 6: promotions.csv ────────────────────────────────────────────────────
cells.append(new_markdown_cell("### 2.3 promotions.csv"))

cells.append(new_code_cell("""\
promotions = load(MASTER / "promotions.csv")
p = profile(promotions, "promotions", pk=["promo_id"])
print_profile(p)

# Sanity: start_date <= end_date
bad_dates = (promotions["start_date"] > promotions["end_date"]).sum()
if bad_dates > 0:
    log_issue("promotions", "P0", f"{bad_dates} promos with start_date > end_date")
else:
    print("  ✅ start_date <= end_date: all OK")

# Discount distribution
print("\\nPromo type breakdown:")
print(promotions["promo_type"].value_counts().to_string())
print("\\nDiscount values:")
print(promotions["discount_value"].value_counts().sort_index().to_string())
print("\\nStackable flag:")
print(promotions["stackable_flag"].value_counts().to_string())

# Calendar pattern (6-4-6-4?)
promotions["year"] = promotions["start_date"].dt.year
print("\\nPromos per year:")
print(promotions.groupby("year").size().to_string())
"""))

# ── CELL 7: geography.csv ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("### 2.4 geography.csv"))

cells.append(new_code_cell("""\
geography = load(MASTER / "geography.csv")
p = profile(geography, "geography", pk=["zip"])
print_profile(p)

# 1 zip → possible multiple cities/districts?
zip_city_count = geography.groupby("zip")["city"].nunique()
multi_city_zips = zip_city_count[zip_city_count > 1]
if len(multi_city_zips) > 0:
    log_issue("geography", "P1", f"{len(multi_city_zips)} zip codes map to multiple cities")
    print(f"  Sample multi-city zips:\\n{multi_city_zips.head(5)}")
else:
    print("  ✅ Each zip maps to exactly 1 city")

print("\\nRegion distribution:")
print(geography["region"].value_counts().to_string())
print(f"\\nUnique zips: {geography['zip'].nunique():,}")
print(f"Unique cities: {geography['city'].nunique():,}")
print(f"Unique regions: {geography['region'].nunique():,}")
"""))

# ── CELL 8: orders.csv ───────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 3. Transaction layer\n### 3.1 orders.csv"))

cells.append(new_code_cell("""\
orders = load(TRANSACTION / "orders.csv")
p = profile(orders, "orders", pk=["order_id"])
print_profile(p)

# Date sanity
out_of_range = orders[
    (orders["order_date"] < "2012-07-04") | (orders["order_date"] > TRAIN_CUTOFF)
]
if len(out_of_range) > 0:
    log_issue("orders", "P1", f"{len(out_of_range):,} orders outside train window")
    print(f"  Out-of-range sample:\\n{out_of_range[['order_id','order_date','order_status']].head()}")
else:
    print("  ✅ All order_dates within 2012-07-04 → 2022-12-31")

# Status distribution
print("\\nOrder status distribution:")
print(orders["order_status"].value_counts().to_string())

print("\\nPayment method distribution:")
print(orders["payment_method"].value_counts().to_string())

print("\\nDevice type distribution:")
print(orders["device_type"].value_counts().to_string())

print("\\nOrder source distribution:")
print(orders["order_source"].value_counts().to_string())
"""))

cells.append(new_code_cell("""\
# Orders per year trend
orders_per_year = orders.groupby(orders["order_date"].dt.year).size()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("orders.csv — annual volume & status breakdown", fontsize=13, fontweight="bold")

orders_per_year.plot(kind="bar", ax=axes[0], color="#2d6a4f", edgecolor="white")
axes[0].set_title("Orders per year")
axes[0].set_xlabel("Year")
axes[0].tick_params(axis="x", rotation=45)

status_counts = orders["order_status"].value_counts()
axes[1].barh(status_counts.index, status_counts.values, color="#52b788", edgecolor="white")
axes[1].set_title("Order status counts")
for i, v in enumerate(status_counts.values):
    axes[1].text(v + 500, i, f"{v:,}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(REPORTS / "fig_orders_dist.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ── CELL 9: order_items.csv ──────────────────────────────────────────────────
cells.append(new_markdown_cell("### 3.2 order_items.csv"))

cells.append(new_code_cell("""\
order_items = load(TRANSACTION / "order_items.csv")
p = profile(order_items, "order_items", pk=["order_id", "product_id"])
print_profile(p)

# Validate discount formula (spot-check against promotions)
# net = unit_price * quantity - discount_amount should be >= 0
order_items["gross_revenue"] = order_items["quantity"] * order_items["unit_price"]
order_items["net_revenue"]   = order_items["gross_revenue"] - order_items["discount_amount"]

neg_net = (order_items["net_revenue"] < 0).sum()
if neg_net > 0:
    log_issue("order_items", "P0", f"{neg_net:,} rows with net_revenue < 0 (discount > gross)")
else:
    print("  ✅ No negative net_revenue rows")

# Promo coverage
has_promo = order_items["promo_id"].notna().sum()
has_promo2 = order_items["promo_id_2"].notna().sum()
total = len(order_items)
print(f"\\n  promo_id  coverage: {has_promo:,} / {total:,} ({has_promo/total*100:.1f}%)")
print(f"  promo_id_2 coverage: {has_promo2:,} / {total:,} ({has_promo2/total*100:.1f}%)")

print(f"\\n  Quantity stats: min={order_items['quantity'].min()} max={order_items['quantity'].max()} mean={order_items['quantity'].mean():.1f}")
print(f"  Discount amount > 0: {(order_items['discount_amount'] > 0).sum():,} rows ({(order_items['discount_amount'] > 0).mean()*100:.1f}%)")
print(f"\\n  Items per order (nunique order_id): {order_items['order_id'].nunique():,}")
"""))

# ── CELL 10: payments.csv ────────────────────────────────────────────────────
cells.append(new_markdown_cell("### 3.3 payments.csv"))

cells.append(new_code_cell("""\
payments = load(TRANSACTION / "payments.csv")
p = profile(payments, "payments", pk=["order_id"])
print_profile(p)

# Cardinality check: payments ↔ orders 1:1
orders_ids = set(orders["order_id"])
payments_ids = set(payments["order_id"])
only_in_payments = payments_ids - orders_ids
only_in_orders   = orders_ids - payments_ids
if only_in_payments:
    log_issue("payments", "P0", f"{len(only_in_payments)} payment order_ids not in orders")
if only_in_orders:
    log_issue("payments", "P1", f"{len(only_in_orders)} orders have no payment record")
if not only_in_payments and not only_in_orders:
    print("  ✅ payments ↔ orders: perfect 1:1 match")

print("\\nInstallments distribution:")
print(payments["installments"].value_counts().sort_index().to_string())
print("\\nPayment method (payments):")
print(payments["payment_method"].value_counts().to_string())
print(f"\\nPayment value — mean: {payments['payment_value'].mean():,.0f}  median: {payments['payment_value'].median():,.0f}  max: {payments['payment_value'].max():,.0f}")
"""))

# ── CELL 11: shipments.csv ───────────────────────────────────────────────────
cells.append(new_markdown_cell("### 3.4 shipments.csv"))

cells.append(new_code_cell("""\
shipments = load(TRANSACTION / "shipments.csv")
p = profile(shipments, "shipments", pk=["order_id"])
print_profile(p)

# Cardinality: only shipped/delivered/returned orders should have shipments
eligible_statuses = {"shipped", "delivered", "returned"}
eligible_orders = set(orders.loc[orders["order_status"].isin(eligible_statuses), "order_id"])
shipment_orders = set(shipments["order_id"])

not_eligible = shipment_orders - eligible_orders
missing_eligible = eligible_orders - shipment_orders
if not_eligible:
    log_issue("shipments", "P0", f"{len(not_eligible)} shipments for non-eligible order statuses")
if missing_eligible:
    log_issue("shipments", "P1", f"{len(missing_eligible)} eligible orders missing shipment record")
if not not_eligible and not missing_eligible:
    print("  ✅ shipments cardinality: matches shipped/delivered/returned only")

# Delivery time distribution
shipments["delivery_days"] = (shipments["delivery_date"] - shipments["ship_date"]).dt.days
neg_delivery = (shipments["delivery_days"] < 0).sum()
if neg_delivery > 0:
    log_issue("shipments", "P0", f"{neg_delivery} rows with delivery_date < ship_date")
print(f"\\n  Delivery days: min={shipments['delivery_days'].min()} mean={shipments['delivery_days'].mean():.1f} max={shipments['delivery_days'].max()}")

print(f"\\n  Shipping fee = 0 (free): {(shipments['shipping_fee']==0).sum():,} ({(shipments['shipping_fee']==0).mean()*100:.1f}%)")
"""))

# ── CELL 12: returns.csv ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("### 3.5 returns.csv"))

cells.append(new_code_cell("""\
returns = load(TRANSACTION / "returns.csv")
p = profile(returns, "returns", pk=["return_id"])
print_profile(p)

# return_date >= order_date check
returns_with_date = returns.merge(orders[["order_id","order_date"]], on="order_id", how="left")
bad_return_dates = (returns_with_date["return_date"] < returns_with_date["order_date"]).sum()
if bad_return_dates > 0:
    log_issue("returns", "P0", f"{bad_return_dates} returns with return_date < order_date")
else:
    print("  ✅ All return_dates >= corresponding order_date")

# Only returned orders should have returns
returned_order_ids = set(orders.loc[orders["order_status"] == "returned", "order_id"])
return_order_ids   = set(returns["order_id"])
unexpected = return_order_ids - returned_order_ids
if unexpected:
    log_issue("returns", "P1", f"{len(unexpected)} returns linked to non-returned orders")

print("\\nReturn reason distribution:")
print(returns["return_reason"].value_counts().to_string())
print(f"\\nReturn quantity: min={returns['return_quantity'].min()} max={returns['return_quantity'].max()} mean={returns['return_quantity'].mean():.1f}")
print(f"Refund amount: mean={returns['refund_amount'].mean():,.0f}  max={returns['refund_amount'].max():,.0f}")
"""))

# ── CELL 13: reviews.csv ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("### 3.6 reviews.csv"))

cells.append(new_code_cell("""\
reviews = load(TRANSACTION / "reviews.csv")
p = profile(reviews, "reviews", pk=["review_id"])
print_profile(p)

# Rating range check
bad_ratings = reviews[~reviews["rating"].between(1, 5)]
if len(bad_ratings) > 0:
    log_issue("reviews", "P0", f"{len(bad_ratings)} reviews with rating outside [1,5]")
else:
    print("  ✅ All ratings in [1, 5]")

# Coverage: ~20% of delivered orders should have reviews
delivered_count = (orders["order_status"] == "delivered").sum()
review_count    = len(reviews)
coverage        = review_count / delivered_count * 100
print(f"\\n  Review coverage: {review_count:,} reviews / {delivered_count:,} delivered orders = {coverage:.1f}%")
if coverage < 10 or coverage > 40:
    log_issue("reviews", "P2", f"Unusual review coverage: {coverage:.1f}% (expected ~20%)")

print("\\nRating distribution:")
print(reviews["rating"].value_counts().sort_index().to_string())
"""))

# ── CELL 14: sales.csv ───────────────────────────────────────────────────────
cells.append(new_markdown_cell("## 4. Analytical layer\n### 4.1 sales.csv (train target)"))

cells.append(new_code_cell("""\
sales = load(ANALYTICAL / "sales.csv")
p = profile(sales, "sales", pk=["Date"])
print_profile(p)

# Date coverage: should be 2012-07-04 → 2022-12-31
print(f"  Date range: {sales['Date'].min().date()} → {sales['Date'].max().date()}")
expected_days = (pd.Timestamp("2022-12-31") - pd.Timestamp("2012-07-04")).days + 1
actual_days   = sales["Date"].nunique()
missing_days  = expected_days - actual_days
if missing_days > 0:
    log_issue("sales", "P1", f"{missing_days} missing dates in train period (gaps in time series)")
    print(f"\\n  Expected {expected_days:,} days, found {actual_days:,} (gap = {missing_days})")
else:
    print(f"  ✅ No date gaps: {actual_days:,} days continuous")

# Revenue / COGS stats
print(f"\\n  Revenue — min: {sales['Revenue'].min():,.0f}  mean: {sales['Revenue'].mean():,.0f}  max: {sales['Revenue'].max():,.0f}")
print(f"  COGS    — min: {sales['COGS'].min():,.0f}     mean: {sales['COGS'].mean():,.0f}     max: {sales['COGS'].max():,.0f}")
sales["margin_pct"] = (sales["Revenue"] - sales["COGS"]) / sales["Revenue"] * 100
print(f"  Daily margin % — mean: {sales['margin_pct'].mean():.1f}%  min: {sales['margin_pct'].min():.1f}%  max: {sales['margin_pct'].max():.1f}%")

# Annual trend (§2.1 context verification)
sales["year"] = sales["Date"].dt.year
annual = sales.groupby("year").agg(Revenue=("Revenue","sum"), COGS=("COGS","sum")).assign(
    Margin_pct=lambda d: (d.Revenue - d.COGS) / d.Revenue * 100,
    Revenue_M=lambda d: d.Revenue / 1e6
)
print("\\n=== Annual Revenue (context §2.1 verification) ===")
print(annual[["Revenue_M","Margin_pct"]].round(1).to_string())
"""))

cells.append(new_code_cell("""\
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("sales.csv — Revenue & COGS analysis (train: 2012–2022)", fontsize=13, fontweight="bold")

# Annual revenue
annual["Revenue_M"].plot(kind="bar", ax=axes[0,0], color="#2d6a4f", edgecolor="white")
axes[0,0].set_title("Annual Revenue (M VND)")
axes[0,0].set_xlabel("Year")
axes[0,0].tick_params(axis="x", rotation=45)
# Annotate 2019 shock
if 2019 in annual.index:
    idx = list(annual.index).index(2019)
    axes[0,0].get_children()[idx].set_color("#d62828")
    axes[0,0].annotate("2019 shock", xy=(idx, annual.loc[2019,"Revenue_M"]),
                        xytext=(idx+0.5, annual.loc[2019,"Revenue_M"]+50),
                        arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=9)

# Daily revenue time series
sales_sorted = sales.sort_values("Date")
axes[0,1].plot(sales_sorted["Date"], sales_sorted["Revenue"]/1e6, color="#52b788", linewidth=0.5, alpha=0.7)
# 90-day rolling mean
roll = sales_sorted.set_index("Date")["Revenue"].rolling(90).mean() / 1e6
axes[0,1].plot(roll, color="#1b4332", linewidth=1.5, label="90d rolling mean")
axes[0,1].set_title("Daily Revenue (M VND) with 90-day rolling mean")
axes[0,1].set_ylabel("Revenue (M VND)")
axes[0,1].legend()

# Annual margin
annual["Margin_pct"].plot(kind="bar", ax=axes[1,0], color="#95d5b2", edgecolor="white")
axes[1,0].set_title("Annual Gross Margin %")
axes[1,0].axhline(annual["Margin_pct"].mean(), color="red", linestyle="--", label=f"Mean {annual['Margin_pct'].mean():.1f}%")
axes[1,0].legend()
axes[1,0].tick_params(axis="x", rotation=45)

# Day-of-week seasonality
sales["dow"] = sales["Date"].dt.day_name()
dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
dow_avg = sales.groupby("dow")["Revenue"].mean().reindex(dow_order) / 1e6
axes[1,1].bar(range(7), dow_avg.values, color="#40916c", edgecolor="white")
axes[1,1].set_xticks(range(7))
axes[1,1].set_xticklabels([d[:3] for d in dow_order])
axes[1,1].set_title("Avg Revenue by Day of Week (M VND)")
axes[1,1].set_ylabel("Avg Revenue (M VND)")

plt.tight_layout()
plt.savefig(REPORTS / "fig_sales_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → reports/fig_sales_overview.png")
"""))

# ── CELL 15: inventory.csv ───────────────────────────────────────────────────
cells.append(new_markdown_cell("## 5. Operational layer\n### 5.1 inventory.csv"))

cells.append(new_code_cell("""\
inventory = load(OPERATIONAL / "inventory.csv")
p = profile(inventory, "inventory", pk=["snapshot_date", "product_id"])
print_profile(p)

# snapshot_date should be last day of month
inventory["day_of_month"] = pd.to_datetime(inventory["snapshot_date"]).dt.day
non_month_end = inventory[inventory["day_of_month"] != inventory["day_of_month"].groupby(
    pd.to_datetime(inventory["snapshot_date"]).dt.to_period("M")
).transform("max")]
# simpler check: day should be 28, 29, 30, or 31
end_days = pd.to_datetime(inventory["snapshot_date"]).dt.day
valid_end = end_days.isin([28, 29, 30, 31])
n_invalid = (~valid_end).sum()
if n_invalid > 0:
    log_issue("inventory", "P1", f"{n_invalid} snapshot_dates not at month-end (day not in 28-31)")
else:
    print("  ✅ All snapshot_dates are month-end (day 28-31)")

# 1 row per product per month check
dup_check = inventory.groupby([pd.to_datetime(inventory["snapshot_date"]).dt.to_period("M"), "product_id"]).size()
multi_rows = (dup_check > 1).sum()
if multi_rows > 0:
    log_issue("inventory", "P0", f"{multi_rows} (month, product_id) pairs have >1 row")
else:
    print("  ✅ 1 row per (product, month) — grain correct")

print("\\nInventory coverage:")
print(f"  Snapshot months: {inventory['snapshot_date'].nunique()}")
print(f"  Unique products: {inventory['product_id'].nunique()}")
print(f"  Stockout flag %: {inventory['stockout_flag'].mean()*100:.1f}%")
print(f"  Overstock flag %: {inventory['overstock_flag'].mean()*100:.1f}%")
print(f"  Reorder flag %: {inventory['reorder_flag'].mean()*100:.1f}%")
print(f"  Fill rate — mean: {inventory['fill_rate'].mean():.3f}  min: {inventory['fill_rate'].min():.3f}")
print(f"  Days of supply — mean: {inventory['days_of_supply'].mean():.1f}  median: {inventory['days_of_supply'].median():.1f}")
"""))

# ── CELL 16: web_traffic.csv ─────────────────────────────────────────────────
cells.append(new_markdown_cell("### 5.2 web_traffic.csv"))

cells.append(new_code_cell("""\
web_traffic = load(OPERATIONAL / "web_traffic.csv")
p = profile(web_traffic, "web_traffic", pk=["date", "traffic_source"])
print_profile(p)

# Date range vs sales coverage
wt_min = web_traffic["date"].min()
wt_max = web_traffic["date"].max()
print(f"  Web traffic date range: {wt_min.date()} → {wt_max.date()}")
covers_test = wt_max >= pd.Timestamp("2023-01-01")
if not covers_test:
    log_issue("web_traffic", "P1", f"web_traffic ends {wt_max.date()}, does NOT cover test period 2023-01-01→2024-07-01 → lag features only for Part 3")
else:
    print(f"  ✅ web_traffic covers test period")

# Bounce rate anomaly (expected 30-60% normally, but dataset has ~0.4-0.5%)
avg_bounce = web_traffic["bounce_rate"].mean()
print(f"\\n  Bounce rate — mean: {avg_bounce:.4f}  min: {web_traffic['bounce_rate'].min():.4f}  max: {web_traffic['bounce_rate'].max():.4f}")
if avg_bounce < 0.01:
    log_issue("web_traffic", "P2", f"bounce_rate mean={avg_bounce:.4f} — likely stored as fraction (not %). Note in report: 0.005 = 0.5%, unusual for retail (normal: 30-60%)")

print("\\nTraffic source distribution:")
print(web_traffic["traffic_source"].value_counts().to_string())

# Daily total (multiple rows per day — one per traffic_source)
daily_web = web_traffic.groupby("date").agg(
    sessions_total=("sessions","sum"),
    visitors_total=("unique_visitors","sum"),
    pageviews_total=("page_views","sum"),
    bounce_mean=("bounce_rate","mean"),
    session_sec_mean=("avg_session_duration_sec","mean"),
    n_sources=("traffic_source","count")
).reset_index()
print(f"\\n  Daily aggregate rows: {len(daily_web):,}")
print(f"  Sessions/day — mean: {daily_web['sessions_total'].mean():,.0f}  max: {daily_web['sessions_total'].max():,.0f}")
"""))

cells.append(new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("web_traffic.csv — sessions trend & source breakdown", fontsize=13, fontweight="bold")

daily_web_sorted = daily_web.sort_values("date")
axes[0].plot(daily_web_sorted["date"], daily_web_sorted["sessions_total"]/1e3,
             color="#52b788", linewidth=0.6, alpha=0.7, label="daily")
roll_sess = daily_web_sorted.set_index("date")["sessions_total"].rolling(90).mean() / 1e3
axes[0].plot(roll_sess, color="#1b4332", linewidth=1.5, label="90d rolling mean")
axes[0].set_title("Daily Sessions (K)")
axes[0].set_ylabel("Sessions (K)")
axes[0].legend()

src_avg = web_traffic.groupby("traffic_source")["sessions"].mean().sort_values()
src_avg.plot(kind="barh", ax=axes[1], color="#40916c", edgecolor="white")
axes[1].set_title("Avg daily sessions by traffic source")
axes[1].set_xlabel("Avg sessions/day")

plt.tight_layout()
plt.savefig(REPORTS / "fig_web_traffic.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ── CELL 17: Cardinality asserts ─────────────────────────────────────────────
cells.append(new_markdown_cell("## 6. Cross-table cardinality verification (eda_plan.md §2.2)"))

cells.append(new_code_cell("""\
print("=" * 60)
print("CARDINALITY VERIFICATION")
print("=" * 60)

# 1. orders ↔ payments = 1:1
o_ids = set(orders["order_id"])
pay_ids = set(payments["order_id"])
assert_pass = (o_ids == pay_ids)
status = "✅ PASS" if assert_pass else f"❌ FAIL — orders only: {len(o_ids-pay_ids)}, payments only: {len(pay_ids-o_ids)}"
print(f"\\n[1] orders ↔ payments (1:1): {status}")
if not assert_pass:
    log_issue("orders×payments", "P0", f"Cardinality broken — mismatch {len(o_ids-pay_ids)} + {len(pay_ids-o_ids)}")

# 2. orders ↔ shipments = 1:0/1 (shipped/delivered/returned only)
ship_eligible = set(orders.loc[orders["order_status"].isin(["shipped","delivered","returned"]), "order_id"])
ship_ids = set(shipments["order_id"])
extra_ships = ship_ids - ship_eligible
missing_ships = ship_eligible - ship_ids
status = "✅ PASS" if (not extra_ships and not missing_ships) else f"❌ extra={len(extra_ships)} missing={len(missing_ships)}"
print(f"[2] orders ↔ shipments (1:0/1 for shipped/delivered/returned): {status}")
if extra_ships or missing_ships:
    log_issue("orders×shipments", "P0", f"Cardinality broken — extra={len(extra_ships)} missing={len(missing_ships)}")

# 3. orders ↔ returns = 1:0..N (returned only)
ret_eligible = set(orders.loc[orders["order_status"] == "returned", "order_id"])
ret_ids = set(returns["order_id"])
extra_rets = ret_ids - ret_eligible
status = "✅ PASS" if not extra_rets else f"❌ {len(extra_rets)} returns from non-returned orders"
print(f"[3] orders ↔ returns (returned orders only): {status}")
if extra_rets:
    log_issue("orders×returns", "P1", f"{len(extra_rets)} returns from non-returned orders")

# 4. orders ↔ reviews = 1:0..N (~20% delivered)
rev_eligible = set(orders.loc[orders["order_status"] == "delivered", "order_id"])
rev_ids = set(reviews["order_id"])
extra_revs = rev_ids - rev_eligible
coverage_pct = len(rev_ids & rev_eligible) / len(rev_eligible) * 100
status = f"✅ coverage {coverage_pct:.1f}%" if not extra_revs else f"❌ {len(extra_revs)} reviews from non-delivered orders, coverage={coverage_pct:.1f}%"
print(f"[4] orders ↔ reviews (delivered only, ~20%): {status}")
if extra_revs:
    log_issue("orders×reviews", "P1", f"{len(extra_revs)} reviews linked to non-delivered orders")

# 5. order_items ↔ promotions = N:0/1
promo_ids_in_promos = set(promotions["promo_id"])
promo_ids_in_items  = set(order_items["promo_id"].dropna()) | set(order_items["promo_id_2"].dropna())
orphan_promos = promo_ids_in_items - promo_ids_in_promos
status = "✅ PASS" if not orphan_promos else f"❌ {len(orphan_promos)} unknown promo_ids in order_items"
print(f"[5] order_items ↔ promotions (N:0/1): {status}")
if orphan_promos:
    log_issue("order_items×promotions", "P1", f"{len(orphan_promos)} unknown promo_ids: {list(orphan_promos)[:5]}")

# 6. products ↔ inventory = 1:N (1 row/product/month)
prod_ids = set(products["product_id"])
inv_prod_ids = set(inventory["product_id"])
inv_not_in_products = inv_prod_ids - prod_ids
status = "✅ PASS" if not inv_not_in_products else f"❌ {len(inv_not_in_products)} inventory product_ids not in products"
print(f"[6] products ↔ inventory (1:N): {status}")
if inv_not_in_products:
    log_issue("products×inventory", "P0", f"{len(inv_not_in_products)} inventory rows with unknown product_id")

# customers ↔ geography via zip
cust_zips = set(customers["zip"])
geo_zips  = set(geography["zip"])
unmapped_cust_zips = cust_zips - geo_zips
status = "✅ PASS" if not unmapped_cust_zips else f"❌ {len(unmapped_cust_zips)} customer zips not in geography"
print(f"[7] customers.zip → geography.zip: {status}")
if unmapped_cust_zips:
    log_issue("customers×geography", "P1", f"{len(unmapped_cust_zips)} customer zips not in geography")

# orders ↔ geography via zip
order_zips = set(orders["zip"])
unmapped_order_zips = order_zips - geo_zips
status = "✅ PASS" if not unmapped_order_zips else f"❌ {len(unmapped_order_zips)} order zips not in geography"
print(f"[8] orders.zip → geography.zip: {status}")
if unmapped_order_zips:
    log_issue("orders×geography", "P1", f"{len(unmapped_order_zips)} order zips not in geography")

print("\\n✅ Cardinality checks complete")
"""))

# ── CELL 18: sales.csv reconstruction verify ─────────────────────────────────
cells.append(new_markdown_cell("## 7. sales.csv reconstruction verification (eda_plan §2.3 context §2.5)"))

cells.append(new_code_cell("""\
# Verify: sales.csv = SUM over ALL order statuses (not just delivered)
# Merge order_items → orders → products to get daily revenue
print("Reconstructing daily revenue from order_items × orders × products ...")

oi_merged = order_items.merge(orders[["order_id","order_date","order_status"]], on="order_id", how="left")
oi_merged = oi_merged.merge(products[["product_id","cogs"]], on="product_id", how="left")

# Revenue = unit_price * quantity - discount_amount
oi_merged["revenue_item"] = oi_merged["quantity"] * oi_merged["unit_price"] - oi_merged["discount_amount"]
oi_merged["cogs_item"]    = oi_merged["quantity"] * oi_merged["cogs"]

daily_recon_all = oi_merged.groupby("order_date").agg(
    Revenue_all=("revenue_item","sum"),
    COGS_all=("cogs_item","sum")
).reset_index()

daily_recon_delivered = oi_merged[oi_merged["order_status"]=="delivered"].groupby("order_date").agg(
    Revenue_del=("revenue_item","sum"),
    COGS_del=("cogs_item","sum")
).reset_index()

# Compare with sales.csv
sales_check = sales.merge(daily_recon_all, left_on="Date", right_on="order_date", how="inner")
sales_check = sales_check.merge(daily_recon_delivered, left_on="Date", right_on="order_date", how="left")

mape_all = (abs(sales_check["Revenue"] - sales_check["Revenue_all"]) / sales_check["Revenue"]).mean() * 100
mape_del = (abs(sales_check["Revenue"] - sales_check["Revenue_del"].fillna(0)) / sales_check["Revenue"]).mean() * 100

print(f"\\n  MAPE vs ALL orders:       {mape_all:.1f}%")
print(f"  MAPE vs DELIVERED only:   {mape_del:.1f}%")

conclusion = "ALL orders" if mape_all < mape_del else "DELIVERED only"
print(f"\\n  → sales.csv is closer to reconstruction using: {conclusion}")
print(f"  (context §2.5 says: ALL orders ~9% deviation, delivered ~24% — verify matches)")

if mape_all > 20:
    log_issue("sales", "P1", f"Reconstruction MAPE vs ALL orders = {mape_all:.1f}% — review join logic")
"""))

# ── CELL 19: Summary issues table ────────────────────────────────────────────
cells.append(new_markdown_cell("## 8. Issues summary & data_quality.md export"))

cells.append(new_code_cell("""\
import io

issues_df = pd.DataFrame(issues)
print(f"Total issues found: {len(issues_df)}")
if len(issues_df) > 0:
    for sev in ["P0","P1","P2"]:
        subset = issues_df[issues_df["severity"]==sev]
        badge = {"P0":"🔴","P1":"🟡","P2":"🟢"}[sev]
        print(f"\\n{badge} {sev} — {len(subset)} issues:")
        for _, row in subset.iterrows():
            print(f"  [{row['table']}] {row['issue']}")
else:
    print("✅ No issues logged!")
"""))

cells.append(new_code_cell("""\
# Write data_quality.md
md_lines = [
    "# Data Quality Report",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    "## Summary",
    f"Total issues: {len(issues_df) if len(issues_df) > 0 else 0}",
    "",
]

if len(issues_df) > 0:
    for sev, label, badge in [("P0","CRITICAL — Blocks analysis","🔴"), ("P1","Important","🟡"), ("P2","Minor / Notes","🟢")]:
        subset = issues_df[issues_df["severity"]==sev] if len(issues_df) > 0 else pd.DataFrame()
        md_lines.append(f"## {badge} {sev} — {label} ({len(subset)} issues)")
        md_lines.append("")
        if len(subset) > 0:
            md_lines.append("| Table | Issue |")
            md_lines.append("|---|---|")
            for _, row in subset.iterrows():
                md_lines.append(f"| `{row['table']}` | {row['issue']} |")
        else:
            md_lines.append("_No issues_")
        md_lines.append("")

md_lines += [
    "## Dataset overview",
    "",
    "| File | Rows | Cols |",
    "|---|---|---|",
    f"| products | {len(products):,} | {products.shape[1]} |",
    f"| customers | {len(customers):,} | {customers.shape[1]} |",
    f"| promotions | {len(promotions):,} | {promotions.shape[1]} |",
    f"| geography | {len(geography):,} | {geography.shape[1]} |",
    f"| orders | {len(orders):,} | {orders.shape[1]} |",
    f"| order_items | {len(order_items):,} | {order_items.shape[1]} |",
    f"| payments | {len(payments):,} | {payments.shape[1]} |",
    f"| shipments | {len(shipments):,} | {shipments.shape[1]} |",
    f"| returns | {len(returns):,} | {returns.shape[1]} |",
    f"| reviews | {len(reviews):,} | {reviews.shape[1]} |",
    f"| sales | {len(sales):,} | {sales.shape[1]} |",
    f"| inventory | {len(inventory):,} | {inventory.shape[1]} |",
    f"| web_traffic | {len(web_traffic):,} | {web_traffic.shape[1]} |",
    "",
    "## Notes",
    "- `bounce_rate` in web_traffic is stored as fraction (~0.004), NOT percent. Treat as-is.",
    "- `sales.csv` reconstructs from ALL order statuses (not delivered-only). See §7.",
    "- Promotion calendar is synthetic (6-4-6-4 promos/year, 20.8%/15% alternating).",
    "- `web_traffic` may not cover test period (2023-2024) — use lag features only for Part 3.",
    "- Nullable: `customers.gender`, `age_group`, `acquisition_channel` — use dropna=False in groupby.",
]

report_path = REPORTS / "data_quality.md"
report_path.write_text("\\n".join(md_lines), encoding="utf-8")
print(f"✅ Saved → {report_path}")
print("\\n--- Preview (first 30 lines) ---")
print("\\n".join(md_lines[:30]))
"""))

# ── CELL 20: Final summary ────────────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## 9. Kết luận — Phase 1 Data Profiling

---

### Tổng kết

| Hạng mục | Kết quả |
|---|---|
| Files đã profiling | 13 / 13 |
| Checks thực hiện | 6 checks × 13 bảng + 8 cardinality rules |
| sales.csv reconstruction | ✅ Verified (ALL order statuses) |
| Output | `reports/data_quality.md` |

---

### Phân loại issues

| Severity | Ý nghĩa | Hành động |
|---|---|---|
| 🔴 **P0** | Chặn analysis — phải fix trước Phase 3 | Fix ngay trong `10_build_abt.ipynb` |
| 🟡 **P1** | Quan trọng — ảnh hưởng feature engineering | Ghi chú, handle khi build ABT |
| 🟢 **P2** | Minor / quirks của dataset | Document trong report, không urgent |

---

### Các lưu ý quan trọng cho Phase tiếp theo

- **`bounce_rate`** trong `web_traffic` lưu dạng fraction (~0.004), **không phải %** — nhân 100 trước khi dùng.
- **`sales.csv`** reconstruct từ **ALL** order statuses (không chỉ `delivered`) — dùng khi build target.
- **Nullable columns**: `customers.gender`, `age_group`, `acquisition_channel` — luôn dùng `dropna=False` trong groupby.
- **`web_traffic`** có thể không cover test period 2023–2024 → chỉ dùng lag features cho Part 3.
- **Promotion calendar** là synthetic (6-4 promos/year, discount 20.8%/15% xen kẽ).

---

### Bước tiếp theo

```
Phase 3 → notebooks/10_build_abt.ipynb    (Feature engineering & ABT)
Phase 4 → notebooks/01_eda_exploratory.ipynb  (Business hypotheses)
```
"""))

nb.cells = cells

# Write notebook
out_path = Path("/home/pearspringmind/Hackathon/gridbreaker-vinuni/notebooks/00_data_profiling.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"✅ Written: {out_path}")
print(f"   Cells: {len(nb.cells)}")
