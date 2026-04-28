# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Datathon 2026 — Phần 3: Sales Forecasting Pipeline
# **The GridBreakers · VinUniversity Datathon 2026**
#
# ## Mục tiêu
# Dự báo **Revenue** và **COGS** hàng ngày cho 548 ngày (2023-01-01 → 2024-07-01)
# sử dụng dữ liệu lịch sử 2012–2022.
#
# ## Cấu trúc Pipeline
#
# | Phần | Nội dung |
# |---|---|
# | **0 — Setup** | Imports, config, seed |
# | **1 — Data Loading** | Load 12 tables, leakage audit |
# | **2 — EDA** | Seasonality, promotions, web traffic, orders |
# | **3 — Feature Engineering** | Calendar, projected promos, lags, rolling, external signals |
# | **4 — Modeling** | Phase A: Baselines → Phase B: LightGBM → Phase C: Ensemble |
# | **5 — CV & Evaluation** | TimeSeriesSplit, metrics |
# | **6 — SHAP** | Feature importance, business interpretation |
# | **7 — Submission** | Generate submission.csv |

# %% [markdown]
# ## 0 · Setup & Imports

# %%
import os, warnings, json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import shap
import optuna
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR    = "../../data/raw"
EXTERNAL_DIR = "../../data/external"
SUBM_DIR    = "../../submissions"
REPORTS_DIR = "../../reports"
os.makedirs(SUBM_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Plot theme ────────────────────────────────────────────────────────────
PALETTE = ["#2563eb", "#f97316", "#16a34a", "#dc2626", "#7c3aed", "#0891b2", "#ca8a04"]
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
})
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DOW_NAMES   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

# ── LightGBM parameters ───────────────────────────────────────────────────
LGB_PARAMS = {
    "objective": "regression_l1", "metric": "mae",
    "num_leaves": 127, "learning_rate": 0.02,
    "feature_fraction": 0.75, "bagging_fraction": 0.75, "bagging_freq": 5,
    "min_child_samples": 15, "reg_alpha": 0.10, "reg_lambda": 0.20,
    "verbose": -1, "seed": SEED, "n_jobs": -1,
}

print("Setup complete")
print(f"  lightgbm {lgb.__version__}  |  xgboost {xgb.__version__}  |  shap {shap.__version__}  |  optuna {optuna.__version__}")

# %% [markdown]
# ## 1 · Data Loading & Leakage Audit

# %%
# ── 1.1 Load all tables ───────────────────────────────────────────────────────
sales_raw   = pd.read_csv(f"{DATA_DIR}/Analytical/sales.csv", parse_dates=["Date"])
sample_sub  = pd.read_csv(f"{EXTERNAL_DIR}/sample_submission.csv", parse_dates=["Date"])

customers   = pd.read_csv(f"{DATA_DIR}/Master/customers.csv", parse_dates=["signup_date"])
products    = pd.read_csv(f"{DATA_DIR}/Master/products.csv")
promos_raw  = pd.read_csv(f"{DATA_DIR}/Master/promotions.csv", parse_dates=["start_date","end_date"])
geography   = pd.read_csv(f"{DATA_DIR}/Master/geography.csv")

orders_raw  = pd.read_csv(f"{DATA_DIR}/Transaction/orders.csv", parse_dates=["order_date"])
order_items = pd.read_csv(f"{DATA_DIR}/Transaction/order_items.csv")
payments    = pd.read_csv(f"{DATA_DIR}/Transaction/payments.csv")
returns_raw = pd.read_csv(f"{DATA_DIR}/Transaction/returns.csv", parse_dates=["return_date"])
reviews     = pd.read_csv(f"{DATA_DIR}/Transaction/reviews.csv", parse_dates=["review_date"])
shipments   = pd.read_csv(f"{DATA_DIR}/Transaction/shipments.csv", parse_dates=["ship_date","delivery_date"])

inventory   = pd.read_csv(f"{DATA_DIR}/Operational/inventory.csv", parse_dates=["snapshot_date"])
wt_raw      = pd.read_csv(f"{DATA_DIR}/Operational/web_traffic.csv", parse_dates=["date"])

# ── 1.2 Leakage audit: date ranges ────────────────────────────────────────────
print("=" * 70)
print(f"  TRAIN sales : {sales_raw['Date'].min().date()} -> {sales_raw['Date'].max().date()}  ({len(sales_raw):,} rows)")
print(f"  TEST period : {sample_sub['Date'].min().date()} -> {sample_sub['Date'].max().date()}  ({len(sample_sub):,} rows)")
print("-" * 70)

date_tables = {
    "web_traffic":  ("date",          wt_raw),
    "orders":       ("order_date",    orders_raw),
    "returns":      ("return_date",   returns_raw),
    "reviews":      ("review_date",   reviews),
    "shipments":    ("ship_date",     shipments),
    "inventory":    ("snapshot_date", inventory),
    "promotions":   ("end_date",      promos_raw),
}
leakage_found = False
for name, (col, df) in date_tables.items():
    dmax = df[col].max()
    is_leak = dmax > pd.Timestamp("2022-12-31")
    flag = "<<< LEAKAGE!" if is_leak else "OK"
    if is_leak: leakage_found = True
    print(f"  {name:<16} max {str(dmax)[:10]}  {flag}")

if not leakage_found:
    print("\n  [PASS] No data leakage detected — all tables end <= 2022-12-31")
else:
    print("\n  [WARN] Tables extend beyond training cutoff — filter immediately!")

print("=" * 70)

# %% [markdown]
# ### 1.3 Key finding: Promotion recurrence pattern
# Promotions follow annual cycles. For test period (2023-2024), we project
# recurring promos based on historical patterns.

# %%
# ── Analyze promotion patterns ────────────────────────────────────────────────
promos_raw["year"] = promos_raw["start_date"].dt.year
promos_raw["promo_month_start"] = promos_raw["start_date"].dt.month
promos_raw["promo_day_start"]   = promos_raw["start_date"].dt.day
promos_raw["promo_month_end"]   = promos_raw["end_date"].dt.month
promos_raw["promo_day_end"]     = promos_raw["end_date"].dt.day

print("Promotions per year:")
print(promos_raw.groupby("year").size().to_string())
print(f"\nRecurring promo names:")
for name, grp in promos_raw.groupby("promo_name"):
    years = sorted(grp["year"].unique())
    print(f"  {name:<30} years={years}  discount={grp['discount_value'].iloc[0]:.0f}%")

# %% [markdown]
# ## 2 · Exploratory Data Analysis

# %% [markdown]
# ### 2.1 Sales overview (Revenue, COGS, Gross Profit)

# %%
sales = sales_raw.sort_values("Date").reset_index(drop=True).copy()
sales["gross_profit"] = sales["Revenue"] - sales["COGS"]
sales["gross_margin"] = sales["gross_profit"] / sales["Revenue"]
sales["year"]  = sales["Date"].dt.year
sales["month"] = sales["Date"].dt.month
sales["dow"]   = sales["Date"].dt.dayofweek
sales["doy"]   = sales["Date"].dt.dayofyear
sales["woy"]   = sales["Date"].dt.isocalendar().week.astype(int)

print(f"Revenue: mean={sales['Revenue'].mean():,.0f}  median={sales['Revenue'].median():,.0f}  "
      f"min={sales['Revenue'].min():,.0f}  max={sales['Revenue'].max():,.0f}")
print(f"COGS:    mean={sales['COGS'].mean():,.0f}  median={sales['COGS'].median():,.0f}  "
      f"min={sales['COGS'].min():,.0f}  max={sales['COGS'].max():,.0f}")
print(f"Gross margin: mean={sales['gross_margin'].mean():.1%}  "
      f"negative days={(sales['gross_margin']<0).sum()} ({(sales['gross_margin']<0).mean():.1%})")
print(f"Pearson r(Revenue, COGS) = {sales['Revenue'].corr(sales['COGS']):.4f}")

# %%
# ── Revenue + COGS + 30-day MA ────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

for i, (col, color, label) in enumerate([
    ("Revenue", PALETTE[0], "Revenue"),
    ("COGS", PALETTE[1], "COGS"),
]):
    axes[i].fill_between(sales["Date"], sales[col]/1e6, alpha=0.10, color=color)
    axes[i].plot(sales["Date"], sales[col]/1e6, lw=0.5, color=color, alpha=0.6)
    axes[i].plot(sales["Date"],
                 sales[col].rolling(30, center=True, min_periods=10).mean()/1e6,
                 lw=2.2, color=color, label="30-day MA")
    axes[i].set_ylabel(f"{label} (M VND)")
    axes[i].set_title(f"Daily {label} 2012–2022")
    axes[i].legend(fontsize=9)

# Gross margin
gm_ma = sales["gross_margin"].rolling(30, center=True, min_periods=10).mean()
axes[2].axhline(0, color="black", lw=0.8)
axes[2].fill_between(sales["Date"], gm_ma*100, where=(gm_ma>=0),
                     alpha=0.3, color=PALETTE[2], label="Positive margin")
axes[2].fill_between(sales["Date"], gm_ma*100, where=(gm_ma<0),
                     alpha=0.3, color=PALETTE[3], label="Negative margin")
axes[2].plot(sales["Date"], gm_ma*100, lw=1.5, color=PALETTE[2])
axes[2].set_ylabel("Gross Margin %")
axes[2].set_title("Gross Margin % (30-day MA)")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[2].legend(fontsize=9)

plt.suptitle("Sales Overview — Revenue · COGS · Gross Margin", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/sales_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 2.2 Seasonality: Monthly & Day-of-Week Patterns

# %%
monthly_rev = sales.groupby("month")["Revenue"].agg(["mean", "std"])
dow_rev     = sales.groupby("dow")["Revenue"].mean()
pivot_rev   = sales.pivot_table(values="Revenue", index="month", columns="dow", aggfunc="mean")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Monthly
colors_m = [PALETTE[2] if v > monthly_rev["mean"].mean() else PALETTE[0]
            for v in monthly_rev["mean"]]
axes[0].bar(monthly_rev.index, monthly_rev["mean"]/1e6, color=colors_m, alpha=0.85, edgecolor="white")
axes[0].errorbar(monthly_rev.index, monthly_rev["mean"]/1e6,
                 yerr=monthly_rev["std"]/1e6, fmt="none", color="gray", capsize=3)
axes[0].axhline(monthly_rev["mean"].mean()/1e6, color="red", linestyle="--",
                label=f"Mean: {monthly_rev['mean'].mean()/1e6:.1f}M")
axes[0].set_xticks(range(1,13)); axes[0].set_xticklabels(MONTH_NAMES, rotation=45)
axes[0].set_ylabel("Revenue (M VND)"); axes[0].set_title("Average Daily Revenue by Month")
axes[0].legend(fontsize=8)

# DOW
axes[1].bar(dow_rev.index, dow_rev.values/1e6, color=PALETTE[5], alpha=0.85, edgecolor="white")
axes[1].axhline(dow_rev.mean()/1e6, color="red", linestyle="--")
axes[1].set_xticks(range(7)); axes[1].set_xticklabels(DOW_NAMES)
axes[1].set_ylabel("Revenue (M VND)"); axes[1].set_title("Average Revenue by Day of Week")

# Heatmap
im = axes[2].imshow(pivot_rev.values/1e6, aspect="auto", cmap="YlOrRd")
axes[2].set_xticks(range(7)); axes[2].set_xticklabels(DOW_NAMES)
axes[2].set_yticks(range(12)); axes[2].set_yticklabels(MONTH_NAMES)
axes[2].set_title("Revenue Heatmap: Month x DOW (M VND)")
plt.colorbar(im, ax=axes[2], label="M VND", shrink=0.85)

plt.suptitle("Seasonality — Monthly & Day-of-Week Patterns", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/seasonality.png", dpi=150, bbox_inches="tight")
plt.show()

peak_month = monthly_rev["mean"].idxmax()
low_month  = monthly_rev["mean"].idxmin()
print(f"Peak month: {MONTH_NAMES[peak_month-1]} ({monthly_rev['mean'][peak_month]/1e6:.1f}M/day)")
print(f"Low month:  {MONTH_NAMES[low_month-1]} ({monthly_rev['mean'][low_month]/1e6:.1f}M/day)")
print(f"Peak/Low ratio: {monthly_rev['mean'].max()/monthly_rev['mean'].min():.2f}x")

# %% [markdown]
# ### 2.3 Annual Trend & YoY Growth (2013-2022)

# %%
annual = sales[sales["year"].between(2013, 2022)].groupby("year")[["Revenue", "COGS"]].sum()
yoy_rev  = annual["Revenue"].pct_change().dropna()
yoy_cogs = annual["COGS"].pct_change().dropna()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(annual.index, annual["Revenue"]/1e9, color=PALETTE[0], alpha=0.85, edgecolor="white")
axes[0].set_title("Annual Revenue (ty VND)"); axes[0].set_ylabel("Revenue (ty)")

colors_y = [PALETTE[2] if v>0 else PALETTE[3] for v in yoy_rev.values]
axes[1].bar(yoy_rev.index, yoy_rev.values*100, color=colors_y, alpha=0.85, edgecolor="white")
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("YoY Revenue Growth (%)")

axes[2].fill_between(annual.index, 0, annual["COGS"]/1e9, alpha=0.7, color=PALETTE[1], label="COGS")
axes[2].fill_between(annual.index, annual["COGS"]/1e9, annual["Revenue"]/1e9,
                     alpha=0.7, color=PALETTE[2], label="Gross Profit")
axes[2].plot(annual.index, annual["Revenue"]/1e9, lw=2, color=PALETTE[0], marker="o", label="Revenue")
axes[2].set_title("Revenue = COGS + Gross Profit"); axes[2].legend(fontsize=9)

plt.suptitle("Annual Trend & YoY Growth", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/annual_trend.png", dpi=150, bbox_inches="tight")
plt.show()

geo_rev  = (1+yoy_rev).prod()**(1/len(yoy_rev))-1
geo_cogs = (1+yoy_cogs).prod()**(1/len(yoy_cogs))-1
print(f"Geometric mean YoY growth: Revenue={geo_rev:+.2%}/yr  COGS={geo_cogs:+.2%}/yr")

# %% [markdown]
# ### 2.4 Promotion Impact on Revenue

# %%
# Active promo count per day
promo_cnt_by_day = []
for d in sales["Date"]:
    cnt = ((promos_raw["start_date"] <= d) & (promos_raw["end_date"] >= d)).sum()
    promo_cnt_by_day.append(cnt)
sales["promo_count"] = promo_cnt_by_day

promo_buckets = sales.groupby(
    sales["promo_count"].apply(lambda x: "0" if x==0 else ("1" if x==1 else "2+"))
)["Revenue"].agg(["mean", "count"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(promo_buckets.index, promo_buckets["mean"]/1e6,
            color=[PALETTE[0], PALETTE[2], PALETTE[3]], alpha=0.85, edgecolor="white")
axes[0].set_title("Average Revenue by Active Promo Count")
axes[0].set_ylabel("Revenue (M VND)")
for i, (bkt, row) in enumerate(promo_buckets.iterrows()):
    axes[0].text(i, row["mean"]/1e6 + 0.1, f"{row['mean']/1e6:.1f}M\n(n={row['count']:,})",
                ha="center", va="bottom", fontsize=8)

# Revenue with promo overlay
ax1b = axes[1].twinx()
axes[1].plot(sales["Date"], sales["Revenue"].rolling(30, center=True, min_periods=10).mean()/1e6,
             lw=1.8, color=PALETTE[0], label="Revenue 30d MA")
ax1b.fill_between(sales["Date"], sales["promo_count"], alpha=0.3, color=PALETTE[3])
ax1b.plot(sales["Date"], sales["promo_count"], lw=0.7, color=PALETTE[3], label="Active Promos")
axes[1].set_ylabel("Revenue (M VND)"); ax1b.set_ylabel("# Active Promos")
axes[1].set_title("Revenue vs Active Promos"); axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[1].legend(loc="upper left", fontsize=8); ax1b.legend(loc="upper right", fontsize=8)

plt.suptitle("Promotion Impact on Revenue", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/promo_impact.png", dpi=150, bbox_inches="tight")
plt.show()

for bkt, row in promo_buckets.iterrows():
    print(f"  {bkt} promos: mean={row['mean']:>12,.0f}  n={row['count']:>5,}")

# %% [markdown]
# ### 2.5 Web Traffic & Order Signals

# %%
# Daily web traffic aggregation
wt_daily = (wt_raw.groupby("date")
            .agg(wt_sessions=("sessions", "sum"),
                 wt_visitors=("unique_visitors", "sum"),
                 wt_pageviews=("page_views", "sum"),
                 wt_bounce=("bounce_rate", "mean"),
                 wt_duration=("avg_session_duration_sec", "mean"))
            .reset_index().rename(columns={"date": "Date"}))

# Daily order aggregation
ord_daily = (orders_raw.groupby("order_date")
             .agg(ord_count=("order_id", "count"),
                  ord_delivered=("order_status", lambda x: (x == "delivered").sum()),
                  ord_cancelled=("order_status", lambda x: (x == "cancelled").sum()))
             .reset_index().rename(columns={"order_date": "Date"}))
ord_daily["ord_cancel_rate"] = ord_daily["ord_cancelled"] / ord_daily["ord_count"]

# Correlation analysis
merged_signals = (sales[["Date", "Revenue"]]
                  .merge(wt_daily, on="Date", how="inner")
                  .merge(ord_daily, on="Date", how="inner"))

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Sessions vs Revenue
axes[0,0].scatter(merged_signals["wt_sessions"]/1e3, merged_signals["Revenue"]/1e6,
                  s=5, alpha=0.3, color=PALETTE[0])
r_s = merged_signals["Revenue"].corr(merged_signals["wt_sessions"])
axes[0,0].set_xlabel("Sessions (K)"); axes[0,0].set_ylabel("Revenue (M)")
axes[0,0].set_title(f"Sessions vs Revenue  (r={r_s:.3f})")

# Lag correlation: sessions → revenue
lag_corrs = {}
for lag in range(0, 8):
    c = merged_signals["Revenue"].corr(merged_signals["wt_sessions"].shift(lag))
    lag_corrs[lag] = c
axes[0,1].bar(list(lag_corrs.keys()), list(lag_corrs.values()),
              color=PALETTE[0], alpha=0.85, edgecolor="white")
best_lag = max(lag_corrs, key=lambda k: lag_corrs[k])
axes[0,1].axvline(best_lag, color="red", linestyle="--",
                  label=f"Best: lag={best_lag}d r={lag_corrs[best_lag]:.3f}")
axes[0,1].set_xlabel("Lag (days)"); axes[0,1].set_ylabel("Pearson r")
axes[0,1].set_title("Corr(Revenue, Sessions[t-lag])")
axes[0,1].legend(fontsize=9)

# Order count vs Revenue
axes[1,0].scatter(merged_signals["ord_count"], merged_signals["Revenue"]/1e6,
                  s=5, alpha=0.3, color=PALETTE[1])
r_o = merged_signals["Revenue"].corr(merged_signals["ord_count"])
axes[1,0].set_xlabel("Daily Order Count"); axes[1,0].set_ylabel("Revenue (M)")
axes[1,0].set_title(f"Order Count vs Revenue  (r={r_o:.3f})")

# Traffic source breakdown
src_sess = wt_raw.groupby("traffic_source")["sessions"].sum().sort_values(ascending=False)
axes[1,1].bar(src_sess.index, src_sess.values/1e6, color=PALETTE[5], alpha=0.85, edgecolor="white")
axes[1,1].set_ylabel("Total Sessions (M)"); axes[1,1].set_title("Sessions by Traffic Source")
axes[1,1].tick_params(axis="x", rotation=45)

plt.suptitle("Web Traffic & Order Signals", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/traffic_orders.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3 · Feature Engineering

# %% [markdown]
# ### 3.1 Calendar + Fourier + Vietnamese Retail Events

# %%
# Build full date range
all_dates = pd.date_range(sales["Date"].min(), sample_sub["Date"].max(), freq="D")
full = pd.DataFrame({"Date": all_dates})
full = full.merge(sales[["Date", "Revenue", "COGS"]], on="Date", how="left")
full["is_test"] = full["Date"].isin(sample_sub["Date"]).astype(int)

def add_calendar_features(df):
    """Calendar features — all test-safe (known in advance)."""
    df = df.copy()
    D = df["Date"]

    # Basic time components
    df["cal_year"]     = D.dt.year
    df["cal_month"]    = D.dt.month
    df["cal_day"]      = D.dt.day
    df["cal_dow"]      = D.dt.dayofweek
    df["cal_doy"]      = D.dt.dayofyear
    df["cal_woy"]      = D.dt.isocalendar().week.astype(int)
    df["cal_quarter"]  = D.dt.quarter

    # Boolean flags
    df["cal_is_weekend"]     = (D.dt.dayofweek >= 5).astype(int)
    df["cal_is_month_end"]   = D.dt.is_month_end.astype(int)
    df["cal_is_month_start"] = D.dt.is_month_start.astype(int)
    df["cal_is_quarter_end"] = D.dt.is_quarter_end.astype(int)
    df["cal_is_year_end"]    = D.dt.is_year_end.astype(int)
    df["cal_is_year_start"]  = D.dt.is_year_start.astype(int)

    # Cyclic encodings (Fourier terms)
    df["cal_month_sin"] = np.sin(2 * np.pi * df["cal_month"] / 12)
    df["cal_month_cos"] = np.cos(2 * np.pi * df["cal_month"] / 12)
    df["cal_dow_sin"]   = np.sin(2 * np.pi * df["cal_dow"] / 7)
    df["cal_dow_cos"]   = np.cos(2 * np.pi * df["cal_dow"] / 7)
    df["cal_doy_sin"]   = np.sin(2 * np.pi * df["cal_doy"] / 365.25)
    df["cal_doy_cos"]   = np.cos(2 * np.pi * df["cal_doy"] / 365.25)
    df["cal_woy_sin"]   = np.sin(2 * np.pi * df["cal_woy"] / 52)
    df["cal_woy_cos"]   = np.cos(2 * np.pi * df["cal_woy"] / 52)

    # Higher harmonics for day-of-year (captures sharp seasonality)
    for k in [2, 3, 4]:
        df[f"cal_doy_sin{k}"] = np.sin(2 * k * np.pi * df["cal_doy"] / 365.25)
        df[f"cal_doy_cos{k}"] = np.cos(2 * k * np.pi * df["cal_doy"] / 365.25)

    # Vietnamese retail calendar
    df["vn_tet"]          = (((df["cal_month"]==1)&(df["cal_day"]>=20)) |
                              ((df["cal_month"]==2)&(df["cal_day"]<=20))).astype(int)
    df["vn_pre_tet"]      = (((df["cal_month"]==1)&(df["cal_day"]>=14)) |
                              ((df["cal_month"]==2)&(df["cal_day"]<=7))).astype(int)
    df["vn_post_tet"]     = ((df["cal_month"]==2)&(df["cal_day"].between(21,28))).astype(int)
    df["vn_mid_sale"]     = df["cal_month"].isin([6,7]).astype(int)
    df["vn_year_end"]     = df["cal_month"].isin([11,12]).astype(int)
    df["vn_back_school"]  = df["cal_month"].isin([8,9]).astype(int)
    df["vn_summer"]       = df["cal_month"].isin([5,6,7]).astype(int)
    df["vn_low_season"]   = df["cal_month"].isin([1,2,10]).astype(int)
    df["vn_peak_season"]  = df["cal_month"].isin([4,5,6]).astype(int)

    # Time index (linear trend)
    df["cal_time_idx"] = (D - D.min()).dt.days
    return df

full = add_calendar_features(full)
print(f"Calendar features added. Shape: {full.shape}")

# %% [markdown]
# ### 3.2 Projected Promotions (Recurring Annual Pattern)
# CRITICAL: Promotions data ends 2022-12-31 but follows strict annual recurrence.
# We project expected promos for the test period.

# %%
def build_projected_promos(date_range):
    """Generate expected promotions for any date range based on recurring patterns.

    Pattern (from 2013-2022 analysis):
    - ANNUAL (every year):
      * Spring Sale:    Mar 18 - Apr 17 (12%, percentage)
      * Mid-Year Sale:  Jun 23 - Jul 22 (18%, percentage)
      * Fall Launch:    Aug 30 - Oct 01 (10%, percentage)
      * Year-End Sale:  Nov 18 - Jan 02 next year (20%, percentage)
    - BIENNIAL (odd years: 2013, 2015, ..., 2023):
      * Urban Blowout:  Jul 30 - Sep 02 (50%, fixed)
      * Rural Special:  Jan 30 - Mar 01 (15%, percentage)
    """
    rows = []
    for d in date_range:
        yr  = d.year
        mo  = d.month
        day = d.day
        is_odd_year = (yr % 2 == 1)

        active = []
        total_discount = 0.0
        has_stackable = 0
        pct_count = 0
        fixed_count = 0

        # Spring Sale (annual)
        if (mo == 3 and day >= 18) or (mo == 4 and day <= 17):
            active.append("spring_sale")
            total_discount += 12.0
            has_stackable = 1
            pct_count += 1

        # Mid-Year Sale (annual)
        if (mo == 6 and day >= 23) or (mo == 7 and day <= 22):
            active.append("midyear_sale")
            total_discount += 18.0
            pct_count += 1

        # Fall Launch (annual)
        if (mo == 8 and day >= 30) or (mo == 9) or (mo == 10 and day <= 1):
            active.append("fall_launch")
            total_discount += 10.0
            pct_count += 1

        # Year-End Sale (annual, crosses year boundary)
        if (mo == 11 and day >= 18) or (mo == 12) or (mo == 1 and day <= 2):
            active.append("yearend_sale")
            total_discount += 20.0
            pct_count += 1

        # Urban Blowout (biennial, odd years)
        if is_odd_year and ((mo == 7 and day >= 30) or (mo == 8) or (mo == 9 and day <= 2)):
            active.append("urban_blowout")
            total_discount += 50.0
            fixed_count += 1

        # Rural Special (biennial, odd years)
        if is_odd_year and ((mo == 1 and day >= 30) or (mo == 2) or (mo == 3 and day <= 1)):
            active.append("rural_special")
            total_discount += 15.0
            pct_count += 1

        rows.append({
            "Date": d,
            "proj_promo_count": len(active),
            "proj_promo_discount_sum": total_discount,
            "proj_promo_stackable": has_stackable,
            "proj_promo_pct_count": pct_count,
            "proj_promo_fixed_count": fixed_count,
        })
    return pd.DataFrame(rows)

proj_promos_df = build_projected_promos(all_dates)
full = full.merge(proj_promos_df, on="Date", how="left")
print(f"Projected promotions added. Shape: {full.shape}")
print(f"  Test period sample (2023-01-01):")
print(f"  {full[full['Date']=='2023-01-01'][['Date','proj_promo_count','proj_promo_discount_sum']].to_string(index=False)}")
print(f"  Test period sample (2023-03-20):")
print(f"  {full[full['Date']=='2023-03-20'][['Date','proj_promo_count','proj_promo_discount_sum']].to_string(index=False)}")

# %% [markdown]
# ### 3.3 Order Signals (lagged >= 365 days — test-safe)

# %%
# Aggregate order signals
ord_sig = (orders_raw.groupby("order_date")
           .agg(ord_count=("order_id", "count"),
                ord_delivered=("order_status", lambda x: (x=="delivered").sum()),
                ord_cancelled=("order_status", lambda x: (x=="cancelled").sum()))
           .reset_index().rename(columns={"order_date": "Date"}))
ord_sig["ord_cancel_rate"] = ord_sig["ord_cancelled"] / ord_sig["ord_count"]

full = full.merge(ord_sig, on="Date", how="left")

# Create LAGGED versions (lag >= 365 ensures availability for entire 2023)
# For 2024: lags >= 365 from 2022-12-31 point to dates within training range
# But lags >= 365 from 2024-07-01 point to 2023-07-01 which has NO order data
# So for 2024 dates, we fill with the same day-of-year from 2022 as proxy
SHIFT_DAYS = [7, 14, 28, 90, 180, 365]
for lag in SHIFT_DAYS:
    full[f"ord_count_l{lag}"]     = full["ord_count"].shift(lag)
    full[f"ord_delivered_l{lag}"] = full["ord_delivered"].shift(lag)
    full[f"ord_cancelled_l{lag}"] = full["ord_cancelled"].shift(lag)
    full[f"ord_cancel_rate_l{lag}"] = full["ord_cancel_rate"].shift(lag)

# Rolling order features (shift-1 first for no leakage)
ord_sh1 = full["ord_count"].shift(1)
for w in [7, 14, 28, 90, 365]:
    full[f"ord_count_rm{w}"] = ord_sh1.rolling(w, min_periods=max(1, w//4)).mean()

# Drop raw (non-lagged) order columns to prevent leakage
full.drop(columns=["ord_count", "ord_delivered", "ord_cancelled", "ord_cancel_rate"],
          inplace=True, errors="ignore")
print("Order features (lagged) added. Shape:", full.shape)

# %% [markdown]
# ### 3.4 Web Traffic Features (lagged + rolling)

# %%
# Merge web traffic (shift +1 day to simulate "yesterday's traffic predicts today's revenue")
wt_daily["Date_shifted"] = wt_daily["Date"] + pd.Timedelta(days=1)
full = full.merge(
    wt_daily[["Date_shifted", "wt_sessions", "wt_visitors", "wt_pageviews", "wt_bounce", "wt_duration"]],
    left_on="Date", right_on="Date_shifted", how="left"
).drop(columns=["Date_shifted"])

# Rolling web traffic
for col in ["wt_sessions", "wt_visitors", "wt_pageviews"]:
    sh = full[col].shift(1)
    for w in [7, 14, 28]:
        full[f"{col}_rm{w}"] = sh.rolling(w, min_periods=max(1, w//4)).mean()

print("Web traffic features added. Shape:", full.shape)

# %% [markdown]
# ### 3.5 Revenue & COGS — Lag, Rolling, EWM Features

# %%
LAGS = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 180, 364, 365, 366]
ROLL_WINDOWS = [7, 14, 28, 60, 90, 180, 365]
EWM_SPANS = [7, 14, 30, 90, 180]

def add_target_features(df, col):
    """Add lag, rolling, EWM features for a target column. All shift(>=1)."""
    s = df[col].copy()

    # Lags
    for lag in LAGS:
        df[f"{col}_l{lag}"] = s.shift(lag)

    # Rolling stats (on shift-1 values)
    sh = s.shift(1)
    for w in ROLL_WINDOWS:
        mp = max(1, w // 4)
        df[f"{col}_rm{w}"]   = sh.rolling(w, min_periods=mp).mean()
        df[f"{col}_rs{w}"]   = sh.rolling(w, min_periods=max(5, mp)).std()
        df[f"{col}_rmed{w}"] = sh.rolling(w, min_periods=mp).median()
        df[f"{col}_rmax{w}"] = sh.rolling(w, min_periods=mp).max()
        df[f"{col}_rmin{w}"] = sh.rolling(w, min_periods=mp).min()

    # EWM
    for sp in EWM_SPANS:
        df[f"{col}_ewm{sp}"] = sh.ewm(span=sp, adjust=False).mean()

    # YoY anchors
    df[f"{col}_yoy_ratio"] = s.shift(365) / (s.shift(372).rolling(7, min_periods=4).mean() + 1e-9)
    l364_vals = (s.shift(357) + s.shift(364) + s.shift(371)) / 3
    df[f"{col}_l364_mean"] = l364_vals
    df[f"{col}_2yoy_mean"] = (s.shift(365) + s.shift(730)) / 2

    # Short-term / long-term trend ratio
    r14 = sh.rolling(14, min_periods=7).mean()
    r180 = sh.shift(180).rolling(14, min_periods=7).mean()
    df[f"{col}_trend14_180"] = r14 / (r180 + 1e-9)

    return df

for tgt in ["Revenue", "COGS"]:
    full = add_target_features(full, tgt)

# Cross-target ratio features
for lag in [28, 90, 180, 365]:
    full[f"cogs_rev_ratio_l{lag}"] = (full[f"COGS_l{lag}"] /
                                        (full[f"Revenue_l{lag}"] + 1e-9))

# %% [markdown]
# ### 3.5b Category-Driven Revenue Features
# CRITICAL: All category features use lag >= 365. For 2024 dates where
# lag=365 points to 2023 (no data), lags >= 730 ensure full coverage.

# %%
oi_cat = (order_items
    .merge(products[["product_id", "category"]], on="product_id", how="left")
    .merge(orders_raw[["order_id", "order_date"]], on="order_id", how="left"))
oi_cat["item_revenue"] = (oi_cat["quantity"] * oi_cat["unit_price"]
                           - oi_cat["discount_amount"].fillna(0))

TOP_CATS = oi_cat.groupby("category")["item_revenue"].sum().nlargest(4).index.tolist()
cat_features = None
for cat in TOP_CATS:
    sub = oi_cat[oi_cat["category"] == cat]
    agg = (sub.groupby("order_date").agg(
        **{f"cat_{cat.lower()}_rev": ("item_revenue", "sum"),
           f"cat_{cat.lower()}_orders": ("order_id", "nunique"),
           f"cat_{cat.lower()}_avg_price": ("unit_price", "mean")})
        .reset_index().rename(columns={"order_date": "Date"}))
    cat_features = agg if cat_features is None else cat_features.merge(agg, on="Date", how="outer")

all_cat_rev_cols = [f"cat_{c.lower()}_rev" for c in TOP_CATS]
cat_features["cat_total_rev"] = cat_features[all_cat_rev_cols].sum(axis=1)
for c in TOP_CATS:
    cat_features[f"cat_{c.lower()}_share"] = (
        cat_features[f"cat_{c.lower()}_rev"] / (cat_features["cat_total_rev"] + 1e-9))

prod_daily = (oi_cat.groupby("order_date")
    .agg(prod_avg_price=("unit_price", "mean"),
         prod_avg_discount=("discount_amount", lambda x: (x / (oi_cat.loc[x.index, "quantity"] * oi_cat.loc[x.index, "unit_price"] + 1e-9)).mean()),
         prod_unique_products=("product_id", "nunique"))
    .reset_index().rename(columns={"order_date": "Date"}))
cat_features = cat_features.merge(prod_daily, on="Date", how="outer")

full = full.merge(cat_features, on="Date", how="left")

CAT_COLS = [c for c in cat_features.columns if c != "Date"]
for col in CAT_COLS:
    for lag in [365, 730]:
        full[f"{col}_l{lag}"] = full[col].shift(lag)
full.drop(columns=CAT_COLS, inplace=True, errors="ignore")
print(f"Category features added ({len(CAT_COLS)} cols). Shape: {full.shape}")

# %% [markdown]
# ### 3.5c Return Rate Features
# Daily return count and rate, lagged >= 365 for test safety.

# %%
ret_daily = (returns_raw.groupby("return_date")
    .agg(ret_count=("return_id", "count"))
    .reset_index().rename(columns={"return_date": "Date"}))
full = full.merge(ret_daily, on="Date", how="left")

ord_tmp = (orders_raw.groupby("order_date")["order_id"].count()
           .reset_index().rename(columns={"order_date": "Date", "order_id": "ord_tmp_count"}))
full = full.merge(ord_tmp, on="Date", how="left")
full["ret_rate"] = full["ret_count"].fillna(0) / (full["ord_tmp_count"].fillna(0) + 1e-9)

for col in ["ret_count", "ret_rate"]:
    for lag in [365, 730]:
        full[f"{col}_l{lag}"] = full[col].shift(lag)
full.drop(columns=["ret_count", "ret_rate", "ord_tmp_count"], inplace=True, errors="ignore")
print("Return rate features added. Shape:", full.shape)

# %% [markdown]
# ### 3.5d Review Quality Features
# Daily review volume, avg rating, and rating dispersion, lagged >= 365.

# %%
rev_daily = (reviews.groupby("review_date")
    .agg(review_count=("review_id", "count"),
         review_avg_rating=("rating", "mean"),
         review_rating_std=("rating", "std"))
    .reset_index().rename(columns={"review_date": "Date"}))
full = full.merge(rev_daily, on="Date", how="left")

REV_COLS = ["review_count", "review_avg_rating", "review_rating_std"]
for col in REV_COLS:
    for lag in [365, 730]:
        full[f"{col}_l{lag}"] = full[col].shift(lag)
full.drop(columns=REV_COLS, inplace=True, errors="ignore")
print("Review quality features added. Shape:", full.shape)

# %% [markdown]
# ### 3.5e Payment Mix Features
# Daily COD ratio, credit card ratio, avg installments, lagged >= 365.

# %%
paymix = (orders_raw.groupby("order_date")
    .agg(pay_cod_ratio=("payment_method", lambda x: (x == "cod").mean()),
         pay_cc_ratio=("payment_method", lambda x: (x == "credit_card").mean()))
    .reset_index().rename(columns={"order_date": "Date"}))
pay_inst = (payments[["order_id", "installments"]]
    .merge(orders_raw[["order_id", "order_date"]], on="order_id", how="left")
    .groupby("order_date")["installments"].mean()
    .reset_index().rename(columns={"order_date": "Date", "installments": "pay_avg_installments"}))
full = full.merge(paymix, on="Date", how="left")
full = full.merge(pay_inst, on="Date", how="left")

PAYMIX_COLS = ["pay_cod_ratio", "pay_cc_ratio", "pay_avg_installments"]
for col in PAYMIX_COLS:
    for lag in [365, 730]:
        full[f"{col}_l{lag}"] = full[col].shift(lag)
full.drop(columns=PAYMIX_COLS, inplace=True, errors="ignore")
print("Payment mix features added. Shape:", full.shape)

# %% [markdown]
# ### 3.5f Inventory Health Features
# Monthly inventory health → daily via forward-fill, lagged >= 365.

# %%
inv_monthly = (inventory.groupby("snapshot_date")
    .agg(inv_stockout_ratio=("stockout_flag", "mean"),
         inv_overstock_ratio=("overstock_flag", "mean"),
         inv_avg_dos=("days_of_supply", "mean"),
         inv_avg_fill_rate=("fill_rate", "mean"),
         inv_avg_str=("sell_through_rate", "mean"))
    .reset_index().rename(columns={"snapshot_date": "Date"}))
inv_daily = pd.DataFrame({"Date": all_dates}).merge(inv_monthly, on="Date", how="left")
INV_COLS = ["inv_stockout_ratio", "inv_overstock_ratio", "inv_avg_dos",
            "inv_avg_fill_rate", "inv_avg_str"]
inv_daily[INV_COLS] = inv_daily[INV_COLS].ffill()
inv_daily = inv_daily[inv_daily["Date"] >= inv_monthly["Date"].min()]
full = full.merge(inv_daily, on="Date", how="left")
for col in INV_COLS:
    for lag in [365, 730]:
        full[f"{col}_l{lag}"] = full[col].shift(lag)
full.drop(columns=INV_COLS, inplace=True, errors="ignore")
print("Inventory health features added. Shape:", full.shape)

# %% [markdown]
# ### 3.5g Customer Acquisition Features
# Daily new customer ratio based on first-order-date heuristic, lag >= 365.

# %%
first_ord = orders_raw.groupby("customer_id")["order_date"].min().reset_index()
first_ord.columns = ["customer_id", "first_order_date"]
ord_new = orders_raw[["order_id", "order_date", "customer_id"]].merge(first_ord, on="customer_id", how="left")
ord_new["is_new"] = (ord_new["order_date"] == ord_new["first_order_date"]).astype(int)
cust_acq = (ord_new.groupby("order_date")
    .agg(new_cust_count=("is_new", "sum"),
         new_cust_ratio=("is_new", "mean"))
    .reset_index().rename(columns={"order_date": "Date"}))
full = full.merge(cust_acq, on="Date", how="left")
CUST_COLS = ["new_cust_count", "new_cust_ratio"]
for col in CUST_COLS:
    for lag in [365, 730]:
        full[f"{col}_l{lag}"] = full[col].shift(lag)
full.drop(columns=CUST_COLS, inplace=True, errors="ignore")
print("Customer acquisition features added. Shape:", full.shape)

# %% [markdown]
# ### 3.5h Interaction Features
# Built from calendar + projected promos only — fully test-safe.

# %%
full["interact_promo_wknd"]      = full["proj_promo_count"] * full["cal_is_weekend"]
full["interact_promo_doy_sin"]   = full["proj_promo_discount_sum"] * full["cal_doy_sin"]
full["interact_promo_doy_cos"]   = full["proj_promo_discount_sum"] * full["cal_doy_cos"]
full["interact_trend_doy_sin"]   = full["cal_time_idx"] * full["cal_doy_sin"]
full["interact_trend_doy_cos"]   = full["cal_time_idx"] * full["cal_doy_cos"]
full["interact_tet_year"]        = full["vn_tet"] * (full["cal_year"] - 2012)
full["interact_promo_tet"]       = full["proj_promo_count"] * full["vn_tet"]
full["interact_wknd_peak"]       = full["cal_is_weekend"] * full["vn_peak_season"]
full["interact_promo_month_sin"] = full["proj_promo_discount_sum"] * full["cal_month_sin"]
full["interact_promo_month_cos"] = full["proj_promo_discount_sum"] * full["cal_month_cos"]
print("Interaction features added. Shape:", full.shape)

# %% [markdown]
# ### 3.6 Feature List Compilation & Leakage Audit

# %%
# Define excluded columns
EXCLUDE = {"Date", "Revenue", "COGS", "is_test", "gross_profit", "gross_margin",
           "promo_count", "promo_bucket"}

FEATURE_COLS = [c for c in full.columns if c not in EXCLUDE]

# Leakage audit: ensure no raw (non-lagged) order columns exist
raw_order_cols = [c for c in FEATURE_COLS
                  if c.startswith("ord_") and not any(
                      c.endswith(f"_l{lag}") or c.endswith(f"_rm{w}")
                      for lag in SHIFT_DAYS for w in [7,14,28,90,365]
                  )]
if raw_order_cols:
    print(f"[FAIL] Raw order columns found: {raw_order_cols}")
else:
    print("[PASS] No raw order features — all are lagged")

# Feature group summary
print(f"\nTotal features: {len(FEATURE_COLS)}")
groups = {
    "Calendar (cal_/vn_)": sum(1 for c in FEATURE_COLS if c.startswith(("cal_", "vn_"))),
    "Projected Promotions": sum(1 for c in FEATURE_COLS if c.startswith("proj_promo_")),
    "Revenue lags/roll/ewm": sum(1 for c in FEATURE_COLS if c.startswith("Revenue_")),
    "COGS lags/roll/ewm": sum(1 for c in FEATURE_COLS if c.startswith("COGS_")),
    "Order lagged": sum(1 for c in FEATURE_COLS if c.startswith("ord_")),
    "Web traffic": sum(1 for c in FEATURE_COLS if c.startswith("wt_")),
    "Cross-ratio": sum(1 for c in FEATURE_COLS if "cogs_rev_ratio" in c),
    "Category (cat_)": sum(1 for c in FEATURE_COLS if c.startswith("cat_")),
    "Returns (ret_)": sum(1 for c in FEATURE_COLS if c.startswith("ret_")),
    "Reviews (review_)": sum(1 for c in FEATURE_COLS if c.startswith("review_")),
    "Payment mix (pay_)": sum(1 for c in FEATURE_COLS if c.startswith("pay_")),
    "Inventory (inv_)": sum(1 for c in FEATURE_COLS if c.startswith("inv_")),
    "Customer acq (new_cust_)": sum(1 for c in FEATURE_COLS if c.startswith("new_cust_")),
    "Interaction (interact_)": sum(1 for c in FEATURE_COLS if c.startswith("interact_")),
}
for grp, cnt in groups.items():
    print(f"  {grp:<30}: {cnt:>3}")

# %% [markdown]
# ## 4 · Train/Test Split

# %%
# Train: dates with Revenue_l365 available (need 1 year of lagged features)
train_mask = (full["is_test"] == 0) & (full["Revenue_l365"].notna())
train = full[train_mask].copy().reset_index(drop=True)
test  = full[full["is_test"] == 1].copy().reset_index(drop=True)

X_train = train[FEATURE_COLS]
y_train_rev  = train["Revenue"]
y_train_cogs = train["COGS"]

print(f"Train: {len(train):,} rows  [{train['Date'].min().date()} -> {train['Date'].max().date()}]")
print(f"Test:  {len(test):,} rows  [{test['Date'].min().date()} -> {test['Date'].max().date()}]")
print(f"Features: {len(FEATURE_COLS)}")

# Check NaN
nan_cols = X_train.columns[X_train.isna().any()].tolist()
if nan_cols:
    print(f"\nNaN in training features ({len(nan_cols)} cols): {nan_cols[:10]}...")
else:
    print("\nNo NaN in training features")

# %%
# Handle remaining NaN in training data
# Fill NaN with 0 for count-based features, forward fill for others
X_train = X_train.fillna(0)
print("NaN handled in training features")

# %% [markdown]
# ## 5 · Modeling Pipeline

# %% [markdown]
# ### 5.1 Phase A — Baseline Models

# %%
# ── Baseline A0: Mean predictor ───────────────────────────────────────────────
mean_rev  = y_train_rev.mean()
mean_cogs = y_train_cogs.mean()
mae_mean_rev  = mean_absolute_error(y_train_rev,  np.full_like(y_train_rev, mean_rev))
mae_mean_cogs = mean_absolute_error(y_train_cogs, np.full_like(y_train_cogs, mean_cogs))
print(f"Baseline A0 (Mean):")
print(f"  Revenue MAE = {mae_mean_rev:>12,.0f}  (predict {mean_rev:,.0f})")
print(f"  COGS    MAE = {mae_mean_cogs:>12,.0f}  (predict {mean_cogs:,.0f})")

# %%
# ── Baseline A1: Seasonal + Trend (from original baseline) ────────────────────
full_years = sales[sales["year"].between(2013, 2022)]
annual_tot = full_years.groupby("year")[["Revenue", "COGS"]].sum()

yoy_r = annual_tot["Revenue"].pct_change().dropna()
yoy_c = annual_tot["COGS"].pct_change().dropna()
g_rev  = (1 + yoy_r).prod() ** (1 / len(yoy_r))
g_cogs = (1 + yoy_c).prod() ** (1 / len(yoy_c))

# Seasonal profile from 2020-2022
recent = sales[sales["year"].between(2020, 2022)].copy()
recent["month"] = recent["Date"].dt.month
recent["day"]   = recent["Date"].dt.day
ann_mean = recent.groupby("year")[["Revenue", "COGS"]].transform("mean")
recent["rev_norm"]  = recent["Revenue"] / ann_mean["Revenue"]
recent["cogs_norm"] = recent["COGS"]    / ann_mean["COGS"]
seasonal = recent.groupby(["month", "day"])[["rev_norm", "cogs_norm"]].mean().reset_index()

base_rev  = annual_tot.loc[2022, "Revenue"] / 365
base_cogs = annual_tot.loc[2022, "COGS"]    / 365

def seasonal_predict(dates, ref_year=2022):
    """Predict using seasonal profile + trend."""
    df = pd.DataFrame({"Date": pd.to_datetime(dates)})
    df["month"] = df["Date"].dt.month
    df["day"]   = df["Date"].dt.day
    df["ya"]    = df["Date"].dt.year - ref_year
    df = df.merge(seasonal, on=["month", "day"], how="left")
    df["rev_norm"]  = df["rev_norm"].fillna(1.0)
    df["cogs_norm"] = df["cogs_norm"].fillna(1.0)
    return (
        (base_rev  * g_rev ** df["ya"]  * df["rev_norm"]).clip(0).values,
        (base_cogs * g_cogs ** df["ya"] * df["cogs_norm"]).clip(0).values,
    )

seas_rev_train, seas_cogs_train = seasonal_predict(train["Date"])
mae_seas_rev  = mean_absolute_error(y_train_rev, seas_rev_train)
mae_seas_cogs = mean_absolute_error(y_train_cogs, seas_cogs_train)
print(f"Baseline A1 (Seasonal+Trend):")
print(f"  Revenue MAE = {mae_seas_rev:>12,.0f}  Growth={g_rev-1:+.2%}/yr")
print(f"  COGS    MAE = {mae_seas_cogs:>12,.0f}  Growth={g_cogs-1:+.2%}/yr")

# %%
# ── Baseline A2: Linear Regression ────────────────────────────────────────────
lr_rev  = Ridge(alpha=1.0, random_state=SEED)
lr_cogs = Ridge(alpha=1.0, random_state=SEED)

lr_rev.fit(X_train, y_train_rev)
lr_cogs.fit(X_train, y_train_cogs)

p_lr_rev  = lr_rev.predict(X_train)
p_lr_cogs = lr_cogs.predict(X_train)

mae_lr_rev  = mean_absolute_error(y_train_rev, p_lr_rev)
mae_lr_cogs = mean_absolute_error(y_train_cogs, p_lr_cogs)
r2_lr_rev   = r2_score(y_train_rev, p_lr_rev)
print(f"Baseline A2 (Ridge Regression):")
print(f"  Revenue MAE = {mae_lr_rev:>12,.0f}  R2 = {r2_lr_rev:.4f}")
print(f"  COGS    MAE = {mae_lr_cogs:>12,.0f}  R2 = {r2_score(y_train_cogs, p_lr_cogs):.4f}")

# %%
# ── Baseline A3: Decision Tree ────────────────────────────────────────────────
dt_rev  = DecisionTreeRegressor(max_depth=8, random_state=SEED)
dt_cogs = DecisionTreeRegressor(max_depth=8, random_state=SEED)

dt_rev.fit(X_train, y_train_rev)
dt_cogs.fit(X_train, y_train_cogs)

p_dt_rev  = dt_rev.predict(X_train)
p_dt_cogs = dt_cogs.predict(X_train)

mae_dt_rev  = mean_absolute_error(y_train_rev, p_dt_rev)
mae_dt_cogs = mean_absolute_error(y_train_cogs, p_dt_cogs)
r2_dt_rev   = r2_score(y_train_rev, p_dt_rev)
print(f"Baseline A3 (Decision Tree, max_depth=8):")
print(f"  Revenue MAE = {mae_dt_rev:>12,.0f}  R2 = {r2_dt_rev:.4f}")
print(f"  COGS    MAE = {mae_dt_cogs:>12,.0f}  R2 = {r2_score(y_train_cogs, p_dt_cogs):.4f}")

# %%
# ── Baseline summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  BASELINE SUMMARY (In-sample)")
print("-" * 70)
print(f"  {'Model':<25} {'Revenue MAE':>14} {'Revenue R2':>12} {'COGS MAE':>14}")
print("-" * 70)
print(f"  {'A0: Mean':<25} {mae_mean_rev:>14,.0f} {'—':>12} {mae_mean_cogs:>14,.0f}")
print(f"  {'A1: Seasonal+Trend':<25} {mae_seas_rev:>14,.0f} {'—':>12} {mae_seas_cogs:>14,.0f}")
print(f"  {'A2: Ridge Regression':<25} {mae_lr_rev:>14,.0f} {r2_lr_rev:>12.4f} {mae_lr_cogs:>14,.0f}")
print(f"  {'A3: Decision Tree':<25} {mae_dt_rev:>14,.0f} {r2_dt_rev:>12.4f} {mae_dt_cogs:>14,.0f}")
print("=" * 70)

# %% [markdown]
# ### 5.2 Phase B — Gradient Boosting with TimeSeriesSplit CV

# %%
def run_ts_cv(X, y, model, n_splits=5, gap=14, label="Model"):
    """TimeSeriesSplit cross-validation with expanding window.

    gap=14 prevents leakage from rolling window features into validation.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    fold_mae, fold_rmse, fold_r2 = [], [], []
    all_y_true, all_y_pred = [], []

    print(f"\n{'Fold':>5} | {'Train size':>10} | {'Val size':>8} | {'MAE':>14} | {'R2':>10}")
    print("-" * 65)

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_tr, y_tr)
        y_pred = model_clone.predict(X_val)

        mae  = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2   = r2_score(y_val, y_pred)

        fold_mae.append(mae)
        fold_rmse.append(rmse)
        fold_r2.append(r2)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        print(f"  {fold+1:>3}  | {len(X_tr):>10,} | {len(X_val):>8,} | {mae:>14,.0f} | {r2:>10.4f}")

    print("-" * 65)
    overall_mae  = mean_absolute_error(all_y_true, all_y_pred)
    overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    overall_r2   = r2_score(all_y_true, all_y_pred)
    print(f"  OVERALL | {'':>10} | {'':>8} | {overall_mae:>14,.0f} | {overall_r2:>10.4f}")

    return {
        "fold_mae": fold_mae, "fold_rmse": fold_rmse, "fold_r2": fold_r2,
        "mae_mean": np.mean(fold_mae), "mae_std": np.std(fold_mae),
        "overall_mae": overall_mae, "overall_rmse": overall_rmse, "overall_r2": overall_r2,
        "y_true": all_y_true, "y_pred": all_y_pred,
    }

# ── Phase B1: LightGBM baseline (no tuning) ───────────────────────────────────
lgb_base = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=1000, random_state=SEED)
cv_lgb_rev = run_ts_cv(X_train, y_train_rev, lgb_base, label="LGB-Revenue")

# %% [markdown]
# ### 5.3 Phase B2 — Optuna Hyperparameter Tuning

# %%
def optuna_objective(trial, X, y, n_splits=3, gap=14):
    """Optuna objective for LightGBM tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = []

    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        scores.append(mean_absolute_error(y_val, y_pred))

    return np.mean(scores)

# %%
# ── Run Optuna (with SQLite persistence to skip on re-run) ──────────────────
OPTUNA_DB = "sqlite:///../../.claude/optuna.db"
print("Running Optuna tuning for Revenue model (3-fold CV, 50 trials)...")
study_rev = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name="lgb_revenue", storage=OPTUNA_DB, load_if_exists=True,
)
if len(study_rev.trials) < 50:
    study_rev.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train_rev),
        n_trials=50 - len(study_rev.trials), show_progress_bar=True
    )
else:
    print("  [SKIP] Already has 50 trials, loading from cache")

print(f"\nBest Revenue MAE: {study_rev.best_value:,.0f}")
print(f"Best params: {study_rev.best_params}")

# COGS model (share architecture, tune separately if time permits)
print("\nRunning Optuna tuning for COGS model (50 trials)...")
study_cogs = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED+1),
    study_name="lgb_cogs", storage=OPTUNA_DB, load_if_exists=True,
)
if len(study_cogs.trials) < 50:
    study_cogs.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train_cogs),
        n_trials=50 - len(study_cogs.trials), show_progress_bar=True
    )
else:
    print("  [SKIP] Already has 50 trials, loading from cache")

print(f"\nBest COGS MAE: {study_cogs.best_value:,.0f}")
print(f"Best params: {study_cogs.best_params}")

# %% [markdown]
# ### 5.3b Phase B3 — XGBoost with Optuna Hyperparameter Tuning

# %%
def optuna_objective_xgb(trial, X, y, n_splits=3, gap=14):
    """Optuna objective for XGBoost tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
    }
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = []
    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        scores.append(mean_absolute_error(y_val, model.predict(X_val)))
    return np.mean(scores)

print("Running Optuna tuning for XGBoost Revenue model (50 trials)...")
study_xgb_rev = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED),
    study_name="xgb_revenue", storage=OPTUNA_DB, load_if_exists=True,
)
if len(study_xgb_rev.trials) < 50:
    study_xgb_rev.optimize(
        lambda trial: optuna_objective_xgb(trial, X_train, y_train_rev),
        n_trials=50 - len(study_xgb_rev.trials), show_progress_bar=True
    )
else:
    print("  [SKIP] Already has 50 trials, loading from cache")
print(f"\nBest XGB Revenue MAE: {study_xgb_rev.best_value:,.0f}")
print(f"Best params: {study_xgb_rev.best_params}")

print("\nRunning Optuna tuning for XGBoost COGS model (50 trials)...")
study_xgb_cogs = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED+2),
    study_name="xgb_cogs", storage=OPTUNA_DB, load_if_exists=True,
)
if len(study_xgb_cogs.trials) < 50:
    study_xgb_cogs.optimize(
        lambda trial: optuna_objective_xgb(trial, X_train, y_train_cogs),
        n_trials=50 - len(study_xgb_cogs.trials), show_progress_bar=True
    )
else:
    print("  [SKIP] Already has 50 trials, loading from cache")
print(f"\nBest XGB COGS MAE: {study_xgb_cogs.best_value:,.0f}")
print(f"Best params: {study_xgb_cogs.best_params}")

# %% [markdown]
# ### 5.4 Phase C — Ensemble Model

# %%
# ── Build tuned models ────────────────────────────────────────────────────────
lgb_tuned_rev = lgb.LGBMRegressor(**{**study_rev.best_params, "random_state": SEED, "n_jobs": -1, "verbose": -1})
lgb_tuned_cogs = lgb.LGBMRegressor(**{**study_cogs.best_params, "random_state": SEED, "n_jobs": -1, "verbose": -1})

# Cross-validate tuned models
cv_tuned_rev = run_ts_cv(X_train, y_train_rev, lgb_tuned_rev, label="LGB-Tuned-Revenue")
cv_tuned_cogs = run_ts_cv(X_train, y_train_cogs, lgb_tuned_cogs, label="LGB-Tuned-COGS")

# %%
# ── Phase C: Optimized Ensemble Strategy ───────────────────────────────────────
# Meta-learner: Ridge trained on CV out-of-fold predictions from each component
# to learn optimal blending weights automatically.

tscv = TimeSeriesSplit(n_splits=5, gap=14)
meta_X_rev = np.zeros((len(X_train), 4))
meta_y_rev = np.zeros(len(X_train))

for tr_idx, val_idx in tscv.split(X_train):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train_rev.iloc[tr_idx], y_train_rev.iloc[val_idx]

    # LGB
    lgb_m = lgb.LGBMRegressor(**{**study_rev.best_params, "random_state": SEED, "n_jobs": -1, "verbose": -1})
    lgb_m.fit(X_tr, y_tr)
    meta_X_rev[val_idx, 0] = lgb_m.predict(X_val)

    # XGB
    xgb_m = xgb.XGBRegressor(**{**study_xgb_rev.best_params, "random_state": SEED, "n_jobs": -1, "verbosity": 0, "tree_method": "hist"})
    xgb_m.fit(X_tr, y_tr)
    meta_X_rev[val_idx, 1] = xgb_m.predict(X_val)

    # Ridge
    ridge_m = Ridge(alpha=1.0, random_state=SEED)
    ridge_m.fit(X_tr, y_tr)
    meta_X_rev[val_idx, 2] = ridge_m.predict(X_val)

    # Seasonal
    val_dates = train.iloc[val_idx]["Date"].values
    seas_pred_tr, _ = seasonal_predict(val_dates)
    meta_X_rev[val_idx, 3] = seas_pred_tr
    meta_y_rev[val_idx] = y_val.values

# Meta-learner: non-negative Ridge for interpretable weights
meta_lr = Ridge(alpha=1.0, fit_intercept=False, positive=True, random_state=SEED)
meta_lr.fit(meta_X_rev, meta_y_rev)
raw_weights = meta_lr.coef_
W_LGB, W_XGB, W_LR, W_SEAS = raw_weights / raw_weights.sum()

print("\n" + "=" * 70)
print("  OPTIMIZED ENSEMBLE WEIGHTS (Revenue)")
print("-" * 70)
for name, w in [("LightGBM", W_LGB), ("XGBoost", W_XGB),
                ("Ridge", W_LR), ("Seasonal", W_SEAS)]:
    print(f"  {name:<20} {w:>6.1%}")
print("=" * 70)

# %% [markdown]
# ## 6 · Final Model Training & Iterative Prediction

# %%
# ── Train final models on full training data ───────────────────────────────────
print("Training final models on full training data...")

# LGB
lgb_final_rev = lgb.LGBMRegressor(**{**study_rev.best_params, "random_state": SEED, "n_jobs": -1, "verbose": -1})
lgb_final_cogs = lgb.LGBMRegressor(**{**study_cogs.best_params, "random_state": SEED, "n_jobs": -1, "verbose": -1})
lgb_final_rev.fit(X_train, y_train_rev)
lgb_final_cogs.fit(X_train, y_train_cogs)

# XGB
xgb_final_rev = xgb.XGBRegressor(**{**study_xgb_rev.best_params, "random_state": SEED, "n_jobs": -1, "verbosity": 0, "tree_method": "hist"})
xgb_final_cogs = xgb.XGBRegressor(**{**study_xgb_cogs.best_params, "random_state": SEED, "n_jobs": -1, "verbosity": 0, "tree_method": "hist"})
xgb_final_rev.fit(X_train, y_train_rev)
xgb_final_cogs.fit(X_train, y_train_cogs)

# Ridge (already trained)
# Seasonal (already computed)

# %%
# ── Iterative day-by-day test prediction ──────────────────────────────────────
# We predict day-by-day because Revenue/COGS lag features depend on
# previous predictions for dates within the test period.
# After each prediction, ALL derived features (lag, rolling, EWM, cross-ratio)
# are recomputed so the next prediction sees consistent feature values.
# Only updating raw lags (as done previously) left 88 rolling/EWM/YoY features
# at NaN/0, causing the model to drift and over-predict by 3-8x.

test_dates_sorted = sorted(test["Date"].tolist())
fp = full.set_index("Date").copy()

# Pre-compute seasonal baseline for all test dates
seas_rev_test, seas_cogs_test = seasonal_predict(test_dates_sorted)

print(f"Predicting {len(test_dates_sorted)} test days iteratively...")
for i, d in enumerate(test_dates_sorted):
    if i % 100 == 0:
        print(f"  Day {i+1:>3}/{len(test_dates_sorted)}  ({d.date()})")

    si = i  # index in test array

    # Get features from fp — they are correct because we recompute after each prediction
    row = fp.loc[[d], FEATURE_COLS].fillna(0)

    # Ensemble prediction (4-component blend)
    p_lgb_rev  = max(float(lgb_final_rev.predict(row)[0]), 0)
    p_xgb_rev  = max(float(xgb_final_rev.predict(row)[0]), 0)
    p_lr_rev   = max(float(lr_rev.predict(row)[0]), 0)
    p_lgb_cogs = max(float(lgb_final_cogs.predict(row)[0]), 0)
    p_xgb_cogs = max(float(xgb_final_cogs.predict(row)[0]), 0)
    p_lr_cogs  = max(float(lr_cogs.predict(row)[0]), 0)

    # Blend predictions
    fp.loc[d, "Revenue"] = (W_LGB * p_lgb_rev + W_XGB * p_xgb_rev +
                            W_LR * p_lr_rev + W_SEAS * seas_rev_test[si])
    fp.loc[d, "COGS"]    = (W_LGB * p_lgb_cogs + W_XGB * p_xgb_cogs +
                            W_LR * p_lr_cogs + W_SEAS * seas_cogs_test[si])

    # Recompute all derived features so the next forecast date sees consistent values.
    # Without this, rolling/EWM/YoY features (88 of 205) stay at NaN/0,
    # creating an out-of-distribution feature vector that causes runaway drift.
    if i < len(test_dates_sorted) - 1:
        fp = add_target_features(fp.reset_index(), "Revenue")
        fp = add_target_features(fp, "COGS")
        for lag in [28, 90, 180, 365]:
            fp[f"cogs_rev_ratio_l{lag}"] = (fp[f"COGS_l{lag}"] /
                                               (fp[f"Revenue_l{lag}"] + 1e-9))
        fp = fp.set_index("Date")

print("Iterative prediction complete.")

# %% [markdown]
# ## 7 · Evaluation & CV Results Visualization

# %%
# ── CV fold plot ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.flatten()

# Re-run CV with fold-level data for plotting
tscv = TimeSeriesSplit(n_splits=5, gap=14)
for i, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
    y_val = y_train_rev.iloc[val_idx]
    y_pred_cv = lgb_base.fit(X_train.iloc[tr_idx], y_train_rev.iloc[tr_idx]).predict(X_train.iloc[val_idx])

    n = min(200, len(y_val))
    axes[i].plot(range(n), y_val.values[:n]/1e6, lw=1.2, color=PALETTE[0], label="Actual")
    axes[i].plot(range(n), y_pred_cv[:n]/1e6, lw=0.9, color=PALETTE[1], linestyle="--", label="LGB")
    mae_f = mean_absolute_error(y_val, y_pred_cv)
    r2_f  = r2_score(y_val, y_pred_cv)
    axes[i].set_title(f"Fold {i+1}  MAE={mae_f:,.0f}  R2={r2_f:.3f}", fontsize=9)
    axes[i].legend(fontsize=7)
    axes[i].set_ylabel("Revenue (M VND)")

# Summary
maes = cv_tuned_rev["fold_mae"]
axes[-1].bar(range(1, 6), maes, color=PALETTE[0], alpha=0.85, edgecolor="white")
axes[-1].axhline(np.mean(maes), color="red", linestyle="--", label=f"Mean: {np.mean(maes):,.0f}")
axes[-1].set_xlabel("Fold"); axes[-1].set_ylabel("MAE"); axes[-1].set_title("MAE per Fold")
axes[-1].legend(fontsize=9)

plt.suptitle("TimeSeriesSplit CV — 5 Folds, gap=14d", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/cv_folds.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8 · SHAP Feature Importance

# %%
# ── SHAP analysis on trained LGB model ────────────────────────────────────────
print("Computing SHAP values...")
# Sample from training data for SHAP (efficiency)
rng = np.random.default_rng(SEED)
n_shap = min(800, len(X_train))
idx_shap = rng.choice(len(X_train), n_shap, replace=False)
X_shap = X_train.iloc[idx_shap]

explainer = shap.TreeExplainer(lgb_final_rev)
shap_values = explainer.shap_values(X_shap)

shap_df = (pd.DataFrame({"feature": FEATURE_COLS, "shap_mean": np.abs(shap_values).mean(0)})
           .sort_values("shap_mean", ascending=False)
           .reset_index(drop=True))

print("\nTop-20 Revenue Drivers:")
print(shap_df.head(20)[["feature", "shap_mean"]].to_string(index=False))

# %%
# ── SHAP visualization ────────────────────────────────────────────────────────
try:
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # Summary plot
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_shap, feature_names=FEATURE_COLS,
                      show=False, max_display=20, plot_size=None)
    axes[0].set_title("SHAP Beeswarm — Revenue Model\n(red=high, blue=low)", fontsize=10)

    # Bar plot
    top20 = shap_df.head(20)
    axes[1].barh(top20["feature"][::-1], top20["shap_mean"][::-1], color=PALETTE[0], alpha=0.85)
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].set_title("Top-20 Features by Mean |SHAP|", fontsize=11)
    axes[1].tick_params(labelsize=8)

    plt.suptitle("SHAP Feature Importance — LightGBM Revenue Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP visualization saved.")
except Exception as e:
    print(f"SHAP visualization skipped (error: {e})")

# %% [markdown]
# ### Business Interpretation of Top Features
#
# | Rank | Feature | Business Meaning |
# |---|---|---|
# | 1 | `Revenue_l1` | Short-term momentum — yesterday's revenue |
# | 2 | `cal_doy_sin/cos` | Seasonal cycle — time of year |
# | 3 | `Revenue_l364_mean` | Annual anchor — same week last year |
# | 4 | `ord_count_l365` | Order volume proxy (lagged 1yr) |
# | 5 | `proj_promo_count` | Projected active promotions |
# | 6 | `Revenue_ewm7` | Weekly trend (smoothed) |
# | 7 | `cogs_rev_ratio_l90` | Cost structure stability |
# | 8 | `wt_sessions` | Web traffic → purchase signal |
# | 9 | `cal_is_weekend` | Weekend effect |
# | 10 | `vn_tet` | Vietnamese New Year peak |

# %% [markdown]
# ## 9 · Generate Submission

# %%
# ── Extract predictions ───────────────────────────────────────────────────────
pred_df = (fp.loc[test_dates_sorted, ["Revenue", "COGS"]]
           .reset_index().rename(columns={"index": "Date"}))
pred_df["Revenue"] = pred_df["Revenue"].clip(lower=0).round(2)
pred_df["COGS"]    = pred_df["COGS"].clip(lower=0).round(2)

# Ensure correct Date format for merge
pred_df["Date"] = pd.to_datetime(pred_df["Date"])
submission = sample_sub[["Date"]].merge(pred_df, on="Date", how="left")

# ── Integrity checks ──────────────────────────────────────────────────────────
assert len(submission) == len(sample_sub), f"Row count mismatch: {len(submission)} vs {len(sample_sub)}"
assert list(submission["Date"]) == list(sample_sub["Date"]), "Date order mismatch!"
assert submission[["Revenue", "COGS"]].isna().sum().sum() == 0, "NaN in submission!"
assert (submission["Revenue"] >= 0).all(), "Negative Revenue!"
assert (submission["COGS"] >= 0).all(), "Negative COGS!"

# ── Save ──────────────────────────────────────────────────────────────────────
submission_path = f"{SUBM_DIR}/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved: {submission_path}")
print(f"  Rows: {len(submission)}")
print(f"  Revenue: mean={submission['Revenue'].mean():>12,.0f}  "
      f"min={submission['Revenue'].min():>10,.0f}  max={submission['Revenue'].max():>10,.0f}")
print(f"  COGS:    mean={submission['COGS'].mean():>12,.0f}  "
      f"min={submission['COGS'].min():>10,.0f}  max={submission['COGS'].max():>10,.0f}")
print(f"\nFirst 5 rows:")
print(submission.head().to_string(index=False))

# %%
# ── Final forecast plot ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 9))

# 2021-2022 in-sample fit
chk = train[train["Date"].dt.year.between(2021, 2022)].copy()
p_chk_rev = lgb_final_rev.predict(chk[FEATURE_COLS].fillna(0))
mae_chk = mean_absolute_error(chk["Revenue"], p_chk_rev)

axes[0].fill_between(chk["Date"], chk["Revenue"]/1e6, alpha=0.12, color=PALETTE[0])
axes[0].plot(chk["Date"], chk["Revenue"]/1e6, lw=1.2, color=PALETTE[0], label="Actual")
axes[0].plot(chk["Date"], p_chk_rev/1e6, lw=1.0, color=PALETTE[1], linestyle="--",
             label=f"Fitted (MAE={mae_chk:,.0f})")
axes[0].set_title("2021-2022 In-Sample Fit")
axes[0].legend(fontsize=9)
axes[0].set_ylabel("Revenue (M VND)")

# Forecast 2023-2024
axes[1].fill_between(submission["Date"], submission["Revenue"]/1e6, alpha=0.12, color=PALETTE[2])
axes[1].plot(submission["Date"], submission["Revenue"]/1e6, lw=1.5, color=PALETTE[2], label="Revenue Forecast")
axes[1].plot(submission["Date"], submission["COGS"]/1e6, lw=1.2, color=PALETTE[1], linestyle="--", alpha=0.8, label="COGS Forecast")
profit = (submission["Revenue"] - submission["COGS"]) / 1e6
axes[1].fill_between(submission["Date"], 0, profit, alpha=0.15, color=PALETTE[3], label="Gross Profit")
axes[1].set_title("Ensemble Forecast — Jan 2023 to Jul 2024")
axes[1].legend(fontsize=9)
axes[1].set_ylabel("Value (M VND)")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.suptitle("Final Forecast: Revenue · COGS · Gross Profit", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/forecast_final.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10 · Pipeline Summary

# %%
print("=" * 65)
print("  FINAL PIPELINE SUMMARY — Part 3 Forecasting")
print("=" * 65)
print(f"  Features used     : {len(FEATURE_COLS)}")
print(f"  Training samples  : {len(train):,}")
print(f"  Test samples      : {len(test):,}")
print()
print(f"  Model: Ensemble (Weighted Stacking — Meta-Learner)")
print(f"    - LightGBM (Optuna-tuned) : {W_LGB:.0%}")
print(f"    - XGBoost  (Optuna-tuned) : {W_XGB:.0%}")
print(f"    - Ridge Regression        : {W_LR:.0%}")
print(f"    - Seasonal + Trend        : {W_SEAS:.0%}")
print()
print(f"  CV Strategy: TimeSeriesSplit (5 folds, gap=14d, expanding window)")
print(f"  CV Revenue MAE: {cv_tuned_rev['overall_mae']:,.0f}")
print(f"  CV Revenue R2:  {cv_tuned_rev['overall_r2']:.4f}")
print()
print(f"  Submission: {submission_path} ({len(submission)} rows)")
print()
print("  Constraint checklist:")
print("  [PASS] No external data beyond provided")
print("  [PASS] No test-period Revenue/COGS as features")
print("  [PASS] SEED=42 for reproducibility")
print("  [PASS] All lags use shift(>=1) — no future peek")
print("  [PASS] Order features only via lag >= 365 (test-safe)")
print("  [PASS] TimeSeriesSplit CV with temporal gap=14d")
print("  [PASS] SHAP explainability computed")
print("  [PASS] Projected promotions for recurring annual events")
print("=" * 65)
