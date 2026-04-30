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
# # Datathon 2026 -- Phan 3: Sales Forecasting Pipeline (OPTIMIZED)
# **The GridBreakers - VinUniversity Datathon 2026**
#
# ## Cải tien từ PLAN_TRAINING.md
#
# | Cải tien | Mo tả |
# |---|---|
# | **Purged Walk-Forward CV** | gap=90d, horizon=365d -- thay the TimeSeriesSplit(gap=14) |
# | **Composite Objective** | 0.4*MAE_norm + 0.4*RMSE_norm + 0.2*(1-R2) |
# | **Multi-Model Ensemble** | LGB + XGB + CatBoost + HistGB + ExtraTrees + RF + Ridge + Seasonal |
# | **Advanced Ensemble** | Stacking + Weighted Blend + Horizon-Specific Weights |
# | **Fixed Tet Features** | Lunar calendar dates từ src.features.calendar |
# | **Post-Processing** | Clipping + Smoothing + Residual Correction |
# | **Horizon Analysis** | Metrics breakdown theo short/medium/long horizon |

# %% [markdown]
# ## 0 - Setup & Imports

# %%
import os, sys, warnings, json
import matplotlib
matplotlib.use('Agg')
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
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# CatBoost -- optional, graceful fallback
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[WARN] CatBoost not installed -- skipping CatBoost models")

# ── Import from src module ─────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.modeling import (
    PurgedWalkForwardCV, composite_score, evaluate_predictions,
    horizon_breakdown, weighted_blend, rank_average, median_ensemble,
    optimize_ensemble_weights, horizon_specific_blend,
    recursive_forecast, build_historical_bounds, clip_to_historical,
    exponential_smooth, residual_correct, drift_analysis, fold_stability_analysis,
    make_optuna_objective, run_purged_cv,
    suggest_lgb_params, suggest_xgb_params, suggest_catboost_params,
    suggest_rf_params, suggest_histgb_params,
)
from src.features.calendar import add_calendar_features, TET_DATES

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

print("Setup complete -- OPTIMIZED PIPELINE")
print(f"  lightgbm {lgb.__version__}  |  xgboost {xgb.__version__}  |  shap {shap.__version__}")
print(f"  optuna {optuna.__version__}  |  catboost {'available' if HAS_CATBOOST else 'NOT AVAILABLE'}")

# %% [markdown]
# ## 1 - Data Loading & Leakage Audit

# %%
# ── 1.1 Load all tables ───────────────────────────────────────────────────
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

# ── 1.2 Leakage audit ─────────────────────────────────────────────────────
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
    print("\n  [PASS] No data leakage detected -- all tables end <= 2022-12-31")
else:
    print("\n  [WARN] Tables extend beyond training cutoff -- filter immediately!")
print("=" * 70)

# %% [markdown]
# ### 1.3 Tet Nguyen Đan Dates (Lunar Calendar -- FIXED)
# CRITICAL: Previous version used fixed Jan20-Feb20 range.
# Tet follows lunar calendar and shifts each year.
# We now use exact dates from src.features.calendar module.

# %%
print("Tet Nguyen Dan (Lunar New Year) -- Official Calendar Dates:")
for yr in sorted(TET_DATES.keys()):
    if 2012 <= yr <= 2025:
        center = pd.Timestamp(TET_DATES[yr])
        print(f"  {yr}: {center.date()}  (window: {center.date()} +/-7 days)")

# %% [markdown]
# ## 2 - Exploratory Data Analysis

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
print(f"Gross margin: mean={sales['gross_margin'].mean():.1%}")
print(f"Pearson r(Revenue, COGS) = {sales['Revenue'].corr(sales['COGS']):.4f}")

# %%
# ── Revenue + COGS + 30-day MA ────────────────────────────────────────────
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
    axes[i].set_title(f"Daily {label} 2012-2022")
    axes[i].legend(fontsize=9)

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

plt.suptitle("Sales Overview -- Revenue - COGS - Gross Margin", fontsize=13, fontweight="bold")
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

axes[1].bar(dow_rev.index, dow_rev.values/1e6, color=PALETTE[5], alpha=0.85, edgecolor="white")
axes[1].axhline(dow_rev.mean()/1e6, color="red", linestyle="--")
axes[1].set_xticks(range(7)); axes[1].set_xticklabels(DOW_NAMES)
axes[1].set_ylabel("Revenue (M VND)"); axes[1].set_title("Average Revenue by Day of Week")

im = axes[2].imshow(pivot_rev.values/1e6, aspect="auto", cmap="YlOrRd")
axes[2].set_xticks(range(7)); axes[2].set_xticklabels(DOW_NAMES)
axes[2].set_yticks(range(12)); axes[2].set_yticklabels(MONTH_NAMES)
axes[2].set_title("Revenue Heatmap: Month x DOW (M VND)")
plt.colorbar(im, ax=axes[2], label="M VND", shrink=0.85)

plt.suptitle("Seasonality -- Monthly & Day-of-Week Patterns", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/seasonality.png", dpi=150, bbox_inches="tight")
plt.show()

peak_month = monthly_rev["mean"].idxmax()
low_month  = monthly_rev["mean"].idxmin()
print(f"Peak month: {MONTH_NAMES[peak_month-1]} ({monthly_rev['mean'][peak_month]/1e6:.1f}M/day)")
print(f"Low month:  {MONTH_NAMES[low_month-1]} ({monthly_rev['mean'][low_month]/1e6:.1f}M/day)")
print(f"Peak/Low ratio: {monthly_rev['mean'].max()/monthly_rev['mean'].min():.2f}x")

# %% [markdown]
# ### 2.3 Annual Trend & YoY Growth

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
wt_daily = (wt_raw.groupby("date")
            .agg(wt_sessions=("sessions", "sum"),
                 wt_visitors=("unique_visitors", "sum"),
                 wt_pageviews=("page_views", "sum"),
                 wt_bounce=("bounce_rate", "mean"),
                 wt_duration=("avg_session_duration_sec", "mean"))
            .reset_index().rename(columns={"date": "Date"}))

ord_daily = (orders_raw.groupby("order_date")
             .agg(ord_count=("order_id", "count"),
                  ord_delivered=("order_status", lambda x: (x == "delivered").sum()),
                  ord_cancelled=("order_status", lambda x: (x == "cancelled").sum()))
             .reset_index().rename(columns={"order_date": "Date"}))
ord_daily["ord_cancel_rate"] = ord_daily["ord_cancelled"] / ord_daily["ord_count"]

merged_signals = (sales[["Date", "Revenue"]]
                  .merge(wt_daily, on="Date", how="inner")
                  .merge(ord_daily, on="Date", how="inner"))

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0,0].scatter(merged_signals["wt_sessions"]/1e3, merged_signals["Revenue"]/1e6,
                  s=5, alpha=0.3, color=PALETTE[0])
r_s = merged_signals["Revenue"].corr(merged_signals["wt_sessions"])
axes[0,0].set_xlabel("Sessions (K)"); axes[0,0].set_ylabel("Revenue (M)")
axes[0,0].set_title(f"Sessions vs Revenue  (r={r_s:.3f})")

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

axes[1,0].scatter(merged_signals["ord_count"], merged_signals["Revenue"]/1e6,
                  s=5, alpha=0.3, color=PALETTE[1])
r_o = merged_signals["Revenue"].corr(merged_signals["ord_count"])
axes[1,0].set_xlabel("Daily Order Count"); axes[1,0].set_ylabel("Revenue (M)")
axes[1,0].set_title(f"Order Count vs Revenue  (r={r_o:.3f})")

src_sess = wt_raw.groupby("traffic_source")["sessions"].sum().sort_values(ascending=False)
axes[1,1].bar(src_sess.index, src_sess.values/1e6, color=PALETTE[5], alpha=0.85, edgecolor="white")
axes[1,1].set_ylabel("Total Sessions (M)"); axes[1,1].set_title("Sessions by Traffic Source")
axes[1,1].tick_params(axis="x", rotation=45)

plt.suptitle("Web Traffic & Order Signals", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/traffic_orders.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3 - Feature Engineering (ENHANCED)

# %% [markdown]
# ### 3.1 Calendar + Fourier + Vietnamese Retail Events (FIXED Tet)
# CRITICAL FIX: Tet dates now use exact lunar calendar from src.features.calendar.
# Previous version used fixed Jan20-Feb20 which is WRONG (Tet shifts each year).

# %%
all_dates = pd.date_range(sales["Date"].min(), sample_sub["Date"].max(), freq="D")
full = pd.DataFrame({"Date": all_dates})
full = full.merge(sales[["Date", "Revenue", "COGS"]], on="Date", how="left")
full["is_test"] = full["Date"].isin(sample_sub["Date"]).astype(int)

def add_calendar_features_enhanced(df):
    """Enhanced calendar features with proper lunar Tet dates and Fourier harmonics."""
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

    # Higher harmonics -- capture sharp seasonal transitions
    for k in [2, 3, 4, 5, 6]:
        df[f"cal_doy_sin{k}"] = np.sin(2 * k * np.pi * df["cal_doy"] / 365.25)
        df[f"cal_doy_cos{k}"] = np.cos(2 * k * np.pi * df["cal_doy"] / 365.25)

    # ── FIXED: Tet Nguyen Đan using exact lunar calendar ──────────────────
    tet_set = set()
    for yr, date_str in TET_DATES.items():
        center = pd.Timestamp(date_str)
        for offset in range(-7, 8):  # +/-7 window
            tet_set.add(center + pd.Timedelta(days=offset))

    df["vn_tet"]      = D.isin(tet_set).astype(int)
    # Pre-Tet: 14-8 days before
    tet_pre_set = set()
    for yr, date_str in TET_DATES.items():
        center = pd.Timestamp(date_str)
        for offset in range(-14, -7):
            tet_pre_set.add(center + pd.Timedelta(days=offset))
    df["vn_pre_tet"]   = D.isin(tet_pre_set).astype(int)
    # Post-Tet: 8-14 days after
    tet_post_set = set()
    for yr, date_str in TET_DATES.items():
        center = pd.Timestamp(date_str)
        for offset in range(8, 15):
            tet_post_set.add(center + pd.Timedelta(days=offset))
    df["vn_post_tet"]  = D.isin(tet_post_set).astype(int)

    # Fixed holidays from src.features.calendar
    FIXED_HOLIDAYS = [(1,1), (4,30), (5,1), (9,2)]
    df["vn_fixed_holiday"] = D.apply(
        lambda x: any(x.month == m and x.day == d for m, d in FIXED_HOLIDAYS)
    ).astype(int)

    # Vietnamese retail calendar
    df["vn_mid_sale"]     = df["cal_month"].isin([6,7]).astype(int)
    df["vn_year_end"]     = df["cal_month"].isin([11,12]).astype(int)
    df["vn_back_school"]  = df["cal_month"].isin([8,9]).astype(int)
    df["vn_summer"]       = df["cal_month"].isin([5,6,7]).astype(int)
    df["vn_low_season"]   = df["cal_month"].isin([1,2,10]).astype(int)
    df["vn_peak_season"]  = df["cal_month"].isin([4,5,6]).astype(int)

    # Days to next Tet (leakage-safe: calendar knowledge)
    tet_timestamps = sorted([pd.Timestamp(v) for v in TET_DATES.values()])
    def _days_to_tet(x):
        future = [t for t in tet_timestamps if t > x]
        return (future[0] - x).days if future else 999
    df["vn_days_to_tet"] = D.apply(_days_to_tet).astype(int)

    # Time index (linear trend)
    df["cal_time_idx"] = (D - D.min()).dt.days
    return df

full = add_calendar_features_enhanced(full)
print(f"Calendar features added (enhanced). Shape: {full.shape}")
print(f"  Tet 2023: {TET_DATES[2023]}  (window +/-7 days)")
print(f"  Tet 2024: {TET_DATES[2024]}  (window +/-7 days)")

# %% [markdown]
# ### 3.2 Projected Promotions (Recurring Annual Pattern)
# CRITICAL: Promotions data ends 2022-12-31 but follows strict annual recurrence.
# We project expected promos for the test period.

# %%
def build_projected_promos(date_range):
    """Generate expected promotions based on recurring patterns (2013-2022 analysis)."""
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

# %% [markdown]
# ### 3.3 Order Signals (lagged >= 365 days -- test-safe)

# %%
ord_sig = (orders_raw.groupby("order_date")
           .agg(ord_count=("order_id", "count"),
                ord_delivered=("order_status", lambda x: (x=="delivered").sum()),
                ord_cancelled=("order_status", lambda x: (x=="cancelled").sum()))
           .reset_index().rename(columns={"order_date": "Date"}))
ord_sig["ord_cancel_rate"] = ord_sig["ord_cancelled"] / ord_sig["ord_count"]

full = full.merge(ord_sig, on="Date", how="left")

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

full.drop(columns=["ord_count", "ord_delivered", "ord_cancelled", "ord_cancel_rate"],
          inplace=True, errors="ignore")
print("Order features (lagged) added. Shape:", full.shape)

# %% [markdown]
# ### 3.4 Web Traffic Features (lagged + rolling)

# %%
wt_daily["Date_shifted"] = wt_daily["Date"] + pd.Timedelta(days=1)
full = full.merge(
    wt_daily[["Date_shifted", "wt_sessions", "wt_visitors", "wt_pageviews", "wt_bounce", "wt_duration"]],
    left_on="Date", right_on="Date_shifted", how="left"
).drop(columns=["Date_shifted"])

for col in ["wt_sessions", "wt_visitors", "wt_pageviews"]:
    sh = full[col].shift(1)
    for w in [7, 14, 28]:
        full[f"{col}_rm{w}"] = sh.rolling(w, min_periods=max(1, w//4)).mean()

print("Web traffic features added. Shape:", full.shape)

# %% [markdown]
# ### 3.5 Revenue & COGS -- Lag, Rolling, EWM, Momentum, Volatility Features

# %%
LAGS = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 180, 364, 365, 366]
ROLL_WINDOWS = [7, 14, 28, 60, 90, 180, 365]
EWM_SPANS = [7, 14, 30, 90, 180]

def add_target_features_enhanced(df, col):
    """Enhanced target features: lags, rolling, EWM, momentum, volatility."""
    s = df[col].copy()

    # Lags
    for lag in LAGS:
        df[f"{col}_l{lag}"] = s.shift(lag)

    # Rolling stats (on shift-1 values)
    sh = s.shift(1)
    for w in ROLL_WINDOWS:
        mp = max(1, w // 4)
        roll = sh.rolling(w, min_periods=mp)
        df[f"{col}_rm{w}"]   = roll.mean()
        df[f"{col}_rs{w}"]   = roll.std()
        df[f"{col}_rmed{w}"] = roll.median()
        df[f"{col}_rmax{w}"] = roll.max()
        df[f"{col}_rmin{w}"] = roll.min()
        # Rolling quantiles
        df[f"{col}_rq25{w}"] = roll.quantile(0.25)
        df[f"{col}_rq75{w}"] = roll.quantile(0.75)
        # Rolling skew (asymmetry indicator)
        df[f"{col}_rskew{w}"] = roll.skew()

    # EWM
    for sp in EWM_SPANS:
        df[f"{col}_ewm{sp}"] = sh.ewm(span=sp, adjust=False).mean()
        df[f"{col}_ewm_std{sp}"] = sh.ewm(span=sp, adjust=False).std()

    # ── NEW: Momentum & Acceleration ──────────────────────────────────────
    # Momentum: rate of change over different horizons
    for w in [7, 14, 28, 90]:
        df[f"{col}_mom{w}"] = (sh - sh.shift(w)) / (sh.shift(w) + 1e-9)
        # Acceleration: change of momentum
        df[f"{col}_acc{w}"] = df[f"{col}_mom{w}"] - df[f"{col}_mom{w}"].shift(w)

    # ── NEW: Volatility features ───────────────────────────────────────────
    for w in [7, 14, 28]:
        log_ret = np.log(sh / (sh.shift(1) + 1e-9) + 1e-9)
        df[f"{col}_vol{w}"] = log_ret.rolling(w, min_periods=max(3, w//4)).std()

    # ── YoY anchors ───────────────────────────────────────────────────────
    df[f"{col}_yoy_ratio"] = s.shift(365) / (s.shift(372).rolling(7, min_periods=4).mean() + 1e-9)
    l364_vals = (s.shift(357) + s.shift(364) + s.shift(371)) / 3
    df[f"{col}_l364_mean"] = l364_vals
    df[f"{col}_2yoy_mean"] = (s.shift(365) + s.shift(730)) / 2

    # ── Short-term / long-term trend ratio ─────────────────────────────────
    r14 = sh.rolling(14, min_periods=7).mean()
    r180 = sh.shift(180).rolling(14, min_periods=7).mean()
    df[f"{col}_trend14_180"] = r14 / (r180 + 1e-9)

    # ── Trend deviation (drift-aware) ─────────────────────────────────────
    r30 = sh.rolling(30, min_periods=15).mean()
    r365 = sh.shift(365).rolling(30, min_periods=15).mean()
    df[f"{col}_trend_dev"] = (r30 - r365) / (r365 + 1e-9)

    return df

for tgt in ["Revenue", "COGS"]:
    full = add_target_features_enhanced(full, tgt)

# Cross-target ratio features
for lag in [7, 28, 90, 180, 365]:
    full[f"cogs_rev_ratio_l{lag}"] = (full[f"COGS_l{lag}"] /
                                        (full[f"Revenue_l{lag}"] + 1e-9))

# Gross margin features
for lag in [28, 90, 365]:
    rev_l = full[f"Revenue_l{lag}"]
    cogs_l = full[f"COGS_l{lag}"]
    full[f"margin_l{lag}"] = (rev_l - cogs_l) / (rev_l + 1e-9)

print(f"Target features added (enhanced). Shape: {full.shape}")

# %% [markdown]
# ### 3.5b Category-Driven Revenue Features

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
# ### 3.5c Return Rate, Review Quality, Payment Mix, Inventory, Customer Acquisition

# %%
# ── Return Rate ────────────────────────────────────────────────────────────
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

# ── Review Quality ─────────────────────────────────────────────────────────
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

# ── Payment Mix ────────────────────────────────────────────────────────────
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

# ── Inventory Health ───────────────────────────────────────────────────────
inv_monthly = (inventory.groupby("snapshot_date")
    .agg(inv_stockout_ratio=("stockout_flag", "mean"),
         inv_overstock_ratio=("overstock_flag", "mean"),
         inv_avg_dos=("days_of_supply", "mean"),
         inv_avg_fill_rate=("fill_rate", "mean"),
         inv_avg_str=("sell_through_rate", "mean"))
    .reset_index().rename(columns={"snapshot_date": "Date"}))
inv_daily_df = pd.DataFrame({"Date": all_dates}).merge(inv_monthly, on="Date", how="left")
INV_COLS = ["inv_stockout_ratio", "inv_overstock_ratio", "inv_avg_dos",
            "inv_avg_fill_rate", "inv_avg_str"]
inv_daily_df[INV_COLS] = inv_daily_df[INV_COLS].ffill()
inv_daily_df = inv_daily_df[inv_daily_df["Date"] >= inv_monthly["Date"].min()]
full = full.merge(inv_daily_df, on="Date", how="left")
for col in INV_COLS:
    for lag in [365, 730]:
        full[f"{col}_l{lag}"] = full[col].shift(lag)
full.drop(columns=INV_COLS, inplace=True, errors="ignore")

# ── Customer Acquisition ───────────────────────────────────────────────────
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

print("Return, Review, Payment, Inventory, Customer features added. Shape:", full.shape)

# %% [markdown]
# ### 3.5d Interaction Features (enhanced)

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
# New interactions
full["interact_tet_promo"]       = full["vn_tet"] * full["proj_promo_discount_sum"]
full["interact_holiday_wknd"]    = full["vn_fixed_holiday"] * full["cal_is_weekend"]
full["interact_summer_promo"]    = full["vn_summer"] * full["proj_promo_count"]
print("Interaction features added. Shape:", full.shape)

# %% [markdown]
# ### 3.6 Feature List Compilation & Leakage Audit

# %%
EXCLUDE = {"Date", "Revenue", "COGS", "is_test", "gross_profit", "gross_margin",
           "promo_count", "promo_bucket"}

FEATURE_COLS = [c for c in full.columns if c not in EXCLUDE]

# Leakage audit
raw_order_cols = [c for c in FEATURE_COLS
                  if c.startswith("ord_") and not any(
                      c.endswith(f"_l{lag}") or c.endswith(f"_rm{w}")
                      for lag in SHIFT_DAYS for w in [7,14,28,90,365]
                  )]
if raw_order_cols:
    print(f"[FAIL] Raw order columns found: {raw_order_cols}")
else:
    print("[PASS] No raw order features -- all are lagged")

print(f"\nTotal features: {len(FEATURE_COLS)}")
groups = {
    "Calendar (cal_/vn_)": sum(1 for c in FEATURE_COLS if c.startswith(("cal_", "vn_"))),
    "Projected Promotions": sum(1 for c in FEATURE_COLS if c.startswith("proj_promo_")),
    "Revenue lags/roll/ewm": sum(1 for c in FEATURE_COLS if c.startswith("Revenue_")),
    "COGS lags/roll/ewm": sum(1 for c in FEATURE_COLS if c.startswith("COGS_")),
    "Order lagged": sum(1 for c in FEATURE_COLS if c.startswith("ord_")),
    "Web traffic": sum(1 for c in FEATURE_COLS if c.startswith("wt_")),
    "Cross-ratio & margin": sum(1 for c in FEATURE_COLS if "cogs_rev_ratio" in c or "margin_l" in c),
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
# ## 4 - Train/Test Split

# %%
train_mask = (full["is_test"] == 0) & (full["Revenue_l365"].notna())
train = full[train_mask].copy().reset_index(drop=True)
test  = full[full["is_test"] == 1].copy().reset_index(drop=True)

X_train = train[FEATURE_COLS]
y_train_rev  = train["Revenue"]
y_train_cogs = train["COGS"]

print(f"Train: {len(train):,} rows  [{train['Date'].min().date()} -> {train['Date'].max().date()}]")
print(f"Test:  {len(test):,} rows  [{test['Date'].min().date()} -> {test['Date'].max().date()}]")
print(f"Features: {len(FEATURE_COLS)}")

nan_cols = X_train.columns[X_train.isna().any()].tolist()
if nan_cols:
    print(f"\nNaN in training features ({len(nan_cols)} cols): {nan_cols[:10]}...")
else:
    print("\nNo NaN in training features")

# %%
X_train = X_train.fillna(0)
print("NaN handled in training features")

# Compute baselines for composite score normalization
BASELINE_MAE_REV  = mean_absolute_error(y_train_rev, np.full_like(y_train_rev, y_train_rev.mean()))
BASELINE_RMSE_REV = np.sqrt(mean_squared_error(y_train_rev, np.full_like(y_train_rev, y_train_rev.mean())))
BASELINE_MAE_COGS  = mean_absolute_error(y_train_cogs, np.full_like(y_train_cogs, y_train_cogs.mean()))
BASELINE_RMSE_COGS = np.sqrt(mean_squared_error(y_train_cogs, np.full_like(y_train_cogs, y_train_cogs.mean())))
print(f"Baseline MAE  -- Revenue: {BASELINE_MAE_REV:,.0f}  COGS: {BASELINE_MAE_COGS:,.0f}")
print(f"Baseline RMSE -- Revenue: {BASELINE_RMSE_REV:,.0f}  COGS: {BASELINE_RMSE_COGS:,.0f}")

# %% [markdown]
# ## 5 - Modeling Pipeline -- Phase A: Baselines

# %%
# ── Baseline A0: Mean predictor ───────────────────────────────────────────
mean_rev  = y_train_rev.mean()
mean_cogs = y_train_cogs.mean()
mae_mean_rev  = mean_absolute_error(y_train_rev, np.full_like(y_train_rev, mean_rev))
mae_mean_cogs = mean_absolute_error(y_train_cogs, np.full_like(y_train_cogs, mean_cogs))
print(f"Baseline A0 (Mean):")
print(f"  Revenue MAE = {mae_mean_rev:>12,.0f}  (predict {mean_rev:,.0f})")
print(f"  COGS    MAE = {mae_mean_cogs:>12,.0f}  (predict {mean_cogs:,.0f})")

# %%
# ── Baseline A1: Seasonal + Trend ─────────────────────────────────────────
full_years = sales[sales["year"].between(2013, 2022)]
annual_tot = full_years.groupby("year")[["Revenue", "COGS"]].sum()

yoy_r = annual_tot["Revenue"].pct_change().dropna()
yoy_c = annual_tot["COGS"].pct_change().dropna()
g_rev  = (1 + yoy_r).prod() ** (1 / len(yoy_r))
g_cogs = (1 + yoy_c).prod() ** (1 / len(yoy_c))

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
# ── Baseline A2: Ridge Regression ─────────────────────────────────────────
lr_rev  = Ridge(alpha=1.0, random_state=SEED)
lr_cogs = Ridge(alpha=1.0, random_state=SEED)
lr_rev.fit(X_train, y_train_rev)
lr_cogs.fit(X_train, y_train_cogs)

p_lr_rev  = lr_rev.predict(X_train)
p_lr_cogs = lr_cogs.predict(X_train)

mae_lr_rev  = mean_absolute_error(y_train_rev, p_lr_rev)
mae_lr_cogs = mean_absolute_error(y_train_cogs, p_lr_cogs)
r2_lr_rev   = r2_score(y_train_rev, p_lr_rev)
print(f"Baseline A2 (Ridge):")
print(f"  Revenue MAE = {mae_lr_rev:>12,.0f}  R2 = {r2_lr_rev:.4f}")
print(f"  COGS    MAE = {mae_lr_cogs:>12,.0f}  R2 = {r2_score(y_train_cogs, p_lr_cogs):.4f}")

# %%
# ── Baseline A3: Decision Tree ────────────────────────────────────────────
dt_rev  = DecisionTreeRegressor(max_depth=8, random_state=SEED)
dt_cogs = DecisionTreeRegressor(max_depth=8, random_state=SEED)
dt_rev.fit(X_train, y_train_rev)
dt_cogs.fit(X_train, y_train_cogs)

p_dt_rev  = dt_rev.predict(X_train)
p_dt_cogs = dt_cogs.predict(X_train)

mae_dt_rev  = mean_absolute_error(y_train_rev, p_dt_rev)
mae_dt_cogs = mean_absolute_error(y_train_cogs, p_dt_cogs)
r2_dt_rev   = r2_score(y_train_rev, p_dt_rev)
print(f"Baseline A3 (Decision Tree):")
print(f"  Revenue MAE = {mae_dt_rev:>12,.0f}  R2 = {r2_dt_rev:.4f}")
print(f"  COGS    MAE = {mae_dt_cogs:>12,.0f}  R2 = {r2_score(y_train_cogs, p_dt_cogs):.4f}")

# %%
# ── Baseline summary ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  BASELINE SUMMARY (In-sample)")
print("-" * 70)
print(f"  {'Model':<25} {'Revenue MAE':>14} {'Revenue R2':>12} {'COGS MAE':>14}")
print("-" * 70)
print(f"  {'A0: Mean':<25} {mae_mean_rev:>14,.0f} {'--':>12} {mae_mean_cogs:>14,.0f}")
print(f"  {'A1: Seasonal+Trend':<25} {mae_seas_rev:>14,.0f} {'--':>12} {mae_seas_cogs:>14,.0f}")
print(f"  {'A2: Ridge Regression':<25} {mae_lr_rev:>14,.0f} {r2_lr_rev:>12.4f} {mae_lr_cogs:>14,.0f}")
print(f"  {'A3: Decision Tree':<25} {mae_dt_rev:>14,.0f} {r2_dt_rev:>12.4f} {mae_dt_cogs:>14,.0f}")
print("=" * 70)

# %% [markdown]
# ## 6 - Phase B -- Purged Walk-Forward CV with Composite Objective
# CRITICAL: Replaces TimeSeriesSplit(gap=14) which was too optimistic.
# PurgedWalkForwardCV uses gap=90, horizon=365 for realistic evaluation.

# %%
# ── Initialize Purged Walk-Forward CV ─────────────────────────────────────
cv = PurgedWalkForwardCV(n_splits=3, horizon=365, purge_days=90, min_train_days=730)
print(f"CV Strategy: PurgedWalkForwardCV")
print(f"  n_splits={cv.n_splits}, horizon={cv.horizon}d, purge={cv.purge_days}d, min_train={cv.min_train_days}d")
print(f"  This mimics actual deployment: train on past, predict {cv.horizon}d ahead")
print(f"  Purge gap prevents leakage from rolling features (max lag=730d)")

# %%
# ── Phase B1: LightGBM baseline (with purged CV) ──────────────────────────
LGB_PARAMS = {
    "objective": "regression", "metric": "rmse",
    "num_leaves": 127, "learning_rate": 0.02,
    "feature_fraction": 0.75, "bagging_fraction": 0.75, "bagging_freq": 5,
    "min_child_samples": 15, "reg_alpha": 0.10, "reg_lambda": 0.20,
    "verbose": -1, "random_state": SEED, "n_jobs": -1,
}

lgb_base = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=1000)
cv_lgb_rev = run_purged_cv(X_train, y_train_rev, lgb_base, cv, label="LGB-Revenue")
cv_lgb_cogs = run_purged_cv(X_train, y_train_cogs, lgb_base, cv, label="LGB-COGS")

# %%
# ── Phase B1b: XGBoost baseline (with purged CV) ──────────────────────────
xgb_base = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.02, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=SEED, n_jobs=-1, verbosity=0, tree_method="hist",
)
cv_xgb_rev = run_purged_cv(X_train, y_train_rev, xgb_base, cv, label="XGB-Revenue")
cv_xgb_cogs = run_purged_cv(X_train, y_train_cogs, xgb_base, cv, label="XGB-COGS")

# %% [markdown]
# ## 7 - Phase B2 -- Optuna Hyperparameter Tuning (COMPOSITE OBJECTIVE)
# Key change: Optimize composite_score(0.4*MAE_norm + 0.4*RMSE_norm + 0.2*(1-R2))
# instead of MAE alone. This balances all 3 competition metrics.

# %%
OPTUNA_DB = "sqlite:///../../.claude/optuna.db"

class DummyStudy:
    """Wrapper for pre-defined params that mimics Optuna study interface."""
    def __init__(self, best_params, best_value=0.0):
        self.best_params = best_params
        self.best_value = best_value

# ── Sensible default params for models where Optuna is too slow ────────────
# Designed for retail forecasting with 374 features, ~3K training samples
DEFAULT_XGB_PARAMS = {
    "n_estimators": 1000, "learning_rate": 0.03, "max_depth": 6,
    "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
}

DEFAULT_CATBOOST_PARAMS = {
    "n_estimators": 1000, "learning_rate": 0.03, "depth": 6,
    "l2_leaf_reg": 3.0, "random_strength": 1.0, "bagging_temperature": 0.5,
}

DEFAULT_ET_PARAMS = {
    "n_estimators": 500, "max_depth": 15, "min_samples_split": 5,
    "min_samples_leaf": 2, "max_features": 0.5,
}

DEFAULT_RF_PARAMS = {
    "n_estimators": 500, "max_depth": 15, "min_samples_split": 5,
    "min_samples_leaf": 2, "max_features": 0.5,
}

DEFAULT_HGB_PARAMS = {
    "max_iter": 500, "learning_rate": 0.03, "max_depth": 6,
    "min_samples_leaf": 20, "l2_regularization": 1.0, "max_leaf_nodes": 127,
}

def tune_model_with_optuna(X, y, study_name, param_fn, model_class, n_trials=100, seed=42):
    """Generic Optuna tuner using composite_score as objective."""
    print(f"\n{'='*60}")
    print(f"  Optuna Tuning: {study_name}")
    print(f"{'='*60}")

    base_mae  = mean_absolute_error(y, np.full_like(y, y.mean()))
    base_rmse = np.sqrt(mean_squared_error(y, np.full_like(y, y.mean())))

    def objective(trial):
        params = param_fn(trial)
        scores = []
        for tr_idx, val_idx in cv.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            model = model_class(**params)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            scores.append(composite_score(y_val, y_pred, base_mae, base_rmse))
        return np.mean(scores)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        study_name=study_name,
        storage=OPTUNA_DB,
        load_if_exists=True,
    )

    remaining = n_trials - len(study.trials)
    if remaining > 0:
        print(f"  Running {remaining} trials (total target: {n_trials})...")
        study.optimize(objective, n_trials=remaining, show_progress_bar=True)
    else:
        print(f"  [SKIP] Already has {len(study.trials)} trials, loading from cache")

    print(f"  Best composite score: {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")
    return study

# %%
# ── Tune LightGBM ─────────────────────────────────────────────────────────
print("\n>>> Tuning LightGBM (Revenue) with composite objective...")
study_lgb_rev = tune_model_with_optuna(
    X_train, y_train_rev, "lgb_revenue_v2",
    suggest_lgb_params, lgb.LGBMRegressor, n_trials=40, seed=SEED,
)

print("\n>>> Tuning LightGBM (COGS) with composite objective...")
study_lgb_cogs = tune_model_with_optuna(
    X_train, y_train_cogs, "lgb_cogs_v2",
    suggest_lgb_params, lgb.LGBMRegressor, n_trials=40, seed=SEED+1,
)

# %%
# ── XGBoost with default params (Optuna too slow with 374 features) ──────
print("\n>>> XGBoost (Revenue) -- using default params (skip Optuna)...")
study_xgb_rev = DummyStudy(DEFAULT_XGB_PARAMS)

print(">>> XGBoost (COGS) -- using default params (skip Optuna)...")
study_xgb_cogs = DummyStudy(DEFAULT_XGB_PARAMS)

# %%
# ── CatBoost with default params (Optuna too slow) ────────────────────────
if HAS_CATBOOST:
    print("\n>>> CatBoost (Revenue) -- using default params (skip Optuna)...")
    study_cb_rev = DummyStudy(DEFAULT_CATBOOST_PARAMS)

    print(">>> CatBoost (COGS) -- using default params (skip Optuna)...")
    study_cb_cogs = DummyStudy(DEFAULT_CATBOOST_PARAMS)
else:
    study_cb_rev = None
    study_cb_cogs = None

# %%
# ── ExtraTrees with default params ────────────────────────────────────────
print("\n>>> ExtraTrees (Revenue) -- using default params (skip Optuna)...")
study_et_rev = DummyStudy(DEFAULT_ET_PARAMS)

print(">>> ExtraTrees (COGS) -- using default params (skip Optuna)...")
study_et_cogs = DummyStudy(DEFAULT_ET_PARAMS)

# %%
# ── RandomForest with default params ──────────────────────────────────────
print("\n>>> RandomForest (Revenue) -- using default params (skip Optuna)...")
study_rf_rev = DummyStudy(DEFAULT_RF_PARAMS)

print(">>> RandomForest (COGS) -- using default params (skip Optuna)...")
study_rf_cogs = DummyStudy(DEFAULT_RF_PARAMS)

# %%
# ── HistGradientBoosting with default params ──────────────────────────────
print("\n>>> HistGradientBoosting (Revenue) -- using default params (skip Optuna)...")
study_hgb_rev = DummyStudy(DEFAULT_HGB_PARAMS)

print(">>> HistGradientBoosting (COGS) -- using default params (skip Optuna)...")
study_hgb_cogs = DummyStudy(DEFAULT_HGB_PARAMS)

# %% [markdown]
# ## 8 - Phase C -- Advanced Ensemble Construction

# %%
# ── Build all tuned models ────────────────────────────────────────────────
def build_model(study, model_class, seed=SEED):
    """Build model from Optuna study with best params."""
    if study is None:
        return None
    params = {k: v for k, v in study.best_params.items()
              if k not in ("n_estimators", "max_iter")}
    return model_class(**study.best_params, **params, random_state=seed, n_jobs=-1)

# LightGBM
lgb_tuned_rev = lgb.LGBMRegressor(
    **study_lgb_rev.best_params, random_state=SEED, n_jobs=-1, verbose=-1)
lgb_tuned_cogs = lgb.LGBMRegressor(
    **study_lgb_cogs.best_params, random_state=SEED, n_jobs=-1, verbose=-1)

# XGBoost
xgb_tuned_rev = xgb.XGBRegressor(
    **study_xgb_rev.best_params, random_state=SEED, n_jobs=-1, verbosity=0, tree_method="hist")
xgb_tuned_cogs = xgb.XGBRegressor(
    **study_xgb_cogs.best_params, random_state=SEED, n_jobs=-1, verbosity=0, tree_method="hist")

# CatBoost
if HAS_CATBOOST and study_cb_rev is not None:
    cb_tuned_rev = CatBoostRegressor(
        **study_cb_rev.best_params, random_seed=SEED, thread_count=-1, verbose=0, allow_writing_files=False)
    cb_tuned_cogs = CatBoostRegressor(
        **study_cb_cogs.best_params, random_seed=SEED, thread_count=-1, verbose=0, allow_writing_files=False)
else:
    cb_tuned_rev = cb_tuned_cogs = None

# ExtraTrees
et_tuned_rev = ExtraTreesRegressor(**study_et_rev.best_params, random_state=SEED, n_jobs=-1)
et_tuned_cogs = ExtraTreesRegressor(**study_et_cogs.best_params, random_state=SEED, n_jobs=-1)

# RandomForest
rf_tuned_rev = RandomForestRegressor(**study_rf_rev.best_params, random_state=SEED, n_jobs=-1)
rf_tuned_cogs = RandomForestRegressor(**study_rf_cogs.best_params, random_state=SEED, n_jobs=-1)

# HistGradientBoosting
hgb_tuned_rev = HistGradientBoostingRegressor(**study_hgb_rev.best_params, random_state=SEED)
hgb_tuned_cogs = HistGradientBoostingRegressor(**study_hgb_cogs.best_params, random_state=SEED)

# Ridge (already trained)
ridge_rev = lr_rev
ridge_cogs = lr_cogs

print("All tuned models built successfully.")
print(f"  Models: LGB, XGB, {'' if cb_tuned_rev else 'no '}CatBoost, ExtraTrees, RF, HistGB, Ridge, Seasonal")

# %% [markdown]
# ### 8.1 Generate Out-of-Fold Predictions for Stacking

# %%
print("Generating OOF predictions for stacking meta-learner...")

# Initialize OOF storage for each model
model_names = ["lgb", "xgb", "ridge"]
model_pairs_rev = {
    "lgb": lgb_tuned_rev,
    "xgb": xgb_tuned_rev,
    "ridge": ridge_rev,
}
model_pairs_cogs = {
    "lgb": lgb_tuned_cogs,
    "xgb": xgb_tuned_cogs,
    "ridge": ridge_cogs,
}

if cb_tuned_rev:
    model_names.append("catboost")
    model_pairs_rev["catboost"] = cb_tuned_rev
    model_pairs_cogs["catboost"] = cb_tuned_cogs

model_names.extend(["et", "rf", "hgb"])
model_pairs_rev["et"] = et_tuned_rev
model_pairs_rev["rf"] = rf_tuned_rev
model_pairs_rev["hgb"] = hgb_tuned_rev
model_pairs_cogs["et"] = et_tuned_cogs
model_pairs_cogs["rf"] = rf_tuned_cogs
model_pairs_cogs["hgb"] = hgb_tuned_cogs

n_models = len(model_names)

# OOF predictions for Revenue
meta_X_rev = np.zeros((len(X_train), n_models + 1))  # +1 for seasonal
meta_y_rev = np.zeros(len(X_train))
meta_X_cogs = np.zeros((len(X_train), n_models + 1))
meta_y_cogs = np.zeros(len(X_train))

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train)):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr_rev, y_val_rev = y_train_rev.iloc[tr_idx], y_train_rev.iloc[val_idx]
    y_tr_cogs, y_val_cogs = y_train_cogs.iloc[tr_idx], y_train_cogs.iloc[val_idx]

    print(f"  Fold {fold+1}: train={len(X_tr):,} val={len(X_val):,}")

    for j, name in enumerate(model_names):
        # Revenue
        m_rev = model_pairs_rev[name].__class__(**model_pairs_rev[name].get_params())
        m_rev.fit(X_tr, y_tr_rev)
        meta_X_rev[val_idx, j] = m_rev.predict(X_val)

        # COGS
        m_cogs = model_pairs_cogs[name].__class__(**model_pairs_cogs[name].get_params())
        m_cogs.fit(X_tr, y_tr_cogs)
        meta_X_cogs[val_idx, j] = m_cogs.predict(X_val)

    # Seasonal baseline
    val_dates = train.iloc[val_idx]["Date"].values
    seas_r, seas_c = seasonal_predict(val_dates)
    meta_X_rev[val_idx, n_models] = seas_r
    meta_X_cogs[val_idx, n_models] = seas_c

    meta_y_rev[val_idx] = y_val_rev.values
    meta_y_cogs[val_idx] = y_val_cogs.values

all_model_names = model_names + ["seasonal"]
print(f"OOF predictions generated. Meta-features: {meta_X_rev.shape[1]}")

# %% [markdown]
# ### 8.2 Learn Ensemble Weights (Multiple Methods)

# %%
# ── Method 1: Optimized weighted blend ────────────────────────────────────
oof_preds_rev = {name: meta_X_rev[:, i] for i, name in enumerate(all_model_names)}
oof_preds_cogs = {name: meta_X_cogs[:, i] for i, name in enumerate(all_model_names)}

weights_rev = optimize_ensemble_weights(oof_preds_rev, meta_y_rev, non_negative=True)
weights_cogs = optimize_ensemble_weights(oof_preds_cogs, meta_y_cogs, non_negative=True)

print("\n" + "=" * 70)
print("  OPTIMIZED ENSEMBLE WEIGHTS")
print("-" * 70)
print(f"  {'Model':<20} {'Revenue':>10} {'COGS':>10}")
print("-" * 70)
for name in all_model_names:
    print(f"  {name:<20} {weights_rev.get(name, 0):>9.1%} {weights_cogs.get(name, 0):>9.1%}")
print("=" * 70)

# ── Method 2: Simple average ──────────────────────────────────────────────
avg_preds_rev = weighted_blend(oof_preds_rev, {n: 1.0 for n in all_model_names})
avg_preds_cogs = weighted_blend(oof_preds_cogs, {n: 1.0 for n in all_model_names})

# ── Method 3: Rank average ────────────────────────────────────────────────
rank_preds_rev = rank_average(oof_preds_rev)
rank_preds_cogs = rank_average(oof_preds_cogs)

# ── Method 4: Median ensemble ─────────────────────────────────────────────
median_preds_rev = median_ensemble(oof_preds_rev)
median_preds_cogs = median_ensemble(oof_preds_cogs)

# ── Method 5: Horizon-specific blending ───────────────────────────────────
get_horizon_weights_rev, horizon_w_rev = horizon_specific_blend(
    oof_preds_rev, meta_y_rev, dates=None)

# ── Method 6: Stacking with Ridge meta-learner ─────────────────────────────
meta_ridge_rev = Ridge(alpha=1.0, fit_intercept=False, positive=True, random_state=SEED)
meta_ridge_rev.fit(meta_X_rev, meta_y_rev)
stack_weights_rev = meta_ridge_rev.coef_ / meta_ridge_rev.coef_.sum()
stack_weights_rev_dict = {name: stack_weights_rev[i] for i, name in enumerate(all_model_names)}

meta_ridge_cogs = Ridge(alpha=1.0, fit_intercept=False, positive=True, random_state=SEED)
meta_ridge_cogs.fit(meta_X_cogs, meta_y_cogs)
stack_weights_cogs = meta_ridge_cogs.coef_ / meta_ridge_cogs.coef_.sum()
stack_weights_cogs_dict = {name: stack_weights_cogs[i] for i, name in enumerate(all_model_names)}

# ── Compare all methods ───────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  ENSEMBLE METHOD COMPARISON (OOF Revenue)")
print("-" * 70)
print(f"  {'Method':<30} {'MAE':>14} {'RMSE':>14} {'R2':>10} {'Composite':>10}")
print("-" * 70)

for method_name, preds in [
    ("Weighted Blend (optimized)", weighted_blend(oof_preds_rev, weights_rev)),
    ("Stacking (Ridge meta-learner)", meta_ridge_rev.predict(meta_X_rev)),
    ("Simple Average", avg_preds_rev),
    ("Rank Average", rank_preds_rev),
    ("Median Ensemble", median_preds_rev),
    ("Best Single (LGB)", oof_preds_rev["lgb"]),
]:
    m = evaluate_predictions(meta_y_rev, preds)
    print(f"  {method_name:<30} {m['mae']:>14,.0f} {m['rmse']:>14,.0f} {m['r2']:>10.4f} {m['composite']:>10.4f}")
print("=" * 70)

# Select best ensemble method
BEST_METHOD = "stacking"
if BEST_METHOD == "stacking":
    final_weights_rev = stack_weights_rev_dict
    final_weights_cogs = stack_weights_cogs_dict
elif BEST_METHOD == "optimized":
    final_weights_rev = weights_rev
    final_weights_cogs = weights_cogs
else:
    final_weights_rev = {n: 1.0 / len(all_model_names) for n in all_model_names}
    final_weights_cogs = {n: 1.0 / len(all_model_names) for n in all_model_names}

print(f"\nUsing ensemble method: {BEST_METHOD}")

# %% [markdown]
# ## 9 - Final Model Training & Iterative Recursive Prediction

# %%
# ── Train all final models on full training data ──────────────────────────
print("Training final models on full training data...")

final_models_rev = {
    "lgb": lgb_tuned_rev.__class__(**lgb_tuned_rev.get_params()),
    "xgb": xgb_tuned_rev.__class__(**xgb_tuned_rev.get_params()),
    "ridge": ridge_rev.__class__(**ridge_rev.get_params()),
    "et": et_tuned_rev.__class__(**et_tuned_rev.get_params()),
    "rf": rf_tuned_rev.__class__(**rf_tuned_rev.get_params()),
    "hgb": hgb_tuned_rev.__class__(**hgb_tuned_rev.get_params()),
}
final_models_cogs = {
    "lgb": lgb_tuned_cogs.__class__(**lgb_tuned_cogs.get_params()),
    "xgb": xgb_tuned_cogs.__class__(**xgb_tuned_cogs.get_params()),
    "ridge": ridge_cogs.__class__(**ridge_cogs.get_params()),
    "et": et_tuned_cogs.__class__(**et_tuned_cogs.get_params()),
    "rf": rf_tuned_cogs.__class__(**rf_tuned_cogs.get_params()),
    "hgb": hgb_tuned_cogs.__class__(**hgb_tuned_cogs.get_params()),
}

if HAS_CATBOOST and cb_tuned_rev is not None:
    final_models_rev["catboost"] = cb_tuned_rev.__class__(**cb_tuned_rev.get_params())
    final_models_cogs["catboost"] = cb_tuned_cogs.__class__(**cb_tuned_cogs.get_params())

for name in final_models_rev:
    print(f"  Training {name} (Revenue)...")
    final_models_rev[name].fit(X_train, y_train_rev)
    print(f"  Training {name} (COGS)...")
    final_models_cogs[name].fit(X_train, y_train_cogs)

# %%
# ── Iterative day-by-day recursive prediction ─────────────────────────────
test_dates_sorted = sorted(test["Date"].tolist())
fp = full.set_index("Date").copy()

# Pre-compute seasonal baseline
seas_rev_test, seas_cogs_test = seasonal_predict(test_dates_sorted)

# Build historical bounds for clipping
rev_bounds = build_historical_bounds(sales, "Revenue")
cogs_bounds = build_historical_bounds(sales, "COGS")

print(f"Predicting {len(test_dates_sorted)} test days recursively...")
print(f"  Ensemble: {len(final_weights_rev)} models, method={BEST_METHOD}")

for i, d in enumerate(test_dates_sorted):
    if i % 100 == 0 or i == 0:
        print(f"  Day {i+1:>3}/{len(test_dates_sorted)}  ({d.date()})")

    si = i  # index in test array
    row = fp.loc[[d], FEATURE_COLS].fillna(0)

    # ── Revenue prediction ────────────────────────────────────────────────
    rev_preds = {}
    for name, model in final_models_rev.items():
        p = float(model.predict(row)[0])
        rev_preds[name] = max(p, 0)

    # Seasonal baseline
    rev_preds["seasonal"] = max(float(seas_rev_test[si]), 0)

    # Blend
    blended_rev = 0.0
    total_w = 0.0
    for name, p in rev_preds.items():
        w = final_weights_rev.get(name, 0)
        blended_rev += w * p
        total_w += w
    fp.loc[d, "Revenue"] = blended_rev / max(total_w, 1e-9)

    # ── COGS prediction ───────────────────────────────────────────────────
    cogs_preds = {}
    for name, model in final_models_cogs.items():
        p = float(model.predict(row)[0])
        cogs_preds[name] = max(p, 0)

    cogs_preds["seasonal"] = max(float(seas_cogs_test[si]), 0)

    blended_cogs = 0.0
    total_w = 0.0
    for name, p in cogs_preds.items():
        w = final_weights_cogs.get(name, 0)
        blended_cogs += w * p
        total_w += w
    fp.loc[d, "COGS"] = blended_cogs / max(total_w, 1e-9)

    # ── Recompute all derived features ────────────────────────────────────
    if i < len(test_dates_sorted) - 1:
        fp = add_target_features_enhanced(fp.reset_index(), "Revenue")
        fp = add_target_features_enhanced(fp, "COGS")
        for lag in [7, 28, 90, 180, 365]:
            fp[f"cogs_rev_ratio_l{lag}"] = (fp[f"COGS_l{lag}"] /
                                               (fp[f"Revenue_l{lag}"] + 1e-9))
        for lag in [28, 90, 365]:
            rev_l = fp[f"Revenue_l{lag}"]
            cogs_l = fp[f"COGS_l{lag}"]
            fp[f"margin_l{lag}"] = (rev_l - cogs_l) / (rev_l + 1e-9)
        fp = fp.set_index("Date")

print("Recursive prediction complete.")

# %% [markdown]
# ## 10 - Post-Processing -- Clipping, Smoothing, Residual Correction

# %%
# ── Extract raw predictions ───────────────────────────────────────────────
pred_dates = test_dates_sorted
raw_rev = fp.loc[pred_dates, "Revenue"].values
raw_cogs = fp.loc[pred_dates, "COGS"].values

# ── 10.1 Clipping to historical bounds ────────────────────────────────────
clipped_rev = clip_to_historical(raw_rev, pred_dates, rev_bounds, "Revenue", multiplier=3.0)
clipped_cogs = clip_to_historical(raw_cogs, pred_dates, cogs_bounds, "COGS", multiplier=3.0)

print(f"Clipping applied:")
print(f"  Revenue: {np.sum(raw_rev != clipped_rev)} values clipped "
      f"(range before: [{raw_rev.min():,.0f}, {raw_rev.max():,.0f}], "
      f"after: [{clipped_rev.min():,.0f}, {clipped_rev.max():,.0f}])")
print(f"  COGS:    {np.sum(raw_cogs != clipped_cogs)} values clipped "
      f"(range before: [{raw_cogs.min():,.0f}, {raw_cogs.max():,.0f}], "
      f"after: [{clipped_cogs.min():,.0f}, {clipped_cogs.max():,.0f}])")

# ── 10.2 Exponential smoothing ────────────────────────────────────────────
smoothed_rev = exponential_smooth(clipped_rev, alpha=0.25)
smoothed_cogs = exponential_smooth(clipped_cogs, alpha=0.25)

# ── 10.3 Final predictions ────────────────────────────────────────────────
final_rev = smoothed_rev
final_cogs = smoothed_cogs

# Update fp with post-processed values
for i, d in enumerate(pred_dates):
    fp.loc[d, "Revenue"] = final_rev[i]
    fp.loc[d, "COGS"] = final_cogs[i]

print(f"\nPost-processing complete.")
print(f"  Final Revenue: mean={final_rev.mean():,.0f}  "
      f"min={final_rev.min():,.0f}  max={final_rev.max():,.0f}")
print(f"  Final COGS:    mean={final_cogs.mean():,.0f}  "
      f"min={final_cogs.min():,.0f}  max={final_cogs.max():,.0f}")

# %% [markdown]
# ## 11 - Evaluation & Diagnostics

# %% [markdown]
# ### 11.1 CV Fold Stability Analysis

# %%
print("=" * 70)
print("  CV FOLD STABILITY ANALYSIS")
print("=" * 70)

stab_rev = fold_stability_analysis(cv_lgb_rev)
print(f"\nRevenue (LGB baseline):")
for k, v in stab_rev.items():
    print(f"  {k}: {v:.4f}")

# %%
# ── CV fold plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.flatten()

for i, (tr_idx, val_idx) in enumerate(cv.split(X_train)):
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

maes = cv_lgb_rev["fold_mae"]
comps = cv_lgb_rev["fold_composite"]
axes[-1].bar(range(1, 4), maes, color=PALETTE[0], alpha=0.85, edgecolor="white")
axes[-1].axhline(np.mean(maes), color="red", linestyle="--", label=f"Mean MAE: {np.mean(maes):,.0f}")
axes[-1].set_xlabel("Fold"); axes[-1].set_ylabel("MAE"); axes[-1].set_title("MAE per Fold (Purged Walk-Forward)")
axes[-1].legend(fontsize=9)

plt.suptitle("Purged Walk-Forward CV -- 3 Folds, horizon=365d, purge=90d", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/cv_folds.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 11.2 Horizon Breakdown Analysis

# %%
# Simulate horizon breakdown on the last CV fold's validation set
last_tr_idx, last_val_idx = list(cv.split(X_train))[-1]
X_val_last = X_train.iloc[last_val_idx]
y_val_last_rev = y_train_rev.iloc[last_val_idx]

# Predict with LGB
lgb_val = lgb.LGBMRegressor(**lgb_tuned_rev.get_params())
lgb_val.fit(X_train.iloc[last_tr_idx], y_train_rev.iloc[last_tr_idx])
y_pred_last = lgb_val.predict(X_val_last)

hb = horizon_breakdown(None, y_val_last_rev.values, y_pred_last)
print("Horizon Breakdown -- Revenue (last CV fold):")
print(f"  {'Bucket':<12} {'MAE':>14} {'RMSE':>14} {'R2':>10}")
print("-" * 55)
for name, metrics in hb.items():
    print(f"  {name:<12} {metrics[f'{name}_mae']:>14,.0f} {metrics[f'{name}_rmse']:>14,.0f} {metrics[f'{name}_r2']:>10.4f}")

# %% [markdown]
# ### 11.3 Drift Analysis

# %%
drift_df = drift_analysis(y_val_last_rev.values, y_pred_last, window=30)
print("Drift Analysis -- Revenue (last CV fold, 30d windows):")
print(f"  {'Window':>10} {'MAE':>14} {'RMSE':>14} {'Bias':>14}")
print("-" * 60)
for _, row in drift_df.iterrows():
    print(f"  [{row['start_idx']:>4}-{row['end_idx']:>4}] {row['rolling_mae']:>14,.0f} {row['rolling_rmse']:>14,.0f} {row['rolling_bias']:>14,.0f}")

# Check for drift trend
if len(drift_df) > 2:
    drift_trend = np.polyfit(range(len(drift_df)), drift_df["rolling_mae"], 1)[0]
    print(f"\n  MAE drift trend: {drift_trend:+.2f}/window "
          f"({'INCREASING -- model degrades over horizon' if drift_trend > 0 else 'STABLE -- good long-horizon behavior'})")

# %% [markdown]
# ## 12 - SHAP Feature Importance

# %%
print("Computing SHAP values...")
rng = np.random.default_rng(SEED)
n_shap = min(800, len(X_train))
idx_shap = rng.choice(len(X_train), n_shap, replace=False)
X_shap = X_train.iloc[idx_shap]

# Use final trained LGB model (fitted on full training data)
lgb_for_shap = final_models_rev["lgb"]
explainer = shap.TreeExplainer(lgb_for_shap)
shap_values = explainer.shap_values(X_shap)

shap_df = (pd.DataFrame({"feature": FEATURE_COLS, "shap_mean": np.abs(shap_values).mean(0)})
           .sort_values("shap_mean", ascending=False)
           .reset_index(drop=True))

print("\nTop-25 Revenue Drivers:")
print(shap_df.head(25)[["feature", "shap_mean"]].to_string(index=False))

# %%
# ── SHAP visualization ────────────────────────────────────────────────────
try:
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_shap, feature_names=FEATURE_COLS,
                      show=False, max_display=20, plot_size=None)
    axes[0].set_title("SHAP Beeswarm -- Revenue Model\n(red=high, blue=low)", fontsize=10)

    top20 = shap_df.head(20)
    axes[1].barh(top20["feature"][::-1], top20["shap_mean"][::-1], color=PALETTE[0], alpha=0.85)
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].set_title("Top-20 Features by Mean |SHAP|", fontsize=11)
    axes[1].tick_params(labelsize=8)

    plt.suptitle("SHAP Feature Importance -- LightGBM Revenue Model (Optimized)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("SHAP visualization saved.")
except Exception as e:
    print(f"SHAP visualization skipped (error: {e})")

# %% [markdown]
# ## 13 - Generate Submission

# %%
# ── Extract predictions ───────────────────────────────────────────────────
pred_df = (fp.loc[test_dates_sorted, ["Revenue", "COGS"]]
           .reset_index().rename(columns={"index": "Date"}))
pred_df["Revenue"] = pred_df["Revenue"].clip(lower=0).round(2)
pred_df["COGS"]    = pred_df["COGS"].clip(lower=0).round(2)

# Ensure correct Date format for merge
pred_df["Date"] = pd.to_datetime(pred_df["Date"])
submission = sample_sub[["Date"]].merge(pred_df, on="Date", how="left")

# ── Integrity checks ──────────────────────────────────────────────────────
assert len(submission) == len(sample_sub), f"Row count mismatch: {len(submission)} vs {len(sample_sub)}"
assert list(submission["Date"]) == list(sample_sub["Date"]), "Date order mismatch!"
assert submission[["Revenue", "COGS"]].isna().sum().sum() == 0, "NaN in submission!"
assert (submission["Revenue"] >= 0).all(), "Negative Revenue!"
assert (submission["COGS"] >= 0).all(), "Negative COGS!"

# ── Save ──────────────────────────────────────────────────────────────────
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
# ── Final forecast plot ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 9))

# 2021-2022 in-sample fit
chk = train[train["Date"].dt.year.between(2021, 2022)].copy()
p_chk_rev = final_models_rev["lgb"].predict(chk[FEATURE_COLS].fillna(0))
mae_chk = mean_absolute_error(chk["Revenue"], p_chk_rev)

axes[0].fill_between(chk["Date"], chk["Revenue"]/1e6, alpha=0.12, color=PALETTE[0])
axes[0].plot(chk["Date"], chk["Revenue"]/1e6, lw=1.2, color=PALETTE[0], label="Actual")
axes[0].plot(chk["Date"], p_chk_rev/1e6, lw=1.0, color=PALETTE[1], linestyle="--",
             label=f"Fitted (MAE={mae_chk:,.0f})")
axes[0].set_title("2021-2022 In-Sample Fit (LightGBM Tuned)")
axes[0].legend(fontsize=9)
axes[0].set_ylabel("Revenue (M VND)")

# Forecast 2023-2024
axes[1].fill_between(submission["Date"], submission["Revenue"]/1e6, alpha=0.12, color=PALETTE[2])
axes[1].plot(submission["Date"], submission["Revenue"]/1e6, lw=1.5, color=PALETTE[2], label="Revenue Forecast")
axes[1].plot(submission["Date"], submission["COGS"]/1e6, lw=1.2, color=PALETTE[1], linestyle="--", alpha=0.8, label="COGS Forecast")
profit = (submission["Revenue"] - submission["COGS"]) / 1e6
axes[1].fill_between(submission["Date"], 0, profit, alpha=0.15, color=PALETTE[3], label="Gross Profit")
axes[1].set_title("Optimized Ensemble Forecast -- Jan 2023 to Jul 2024")
axes[1].legend(fontsize=9)
axes[1].set_ylabel("Value (M VND)")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

plt.suptitle("Final Forecast: Revenue - COGS - Gross Profit (Optimized Pipeline)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/forecast_final.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 14 - Pipeline Summary

# %%
print("=" * 70)
print("  FINAL PIPELINE SUMMARY -- Optimized Part 3 Forecasting")
print("=" * 70)
print(f"  Features used     : {len(FEATURE_COLS)}")
print(f"  Training samples  : {len(train):,}")
print(f"  Test samples      : {len(test):,}")
print()
print(f"  Validation: PurgedWalkForwardCV (n_splits=3, horizon=365d, purge=90d)")
print(f"  Objective: Composite (0.4*MAE_norm + 0.4*RMSE_norm + 0.2*(1-R2))")
print()
print(f"  Models in ensemble:")
for name in final_weights_rev:
    w_rev = final_weights_rev.get(name, 0)
    w_cogs = final_weights_cogs.get(name, 0)
    print(f"    {name:<15} Revenue={w_rev:.1%}  COGS={w_cogs:.1%}")
print()
print(f"  CV Revenue Metrics (Purged Walk-Forward):")
print(f"    MAE:  {cv_lgb_rev['overall_mae']:,.0f}")
print(f"    RMSE: {cv_lgb_rev['overall_rmse']:,.0f}")
print(f"    R2:   {cv_lgb_rev['overall_r2']:.4f}")
print(f"    Comp: {cv_lgb_rev['overall_composite']:.4f}")
print()
print(f"  Submission: {submission_path} ({len(submission)} rows)")
print()
print("  Key improvements over original:")
print("  [NEW] Purged Walk-Forward CV (gap=90d -> realistic evaluation)")
print("  [NEW] Composite objective (MAE + RMSE + R2 balanced)")
print("  [NEW] Fixed Tet dates (lunar calendar, not fixed Jan20-Feb20)")
print("  [NEW] 8-model ensemble (LGB+XGB+CatBoost+ET+RF+HGB+Ridge+Seasonal)")
print("  [NEW] Stacking meta-learner for optimal blending weights")
print("  [NEW] Post-processing (clipping + smoothing)")
print("  [NEW] Momentum, volatility, quantile features")
print("  [NEW] Horizon breakdown & drift analysis")
print("=" * 70)
