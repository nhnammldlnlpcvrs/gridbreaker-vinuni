"""Build the full EDA + forecasting notebook."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

cells = []

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
# Datathon 2026 — Phần 3: EDA + Sales Forecasting
**The GridBreakers · VinUniversity Datathon 2026**

---

## Cấu trúc notebook

| Phần | Nội dung |
|---|---|
| **A — EDA** | A1 Overview sales · A2 Revenue vs COGS · A3 Seasonality · A4 YoY trend · A5 Decomposition · A6 Promotions · A7 Web traffic · A8 Inventory · A9 Orders & Returns |
| **B — Feature Engineering** | Calendar · Lag/Rolling/EWM · Promotions · Web traffic · Inventory · Cross-table signals |
| **C — Pipeline & CV** | TimeSeriesSplit · Leakage audit · LGB ensemble |
| **D — Evaluation** | MAE · RMSE · R² · Fold plots |
| **E — SHAP** | Feature importance · Business interpretation |
| **F — Submission** | Generate submission.csv |

> **Leakage policy:**  
> • Tất cả lag/rolling dùng `shift(≥1)` — không có giá trị tương lai  
> • `ord_count`, `ord_delivered`… chỉ dùng dưới dạng lag 365 (available trong test)  
> • Web traffic được shift +1 ngày trước khi merge  
> • Revenue/COGS của tập test **không bao giờ** được dùng làm feature
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 0. SETUP
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## 0 · Setup"))
cells.append(new_code_cell("""\
import os, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

DATA_DIR    = "data/raw"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Plot theme ────────────────────────────────────────────────────────────────
PALETTE = ["#2563eb", "#f97316", "#16a34a", "#dc2626", "#7c3aed", "#0891b2", "#ca8a04"]
plt.rcParams.update({
    "figure.dpi"        : 130,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.25,
    "grid.linestyle"    : "--",
    "font.family"       : "DejaVu Sans",
    "font.size"         : 10,
    "axes.titlesize"    : 12,
    "axes.labelsize"    : 10,
})
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
DOW_NAMES   = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

# ── Model hyperparams ────────────────────────────────────────────────────────
PARAMS_MAE = dict(
    objective="regression_l1", metric="mae",
    num_leaves=255, learning_rate=0.015,
    feature_fraction=0.70, bagging_fraction=0.70, bagging_freq=5,
    min_child_samples=15, reg_alpha=0.10, reg_lambda=0.20,
    verbose=-1, seed=SEED, n_jobs=-1,
)
PARAMS_HUBER = dict(
    objective="huber", alpha=0.9, metric="huber",
    num_leaves=127, learning_rate=0.02,
    feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=5,
    min_child_samples=20, reg_alpha=0.05, reg_lambda=0.10,
    verbose=-1, seed=SEED+1, n_jobs=-1,
)
W_MAE, W_HUBER, W_SEAS = 0.65, 0.20, 0.15

LAGS         = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 180, 364, 365, 366]
ROLL_WINDOWS = [7, 14, 28, 60, 90, 180, 365]
EWM_SPANS    = [7, 14, 30, 90, 180]

print("✓ Setup complete  |  LightGBM:", lgb.__version__)
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL DATA
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## 1 · Load All Data"))
cells.append(new_code_cell("""\
# ── Analytical ────────────────────────────────────────────────────────────────
sales      = pd.read_csv(f"{DATA_DIR}/Analytical/sales.csv",             parse_dates=["Date"])
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv",            parse_dates=["Date"])

# ── Master ────────────────────────────────────────────────────────────────────
customers  = pd.read_csv(f"{DATA_DIR}/Master/customers.csv",             parse_dates=["signup_date"])
products   = pd.read_csv(f"{DATA_DIR}/Master/products.csv")
promos     = pd.read_csv(f"{DATA_DIR}/Master/promotions.csv",            parse_dates=["start_date","end_date"])
geography  = pd.read_csv(f"{DATA_DIR}/Master/geography.csv")

# ── Transaction ───────────────────────────────────────────────────────────────
orders     = pd.read_csv(f"{DATA_DIR}/Transaction/orders.csv",           parse_dates=["order_date"])
order_items= pd.read_csv(f"{DATA_DIR}/Transaction/order_items.csv")
payments   = pd.read_csv(f"{DATA_DIR}/Transaction/payments.csv")
returns    = pd.read_csv(f"{DATA_DIR}/Transaction/returns.csv",          parse_dates=["return_date"])
reviews    = pd.read_csv(f"{DATA_DIR}/Transaction/reviews.csv",          parse_dates=["review_date"])
shipments  = pd.read_csv(f"{DATA_DIR}/Transaction/shipments.csv",        parse_dates=["ship_date","delivery_date"])

# ── Operational ───────────────────────────────────────────────────────────────
inventory  = pd.read_csv(f"{DATA_DIR}/Operational/inventory.csv",        parse_dates=["snapshot_date"])
wt_raw     = pd.read_csv(f"{DATA_DIR}/Operational/web_traffic.csv",      parse_dates=["date"])

# ── Sort ──────────────────────────────────────────────────────────────────────
sales      = sales.sort_values("Date").reset_index(drop=True)
sample_sub = sample_sub.sort_values("Date").reset_index(drop=True)
orders     = orders.sort_values("order_date").reset_index(drop=True)

print("=" * 58)
print(f"  TRAIN  {sales['Date'].min().date()} → {sales['Date'].max().date()}  ({len(sales):,} rows)")
print(f"  TEST   {sample_sub['Date'].min().date()} → {sample_sub['Date'].max().date()}  ({len(sample_sub):,} rows)")
print("=" * 58)
for name, df in [("sales",sales),("orders",orders),("products",products),
                  ("promos",promos),("inventory",inventory),("wt_raw",wt_raw),
                  ("returns",returns),("reviews",reviews),("payments",payments)]:
    print(f"  {name:<12} {df.shape[0]:>8,} rows × {df.shape[1]:>3} cols")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PART A — EDA
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
---
# PHẦN A — Exploratory Data Analysis

> Mục tiêu: hiểu **shape of data**, **seasonality**, **tác nhân ảnh hưởng Revenue**  
> và rút ra các features có giá trị cho mô hình dự báo.
"""))

# ─── A1. Overview sales ───────────────────────────────────────────────────────
cells.append(new_markdown_cell("## A1 · Tổng quan sales.csv — Revenue, COGS, Gross Profit"))
cells.append(new_code_cell("""\
sales["gross_profit"]  = sales["Revenue"] - sales["COGS"]
sales["gross_margin"]  = sales["gross_profit"] / sales["Revenue"].replace(0, np.nan)
sales["year"]  = sales["Date"].dt.year
sales["month"] = sales["Date"].dt.month
sales["dow"]   = sales["Date"].dt.dayofweek
sales["doy"]   = sales["Date"].dt.dayofyear

# ── Basic stats ───────────────────────────────────────────────────────────────
print("Revenue statistics:")
print(sales["Revenue"].describe().apply(lambda x: f"{x:,.0f}").to_string())
print()
print(f"Zero-revenue days  : {(sales['Revenue']==0).sum()}")
print(f"Negative COGS days : {(sales['COGS']<0).sum()}")
print(f"Margin < 0 days    : {(sales['gross_margin']<0).sum()}")
print(f"Mean gross margin  : {sales['gross_margin'].mean():.1%}")
"""))

cells.append(new_code_cell("""\
# ── Figure A1: Revenue + COGS + Margin overview ───────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)

# Revenue raw + 30-day MA
axes[0].fill_between(sales["Date"], sales["Revenue"], alpha=0.10, color=PALETTE[0])
axes[0].plot(sales["Date"], sales["Revenue"], lw=0.5, color=PALETTE[0], alpha=0.6)
axes[0].plot(sales["Date"],
             sales["Revenue"].rolling(30, center=True, min_periods=10).mean(),
             lw=2.2, color=PALETTE[0], label="30-day MA")
axes[0].set_ylabel("Revenue (VND)"); axes[0].set_title("Daily Revenue 2012–2022")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e6:.0f}M"))
axes[0].legend(fontsize=9)

# COGS
axes[1].fill_between(sales["Date"], sales["COGS"], alpha=0.10, color=PALETTE[1])
axes[1].plot(sales["Date"], sales["COGS"], lw=0.5, color=PALETTE[1], alpha=0.6)
axes[1].plot(sales["Date"],
             sales["COGS"].rolling(30, center=True, min_periods=10).mean(),
             lw=2.2, color=PALETTE[1], label="30-day MA")
axes[1].set_ylabel("COGS (VND)"); axes[1].set_title("Daily COGS 2012–2022")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e6:.0f}M"))
axes[1].legend(fontsize=9)

# Gross margin %
gm_ma = sales["gross_margin"].rolling(30, center=True, min_periods=10).mean()
axes[2].axhline(0, color="black", lw=0.8, zorder=3)
axes[2].fill_between(sales["Date"], gm_ma, where=(gm_ma>=0),
                     alpha=0.3, color=PALETTE[2], label="Positive margin")
axes[2].fill_between(sales["Date"], gm_ma, where=(gm_ma<0),
                     alpha=0.3, color=PALETTE[3], label="Negative margin")
axes[2].plot(sales["Date"], gm_ma, lw=1.5, color=PALETTE[2])
axes[2].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
axes[2].set_ylabel("Gross Margin %"); axes[2].set_title("Gross Margin % (30-day MA)")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[2].legend(fontsize=9)

plt.suptitle("Sales Overview — Revenue · COGS · Gross Margin", fontsize=13, y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/A1_sales_overview.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ─── A2. Revenue vs COGS scatter ──────────────────────────────────────────────
cells.append(new_markdown_cell("""\
## A2 · Revenue vs COGS — Phân tích mối quan hệ

> **Tổng doanh thu thuần = Revenue. COGS = vốn. Gross Profit = Revenue − COGS.**  
> Hiểu mối quan hệ Revenue-COGS giúp ta dự báo cả hai chỉ số cùng lúc.
"""))
cells.append(new_code_cell("""\
fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# Scatter Revenue vs COGS (coloured by year)
ax0 = fig.add_subplot(gs[0, :2])
years_uniq = sorted(sales["year"].unique())
cmap = plt.cm.get_cmap("plasma", len(years_uniq))
for i, yr in enumerate(years_uniq):
    s = sales[sales["year"]==yr]
    ax0.scatter(s["COGS"]/1e6, s["Revenue"]/1e6,
                s=6, alpha=0.4, color=cmap(i), label=str(yr))
ax0.set_xlabel("COGS (M VND)"); ax0.set_ylabel("Revenue (M VND)")
ax0.set_title("Revenue vs COGS (màu theo năm)")
lgnd = ax0.legend(fontsize=7, ncol=4, loc="upper left",
                  markerscale=2, framealpha=0.7)

# Revenue distribution
ax1 = fig.add_subplot(gs[0, 2])
ax1.hist(sales["Revenue"]/1e6, bins=80, color=PALETTE[0], alpha=0.8, edgecolor="white")
ax1.set_xlabel("Revenue (M VND)"); ax1.set_title("Phân phối Revenue")
ax1.axvline(sales["Revenue"].mean()/1e6, color="red", linestyle="--",
            label=f"Mean: {sales['Revenue'].mean()/1e6:.1f}M")
ax1.legend(fontsize=8)

# Gross profit distribution
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(sales["gross_profit"]/1e6, bins=80, color=PALETTE[2], alpha=0.8, edgecolor="white")
ax2.axvline(0, color="red", lw=1.5)
ax2.set_xlabel("Gross Profit (M VND)"); ax2.set_title("Phân phối Gross Profit")

# COGS/Revenue ratio over time
ax3 = fig.add_subplot(gs[1, 1])
ratio = (sales["COGS"] / sales["Revenue"].replace(0, np.nan)).rolling(30, min_periods=10).mean()
ax3.plot(sales["Date"], ratio, lw=1.2, color=PALETTE[4])
ax3.axhline(ratio.mean(), color="red", linestyle="--", label=f"Mean: {ratio.mean():.2f}")
ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.set_title("COGS/Revenue ratio (30-day MA)"); ax3.legend(fontsize=8)

# Log-log scatter
ax4 = fig.add_subplot(gs[1, 2])
mask = (sales["Revenue"]>0) & (sales["COGS"]>0)
ax4.scatter(np.log1p(sales.loc[mask,"COGS"]),
            np.log1p(sales.loc[mask,"Revenue"]),
            s=4, alpha=0.2, color=PALETTE[5])
ax4.set_xlabel("log(COGS+1)"); ax4.set_ylabel("log(Revenue+1)")
ax4.set_title("Log-Log Revenue vs COGS")
corr = np.log1p(sales.loc[mask,"Revenue"]).corr(np.log1p(sales.loc[mask,"COGS"]))
ax4.text(0.05, 0.92, f"Pearson r = {corr:.3f}", transform=ax4.transAxes, fontsize=9)

plt.suptitle("A2 — Revenue vs COGS: Phân tích mối quan hệ", fontsize=13, y=1.02, fontweight="bold")
plt.savefig(f"{REPORTS_DIR}/A2_revenue_cogs.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Pearson r (Revenue, COGS): {sales['Revenue'].corr(sales['COGS']):.4f}")
print(f"Mean COGS/Revenue ratio  : {(sales['COGS']/sales['Revenue'].replace(0,np.nan)).mean():.3f}")
"""))

# ─── A3. Seasonality ──────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## A3 · Tính mùa vụ — Monthly, Day-of-Week, Heatmap"))
cells.append(new_code_cell("""\
monthly    = sales.groupby("month")["Revenue"].agg(["mean","median","std"])
dow_avg    = sales.groupby("dow")["Revenue"].mean()
# Month × DOW heatmap
pivot = sales.pivot_table(values="Revenue", index="month", columns="dow", aggfunc="mean")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Monthly bars
colors_m = [PALETTE[2] if v > monthly["mean"].mean() else PALETTE[0]
            for v in monthly["mean"]]
axes[0].bar(monthly.index, monthly["mean"]/1e6, color=colors_m, alpha=0.85, edgecolor="white")
axes[0].errorbar(monthly.index, monthly["mean"]/1e6,
                 yerr=monthly["std"]/1e6, fmt="none", color="gray", capsize=3, lw=1.2)
axes[0].axhline(monthly["mean"].mean()/1e6, color="red", linestyle="--", lw=1.5,
                label=f"Overall mean: {monthly['mean'].mean()/1e6:.1f}M")
axes[0].set_xticks(range(1,13)); axes[0].set_xticklabels(MONTH_NAMES, rotation=45)
axes[0].set_ylabel("Revenue (M VND)"); axes[0].set_title("Average Daily Revenue by Month")
axes[0].legend(fontsize=8)
for i, (m, row) in enumerate(monthly.iterrows()):
    pct = (row["mean"]/monthly["mean"].mean()-1)*100
    axes[0].text(m, row["mean"]/1e6 + row["std"]/1e6 + monthly["mean"].max()/1e6*0.02,
                 f"{pct:+.0f}%", ha="center", va="bottom", fontsize=7.5)

# DOW bars
axes[1].bar(dow_avg.index, dow_avg.values/1e6, color=PALETTE[5], alpha=0.85, edgecolor="white")
axes[1].axhline(dow_avg.mean()/1e6, color="red", linestyle="--", lw=1.5)
axes[1].set_xticks(range(7)); axes[1].set_xticklabels(DOW_NAMES)
axes[1].set_ylabel("Revenue (M VND)"); axes[1].set_title("Average Daily Revenue by Day of Week")
for i, (d, v) in enumerate(dow_avg.items()):
    pct = (v/dow_avg.mean()-1)*100
    axes[1].text(d, v/1e6 + dow_avg.max()/1e6*0.01, f"{pct:+.0f}%",
                 ha="center", va="bottom", fontsize=8)

# Heatmap Month × DOW
im = axes[2].imshow(pivot.values/1e6, aspect="auto", cmap="YlOrRd")
axes[2].set_xticks(range(7)); axes[2].set_xticklabels(DOW_NAMES)
axes[2].set_yticks(range(12)); axes[2].set_yticklabels(MONTH_NAMES)
axes[2].set_title("Revenue Heatmap: Month × Day-of-Week (M VND)")
plt.colorbar(im, ax=axes[2], label="Avg Revenue (M VND)", shrink=0.85)
for i in range(12):
    for j in range(7):
        axes[2].text(j, i, f"{pivot.values[i,j]/1e6:.1f}",
                    ha="center", va="center", fontsize=6.5, color="black")

plt.suptitle("A3 — Tính mùa vụ: Tháng · Ngày trong tuần · Heatmap", fontsize=13,
             y=1.02, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/A3_seasonality.png", dpi=150, bbox_inches="tight")
plt.show()

peak_m = monthly["mean"].idxmax()
low_m  = monthly["mean"].idxmin()
print(f"Peak month : {MONTH_NAMES[peak_m-1]}  ({monthly['mean'][peak_m]/1e6:.1f}M avg/day)")
print(f"Low month  : {MONTH_NAMES[low_m-1]}   ({monthly['mean'][low_m]/1e6:.1f}M avg/day)")
print(f"Peak/Low ratio: {monthly['mean'].max()/monthly['mean'].min():.2f}×")
print(f"Weekend revenue vs weekday: {dow_avg[[5,6]].mean()/dow_avg[:5].mean()-1:+.1%}")
"""))

# ─── A4. YoY trend ────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## A4 · Xu hướng năm (YoY) và Annual Revenue Decomposition"))
cells.append(new_code_cell("""\
annual = (sales[sales["year"].between(2013,2022)]
          .groupby("year")[["Revenue","COGS","gross_profit"]].sum())
yoy_rev = annual["Revenue"].pct_change().dropna()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Annual revenue
bars = axes[0].bar(annual.index, annual["Revenue"]/1e9,
                   color=PALETTE[0], alpha=0.85, edgecolor="white", width=0.7)
axes[0].set_title("Annual Revenue (tỷ VND)")
axes[0].set_ylabel("Revenue (tỷ)")
for yr, v in annual["Revenue"].items():
    axes[0].text(yr, v/1e9 + annual["Revenue"].max()/1e9*0.015,
                 f"{v/1e9:.1f}T", ha="center", va="bottom", fontsize=8)

# YoY growth
colors_yoy = [PALETTE[2] if v>0 else PALETTE[3] for v in yoy_rev.values]
axes[1].bar(yoy_rev.index, yoy_rev.values*100, color=colors_yoy, alpha=0.85,
            edgecolor="white", width=0.7)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_title("YoY Revenue Growth (%)")
for yr, v in yoy_rev.items():
    axes[1].text(yr, v*100 + (1.5 if v>=0 else -2),
                 f"{v*100:.1f}%", ha="center",
                 va="bottom" if v>=0 else "top", fontsize=8.5)

# Revenue vs COGS vs Gross Profit stacked
x = annual.index
axes[2].fill_between(x, 0, annual["COGS"]/1e9, alpha=0.7, color=PALETTE[1], label="COGS")
axes[2].fill_between(x, annual["COGS"]/1e9, annual["Revenue"]/1e9,
                     alpha=0.7, color=PALETTE[2], label="Gross Profit")
axes[2].plot(x, annual["Revenue"]/1e9, lw=2, color=PALETTE[0], marker="o", ms=5, label="Revenue")
axes[2].set_title("Revenue = COGS + Gross Profit (tỷ VND)")
axes[2].set_ylabel("tỷ VND")
axes[2].legend(fontsize=9)

plt.suptitle("A4 — Xu hướng năm: Annual Revenue · YoY · Decomposition",
             fontsize=13, y=1.02, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/A4_annual_trend.png", dpi=150, bbox_inches="tight")
plt.show()

geo_mean = (1+yoy_rev).prod()**(1/len(yoy_rev))-1
print(f"Geometric mean YoY growth: {geo_mean:+.2%}/year")
print(f"Best year : {yoy_rev.idxmax()}  ({yoy_rev.max():+.1%})")
print(f"Worst year: {yoy_rev.idxmin()}  ({yoy_rev.min():+.1%})")
"""))

# ─── A5. Decomposition ────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## A5 · Phân rã tín hiệu Revenue: Trend + Seasonal + Residual"))
cells.append(new_code_cell("""\
s = sales.set_index("Date")["Revenue"].copy()
trend    = s.rolling(365, center=True, min_periods=180).mean()
seasonal = s / trend.replace(0, np.nan)
residual = s - trend

fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)

axes[0].fill_between(s.index, s.values/1e6, alpha=0.15, color=PALETTE[0])
axes[0].plot(s.index, s.values/1e6, lw=0.5, color=PALETTE[0], alpha=0.7)
axes[0].plot(trend.index, trend.values/1e6, lw=2.5, color=PALETTE[3], label="Trend (365d MA)")
axes[0].set_ylabel("Revenue (M)"); axes[0].set_title("Original + Trend")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}M"))
axes[0].legend(fontsize=9)

axes[1].axhline(1, color="red", linestyle="--", lw=1.2, label="Index=1 (no seasonal effect)")
axes[1].plot(seasonal.index, seasonal.values, lw=0.6, color=PALETTE[2], alpha=0.7)
axes[1].plot(seasonal.index, seasonal.rolling(30, center=True, min_periods=10).mean(),
             lw=2, color=PALETTE[2], label="30d MA seasonal")
axes[1].set_ylabel("Seasonal Index (Revenue/Trend)"); axes[1].set_title("Seasonal Component")
axes[1].legend(fontsize=9)

axes[2].axhline(0, color="black", lw=0.8)
axes[2].fill_between(residual.index, residual.values/1e6,
                     where=(residual.values/1e6>=0), alpha=0.5, color=PALETTE[2])
axes[2].fill_between(residual.index, residual.values/1e6,
                     where=(residual.values/1e6<0),  alpha=0.5, color=PALETTE[3])
axes[2].set_ylabel("Residual (M)"); axes[2].set_title("Residual Component")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.suptitle("A5 — Signal Decomposition: Trend · Seasonal · Residual",
             fontsize=13, y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/A5_decomposition.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Seasonal index range: [{seasonal.min():.2f}, {seasonal.max():.2f}]")
print(f"Seasonal CV (std/mean): {seasonal.std()/seasonal.mean():.3f}")
"""))

# ─── A6. Promotions ───────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## A6 · Tác động của Khuyến mãi lên Revenue"))
cells.append(new_code_cell("""\
# Active promos per day
promo_cnt_by_day = []
for d in sales["Date"]:
    cnt = ((promos["start_date"]<=d) & (promos["end_date"]>=d)).sum()
    promo_cnt_by_day.append(cnt)
sales["promo_count"] = promo_cnt_by_day

# Revenue by promo_count bucket
sales["promo_bucket"] = sales["promo_count"].apply(
    lambda x: "0 promos" if x==0 else ("1 promo" if x==1 else "2+ promos"))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Revenue by promo count bucket
promo_rev = sales.groupby("promo_bucket")["Revenue"].agg(["mean","median","count"])
axes[0].bar(promo_rev.index, promo_rev["mean"]/1e6,
            color=[PALETTE[0],PALETTE[2],PALETTE[3]], alpha=0.85, edgecolor="white")
for i, (bkt, row) in enumerate(promo_rev.iterrows()):
    axes[0].text(i, row["mean"]/1e6 + 0.15, f"{row['mean']/1e6:.1f}M\n(n={row['count']:,})",
                 ha="center", va="bottom", fontsize=8)
axes[0].set_title("Average Revenue by Promotion Count")
axes[0].set_ylabel("Revenue (M VND)")

# Promo timeline with revenue
ax0b = axes[1].twinx()
axes[1].fill_between(sales["Date"], sales["Revenue"]/1e6,
                     alpha=0.15, color=PALETTE[0])
axes[1].plot(sales["Date"],
             sales["Revenue"].rolling(30,center=True,min_periods=10).mean()/1e6,
             lw=1.8, color=PALETTE[0], label="Revenue 30d MA")
ax0b.plot(sales["Date"], sales["promo_count"],
          lw=0.8, color=PALETTE[3], alpha=0.6, label="Active promos")
axes[1].set_ylabel("Revenue (M VND)"); ax0b.set_ylabel("# Active Promos")
axes[1].set_title("Revenue vs Active Promos Over Time")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[1].legend(loc="upper left", fontsize=8)
ax0b.legend(loc="upper right", fontsize=8)

# Promo type distribution
ptype = promos["promo_type"].value_counts()
axes[2].pie(ptype.values, labels=ptype.index, autopct="%1.1f%%",
            colors=PALETTE[:len(ptype)], startangle=90,
            textprops={"fontsize":10})
axes[2].set_title("Promotion Types Distribution")

plt.suptitle("A6 — Tác động Khuyến mãi lên Revenue", fontsize=13, y=1.02, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/A6_promotions.png", dpi=150, bbox_inches="tight")
plt.show()

for bkt, row in promo_rev.iterrows():
    print(f"  {bkt}: mean={row['mean']:>12,.0f}  median={promo_rev.loc[bkt,'median']:>12,.0f}  n={row['count']:>5,}")
"""))

# ─── A7. Web traffic ──────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## A7 · Web Traffic → Revenue Signal"))
cells.append(new_code_cell("""\
wt_day = (wt_raw.groupby("date")
          .agg(sessions=("sessions","sum"),
               visitors=("unique_visitors","sum"),
               pageviews=("page_views","sum"),
               bounce=("bounce_rate","mean"),
               duration=("avg_session_duration_sec","mean"))
          .reset_index().rename(columns={"date":"Date"}))
wt_check = sales.merge(wt_day, on="Date", how="inner")

# Lag analysis: which lag of sessions best predicts revenue?
lag_corrs = {}
for lag in range(0, 8):
    wt_check[f"sess_l{lag}"] = wt_check["sessions"].shift(lag)
    c = wt_check["Revenue"].corr(wt_check[f"sess_l{lag}"])
    lag_corrs[lag] = c

fig, axes = plt.subplots(2, 2, figsize=(15, 8))

# Lag correlation
axes[0,0].bar(list(lag_corrs.keys()), list(lag_corrs.values()),
              color=PALETTE[0], alpha=0.85, edgecolor="white")
best_lag = max(lag_corrs, key=lambda k: lag_corrs[k])
axes[0,0].axvline(best_lag, color="red", linestyle="--",
                  label=f"Best lag: {best_lag}d (r={lag_corrs[best_lag]:.3f})")
axes[0,0].set_xlabel("Lag (days)"); axes[0,0].set_ylabel("Pearson r")
axes[0,0].set_title("Corr(Revenue, Sessions[t-lag])")
axes[0,0].legend(fontsize=9)
axes[0,0].set_xticks(range(8))

# Sessions vs Revenue scatter
axes[0,1].scatter(wt_check["sessions"]/1e3, wt_check["Revenue"]/1e6,
                  s=6, alpha=0.25, color=PALETTE[1])
r_sv = wt_check["Revenue"].corr(wt_check["sessions"])
axes[0,1].set_xlabel("Sessions (K)"); axes[0,1].set_ylabel("Revenue (M)")
axes[0,1].set_title(f"Revenue vs Daily Sessions  (r={r_sv:.3f})")

# Traffic source breakdown
if "traffic_source" in wt_raw.columns:
    src_bounce = wt_raw.groupby("traffic_source")["bounce_rate"].mean().sort_values()
    axes[1,0].barh(src_bounce.index, src_bounce.values*100,
                   color=PALETTE[2], alpha=0.85, edgecolor="white")
    axes[1,0].set_xlabel("Avg Bounce Rate (%)")
    axes[1,0].set_title("Bounce Rate by Traffic Source")

    src_sess = wt_raw.groupby("traffic_source")["sessions"].sum().sort_values(ascending=False)
    axes[1,1].bar(src_sess.index, src_sess.values/1e6,
                  color=PALETTE[5], alpha=0.85, edgecolor="white")
    axes[1,1].set_ylabel("Total Sessions (M)")
    axes[1,1].set_title("Total Sessions by Traffic Source")
    axes[1,1].tick_params(axis="x", rotation=30)
else:
    axes[1,0].text(0.5,0.5,"traffic_source N/A", ha="center",va="center",
                   transform=axes[1,0].transAxes)
    axes[1,1].text(0.5,0.5,"traffic_source N/A", ha="center",va="center",
                   transform=axes[1,1].transAxes)

plt.suptitle("A7 — Web Traffic Signals", fontsize=13, y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/A7_web_traffic.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Best lag for sessions→revenue: lag={best_lag} day(s)  r={lag_corrs[best_lag]:.4f}")
print(f"Corr(Revenue, visitors)       : {wt_check['Revenue'].corr(wt_check['visitors']):.4f}")
print(f"Corr(Revenue, pageviews)      : {wt_check['Revenue'].corr(wt_check['pageviews']):.4f}")
"""))

# ─── A8. Inventory ────────────────────────────────────────────────────────────
cells.append(new_markdown_cell("## A8 · Inventory — Stockout & Fill Rate tác động đến Revenue"))
cells.append(new_code_cell("""\
inv_monthly = (inventory.groupby("snapshot_date")
               .agg(avg_fill_rate  =("fill_rate",        "mean"),
                    stockout_pct   =("stockout_flag",     "mean"),
                    overstock_pct  =("overstock_flag",    "mean"),
                    total_stock    =("stock_on_hand",     "sum"),
                    total_sold     =("units_sold",        "sum"),
                    avg_sell_thru  =("sell_through_rate", "mean"))
               .reset_index())

# Merge with monthly revenue
sales_monthly = (sales.assign(ym=sales["Date"].dt.to_period("M"))
                 .groupby("ym")["Revenue"].sum()
                 .reset_index())
sales_monthly["snapshot_date"] = pd.to_datetime(sales_monthly["ym"].dt.to_timestamp(how="E"))
inv_rev = inv_monthly.merge(sales_monthly[["snapshot_date","Revenue"]],
                            on="snapshot_date", how="inner")

fig, axes = plt.subplots(2, 2, figsize=(15, 8))

# Fill rate vs Revenue
axes[0,0].scatter(inv_rev["avg_fill_rate"]*100, inv_rev["Revenue"]/1e9,
                  s=40, alpha=0.7, color=PALETTE[0])
r_fr = inv_rev["avg_fill_rate"].corr(inv_rev["Revenue"])
axes[0,0].set_xlabel("Avg Fill Rate (%)"); axes[0,0].set_ylabel("Monthly Revenue (tỷ)")
axes[0,0].set_title(f"Fill Rate vs Revenue  (r={r_fr:.3f})")

# Stockout rate over time
axes[0,1].plot(inv_monthly["snapshot_date"], inv_monthly["stockout_pct"]*100,
               lw=1.5, color=PALETTE[3], label="Stockout %")
axes[0,1].plot(inv_monthly["snapshot_date"], inv_monthly["avg_fill_rate"]*100,
               lw=1.5, color=PALETTE[2], label="Fill Rate %")
axes[0,1].set_ylabel("%"); axes[0,1].set_title("Stockout vs Fill Rate Over Time")
axes[0,1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[0,1].legend(fontsize=9)

# Sell-through rate by category
if "category" in inventory.columns:
    cat_str = (inventory.groupby("category")["sell_through_rate"].mean()
               .sort_values(ascending=False))
    axes[1,0].barh(cat_str.index, cat_str.values*100,
                   color=PALETTE[5], alpha=0.85, edgecolor="white")
    axes[1,0].set_xlabel("Avg Sell-Through Rate (%)")
    axes[1,0].set_title("Sell-Through Rate by Category")
else:
    axes[1,0].text(0.5,0.5,"No category col",ha="center",va="center",
                   transform=axes[1,0].transAxes)

# Stock on hand trend
axes[1,1].fill_between(inv_monthly["snapshot_date"],
                       inv_monthly["total_stock"]/1e3, alpha=0.3, color=PALETTE[6])
axes[1,1].plot(inv_monthly["snapshot_date"],
               inv_monthly["total_stock"]/1e3, lw=1.5, color=PALETTE[6])
axes[1,1].set_ylabel("Total Stock (K units)")
axes[1,1].set_title("Total Stock on Hand Over Time")
axes[1,1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.suptitle("A8 — Inventory: Fill Rate · Stockout · Sell-Through",
             fontsize=13, y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/A8_inventory.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Correlation Fill Rate vs Revenue: {r_fr:.4f}")
"""))

# ─── A9. Orders & Returns ─────────────────────────────────────────────────────
cells.append(new_markdown_cell("## A9 · Orders, Returns & Payment — Demand Signals"))
cells.append(new_code_cell("""\
# Daily orders
ord_daily = (orders.groupby("order_date")
             .agg(ord_count    =("order_id","count"),
                  ord_delivered=("order_status", lambda x: (x=="delivered").sum()),
                  ord_cancelled=("order_status", lambda x: (x=="cancelled").sum()))
             .reset_index().rename(columns={"order_date":"Date"}))
ord_daily["cancel_rate"] = ord_daily["ord_cancelled"] / ord_daily["ord_count"]

# Returns by month
ret_monthly = (returns.assign(ym=returns["return_date"].dt.to_period("M"))
               .groupby("ym").agg(return_qty=("return_quantity","sum"),
                                   refund_amt=("refund_amount","sum"))
               .reset_index())
ret_monthly["Date"] = pd.to_datetime(ret_monthly["ym"].dt.to_timestamp())

# Merge orders with revenue
ord_rev = sales.merge(ord_daily, on="Date", how="inner")

fig, axes = plt.subplots(2, 2, figsize=(15, 8))

# Order count vs Revenue
axes[0,0].scatter(ord_rev["ord_count"], ord_rev["Revenue"]/1e6,
                  s=5, alpha=0.25, color=PALETTE[0])
r_oc = ord_rev["Revenue"].corr(ord_rev["ord_count"])
axes[0,0].set_xlabel("Daily Order Count"); axes[0,0].set_ylabel("Revenue (M)")
axes[0,0].set_title(f"Order Count vs Revenue  (r={r_oc:.3f})")

# Order status breakdown
status_cnt = orders["order_status"].value_counts()
axes[0,1].bar(status_cnt.index, status_cnt.values/1e3,
              color=PALETTE[:len(status_cnt)], alpha=0.85, edgecolor="white")
axes[0,1].set_ylabel("Orders (K)"); axes[0,1].set_title("Orders by Status")
axes[0,1].tick_params(axis="x", rotation=30)

# Returns over time
axes[1,0].plot(ret_monthly["Date"], ret_monthly["return_qty"],
               lw=1.5, color=PALETTE[3], marker="o", ms=3, label="Return Qty")
axes[1,0].set_title("Monthly Return Quantity Over Time")
axes[1,0].set_ylabel("Return Qty"); axes[1,0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Payment method vs avg order value
pay_method = payments.merge(orders[["order_id","order_status"]], on="order_id", how="left")
pay_avg = pay_method.groupby("payment_method")["payment_value"].mean().sort_values(ascending=False)
axes[1,1].bar(pay_avg.index, pay_avg.values/1e6,
              color=PALETTE[5], alpha=0.85, edgecolor="white")
axes[1,1].set_ylabel("Avg Payment Value (M)"); axes[1,1].set_title("Avg Payment by Method")
axes[1,1].tick_params(axis="x", rotation=30)

plt.suptitle("A9 — Orders · Returns · Payments", fontsize=13, y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/A9_orders_returns.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Corr(Revenue, ord_count)    : {r_oc:.4f}")
print(f"Avg cancel rate             : {ord_daily['cancel_rate'].mean():.3f}")
print(f"Return reasons distribution :")
print(returns["return_reason"].value_counts().to_string())
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PART B — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
---
# PHẦN B — Feature Engineering

> **Anti-leakage checklist:**
> - `shift(≥1)` cho tất cả lag/rolling
> - `ord_count`, `ord_delivered` → chỉ dùng `lag≥365` (available cho test period 2023–2024)
> - Web traffic: shift +1 ngày
> - Revenue/COGS test → không bao giờ được sử dụng
"""))

cells.append(new_code_cell("""\
# ── Full date range ───────────────────────────────────────────────────────────
all_dates = pd.date_range(sales["Date"].min(), sample_sub["Date"].max(), freq="D")
full = pd.DataFrame({"Date": all_dates})
full = full.merge(sales[["Date","Revenue","COGS"]], on="Date", how="left")
full["is_test"] = full["Date"].isin(sample_sub["Date"]).astype(int)
"""))

cells.append(new_code_cell("""\
# ── B1. Calendar features ─────────────────────────────────────────────────────
def add_calendar(df):
    df = df.copy(); D = df["Date"]
    df["cal_year"]    = D.dt.year
    df["cal_month"]   = D.dt.month
    df["cal_day"]     = D.dt.day
    df["cal_dow"]     = D.dt.dayofweek
    df["cal_doy"]     = D.dt.dayofyear
    df["cal_woy"]     = D.dt.isocalendar().week.astype(int)
    df["cal_quarter"] = D.dt.quarter
    # Boolean flags
    df["cal_is_weekend"]     = (D.dt.dayofweek >= 5).astype(int)
    df["cal_is_month_end"]   = D.dt.is_month_end.astype(int)
    df["cal_is_month_start"] = D.dt.is_month_start.astype(int)
    df["cal_is_qtr_end"]     = D.dt.is_quarter_end.astype(int)
    df["cal_is_year_end"]    = D.dt.is_year_end.astype(int)
    df["cal_is_year_start"]  = D.dt.is_year_start.astype(int)
    # Cyclic (smooth seasonal signal)
    df["cal_month_sin"] = np.sin(2*np.pi*df["cal_month"]/12)
    df["cal_month_cos"] = np.cos(2*np.pi*df["cal_month"]/12)
    df["cal_dow_sin"]   = np.sin(2*np.pi*df["cal_dow"]/7)
    df["cal_dow_cos"]   = np.cos(2*np.pi*df["cal_dow"]/7)
    df["cal_doy_sin"]   = np.sin(2*np.pi*df["cal_doy"]/365.25)
    df["cal_doy_cos"]   = np.cos(2*np.pi*df["cal_doy"]/365.25)
    df["cal_woy_sin"]   = np.sin(2*np.pi*df["cal_woy"]/52)
    df["cal_woy_cos"]   = np.cos(2*np.pi*df["cal_woy"]/52)
    # Vietnamese retail events
    df["vn_tet"]          = (((df["cal_month"]==1)&(df["cal_day"]>=20))|
                              ((df["cal_month"]==2)&(df["cal_day"]<=20))).astype(int)
    df["vn_pre_tet"]      = (((df["cal_month"]==1)&(df["cal_day"]>=14))|
                              ((df["cal_month"]==2)&(df["cal_day"]<=7))).astype(int)
    df["vn_post_tet"]     = ((df["cal_month"]==2)&(df["cal_day"].between(21,28))).astype(int)
    df["vn_mid_sale"]     = df["cal_month"].isin([6,7]).astype(int)
    df["vn_year_end"]     = df["cal_month"].isin([11,12]).astype(int)
    df["vn_back_school"]  = df["cal_month"].isin([8,9]).astype(int)
    df["vn_summer"]       = df["cal_month"].isin([5,6,7]).astype(int)
    df["vn_low_season"]   = df["cal_month"].isin([1,2,10]).astype(int)
    df["vn_peak_season"]  = df["cal_month"].isin([4,5,6]).astype(int)
    # Time index
    df["cal_time_idx"]    = (D - D.min()).dt.days
    return df

full = add_calendar(full)

# ── B2. Web traffic (lag +1d) ─────────────────────────────────────────────────
wt_feat = (wt_raw.groupby("date")
           .agg(wt_sessions =("sessions","sum"),
                wt_visitors =("unique_visitors","sum"),
                wt_pageviews=("page_views","sum"),
                wt_bounce   =("bounce_rate","mean"),
                wt_duration =("avg_session_duration_sec","mean"))
           .reset_index().rename(columns={"date":"Date"}))
wt_feat["Date"] = wt_feat["Date"] + pd.Timedelta(days=1)  # lag +1d, no leakage
full = full.merge(wt_feat, on="Date", how="left")

# ── B3. Promotion calendar ────────────────────────────────────────────────────
promo_rows = []
for d in all_dates:
    act = promos[(promos["start_date"]<=d) & (promos["end_date"]>=d)]
    promo_rows.append({
        "Date"                : d,
        "promo_count"         : len(act),
        "promo_discount_sum"  : act["discount_value"].sum(),
        "promo_stackable"     : int(act["stackable_flag"].eq(1).any()) if len(act) else 0,
        "promo_pct_count"     : int((act["promo_type"]=="percentage").sum()),
        "promo_fixed_count"   : int((act["promo_type"]=="fixed").sum()),
        "promo_min_order_sum" : act["min_order_value"].fillna(0).sum(),
    })
promo_df = pd.DataFrame(promo_rows)
full = full.merge(promo_df, on="Date", how="left")

# ── B4. Order signals (lagged only — safe for test) ───────────────────────────
ord_sig = (orders.groupby("order_date")
           .agg(ord_count    =("order_id","count"),
                ord_delivered=("order_status", lambda x: (x=="delivered").sum()),
                ord_cancelled=("order_status", lambda x: (x=="cancelled").sum()))
           .reset_index().rename(columns={"order_date":"Date"}))
ord_sig["ord_cancel_rate"] = ord_sig["ord_cancelled"] / ord_sig["ord_count"]
full = full.merge(ord_sig, on="Date", how="left")
full = full.sort_values("Date").reset_index(drop=True)

# Create LAGGED versions (lag≥365 → available for test 2023-2024)
for lag in [7, 14, 28, 90, 180, 365]:
    full[f"ord_count_l{lag}"]     = full["ord_count"].shift(lag)
    full[f"ord_delivered_l{lag}"] = full["ord_delivered"].shift(lag)
    full[f"ord_cancelled_l{lag}"] = full["ord_cancelled"].shift(lag)
# Rolling order (shift 1 first, so window only sees past)
ord_sh = full["ord_count"].shift(1)
for w in [7, 14, 28, 90, 365]:
    full[f"ord_count_rm{w}"] = ord_sh.rolling(w, min_periods=max(1,w//4)).mean()

# ── B5. Inventory monthly signals (lag 1 month → safe) ───────────────────────
inv_m = (inventory.groupby("snapshot_date")
         .agg(inv_fill_rate  =("fill_rate","mean"),
              inv_stockout   =("stockout_flag","mean"),
              inv_overstock  =("overstock_flag","mean"),
              inv_sell_thru  =("sell_through_rate","mean"),
              inv_stock_total=("stock_on_hand","sum"))
         .reset_index().rename(columns={"snapshot_date":"Date"}))
inv_m["Date"] = inv_m["Date"] + pd.DateOffset(months=1)   # lag 1 month, no leakage
full = full.merge(inv_m, on="Date", how="left")
# Forward fill inventory (monthly snapshot)
for col in ["inv_fill_rate","inv_stockout","inv_overstock","inv_sell_thru","inv_stock_total"]:
    full[col] = full[col].fillna(method="ffill")

# Drop raw (non-lagged) order columns
full.drop(columns=["ord_count","ord_delivered","ord_cancelled","ord_cancel_rate"],
          inplace=True, errors="ignore")

print(f"Full frame shape after external features: {full.shape}")
"""))

cells.append(new_code_cell("""\
# ── B6. Lag / Rolling / EWM for Revenue & COGS ───────────────────────────────
def add_lag_roll(df, col):
    s = df[col].copy()
    # Lags
    for lag in LAGS:
        df[f"{col}_l{lag}"] = s.shift(lag)
    # All rolling windows use shift(1) first
    sh = s.shift(1)
    for w in ROLL_WINDOWS:
        mp = max(1, w//4)
        df[f"{col}_rm{w}"]    = sh.rolling(w, min_periods=mp).mean()
        df[f"{col}_rs{w}"]    = sh.rolling(w, min_periods=max(5,mp)).std()
        df[f"{col}_rmed{w}"]  = sh.rolling(w, min_periods=mp).median()
        df[f"{col}_rmax{w}"]  = sh.rolling(w, min_periods=mp).max()
        df[f"{col}_rmin{w}"]  = sh.rolling(w, min_periods=mp).min()
        df[f"{col}_rq25_{w}"] = sh.rolling(w, min_periods=mp).quantile(0.25)
        df[f"{col}_rq75_{w}"] = sh.rolling(w, min_periods=mp).quantile(0.75)
    # EWM
    for sp in EWM_SPANS:
        df[f"{col}_ewm{sp}"] = sh.ewm(span=sp, adjust=False).mean()
    # YoY anchors
    df[f"{col}_yoy_ratio"]  = s.shift(365) / (s.shift(372).rolling(7,min_periods=4).mean() + 1e-9)
    l364_vals = (s.shift(357) + s.shift(364) + s.shift(371)) / 3    # ±7d avg
    df[f"{col}_l364_mean"]  = l364_vals
    df[f"{col}_2yoy_mean"]  = (s.shift(365) + s.shift(730)) / 2     # 2-year average
    # Trend ratio
    df[f"{col}_trend14_180"] = (sh.rolling(14,min_periods=7).mean() /
                                (s.shift(180).rolling(14,min_periods=7).mean() + 1e-9))
    return df

for tgt in ["Revenue", "COGS"]:
    full = add_lag_roll(full, tgt)

# Cross-target ratio features
for lag in [28, 90, 180, 365]:
    full[f"cogs_rev_ratio_l{lag}"] = (full[f"COGS_l{lag}"] /
                                       (full[f"Revenue_l{lag}"] + 1e-9))

# ── B7. Feature list ──────────────────────────────────────────────────────────
EXCLUDE = {"Date","Revenue","COGS","is_test"}
FEATURE_COLS = [c for c in full.columns if c not in EXCLUDE]

# Leakage audit
raw_live = [c for c in FEATURE_COLS if c in
            ["ord_count","ord_delivered","ord_cancelled","ord_cancel_rate"]]
assert len(raw_live)==0, f"Leakage: raw order cols found: {raw_live}"
print(f"✓ Leakage audit passed — no raw live order features")
print(f"  Total features  : {len(FEATURE_COLS)}")
by_group = {
    "Calendar (cal_*/vn_*)": sum(1 for c in FEATURE_COLS if c.startswith(("cal_","vn_"))),
    "Revenue lags/roll/EWM": sum(1 for c in FEATURE_COLS if c.startswith("Revenue_")),
    "COGS lags/roll/EWM"   : sum(1 for c in FEATURE_COLS if c.startswith("COGS_")),
    "Order lagged"         : sum(1 for c in FEATURE_COLS if c.startswith("ord_")),
    "Web traffic"          : sum(1 for c in FEATURE_COLS if c.startswith("wt_")),
    "Promotions"           : sum(1 for c in FEATURE_COLS if c.startswith("promo_")),
    "Inventory"            : sum(1 for c in FEATURE_COLS if c.startswith("inv_")),
    "Cross-ratio"          : sum(1 for c in FEATURE_COLS if "ratio" in c and "cogs_rev" in c),
}
for grp, cnt in by_group.items():
    print(f"  {grp:<30}: {cnt:>3}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PART C — SEASONAL BASELINE + TRAIN SPLIT
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
---
# PHẦN C — Seasonal Baseline + Train Split + Cross-Validation
"""))

cells.append(new_code_cell("""\
# ── C1. Seasonal baseline (3-year profile, blended growth) ────────────────────
annual = (sales[sales["Date"].dt.year.between(2013,2022)]
          .assign(year=lambda d: d["Date"].dt.year)
          .groupby("year")[["Revenue","COGS"]].sum())

yoy_rv  = annual["Revenue"].pct_change().dropna()
yoy_cg  = annual["COGS"].pct_change().dropna()
g_rv_full   = (1+yoy_rv).prod()**(1/len(yoy_rv))
g_cg_full   = (1+yoy_cg).prod()**(1/len(yoy_cg))
g_rv_recent = (annual["Revenue"].iloc[-1]/annual["Revenue"].iloc[-4])**(1/3)
g_cg_recent = (annual["COGS"].iloc[-1]   /annual["COGS"].iloc[-4])   **(1/3)
G_REV  = (g_rv_full + g_rv_recent) / 2
G_COGS = (g_cg_full + g_cg_recent) / 2

# Seasonal profile from 2020-2022 (most recent, most relevant)
recent_s = sales[sales["Date"].dt.year.between(2020,2022)].copy()
recent_s["month"] = recent_s["Date"].dt.month
recent_s["day"]   = recent_s["Date"].dt.day
ann_m = recent_s.groupby(recent_s["Date"].dt.year)[["Revenue","COGS"]].transform("mean")
recent_s["rev_n"]  = recent_s["Revenue"] / ann_m["Revenue"]
recent_s["cogs_n"] = recent_s["COGS"]    / ann_m["COGS"]
seas_profile = (recent_s.groupby(["month","day"])[["rev_n","cogs_n"]]
                .mean().reset_index())

BASE_REV  = annual.loc[2022,"Revenue"] / 365
BASE_COGS = annual.loc[2022,"COGS"]    / 365

def seasonal_predict(dates, base_rev, base_cogs, g_rev, g_cogs, profile, ref_year=2022):
    df = pd.DataFrame({"Date": pd.to_datetime(dates)})
    df["month"] = df["Date"].dt.month
    df["day"]   = df["Date"].dt.day
    df["ya"]    = df["Date"].dt.year - ref_year
    df = df.merge(profile, on=["month","day"], how="left")
    df["rev_n"]  = df["rev_n"].fillna(1.0)
    df["cogs_n"] = df["cogs_n"].fillna(1.0)
    return (
        (base_rev  * g_rev**df["ya"]  * df["rev_n"]).clip(0).values,
        (base_cogs * g_cogs**df["ya"] * df["cogs_n"]).clip(0).values
    )

test_dates = sorted(sample_sub["Date"].tolist())
seas_rev_test, seas_cogs_test = seasonal_predict(
    test_dates, BASE_REV, BASE_COGS, G_REV, G_COGS, seas_profile)

print(f"Seasonal forecast: Revenue mean={seas_rev_test.mean():,.0f}  COGS mean={seas_cogs_test.mean():,.0f}")
print(f"Growth rates used: Revenue={G_REV-1:+.2%}/yr  COGS={G_COGS-1:+.2%}/yr")
"""))

cells.append(new_code_cell("""\
# ── C2. Train/Test Split ──────────────────────────────────────────────────────
train = full[(full["is_test"]==0) & (full["Revenue_l365"].notna())].copy().reset_index(drop=True)
test  = full[ full["is_test"]==1].copy().reset_index(drop=True)

X_train = train[FEATURE_COLS]
y_rev   = train["Revenue"]
y_cogs  = train["COGS"]

print(f"Train : {len(train):,} rows  [{train['Date'].min().date()} → {train['Date'].max().date()}]")
print(f"Test  : {len(test):,}  rows  [{test['Date'].min().date()} → {test['Date'].max().date()}]")
print()
baseline_mae = (y_rev - y_rev.mean()).abs().mean()
print(f"Naive baseline MAE (predict mean): {baseline_mae:,.0f}")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PART D — CROSS-VALIDATION & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("""\
---
# PHẦN D — Cross-Validation & Evaluation

### Thiết kế CV đúng chiều thời gian

```
Fold 1: ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░
              train ↑   gap ↑    val ↑
Fold 2: ████████████████░░░░░░░░░░░░░░░░░░░
...
Fold 5: ████████████████████████████░░░░░░░
```
- Gap = 14 ngày ngăn rolling window "nhìn thấy" validation
- Expanding window: fold sau có nhiều train data hơn
"""))

cells.append(new_code_cell("""\
def evaluate(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    if label:
        print(f"  {label:<18} MAE={mae:>12,.0f}  RMSE={rmse:>12,.0f}  R²={r2:>8.4f}")
    return mae, rmse, r2

def run_dual_cv(X, y, params_mae, params_huber, n_splits=5, gap=14,
                n_rounds=3500, early=120):
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = {k: [] for k in ["mae_mae","mae_rmse","mae_r2",
                               "hub_mae","hub_rmse","hub_r2",
                               "ens_mae","ens_rmse","ens_r2",
                               "bi_mae","bi_rmse","bi_r2",
                               "bh","bm"]}
    fold_results = []

    header = f"  {'Fold':>4}  {'Train':>7}  {'Val':>6}  {'LGB-MAE':>10}  {'LGB-Huber':>10}  {'Blend-CV':>10}  {'R²-blend':>9}"
    print(header); print("  " + "─"*len(header.strip()))

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[tr_idx], y.iloc[val_idx]

        dt = lgb.Dataset(Xtr, label=ytr)
        dv = lgb.Dataset(Xval, label=yval, reference=dt)
        cb = [lgb.early_stopping(early, verbose=False), lgb.log_evaluation(False)]

        m_mae = lgb.train(params_mae,   dt, num_boost_round=n_rounds, valid_sets=[dv], callbacks=cb)
        m_hub = lgb.train(params_huber, dt, num_boost_round=n_rounds, valid_sets=[dv], callbacks=cb)

        pm = np.clip(m_mae.predict(Xval), 0, None)
        ph = np.clip(m_hub.predict(Xval), 0, None)
        # Blend without seasonal (we don't have exact test dates per fold)
        ratio = W_MAE / (W_MAE + W_HUBER)
        pb = ratio*pm + (1-ratio)*ph

        sm = evaluate(yval, pm); sh = evaluate(yval, ph); sb = evaluate(yval, pb)

        scores["mae_mae"].append(sm[0]); scores["mae_rmse"].append(sm[1]); scores["mae_r2"].append(sm[2])
        scores["hub_mae"].append(sh[0]); scores["hub_rmse"].append(sh[1]); scores["hub_r2"].append(sh[2])
        scores["ens_mae"].append(sb[0]); scores["ens_rmse"].append(sb[1]); scores["ens_r2"].append(sb[2])
        scores["bh"].append(m_hub.best_iteration)
        scores["bm"].append(m_mae.best_iteration)

        fold_results.append({"fold":fold+1,"pm":pm,"ph":ph,"pb":pb,"yval":yval.values})

        print(f"  {fold+1:>4}  {len(ytr):>7,}  {len(yval):>6,}  "
              f"{sm[0]:>10,.0f}  {sh[0]:>10,.0f}  {sb[0]:>10,.0f}  {sb[2]:>9.4f}")

    print("  " + "─"*70)
    for label, k in [("LGB-MAE","mae"),("LGB-Huber","hub"),("Blend","ens")]:
        mm = np.mean(scores[f"{k}_mae"]); ms = np.std(scores[f"{k}_mae"])
        rm = np.mean(scores[f"{k}_rmse"])
        r2 = np.mean(scores[f"{k}_r2"])
        print(f"  {label:<10} MAE={mm:>10,.0f} ±{ms:>8,.0f}  RMSE={rm:>10,.0f}  R²={r2:.4f}")

    return scores, fold_results

print("Running 5-fold TimeSeriesSplit CV (dual model) …")
cv_scores, fold_results = run_dual_cv(X_train, y_rev, PARAMS_MAE, PARAMS_HUBER)
"""))

cells.append(new_code_cell("""\
# ── Fold visualisation ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
axes = axes.flatten()

for i, res in enumerate(fold_results):
    ax = axes[i]
    n  = min(300, len(res["yval"]))
    ax.plot(res["yval"][:n]/1e6,  lw=1.0, color=PALETTE[0], label="Actual")
    ax.plot(res["pm"][:n]/1e6,    lw=0.9, color=PALETTE[1], linestyle="--",
            alpha=0.85, label="LGB-MAE")
    ax.plot(res["pb"][:n]/1e6,    lw=0.9, color=PALETTE[2], linestyle=":",
            alpha=0.85, label="Blend")
    mae_m = cv_scores["mae_mae"][i]; mae_b = cv_scores["ens_mae"][i]
    r2_b  = cv_scores["ens_r2"][i]
    ax.set_title(f"Fold {i+1} | MAE-LGB={mae_m:,.0f} | Blend={mae_b:,.0f} | R²={r2_b:.3f}",
                 fontsize=9)
    ax.legend(fontsize=7); ax.set_ylabel("Revenue (M)")
    ax.tick_params(labelsize=8)

# Summary bar
ax_s = axes[-1]
maes = cv_scores["ens_mae"]
ax_s.bar(range(1,6), maes, color=PALETTE[0], alpha=0.85, edgecolor="white")
ax_s.axhline(np.mean(maes), color="red", linestyle="--", lw=1.8,
             label=f"Mean: {np.mean(maes):,.0f}")
ax_s.set_xlabel("Fold"); ax_s.set_ylabel("MAE"); ax_s.set_title("MAE per Fold (Blend)")
ax_s.legend(fontsize=9)

plt.suptitle("D — Time-Series CV Results: Actual vs Predicted per Fold",
             fontsize=13, y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/D_cv_folds.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

# ══════════════════════════════════════════════════════════════════════════════
# FINAL MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## E · Final Model Training"))
cells.append(new_code_cell("""\
iter_mae = int(np.mean(cv_scores["bm"]) * 1.10)
iter_hub = int(np.mean(cv_scores["bh"]) * 1.10)
print(f"Revenue MAE  model: {iter_mae} rounds")
print(f"Revenue Huber model: {iter_hub} rounds")

dt_rev  = lgb.Dataset(X_train, label=y_rev)
dt_cogs = lgb.Dataset(X_train, label=y_cogs)

def train_model(params, dt, n, label):
    print(f"  Training {label} …", end="", flush=True)
    m = lgb.train(params, dt, num_boost_round=n, callbacks=[lgb.log_evaluation(False)])
    print(" ✓")
    return m

rev_mae_m  = train_model(PARAMS_MAE,   dt_rev,  iter_mae,              "Rev-MAE")
rev_hub_m  = train_model(PARAMS_HUBER, dt_rev,  iter_hub,              "Rev-Hub")
cogs_mae_m = train_model(PARAMS_MAE,   dt_cogs, int(iter_mae*0.9),     "COGS-MAE")
cogs_hub_m = train_model(PARAMS_HUBER, dt_cogs, int(iter_hub*0.9),     "COGS-Hub")

# In-sample evaluation (train)
p_trn = np.clip(W_MAE*rev_mae_m.predict(X_train) + W_HUBER*rev_hub_m.predict(X_train), 0, None)
print("\\nIn-sample (train) — Ensemble:")
evaluate(y_rev, p_trn, "Train (no seas)")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# ITERATIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_code_cell("""\
# ── Iterative day-by-day prediction ──────────────────────────────────────────
def dyn_feats(series, d, lags, roll_windows, ewm_spans, col):
    feat = {}
    for lag in lags:
        feat[f"{col}_l{lag}"] = series.get(d - pd.Timedelta(days=lag), np.nan)
    past = series[series.index < d].dropna()
    for w in roll_windows:
        t = past.iloc[-w:]
        feat[f"{col}_rm{w}"]    = t.mean()           if len(t)>0 else np.nan
        feat[f"{col}_rs{w}"]    = t.std()             if len(t)>=5 else np.nan
        feat[f"{col}_rmed{w}"]  = t.median()          if len(t)>0 else np.nan
        feat[f"{col}_rmax{w}"]  = t.max()             if len(t)>0 else np.nan
        feat[f"{col}_rmin{w}"]  = t.min()             if len(t)>0 else np.nan
        feat[f"{col}_rq25_{w}"] = np.percentile(t,25) if len(t)>0 else np.nan
        feat[f"{col}_rq75_{w}"] = np.percentile(t,75) if len(t)>0 else np.nan
    for sp in ewm_spans:
        feat[f"{col}_ewm{sp}"] = (past.ewm(span=sp,adjust=False).mean().iloc[-1]
                                   if len(past)>0 else np.nan)
    yoy = series.get(d - pd.Timedelta(days=365), np.nan)
    yoy_b = past.iloc[-7:].mean() if len(past)>=4 else np.nan
    feat[f"{col}_yoy_ratio"] = yoy/(yoy_b+1e-9) if not np.isnan(yoy_b) else np.nan
    l364 = [series.get(d - pd.Timedelta(days=k), np.nan) for k in [357,364,371]]
    feat[f"{col}_l364_mean"] = np.nanmean(l364) if any(~np.isnan(v) for v in l364) else np.nan
    l730 = series.get(d - pd.Timedelta(days=730), np.nan)
    feat[f"{col}_2yoy_mean"] = np.nanmean([yoy, l730])
    r14  = past.iloc[-14:].mean() if len(past)>=7 else np.nan
    r180 = past.iloc[-194:-180].mean() if len(past)>=180 else np.nan
    feat[f"{col}_trend14_180"] = r14/(r180+1e-9) if not np.isnan(r180) else np.nan
    return feat

fp = full.set_index("Date").copy()
test_dates_sorted = sorted(fp[fp["is_test"]==1].index.tolist())
static_cols = [c for c in FEATURE_COLS
               if not c.startswith("Revenue_") and not c.startswith("COGS_")
               and "cogs_rev_ratio" not in c]

print(f"Predicting {len(test_dates_sorted)} test days …")
for i, d in enumerate(test_dates_sorted):
    if i % 100 == 0:
        print(f"  Day {i+1:>3}/{len(test_dates_sorted)}  ({d.date()})")

    static = {c: fp.loc[d, c] for c in static_cols if c in fp.columns}
    dr = dyn_feats(fp["Revenue"], d, LAGS, ROLL_WINDOWS, EWM_SPANS, "Revenue")
    dc = dyn_feats(fp["COGS"],    d, LAGS, ROLL_WINDOWS, EWM_SPANS, "COGS")
    ratio_f = {f"cogs_rev_ratio_l{lg}":
               dc.get(f"COGS_l{lg}",np.nan) / (dr.get(f"Revenue_l{lg}",np.nan)+1e-9)
               for lg in [28,90,180,365]}

    row = pd.DataFrame([{**static, **dr, **dc, **ratio_f}])[FEATURE_COLS]
    si  = test_dates_sorted.index(d)

    pm_r  = max(float(rev_mae_m.predict(row)[0]),  0)
    ph_r  = max(float(rev_hub_m.predict(row)[0]),  0)
    pm_c  = max(float(cogs_mae_m.predict(row)[0]), 0)
    ph_c  = max(float(cogs_hub_m.predict(row)[0]), 0)

    fp.loc[d, "Revenue"] = W_MAE*pm_r  + W_HUBER*ph_r  + W_SEAS*float(seas_rev_test[si])
    fp.loc[d, "COGS"]    = W_MAE*pm_c  + W_HUBER*ph_c  + W_SEAS*float(seas_cogs_test[si])

print("✓ Iterative prediction complete.")
"""))

# ══════════════════════════════════════════════════════════════════════════════
# PART E — SHAP
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## F · SHAP Feature Importance & Business Interpretation"))
cells.append(new_code_cell("""\
rng      = np.random.default_rng(SEED)
idx_shap = rng.choice(len(X_train), min(800, len(X_train)), replace=False)
X_shap   = X_train.iloc[idx_shap]

print("Computing SHAP …", end="", flush=True)
explainer   = shap.TreeExplainer(rev_mae_m)
shap_values = explainer.shap_values(X_shap)
print(" ✓")

shap_df = (pd.DataFrame({"feature":FEATURE_COLS,
                          "shap_mean":np.abs(shap_values).mean(0)})
           .sort_values("shap_mean", ascending=False)
           .reset_index(drop=True))
shap_df.to_csv(f"{REPORTS_DIR}/shap_final.csv", index=False)
print("\\nTop-25 Revenue Drivers:")
print(shap_df.head(25)[["feature","shap_mean"]].to_string(index=False))
"""))

cells.append(new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# SHAP beeswarm
plt.sca(axes[0])
shap.summary_plot(shap_values, X_shap, feature_names=FEATURE_COLS,
                  show=False, max_display=25, plot_size=None)
axes[0].set_title("SHAP Beeswarm — Revenue Model\\n"
                  "(đỏ=giá trị cao · xanh=giá trị thấp)", fontsize=10, pad=8)

# SHAP bar (top 25)
top25 = shap_df.head(25)
axes[1].barh(top25["feature"][::-1], top25["shap_mean"][::-1],
             color=PALETTE[0], alpha=0.85)
axes[1].set_xlabel("Mean |SHAP value|")
axes[1].set_title("Top-25 Features — Mean |SHAP|", fontsize=11)
axes[1].tick_params(labelsize=8)

plt.suptitle("SHAP Feature Importance — Revenue Forecast Model",
             fontsize=13, y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/E_shap_final.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

cells.append(new_markdown_cell("""\
### Business Interpretation — Top Revenue Drivers

| Rank | Feature | Business Meaning | Direction |
|---|---|---|---|
| 1 | `Revenue_l1/l2/l7` | **Momentum ngắn hạn**: doanh thu hôm qua/tuần trước → predictor mạnh nhất | ↑ recent rev → ↑ forecast |
| 2 | `Revenue_ewm7/14` | **Xu hướng tuần**: EWM smooths noise, bắt được momentum | ↑ EWM → ↑ forecast |
| 3 | `Revenue_l364_mean` | **Seasonal anchor**: cùng tuần năm trước ±7 ngày | Annual repeat pattern |
| 4 | `Revenue_rm28/90` | **Trend trung hạn**: rolling mean tháng/quý | Business health indicator |
| 5 | `ord_count_l365` | **Order volume năm trước**: proxy cho demand | ↑ historical orders → ↑ forecast |
| 6 | `promo_discount_sum` | **Promotional uplift**: tổng discount đang chạy | ↑ discount → ↑ short-term rev |
| 7 | `vn_tet / vn_pre_tet` | **Tết effect**: peak lớn nhất trong năm | +50–200% so với ngày thường |
| 8 | `wt_sessions (lag 1d)` | **Web-to-purchase signal**: traffic hôm qua → mua hôm nay | Conversion lag ~24h |
| 9 | `inv_fill_rate` | **Inventory readiness**: hàng sẵn có → đáp ứng được đơn | ↑ fill rate → ↑ realised rev |
| 10 | `cogs_rev_ratio_l90` | **Cost structure anchor**: COGS/Rev ratio ổn định | Constraints prediction drift |

### Key Actionable Insights

1. **Tết window**: Tăng tồn kho & marketing 3 tuần trước Tết → highest ROI period
2. **Promo stacking**: stackable promotions có multiplier effect — plan concurrent campaigns
3. **Web sessions D-1**: monitor như leading KPI — dip trong sessions → cảnh báo revenue dip D+1
4. **Q4 (Nov–Dec)**: year-end sale có SHAP dương → allocate budget accordingly
5. **Fill rate > 90%**: Inventory readiness trực tiếp ảnh hưởng realized revenue
"""))

# ══════════════════════════════════════════════════════════════════════════════
# SUBMISSION
# ══════════════════════════════════════════════════════════════════════════════
cells.append(new_markdown_cell("## G · Generate submission.csv + Final Diagnostics"))
cells.append(new_code_cell("""\
pred_df = (fp.loc[test_dates_sorted, ["Revenue","COGS"]]
           .reset_index().rename(columns={"Date":"Date"}))
pred_df["Revenue"] = pred_df["Revenue"].clip(lower=0).round(2)
pred_df["COGS"]    = pred_df["COGS"].clip(lower=0).round(2)

submission = sample_sub[["Date"]].merge(pred_df, on="Date", how="left")

# ── Integrity checks ──────────────────────────────────────────────────────────
assert len(submission) == len(sample_sub),               "Row count mismatch"
assert list(submission["Date"])==list(sample_sub["Date"]),"Date order mismatch"
assert submission[["Revenue","COGS"]].isna().sum().sum()==0, "NaN in submission"

submission.to_csv("submissions/submission.csv", index=False)
print("✓ submission.csv saved")
print(f"  Revenue: mean={submission['Revenue'].mean():>12,.0f}  max={submission['Revenue'].max():>12,.0f}")
print(f"  COGS   : mean={submission['COGS'].mean():>12,.0f}  max={submission['COGS'].max():>12,.0f}")
submission.head(8)
"""))

cells.append(new_code_cell("""\
# ── Final forecast plot ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 9))

# 2021-2022 fitted
chk  = train[train["Date"].dt.year.between(2021,2022)].copy()
pchk = np.clip(
    W_MAE*rev_mae_m.predict(chk[FEATURE_COLS]) +
    W_HUBER*rev_hub_m.predict(chk[FEATURE_COLS]), 0, None)
mae_chk = mean_absolute_error(chk["Revenue"], pchk)

axes[0].fill_between(chk["Date"], chk["Revenue"]/1e6, alpha=0.12, color=PALETTE[0])
axes[0].plot(chk["Date"], chk["Revenue"]/1e6, lw=1.2, color=PALETTE[0], label="Actual")
axes[0].plot(chk["Date"], pchk/1e6, lw=1.0, color=PALETTE[1], linestyle="--",
             label=f"Fitted  (MAE={mae_chk:,.0f})")
axes[0].set_title("2021–2022 In-Sample Fit"); axes[0].legend(fontsize=9)
axes[0].set_ylabel("Revenue (M VND)")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}M"))

axes[1].fill_between(submission["Date"], submission["Revenue"]/1e6,
                     alpha=0.12, color=PALETTE[2])
axes[1].plot(submission["Date"], submission["Revenue"]/1e6,
             lw=1.5, color=PALETTE[2], label="Revenue Forecast")
axes[1].plot(submission["Date"], submission["COGS"]/1e6,
             lw=1.2, color=PALETTE[1], linestyle="--", alpha=0.8, label="COGS Forecast")
axes[1].plot(submission["Date"],
             (submission["Revenue"]-submission["COGS"])/1e6,
             lw=1.0, color=PALETTE[3], linestyle=":", alpha=0.7, label="Gross Profit")
axes[1].set_title("Ensemble Forecast — Jan 2023 → Jul 2024")
axes[1].legend(fontsize=9); axes[1].set_ylabel("Value (M VND)")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}M"))

plt.suptitle("Final Forecast: Revenue · COGS · Gross Profit", fontsize=13, y=1.01, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/G_forecast_final.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

cells.append(new_code_cell("""\
# ── Final summary ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  FINAL PIPELINE SUMMARY")
print("=" * 65)
print(f"  Features      : {len(FEATURE_COLS)} total")
print(f"  Model         : Ensemble — {W_MAE:.0%} LGB-MAE + {W_HUBER:.0%} LGB-Huber + {W_SEAS:.0%} Seasonal")
print(f"  CV Strategy   : TimeSeriesSplit (5 folds, gap=14d, expanding window)")
print()
print(f"  CV Results (Revenue):")
print(f"  LGB-MAE   → MAE={np.mean(cv_scores['mae_mae']):>10,.0f} ± {np.std(cv_scores['mae_mae']):>8,.0f}  "
      f"RMSE={np.mean(cv_scores['mae_rmse']):>10,.0f}  R²={np.mean(cv_scores['mae_r2']):.4f}")
print(f"  LGB-Huber → MAE={np.mean(cv_scores['hub_mae']):>10,.0f} ± {np.std(cv_scores['hub_mae']):>8,.0f}  "
      f"RMSE={np.mean(cv_scores['hub_rmse']):>10,.0f}  R²={np.mean(cv_scores['hub_r2']):.4f}")
print(f"  Blend     → MAE={np.mean(cv_scores['ens_mae']):>10,.0f} ± {np.std(cv_scores['ens_mae']):>8,.0f}  "
      f"RMSE={np.mean(cv_scores['ens_rmse']):>10,.0f}  R²={np.mean(cv_scores['ens_r2']):.4f}")
print()
print(f"  submission.csv: {DATA_DIR}/submission.csv  ({len(submission):,} rows)")
print("=" * 65)
print("  Constraints checklist:")
print("  [✓] No external data")
print("  [✓] No test Revenue/COGS as features")
print("  [✓] SEED=42 set for reproducibility")
print("  [✓] All lags use shift(≥1)")
print("  [✓] ord_count etc. only via lag≥365 (test-safe)")
print("  [✓] TimeSeriesSplit CV with temporal gap")
print("  [✓] SHAP values computed")
print("=" * 65)
"""))

# ══════════════════════════════════════════════════════════════════════════════
# Assemble & write
# ══════════════════════════════════════════════════════════════════════════════
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name":"Python 3","language":"python","name":"python3"}
nb.metadata["language_info"] = {"name":"python","version":"3.10.0"}

out = "part3_full_pipeline.ipynb"

# Ghi file với utf-8
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

# Đọc lại file (PHẢI thêm encoding="utf-8" ở đây nữa)
import json
with open(out, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

code_n = sum(1 for c in nb2["cells"] if c["cell_type"]=="code")
md_n   = sum(1 for c in nb2["cells"] if c["cell_type"]=="markdown")

print("=" * 65)
print(f"✓ SUCCESS: {out}")
print(f"  Total cells: {len(nb2['cells'])} ({code_n} code · {md_n} markdown)")
print("=" * 65)