# Gridbreaker — Fashion E-Commerce Diagnosis

> **Datathon VinUni 2026 · VinTelligence · DS&AI Club**
> Breaking Business Boundaries — from raw CSV to prescriptive strategy in 14 days.

---

## Executive Summary

This Vietnamese fashion e-commerce business ran from 2012 to 2022. The data tells a specific, non-trivial story: **the 2016→2019 revenue collapse was NOT demand destruction**. Web sessions grew every year (6.8M → 11.1M). Customer signups accelerated. Margins stayed stable. The collapse was a supply-side and conversion failure — the business acquired more customers but converted fewer of them.

| # | Finding | Severity |
|---|---|---|
| 1 | Revenue -46% from 2016 peak while traffic +67% — the defining contradiction | P0 |
| 2 | 50.6% of product-months have BOTH stockout AND overstock flags — inventory paradox | P0 |
| 3 | Apr–Jun seasonal peak, not Tết — counter-intuitive for VN retail | P1 |
| 4 | Wednesday > Saturday in revenue — office-worker/buyer pattern | P1 |
| 5 | Promo calendar is synthetic (6-4-6-4 cadence, 2 discount levels) — controllable lever | P2 |

---

## 6-Stage Datathon Pipeline

```
UNDERSTANDING → EXPLORATION → DESIGN → PREPARATION → DASHBOARD → OUTPUT
 (read docs)    (profile+EDA)  (plan)    (clean+ABT)   (Streamlit)   (submission)
```

| Stage | Deliverable | Location |
|---|---|---|
| 1. Understanding | Read exam PDF + data dictionary | `docs/01_understanding/` |
| 2. Exploration | Data profiling, EDA, Part 1 MCQ solutions | `notebooks/02_exploration/` |
| 3. Design | EDA plan v2, data quality report, economic anomalies | `docs/02_exploration/` |
| 4. Preparation | Cleaning rules, ABT builder, joining pipeline | `src/`, `notebooks/04_preparation/` |
| 5. Dashboard | Streamlit multi-page diagnosis app | `src/app/` |
| 6. Output | Kaggle submission, NeurIPS report, forecasting models | `notebooks/06_output/` |

---

## Project Structure

```
gridbreaker-vinuni/
├── data/
│   ├── raw/                          # 13 CSVs across 4 layers (read-only)
│   │   ├── Master/                   # products, customers, promotions, geography
│   │   ├── Transaction/              # orders, order_items, payments, shipments, returns, reviews
│   │   ├── Operational/              # inventory, web_traffic
│   │   └── Analytical/               # sales.csv (train target)
│   ├── interim/                      # Cleaned per-table parquet files
│   └── processed/                    # abt_daily, abt_orders_enriched, abt_customer_cohort
├── src/
│   ├── io.py                         # 13 CSV loaders with explicit dtype schemas + TRAIN_CUTOFF
│   ├── cleaning.py                   # Per-table cleaning + build_promo_daily()
│   ├── joining.py                    # build_daily_abt(), build_orders_enriched(), build_customer_cohort()
│   └── app/                          # Streamlit dashboard
│       ├── main.py                   # Entry point — multi-page navigation
│       ├── utils/
│       │   ├── data_loader.py        # Cached parquet loaders + COLORS palette + CSS injection
│       │   └── chart_helpers.py      # apply_theme(), annotate()
│       ├── components/
│       │   ├── page_header.py        # render_page_header() — badge + title + subtitle
│       │   ├── insight_box.py        # render_insight(), render_section_label()
│       │   └── kpi_glass.py          # Glassmorphism KPI cards
│       └── app_pages/
│           ├── 00_overview.py        # Hero dashboard with revenue × traffic dual-axis
│           ├── 01_revenue_collapse.py# Annotated timeline, waterfall decomposition, scenario slider
│           ├── 02_funnel_customer.py # Session→Review funnel, cohort heatmap, channel LTV, returns
│           ├── 03_inventory.py       # Inventory paradox scatter, stockout rate, lost revenue, fill-rate
│           ├── 04_prescriptive.py    # Recovery Simulator with interactive sliders + roadmap
│           ├── 05_patterns.py        # Monthly seasonality, heatmap, day-of-week (Wed > Sat)
│           ├── 06_promo.py           # Promo calendar Gantt, discount distribution, revenue lift
│           └── 07_geo.py             # Top-20 cities, channel × city, AOV bubble
├── notebooks/
│   ├── 02_exploration/
│   │   ├── 00_data_profiling.ipynb   # L1: schema, distributions, cardinality checks
│   │   ├── 01_eda_exploratory.ipynb  # L1: funnel, structural breaks, anomaly detection
│   │   └── part1_round1_answers.ipynb# Part 1: 10 MCQ solutions with data verification
│   ├── 04_preparation/
│   │   └── 10_build_abt.ipynb        # L2: 3 ABT tables with leakage audit + LTV/RFM enrichment
│   └── 06_output/                    # Part 3 forecasting notebooks
├── docs/
│   ├── 01_understanding/             # Exam PDF, problem statement
│   └── 02_exploration/
│       ├── eda_plan.md               # Complete 15-section EDA plan with 7 hypothesis tests
│       └── data_quality.md           # P0–P2 issues, schema docs, cross-table validation
├── reports/                          # Generated figures and SHAP outputs
├── CLAUDE.md                         # AI coding guidelines for this repo
└── README.md                         # This file
```

---

## Part 1 — Round 1 Multiple Choice (Jupyter Notebook)

### Run Instructions

```bash
# 1. Activate your environment (Python 3.10+)
conda activate datathon

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scipy

# 3. Launch Jupyter from the project root (critical for path resolution)
cd gridbreaker-vinuni
jupyter notebook

# 4. Open and run: notebooks/02_exploration/part1_round1_answers.ipynb
#    Run cells in order — the setup cell auto-resolves data paths.
```

### Answer Table (all 10 questions verified end-to-end)

| Q | Topic | Answer | Verified Value |
|---|---|---|---|
| Q1 | Median time signup→first order | ≈175 days | median = 175 days |
| Q2 | Best-performing order source | Standard | highest order count |
| Q3 | Top return reason | wrong_size | 7,626 returns |
| Q4 | Lowest conversion rate channel | email_campaign | 0.00446 |
| Q5 | Promo type with most orders | Percentage | 38.66% of orders |
| Q6 | Avg orders for customers 55+ | 5.41 orders/customer | mean = 5.41 |
| Q7 | Region with highest revenue | East | 7.29B VND |
| Q8 | Most cancelled payment method | credit_card | 28,452 (47.8%) |
| Q9 | Streetwear return rate | 5.65% | S = 5.65% |
| Q10 | 6-digit zip orders | 24,447 | count = 24,447 |

---

## Part 2 — Streamlit Dashboard

### Run Instructions

```bash
# 1. Activate environment
conda activate datathon

# 2. Install additional dashboard dependencies
pip install streamlit plotly

# 3. Launch from project root
cd gridbreaker-vinuni
streamlit run src/app/main.py

# 4. Open http://localhost:8501 in your browser
```

### Dashboard Pages

| Page | Title | Analysis Level | Key Charts |
|---|---|---|---|
| 00 | Tổng quan (Overview) | Descriptive | Revenue × Traffic dual-axis, KPI row, category mix, DoW |
| 01 | Revenue Collapse | Descriptive→Prescriptive | Annotated timeline, waterfall decomposition, category heatmap |
| 02 | Funnel & Customer | Diagnostic→Prescriptive | Funnel, cohort retention, channel LTV bubble, return reasons |
| 03 | Inventory Paradox | Diagnostic | Stockout × overstock scatter, lost revenue, sell-through violin |
| 04 | Recovery Simulator | Predictive→Prescriptive | Interactive sliders, live projection, effort × impact matrix |
| 05 | Patterns & Timing | Descriptive→Diagnostic | Monthly seasonality, year×month heatmap, DoW analysis |
| 06 | Promo ROI | Diagnostic | Calendar Gantt, discount distribution, revenue lift bars |
| 07 | Geographic | Descriptive | Top-20 cities bar, channel × city, AOV bubble |

### Design System

- **Theme:** Dark cinematic green — `#0A1F1A` background, `#52B788` primary
- **Typography:** Outfit (headings) + Inter (body) + JetBrains Mono (numbers)
- **Cards:** Glass morphism KPI cards with backdrop blur and hover lift
- **Charts:** Plotly dark theme with brand colorway, annotations on key data points
- **Navigation:** 3-section sidebar (DASHBOARD / DIAGNOSIS / STRATEGY)

---

## Part 3 — Forecasting & Output

> **Status:** In design phase. See `notebooks/06_output/` for forecasting pipeline prototypes.

---

## Key Constraints

| Constraint | Value |
|---|---|
| Train cutoff | `2022-12-31` — all analysis leakage-safe |
| Target variable | `sales.Revenue` (daily, VND) |
| Test period | 2023-01-01 → 2024-07-01 (548 days) |
| Random seed | 42 (numpy + sklearn) |
| Submission format | `sample_submission.csv` — 548 rows, `id` + `Revenue` |

---

## Team

**The GridBreakers** — VinTelligence · VinUni DS&AI Club

"Breaking Business Boundaries" — from raw e-commerce data to actionable prescriptive strategy.

---

*Built with Python · Streamlit · Plotly · LightGBM · SHAP*
*Datathon VinUni 2026 — Round 1*
