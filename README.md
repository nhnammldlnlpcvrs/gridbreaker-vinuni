# GridBreakers — Fashion E-Commerce Diagnosis & Forecasting

**Datathon VinUni 2026 · VinTelligence · DS&AI Club**

From raw CSV to prescriptive strategy & 548-day forecast in 14 days.

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
| 3. Design | EDA plan, data quality report, economic anomalies | `docs/02_exploration/` |
| 4. Preparation | Cleaning rules, ABT builder, joining pipeline | `src/`, `notebooks/04_preparation/` |
| 5. Dashboard | Streamlit multi-page diagnosis app | `src/app/` |
| 6. Output | Forecasting pipeline, Kaggle submission, reports | `notebooks/06_output/` |

---

## Project Structure

```
gridbreaker-vinuni/
├── data/
│   ├── raw/                              # 15 CSVs across 4 layers (read-only)
│   │   ├── Analytical/                   # sales.csv (train target)
│   │   ├── Master/                       # products, customers, promotions, geography
│   │   ├── Transaction/                  # orders, order_items, payments, shipments, returns, reviews
│   │   ├── Web/                          # web_traffic
│   │   └── Operational/                  # inventory
│   ├── external/                         # sample_submission.csv
│   ├── interim/                          # Cleaned per-table parquet files
│   └── processed/                        # ABT tables (abt_daily, abt_orders_enriched, abt_customer_cohort)
├── src/
├── notebooks/
│   ├── 02_exploration/
│   │   ├── 00_data_profiling.ipynb       # L1: schema, distributions, cardinality checks
│   │   ├── 01_eda_exploratory.ipynb      # L1: funnel, structural breaks, anomaly detection
│   │   ├── 02_visualization_summary.ipynb# Vietnamese summary of EDA findings
│   │   └── part1_round1_answers.ipynb    # Part 1: 10 MCQ solutions with data verification
│   ├── 03_modeling/
│   │   └── baseline.ipynb                # Early baseline models
│   ├── 04_preparation/
│   │   └── 10_build_abt.ipynb            # L2: 3 ABT tables with leakage audit + LTV/RFM enrichment
│   └── 06_output/
│       ├── part3_pipeline.py             # Main forecasting pipeline (percent-format, editable)
│       └── part3_pipeline.ipynb          # Generated notebook (via jupytext)
├── scripts/                              # Standalone runnable pipeline scripts
├── docs/
│   ├── 01_understanding/                 # Exam PDF, problem statement
│   ...
│   ├── 06_output/                        # Output & submission docs
│   ├── streamlit/                        # Dashboard screenshots & design docs
├── reports/                              # Generated figures and SHAP outputs
│   ├── figures/                          # EDA chart exports
│   ├── tables/                           # LaTeX tables
│   ├── tex/                              # LaTeX report source
├── submissions/
│   ├── sample_submission.csv             # Reference format (548 rows)
│   └── submission.csv                    # Pipeline output
├── requirements.txt
└── README.md
```

---

## Part 1 — Round 1 Multiple Choice (Jupyter Notebook)

### Run Instructions

```bash
# 1. Activate your environment (Python 3.10+)
conda activate datathon

# 2. Install dependencies
pip install -r requirements.txt

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

# 2. Install dashboard dependencies (if not already done)
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
| 08 | Cohort & Hypothesis | Diagnostic | Cohort retention curves, statistical tests, churn analysis |

### Design System

- **Theme:** Dark cinematic green — `#0A1F1A` background, `#52B788` primary

- **Typography:** Outfit (headings) + Inter (body) + JetBrains Mono (numbers)
- **Cards:** Glass morphism KPI cards with backdrop blur and hover lift
- **Charts:** Plotly dark theme with brand colorway, annotations on key data points
- **Navigation:** 3-section sidebar (DASHBOARD / DIAGNOSIS / STRATEGY)

### Executive Dashboard & Business Overview

The dashboard homepage provides a high-level executive summary of the entire business trajectory from 2012 to 2022.  
It is designed to immediately surface the core contradiction behind the company’s decline: revenue collapsed while customer demand indicators continued to rise.

The dashboard combines:
- Financial KPIs
- Traffic analytics
- Customer metrics
- Category composition
- Operational anomalies
- Behavioral patterns

This creates a unified business intelligence layer for rapid diagnosis and strategic decision-making.

---

#### The Great Divergence — Revenue vs Demand

The main overview panel highlights the central finding of the project:

> The business did not fail because demand disappeared.  
> It failed because operational execution and conversion efficiency deteriorated.

Key observations from the dashboard:
- Revenue declined by approximately 44% from the 2016 peak.
- Total orders dropped by 56%, indicating severe conversion deterioration.
- Customer signups continued growing (+129%), proving that top-of-funnel demand remained healthy.
- Gross margin stayed stable at 14%, suggesting pricing and profitability were not the primary issue.
- Average order value remained relatively unchanged, reinforcing that customer willingness to spend did not collapse.

The dual-axis chart (“Revenue vs Traffic — the contradiction”) visualizes the divergence between:
- Rising website sessions
- Falling revenue performance

This divergence strongly suggests:
- Conversion failure
- Inventory inefficiency
- Supply-chain friction
- Funnel leakage

The dashboard also surfaces a major operational anomaly:
- 50.6% of product-months were simultaneously marked as both stockout and overstock.

This “Inventory Paradox” indicates severe supply-chain synchronization problems:
- Some products were unavailable during demand spikes
- Other products accumulated excess inventory without selling

Additional supporting analytics include:
- Product category mix evolution over time
- Payment-method cancellation patterns
- Revenue behavior by day of week

The insight cards at the bottom summarize the economic implications:
- The conversion collapse likely cost the company approximately 4.6 billion VND in unrealized revenue between 2017–2022.
- Website traffic increased by 19% while conversion rate fell from 0.98% to 0.42%.
- Customers continued visiting the platform but increasingly failed to complete purchases.

The dashboard therefore establishes the core narrative for the entire analysis:
the company suffered from operational and conversion breakdowns rather than weak market demand.

<img src="docs/streamlit/dashboard/dashboard_1.png" width="700">

---

#### Executive Diagnostic Summary & Navigation Layer

The second overview dashboard acts as a decision-support and narrative orchestration layer.

Instead of presenting isolated charts, the interface guides users through the investigation process:
1. Identify the divergence between traffic and revenue
2. Diagnose operational bottlenecks
3. Analyze customer retention and conversion behavior
4. Explore recovery strategies and intervention scenarios

The alert panels classify findings into three levels:
- Critical
- Attention
- Insight

Critical findings quantify the macroeconomic impact of operational failure:
- If conversion had remained at the 2016 level, the business could have generated an estimated +4.6B VND in additional revenue.

Attention panels emphasize the contradiction:
- Traffic continued growing while purchasing behavior weakened dramatically.

Insight panels direct users toward the next analytical modules:
- Revenue Collapse diagnostics
- Funnel & Customer analysis
- Recovery Simulator

This structure transforms the dashboard into a guided analytical narrative rather than a static reporting interface.

The system is therefore capable of supporting:
- Executive reporting
- Operational monitoring
- Strategic planning
- Prescriptive business analytics

<img src="docs/streamlit/dashboard/dashboard_2.png" width="700">


#### DIAGNOSIS

<summary><b>01 — Revenue Collapse</b> — Annotated timeline, waterfall decomposition, scenario slider</summary>

Timeline analysis highlighting the 2016–2019 revenue collapse and major operational turning points.

![Revenue Collapse 1](docs/streamlit/diagnosis/revenue_1.png)

Waterfall and scenario analysis explaining how conversion and supply-side failures impacted revenue.

![Revenue Collapse 2](docs/streamlit/diagnosis/revenue_2.png)


<summary><b>02 — Funnel & Customer</b> — Session→Review funnel, cohort heatmap, channel LTV, returns</summary>

Customer funnel visualization from traffic acquisition to purchase and review behavior.

![Funnel Customer 1](docs/streamlit/diagnosis/funel_customer_1.png)

Cohort retention and customer lifetime value analysis across acquisition channels.

![Funnel Customer 2](docs/streamlit/diagnosis/funel_customer_2.png)


<summary><b>03 — Inventory Paradox</b> — Stockout × overstock scatter, lost revenue, fill-rate violin</summary>

Inventory imbalance analysis showing simultaneous stockout and overstock problems.

![Inventory 1](docs/streamlit/diagnosis/inventory_1.png)

Lost-revenue estimation caused by low fill-rate and supply-chain inefficiency.

![Inventory 2](docs/streamlit/diagnosis/iventory_2.png)

Distribution analysis of inventory performance and fulfillment consistency.

![Inventory 3](docs/streamlit/diagnosis/iventory_3.png)


<summary><b>05 — Patterns & Timing</b> — Monthly seasonality, year×month heatmap, day-of-week analysis</summary>

Seasonality analysis identifying recurring monthly and yearly purchasing patterns.

![Patterns 1](docs/streamlit/diagnosis/pattern_timing_1.png)

Heatmap visualization of temporal demand fluctuations across years and months.

![Patterns 2](docs/streamlit/diagnosis/pattern_timing_2.png)

Day-of-week behavioral analysis for operational and marketing optimization.

![Patterns 3](docs/streamlit/diagnosis/pattern_timing_3.png)


<summary><b>06 — Promo ROI</b> — Calendar Gantt, discount distribution, revenue lift bars</summary>

Promotion calendar and campaign scheduling analysis.

![Promo 1](docs/streamlit/diagnosis/promo_1.png)

Revenue uplift and discount effectiveness evaluation for marketing campaigns.

![Promo 2](docs/streamlit/diagnosis/promo_2.png)


<summary><b>07 — Geographic</b> — Top-20 cities, channel × city, AOV bubble chart</summary>

Geographic revenue distribution across major cities and regions.

![Geo 1](docs/streamlit/diagnosis/geo_1.png)

Cross-analysis between customer acquisition channels and geographic performance.

![Geo 2](docs/streamlit/diagnosis/geo_2.png)

Average order value and market opportunity visualization by location.

![Geo 3](docs/streamlit/diagnosis/geo_3.png)


<summary><b>08 — Cohort & Hypothesis</b> — Retention curves, statistical tests, churn analysis</summary>

Customer retention curve analysis across different acquisition cohorts.

![Cohort 1](docs/streamlit/diagnosis/cohort_1.png)

Behavioral segmentation and long-term engagement analysis.

![Cohort 2](docs/streamlit/diagnosis/cohort_2.png)

Hypothesis testing for churn drivers and customer retention factors.

![Cohort 3](docs/streamlit/diagnosis/cohort_3.png)

Statistical validation of operational and marketing assumptions.

![Cohort 4](docs/streamlit/diagnosis/cohort_4.png)

Churn pattern visualization and repeat-purchase analysis.

![Cohort 5](docs/streamlit/diagnosis/cohort_5.png)

Advanced cohort comparison for customer lifecycle evaluation.

![Cohort 6](docs/streamlit/diagnosis/cohort_6.png)


### Strategy & Recovery Planning

The strategy module transforms diagnostic insights into actionable business recovery plans.  
Instead of only explaining why revenue collapsed, the system allows decision-makers to simulate operational improvements, estimate financial uplift, prioritize initiatives, and build an implementation roadmap.

---

#### Recovery Simulator — Interactive Scenario Planning

This module enables users to simulate a 2023 recovery scenario by adjusting operational levers such as conversion rate, stockout reduction, cancellation reduction, AOV uplift, and traffic growth.

The simulator compounds the effect of multiple interventions simultaneously, providing a realistic projection of how operational improvements translate into long-term revenue recovery.

Key observations:
- Conversion improvement has the strongest marginal effect on revenue recovery.
- Inventory stabilization significantly reduces lost sales caused by stockouts.
- Moderate AOV improvements create meaningful compounding gains when paired with higher conversion.
- Traffic growth alone is insufficient without operational optimization.

The live projection chart compares:
- Historical business trajectory
- Baseline “do nothing” scenario
- User-defined recovery strategy

The projected scenario demonstrates that coordinated operational improvements could generate:
- +147% uplift versus the 2022 baseline
- +37% growth relative to the 2016 historical peak

This highlights that the business decline was operationally reversible rather than demand-driven.

<img src="docs/streamlit/strategy/strategy_1.png" width="700">

---

#### Effort × Impact Priority Matrix

The priority matrix evaluates strategic initiatives based on:
- Estimated business impact
- Operational implementation effort
- Expected revenue contribution

The visualization separates:
- Quick wins (low effort, moderate impact)
- Transformational initiatives (high effort, high impact)

Key strategic insights:
- Size-guide fixes and cohort win-back campaigns provide fast, low-cost recovery opportunities.
- Dead-stock SKU reduction improves inventory efficiency while freeing working capital.
- COD-to-prepay nudges generate one of the largest projected revenue uplifts with relatively manageable effort.
- Reorder policy redesign and lead-time optimization require larger operational investment but create durable long-term gains.

The matrix helps leadership prioritize initiatives under limited operational capacity and budget constraints.

Bubble size and color encode projected financial impact, allowing rapid executive interpretation.

<img src="docs/streamlit/strategy/strategy_2.png" width="700">

---

#### 2023 Recovery Roadmap & Gantt Planning

The implementation roadmap converts strategic recommendations into a time-based execution schedule.

The Gantt visualization organizes initiatives across the 2023 timeline and highlights:
- Initiative sequencing
- Cross-functional ownership
- Expected implementation windows
- Coordination dependencies

Execution strategy:
1. Launch quick wins early in Q1:
   - Size-guide improvements
   - Customer win-back campaigns
   - Dead-stock liquidation

2. Use short-term gains to finance larger operational transformation:
   - Reorder policy redesign
   - Lead-time optimization
   - Paid-channel reallocation

3. Transition toward scalable operational recovery by Q3.

The executive insight panel summarizes the overall recovery thesis:
- A balanced portfolio of short-term operational fixes and long-term structural improvements can reverse the post-2016 decline.
- Operational optimization creates significantly higher impact than traffic acquisition alone.
- The simulated strategy portfolio projects up to +147% revenue uplift versus the 2022 baseline.

This module transforms the dashboard from a descriptive analytics tool into a prescriptive decision-support system.

<img src="docs/streamlit/strategy/strategy_3.png" width="700">


---

## Part 3 — Advanced Forecasting Pipeline

The forecasting pipeline predicts **daily Revenue and COGS** for 548 days (2023-01-01 to 2024-07-01) using iterative recursive multi-model forecasting.

### Quick Run

```bash
cd notebooks/06_output
python part3_pipeline.py
```

### Full Workflow (edit .py → convert → run)

```bash
# After editing the percent-format .py file:
jupytext --to notebook part3_pipeline.py -o part3_pipeline.ipynb
python part3_pipeline.py
```

### Pipeline Architecture (14 phases)

| # | Phase | Description |
|---|---|---|
| 1 | Load & Audit | 15 tables loaded with explicit dtypes; leakage audit across all date columns |
| 2 | EDA | 5 comprehensive plots (sales overview, seasonality, annual trend, promo impact, traffic/orders) → `reports/` |
| 3 | Feature Engineering | ~374 features across 14 families (calendar, promos, lags, rolling, EWM, cross-ratios, categories, returns, reviews, payments, inventory, customer acquisition, interactions) |
| 4 | Train/Test Split | Train on non-test rows with valid `Revenue_l365`, test = 548 submission dates |
| 5 | Baselines (A0–A3) | Mean, Seasonal+Trend, Ridge, Decision Tree — establish lower bound |
| 6 | Purged Walk-Forward CV | 3-fold, horizon=365d, purge=90d — realistic production simulation |
| 7 | Optuna Tuning | 40 trials LightGBM (Revenue + COGS) with composite objective; XGBoost/CatBoost/ET/RF/HistGB with tuned defaults |
| 8 | Ensemble Construction | 6 blending methods compared; stacking (Ridge meta-learner) selected as best |
| 9 | Recursive Forecasting | Day-by-day prediction with full feature recomputation each step |
| 10 | Post-Processing | Clipping to historical bounds, exponential smoothing (α=0.25) |
| 11 | Diagnostics | CV fold stability, horizon breakdown (short/medium/long), drift analysis |
| 12 | SHAP Explainability | TreeExplainer on 800-sample subset; top-20 feature importance |
| 13 | Submission | `submissions/submission.csv` — 548 rows, integrity checks |
| 14 | Final Forecast Plot | 2021-2022 in-sample fit + 2023-2024 forecast visualization |

---

### Feature Engineering: 14 Feature Families

The pipeline engineers features from all available data sources. Every external feature uses a lag ≥365 days to prevent data leakage into the test period.

#### 3.1 Calendar Features (~37 features)

Derived solely from the date — these are always available, even for future dates.

| Sub-family | Features | Description |
|---|---|---|
| Time components | `cal_year`, `cal_month`, `cal_day`, `cal_dow`, `cal_doy`, `cal_woy`, `cal_quarter` | Basic date decomposition |
| Boolean flags | `cal_is_weekend`, `cal_is_month_end`, `cal_is_month_start`, `cal_is_quarter_end`, `cal_is_year_end`, `cal_is_year_start` | Calendar event indicators |
| Cyclic encodings | `cal_month_sin/cos`, `cal_dow_sin/cos`, `cal_doy_sin/cos`, `cal_woy_sin/cos` | Fourier sin/cos pairs — capture cyclical nature of time |
| Higher harmonics | `cal_doy_sin{k}/cos{k}` for k=2,3,4,5,6 | Sharp seasonal transitions (e.g., Tết drop, summer spike) |
| Vietnamese holidays | `vn_tet`, `vn_pre_tet`, `vn_post_tet` | Exact lunar calendar Tết dates (±7d window) from `src.features.calendar.TET_DATES` |
| Fixed holidays | `vn_fixed_holiday` | Jan 1, Apr 30, May 1, Sep 2 |
| Retail seasons | `vn_mid_sale`, `vn_year_end`, `vn_back_school`, `vn_summer`, `vn_low_season`, `vn_peak_season` | Vietnamese retail calendar cycles |
| Trend | `cal_time_idx` | Days since first date — linear trend proxy |

**Why Fourier harmonics?** A single sin/cos pair models a smooth sine wave. But retail seasonality has sharp edges (Tết sales plunge, summer peak). Higher harmonics (k=2..6) let the model learn these abrupt transitions.

**Why exact Tết dates?** Tết follows the lunar calendar and shifts by up to 3 weeks between years. Using a fixed Jan 20–Feb 20 window (as earlier versions did) misaligns the feature by up to 21 days.

#### 3.2 Projected Promotions (5 features)

`promotions.csv` ends 2022-12-31 but follows strict annual recurrence patterns. We project expected promos forward:

| Feature | Description |
|---|---|
| `proj_promo_count` | Number of active promos on a given day (0–4) |
| `proj_promo_discount_sum` | Sum of discount percentages from active promos |
| `proj_promo_stackable` | Whether stackable promos are active (0/1) |
| `proj_promo_pct_count` | Number of percentage-based promos |
| `proj_promo_fixed_count` | Number of fixed-amount promos |

**Promo calendar logic:** Spring Sale (Mar 18–Apr 17), Mid-Year (Jun 23–Jul 22), Fall Launch (Aug 30–Oct 1), Year-End (Nov 18–Jan 2), plus biennial Urban Blowout and Rural Special in odd years.

#### 3.3 Revenue & COGS Target Features (~150+ features each)

The core of the model: auto-regressive features that capture the time-series dynamics of Revenue and COGS.

**Lag features** (`Revenue_l1`, `Revenue_l7`, ..., `Revenue_l366`):
14 lags at [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 180, 364, 365, 366] days. Short lags (1–7d) capture momentum. Medium lags (14–60d) capture within-season patterns. Long lags (180–366d) capture YoY anchors.

**Rolling statistics** on shift-1 values (prevents leakage), 7 windows [7, 14, 28, 60, 90, 180, 365] days:
- `_rm{w}` — rolling mean (smoothed level)

- `_rs{w}` — rolling std (volatility)
- `_rmed{w}` — rolling median (robust center)
- `_rmax{w}`, `_rmin{w}` — rolling extremes (range)
- `_rq25{w}`, `_rq75{w}` — rolling quartiles (distribution shape)
- `_rskew{w}` — rolling skewness (asymmetry — rising vs falling)

**EWM features** (Exponentially Weighted Mean), 5 spans [7, 14, 30, 90, 180]:
- `_ewm{span}` — adaptive smoothing, more weight to recent observations

- `_ewm_std{span}` — adaptive volatility

**Momentum & Acceleration** — rate-of-change over [7, 14, 28, 90] days:
- `_mom{w} = (v[t] - v[t-w]) / v[t-w]` — direction and speed

- `_acc{w} = mom[w][t] - mom[w][t-w]` — whether the trend is strengthening or weakening

**Volatility** — log-return rolling std over [7, 14, 28] days:
- `_vol{w}` — captures unstable periods (high vol → harder to predict)

**YoY anchors:**
- `Revenue_yoy_ratio` — this day / same period last year

- `Revenue_l364_mean` — 3-day centered average around lag-364
- `Revenue_2yoy_mean` — average of lag-365 and lag-730

**Trend ratios:**
- `Revenue_trend14_180` — short-term MA / long-term MA (trend strength)

- `Revenue_trend_dev` — deviation of 30d MA from 365d MA YoY baseline

#### 3.4 Order Signals (29 features)

Daily order counts aggregated from `orders` table. All features use lags [7, 14, 28, 90, 180, 365] and rolling means [7, 14, 28, 90, 365] on shift-1 values — zero leakage risk.

#### 3.5 Web Traffic Features (14 features)

Daily sessions, unique visitors, pageviews, bounce rate, avg session duration from `web_traffic`. Merged with t+1 shift to align traffic with next-day sales. Rolling means [7, 14, 28] on shift-1 values.

#### 3.6 Cross-Target & Margin Features (10 features)

- `cogs_rev_ratio_l{7,28,90,180,365}` — COGS/Revenue ratio at various lags (cost efficiency indicator)

- `margin_l{28,90,365}` — Gross margin = (Revenue - COGS) / Revenue

#### 3.7 Category-Driven Features (~30 features)

Top-4 categories by revenue extracted from `order_items × products`. For each category: daily revenue, order count, avg price, and revenue share. All lagged ≥365 days.

#### 3.8 Operational Health Features (~30 features)

| Source | Features | Lag |
|---|---|---|
| Returns | `ret_count`, `ret_rate` | 365, 730 |
| Reviews | `review_count`, `review_avg_rating`, `review_rating_std` | 365, 730 |
| Payments | `pay_cod_ratio`, `pay_cc_ratio`, `pay_avg_installments` | 365, 730 |
| Inventory | `inv_stockout_ratio`, `inv_overstock_ratio`, `inv_avg_dos`, `inv_avg_fill_rate`, `inv_avg_str` | 365, 730 |
| Customer Acq. | `new_cust_count`, `new_cust_ratio` | 365, 730 |

#### 3.9 Interaction Features (13 features)

Multiplicative interactions between calendar events and promotions:
- `interact_promo_wknd` — promo × weekend (do promos work better on weekends?)

- `interact_promo_tet` — promo × Tết (holiday promo effect)
- `interact_tet_promo` — Tết × discount sum (discount sensitivity during Tết)
- `interact_summer_promo` — summer season × promo count
- `interact_wknd_peak` — weekend × peak season
- `interact_holiday_wknd` — fixed holiday × weekend
- `interact_promo_{doy_sin, doy_cos, month_sin, month_cos}` — promo × seasonal position
- `interact_trend_{doy_sin, doy_cos}` — time trend × seasonal position
- `interact_tet_year` — Tết effect over years (decaying or strengthening?)

---

### Model Selection Rationale

#### Why Gradient Boosting (LightGBM) as Primary Learner

Tree-based gradient boosting is the dominant approach for tabular time-series forecasting competitions for several reasons:

1. **Nonlinear interactions**: Unlike linear models, trees automatically discover interaction effects (e.g., "weekend + promo + summer month" → high revenue). Our 13 hand-crafted interaction features help, but trees find the ones we miss.

2. **Feature scale invariance**: Revenue (0 to ~15B VND), COGS, ratios, binary flags — trees don't care about scale differences. No normalization needed.

3. **Missing value handling**: Trees split on present vs. missing naturally. Our early training period has gaps in some lag features.

4. **LightGBM specifically**: Leaf-wise growth (vs. XGBoost's level-wise) converges faster. Native categorical support. Lower memory footprint with 374 features × 3K samples.

#### Why Multi-Model Ensemble

Single models each have blind spots:

| Model | Strength | Weakness |
|---|---|---|
| **LightGBM** (leaf-wise) | Fast convergence, good interactions | Can overfit with small data |
| **XGBoost** (level-wise) | More regularized, stable | Slower, less sharp on interactions |
| **CatBoost** | Ordered boosting, best with categoricals | Heavier, less gain with numeric-only data |
| **ExtraTrees** | Randomized splits = maximum diversity | Higher bias individually |
| **RandomForest** | Bootstrap aggregating = low variance | Weaker on time-series structure |
| **HistGB** (sklearn) | Fast, scikit-learn native | Less tunable than LightGBM |
| **Ridge** (linear) | Can't overfit to noise, extrapolates trends | Misses all nonlinear patterns |
| **Seasonal+Trend** | Fully rule-based, no data needed | Ignores all predictive signals |

Ensembling combines their strengths. The Ridge meta-learner (stacking) learns optimal per-model weights from out-of-fold predictions, weighting each model where it performs best.

#### Why Purged Walk-Forward CV (not standard k-fold)

Standard TimeSeriesSplit with gap=14d produces optimistic estimates because:
- 14-day gap allows rolling features (90d, 180d windows) to leak information across folds

- Training on 2020-2021 to predict 2022 is easier than training on 2012-2022 to predict 2023-2024 (the actual task)

**PurgedWalkForwardCV** uses gap=90d and horizon=365d:
- Each fold trains on data up to time T, then predicts T+90 to T+90+365

- The 90-day purge ensures no rolling window bleeds across the boundary
- This mimics actual deployment: train on past, predict 365 days ahead
- 3 folds × 365-day horizon = covers the full test structure

#### Why Composite Objective

Competition evaluates MAE, RMSE, and R². Optimizing MAE alone can produce:
- Low MAE but high RMSE (large errors on outlier days)

- Good MAE but low R² (model is just predicting the mean)

Composite objective `0.4×MAE_norm + 0.4×RMSE_norm + 0.2×(1-R²)` balances all three. MAE and RMSE share equal weight (80% total), R² gets 20% to prevent mean-regression collapse.

---

### Cross-Validation & Evaluation Strategy

#### Purged Walk-Forward CV Design

```
Fold 1: Train [2012......2020Q2] → Purge 90d → Test [2020Q4......2021Q4]  (365d)
Fold 2: Train [2012......2021Q2] → Purge 90d → Test [2021Q4......2022Q4]  (365d)
Fold 3: Train [2012......2022Q2] → Purge 90d → Test [2022Q4......2022-12-31]
```

#### Diagnostics Run on CV

After CV completes, three diagnostics validate model behavior:

1. **Fold Stability** — MAE/RMSE/R² per fold. If fold-3 (closest to test period) degrades significantly, the model won't generalize to 2023-2024.

2. **Horizon Breakdown** — Metrics computed for short (days 1-90), medium (91-180), and long (181-365) horizons. Degradation over horizon length indicates forecast drift.

3. **Drift Analysis** — Rolling 30d MAE across the prediction window. An upward slope means the model gets progressively worse the further it forecasts — a critical flaw for 548-day prediction.

---

### Recursive Forecasting Mechanism

The test period spans 548 days with NO ground truth for any future date. This means:

1. **Day 1 (2023-01-01)**: Predict Revenue and COGS using all lag features from training data

2. **Feature recomputation**: After prediction, recompute ALL derived features (lags, rolling means, EWM, momentum, volatility, cross-ratios, category features) using the newly predicted values
3. **Day 2 (2023-01-02)**: Use recomputed features to predict the next day
4. **Repeat 548 times**

**Why full recomputation is necessary**: Without recomputing derived features, 88 of ~374 features (all rolling means, EWMs, momentum, volatility beyond simple lags) would use stale values from 2022-12-31. This causes progressive forecast drift — the model's inputs become increasingly out-of-distribution.

The recomputation is the performance bottleneck (~2-3 minutes for 548 days) but eliminates the #1 source of forecast degradation.

---

### Post-Processing Pipeline

Three-stage post-processing corrects known recursive forecasting artifacts:

1. **Clipping to historical bounds** — Predictions outside [min_historical, max_historical × 3] are clamped. This prevents runaway positive feedback loops where an abnormally high prediction feeds into lag features and amplifies subsequent predictions.

2. **Exponential smoothing** (α=0.25) — Reduces day-to-day jitter from model variance. The ensemble already smooths across models; temporal smoothing further stabilizes.

3. **Residual correction** — If systematic bias is detected (e.g., model consistently over-predicts in summer months), a residual model corrects it.

---

### Key Design Decisions

- **Iterative recomputation** of ALL derived features (lags, rolling, EWM, cross-ratios) after each day. Without this, 88 of ~374 features become out-of-distribution and cause forecast drift.

- **Projected promotions**: Since `promotions.csv` ends 2022-12-31, recurring annual promos are projected for the test period based on historical patterns (6-4-6-4 cadence).
- **Purge=90d** in Walk-Forward CV prevents rolling window leakage (max rolling window = 365d; purge ensures no overlap).
- **All external features** (orders, web_traffic, etc.) use lag≥365 to prevent leakage into test period.
- **Negative YoY trend** (-3.8%/yr Revenue, -3.95%/yr COGS) observed in historical data — Revenue declined from ~5B VND/day avg in 2013 to ~4.2B in 2022. The Seasonal+Trend baseline incorporates this geometric decay.
- **Exact lunar Tết dates** from `src.features.calendar.TET_DATES` replace the earlier fixed Jan 20–Feb 20 window (which was off by up to 21 days).
- **Stacking meta-learner** (Ridge with positive weights) selected over simple weighted blend — systematic comparison of 6 ensemble methods on OOF predictions showed stacking produced the lowest composite score.

### Model Stack

| Model | Role | Tuning Method |
|---|---|---|
| **LightGBM** (Optuna-tuned) | Primary learner — highest individual CV score | 40 trials, composite objective |
| **XGBoost** | Diversity model — different tree-building strategy | Default params (tuned by experience) |
| **CatBoost** | Diversity model — ordered boosting | Default params (optional, graceful fallback) |
| **ExtraTrees** | Maximum diversity — randomized splits | Default params |
| **RandomForest** | Bootstrap aggregating — low variance | Default params |
| **HistGradientBoosting** | Sklearn-native — fast histogram-based | Default params |
| **Ridge Regression** | Linear stabilizer — extrapolates trends, zero overfit risk | α=1.0 |
| **Seasonal+Trend** | Rule-based anchor — day-of-year seasonality × geometric trend | No training needed |

### Evaluation Metrics

| Metric | Formula | Competition Weight | What It Measures |
|---|---|---|---|
| **MAE** | `mean(|y - ŷ|)` | 0.4 | Average absolute error in VND — primary metric |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | 0.4 | Penalizes large errors quadratically — outlier sensitivity |
| **R²** | `1 - SS_res / SS_tot` | 0.2 | Variance explained — prevents mean-only prediction |
| **Composite** | `0.4×MAE_norm + 0.4×RMSE_norm + 0.2×(1-R²)` | — | Combined optimization target (lower is better) |

### Data Lineage: 15 Tables → 1 Forecast

```
Analytical/sales.csv  ──────────────┐
Master/promotions.csv ── projected ─┤
Master/products.csv ─── cat share ──┤
Master/customers.csv ── signups ────┤
Master/geography.csv ───────────────┤
Transaction/orders.csv ── agg ──────┤
Transaction/order_items.csv ───┐    ├── 14 feature families
Transaction/payments.csv ──────┤    │      ↓
Transaction/returns.csv ───────┤    │   374 features
Transaction/reviews.csv ───────┤    │      ↓
Transaction/shipments.csv ─────┘    │   8 model ensemble
Operational/inventory.csv ── agg ───┤      ↓
Operational/web_traffic.csv ─ agg ──┘   submission.csv
                                      (548 rows × Revenue + COGS)
```

---

## Dependencies

See `requirements.txt` for the full list. Key packages:

---

## Key Constraints

| Constraint | Value |
|---|---|
| Train cutoff | `2022-12-31` — all analysis leakage-safe |
| Target variables | `sales.Revenue`, `sales.COGS` (daily, VND) |
| Test period | 2023-01-01 → 2024-07-01 (548 days) |
| Random seed | 42 (numpy + sklearn) |
| Submission format | `Date, Revenue, COGS` — 548 rows |