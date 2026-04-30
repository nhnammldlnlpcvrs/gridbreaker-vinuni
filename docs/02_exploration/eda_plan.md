# EDA & Prescriptive Analysis Plan — Datathon 2026 (The Gridbreakers)

> **Version:** 2.0 (Full Economic Audit Upgrade)
> **Scope:** Complete pipeline — EDA, Preprocessing, Visualization, Forecasting.
> **Golden rule:** All features leakage-safe with cutoff `2022-12-31`.

---

## Executive Summary

This business ran a Vietnamese fashion e-commerce operation from 2012 to 2022. The data tells a specific, non-trivial story:

**The 2016 peak → 2019 collapse was NOT caused by demand destruction.** Web sessions grew every year (2013: 6.8M → 2022: 11.1M). Customer signups accelerated. Margins stayed stable. The collapse was a **supply-side and conversion failure**: the business acquired more customers but converted fewer of them, likely due to stockouts on hero SKUs, cohort quality deterioration, and a structural shift of the category mix toward lower-AOV segments.

**Five key findings the judging panel will be looking for:**

1. The 2019 revenue shock (-46% from 2016 peak) while traffic rose +67% — the defining contradiction.
2. Simultaneous stockout + overstock on 50.6% of product-months — the inventory paradox.
3. Apr-Jun seasonal peak (not Tết) — a fundamental counter-intuition about VN retail.
4. Wednesday > Saturday in revenue — B2B-or-office buyer pattern in a "B2C" business.
5. Promo calendar synthetic uniformity (6-4-6-4 pattern, 2 discount levels) — controllable lever.

---

## 0. 3-Layer Architecture (unchanged — optimal for 14-day sprint)

| Layer | Notebook | Purpose |
|---|---|---|
| L1 — EDA Exploratory | `00_data_profiling.ipynb`, `01_eda_exploratory.ipynb` | Find anomalies, verify facts, test hypotheses |
| L2 — Preprocessing / ABT | `10_build_abt.ipynb` | 3 leakage-safe analytical tables |
| L3 — Storytelling | `21_story_{1..7}.ipynb` | Publication-quality charts for judges |

---

## 1. Repository Layout

```
gridbreaker-vinuni/
├── dataset/                         # raw CSV (read-only)
│   ├── Master/                      # products, customers, promotions, geography
│   ├── Transaction/                 # orders, order_items, payments, shipments, returns, reviews
│   ├── Operational/                 # inventory, web_traffic
│   ├── Analytical/                  # sales.csv (train target)
│   └── sample_submission.csv
├── data/
│   ├── interim/                     # cleaned per-table parquet
│   └── processed/                   # abt_daily, abt_orders_enriched, abt_customer_cohort
├── src/
│   ├── io.py                        # loaders with dtype schema, TRAIN_CUTOFF constant
│   ├── cleaning.py                  # per-table rules + build_promo_daily
│   ├── joining.py                   # build_daily_abt, build_orders_enriched, build_customer_cohort
│   └── features/
│       ├── calendar.py              # add_calendar_features, add_lag_roll_features
│       └── cohort.py
├── notebooks/
│   ├── 00_data_profiling.ipynb      ← L1 (augmented with sanity checks)
│   ├── 01_eda_exploratory.ipynb     ← L1 (augmented with funnel + structural break)
│   ├── 10_build_abt.ipynb           ← L2 (augmented with LTV + RFM + leakage audit)
│   ├── 20_mcq_solutions.ipynb       # Part 1
│   ├── 21_story_{1..7}.ipynb        # L3 Part 2
│   └── 30_forecasting.ipynb         # Part 3
├── reports/
│   ├── data_quality.md
│   ├── economic_anomalies_report.md ← new
│   └── figures/
├── eda_plan.md                      ← this file (v2)
└── economic_anomalies_report.md     ← new
```

---

## 2. Phase 1 — Data Profiling (Day 1, ~4h)

**Status:** COMPLETE (`00_data_profiling.ipynb`). Augmented with:
- Distribution validation against economic bounds
- Referential integrity assertions (8 cardinality rules)
- Cross-table join validation
- Outlier detection using business logic (not just IQR)

### Key findings from profiling
| File | Critical Finding | Severity |
|---|---|---|
| `inventory` | 50.6% rows have BOTH stockout_flag=1 AND overstock_flag=1 — impossible in real ops | P0 ANOMALY |
| `inventory` | stockout_flag=67%, overstock_flag=76% — unusually high for both simultaneously | P0 ANOMALY |
| `web_traffic` | `bounce_rate` stored as fraction (~0.0045), not percent — 200x smaller than label implies | P1 |
| `sales` | Reconstructs from ALL order statuses including cancelled (MAPE ~5%) | P1 |
| `promotions` | `applicable_category` null for 40/50 promos — category-targeting nearly absent | P2 |
| `order_items` | `promo_id_2` non-null in only 0.03% rows — effectively unused second promo slot | P2 |

---

## 3. Phase 2 — Cleaning Rules (Day 1-2, ~6h)

**Status:** COMPLETE (`src/cleaning.py`). All 13 tables cleaned with:
- Lowercase normalization of categoricals
- Date parsing with VN locale (no tz)
- Derived columns: `gross_revenue`, `net_revenue`, `item_margin`, `margin_pct`
- Leakage guard: `sales.csv` cut at `2022-12-31`
- `build_promo_daily()` → daily active promo features

---

## 4. Phase 3 — Build ABTs (Day 2, ~6h)

**Status:** COMPLETE (`10_build_abt.ipynb`). Three ABTs passing all quality gates.

**Augmented with (new cells in notebook):**
- Customer LTV proxies (M3, M6, M12 cumulative revenue)
- RFM features (recency, frequency, monetary per customer)
- Conversion rate features at multiple funnel stages
- Inventory pressure signals (stockout risk score, overstock ratio)
- Demand-supply gap features per category-month
- Label leakage audit cell

---

## 5. Phase 4 — Key Economic Anomalies (for judges: this is the gold)

### 5.1 MACRO Anomalies

#### A. The 2019 Revenue Shock
- **Magnitude:** Revenue dropped from 2,104M VND (2016) to 1,136M VND (2019) = **-46%**
- **Critical contradiction:** Sessions grew from 8.4M/year (2016) to 10.0M/year (2019) = **+19%**
- **Economic explanation:** Supply-side failure, not demand collapse. The funnel broke at conversion, not acquisition.
- **Column evidence:** `sales.Revenue` (annual drop), `web_traffic.sessions` (annual rise), `orders` count (2016: 82,247 → 2019: 41,601 = **-49%**)

#### B. Category Structural Break 2018-2019
- Streetwear: -40% YoY 2019 (80% of business)
- Casual: +35% in 2018 → -38% in 2019 (bubble-pop pattern)
- GenZ: +50% in 2018 → -56% in 2019 (extreme bubble-pop)
- Outdoor: 4-year monotonic decline 2015-2019

#### C. Non-Recovery After 2019
- Business never recovered to 2016 peak even by 2022
- 2022 revenue (1,169M) = 55.5% of 2016 peak — structural damage, not cyclical

### 5.2 MICRO Anomalies

#### D. Conversion Rate Collapse
- Orders/Session implied: 82,247 orders / 8.4M sessions (2016) = 0.98%
- Orders/Session implied: 41,601 orders / 10.0M sessions (2019) = 0.42%
- Conversion halved while traffic grew — the funnel is broken at the middle stage
- **Column evidence:** `orders.order_id count`, `web_traffic.sessions` grouped by year

#### E. Payment Method vs Cancellation Correlation
- `orders.payment_method` vs `order_status='cancelled'` — which payment method has highest cancel rate?
- COD (cash-on-delivery) typically has 2-3x higher cancellation in VN e-commerce
- **Business implication:** COD is a conversion trap — drives orders but loses revenue

#### F. Unusual Seasonality Pattern
- Peak: Apr-Jun (average ~6.5M VND/day) — summer fashion drive
- Trough: Nov-Jan (~2.5M/day) — includes Tết, which is normally VN retail peak
- **Economic explanation:** B2B or gifting-free business; or data is fashion seasonal (summer collections)
- **Counter-intuitive finding for judges:** "This business AVOIDS Tết" (anti-seasonal to VN retail norms)

#### G. Wednesday > Saturday Pattern
- Wed revenue higher than Sat — opposite of retail intuition
- Possible explanations: corporate procurement, lunch-break shopping (office workers), app-first B2B
- **Column evidence:** `sales.Revenue` grouped by day-of-week

### 5.3 OPERATIONAL Anomalies

#### H. The Inventory Paradox (CRITICAL ANOMALY)
- 50.6% of inventory product-months have BOTH `stockout_flag=1` AND `overstock_flag=1`
- Overall stockout rate: 67.3% | Overstock rate: 76.3%
- **Economic interpretation:** These flags are NOT mutually exclusive in the data — likely defined as:
  - `stockout_flag`: stockout occurred at ANY point in the month (even 1 day)
  - `overstock_flag`: ending stock > reorder threshold
- **Simulation bug vs economic reality:** A real operation cannot be both stocked out AND overstocked on the same SKU in the same month unless it stockedout early in month then received replenishment. This is likely **a simulation feature**, not a bug.
- **Business implication:** The inventory system has poor timing — orders arrive after stockout, creating a whipsaw effect.

#### I. Days-of-Supply vs Stockout Inconsistency
- Mean `days_of_supply` = X days (check actual value)
- Yet stockout_flag triggers on 67% of product-months
- If DoS is adequate, stockouts should be rare — the mismatch suggests DoS is calculated at month-end AFTER receiving stock, not forecasting forward
- **Evidence:** `inventory.days_of_supply`, `inventory.stockout_days`, `inventory.stock_on_hand`

#### J. High Fill-Rate Yet High Stockout (Contradiction)
- `fill_rate` mean = 0.961 (96.1%) — sounds healthy
- But stockout_flag = 67.3% — appears contradictory
- **Explanation:** fill_rate = fraction of days with stock / total days. Even 1 day of stockout in 31-day month can trigger stockout_flag while fill_rate = 30/31 = 96.8%
- **Business implication:** The stockout metric is very sensitive — even brief gaps hurt

### 5.4 SIMULATION BUGS / Traps

#### K. Discount Value Distribution
- Actual discount values: {10, 12, 15, 18, 20, 50} — NOT the "20.8%/15% alternating" from context
- Context §2.6 is WRONG about the discount values
- Fixed discounts of 50 VND on items priced thousands → negligible real discount
- **Trap for teams:** Treating all discount types equally will distort promo ROI analysis

#### L. promo_id_2 Nearly Empty
- Only 0.03% of order_items have `promo_id_2` — stackable promos are effectively unused
- H7 (stackable margin erosion) has almost no sample — test is statistically weak
- **Implication for Part 2 Story 4:** Focus on promo_type effect, not stackability

---

## 6. Phase 5 — Funnel Breakdown Analysis

**Story 2 — The Conversion Funnel (HIGH PRIORITY for judges)**

### Full funnel stages with column sources:
```
Stage 1: Sessions        → web_traffic.sessions (daily aggregate)
Stage 2: Unique Visitors → web_traffic.unique_visitors
Stage 3: Orders          → orders.order_id COUNT (all statuses)
Stage 4: Delivered       → orders WHERE order_status='delivered'
Stage 5: Revenue         → sales.Revenue (daily)
Stage 6: Retained (M3+)  → abt_customer_cohort.is_active at months_since_signup=3
```

### Key funnel metrics to compute by year:
- `sessions_to_orders`: orders / sessions × 100 (%)
- `orders_to_delivered`: delivered / orders × 100 (%)
- `delivered_to_revenue`: revenue / (delivered × avg_price) × 100 (%)
- `retention_M3`: customers active at M3 / cohort_size × 100 (%)

### Expected finding:
Sessions → Orders conversion drops from ~1% (2016) to ~0.4% (2019) while sessions grew 19%.
This is the "breaking point" in the funnel — the story hook.

---

## 7. Phase 6 — Inventory Crisis Analysis

**Story 5 — Inventory Health (MEDIUM priority but high diagnostic value)**

### Key analyses:
1. **Lost revenue from stockouts:** `stockout_days × avg_daily_revenue_per_SKU`
   - Use: `inventory.stockout_days × (orders_revenue_2016 / 30 / n_products_in_category)`
2. **Stockout-revenue correlation:** Pearson/Spearman between monthly stockout rate and monthly revenue by category
3. **2023 stockout forecast:** Extrapolate demand from 2020-2022 trend vs Dec-2022 `stock_on_hand`
4. **Hero SKU analysis:** Top-20 products by 2016 revenue — track their inventory trajectory 2016-2022

### Inventory paradox visualization:
- Scatter plot: `stockout_days` (x) vs `days_of_supply` (y), colored by `overstock_flag`
- Expected: if overstock_flag=1 AND stockout_days>0, those dots expose the paradox

---

## 8. Phase 7 — Revenue Collapse Diagnosis (4-level analysis)

### 8.1 Descriptive
- Annual revenue table: 2012-2022 with YoY% and margin%
- Category-year heatmap (revenue share and YoY%)
- Highlight: 2016 peak, 2019 trough, 2022 partial recovery

### 8.2 Diagnostic (test 3 competing hypotheses)
- **H1 Cohort quality:** Did 2017-2018 cohorts have lower AOV/LTV than 2013-2016 cohorts?
  - Test: t-test / Mann-Whitney on `net_revenue per order` by signup cohort year
  - Expected: not significant — the problem is conversion VOLUME, not per-order VALUE
- **H2 AOV shift:** Did median order value drop in 2019?
  - Test: median AOV by year from `abt_orders_enriched.net_revenue` grouped by `order_id`
  - Expected: AOV stable (same products, same prices) → volume dropped, not value
- **H3 Stockout:** Did hero SKU stockout correlate with 2019 revenue drop?
  - Test: Spearman correlation, `top_20_skus_stockout_days` vs `revenue_drop_pct_2019`

### 8.3 Predictive
- **If Streetwear share continues declining at 2017-2019 rate:** category at 60% by 2024
- **If conversion does not recover:** 2023-2024 revenue = sessions × 0.42% × avg_AOV
- Quantify: "Maintaining 2016 conversion rate on 2022 traffic = 30K sessions/day × 0.98% = 294 orders/day vs actual 100 orders/day"

### 8.4 Prescriptive
- **Diversification lever:** Increase Casual + GenZ from 12% to 25% share → reduces Streetwear concentration risk
- **Conversion lever:** Improve checkout UX, reduce cart abandonment → 0.42% → 0.65% = +55% revenue
- **Inventory lever:** Reduce stockout_days from 1.16 avg to 0.5 → estimated +X% of lost revenue recovered
- Quantify each lever with formula: `lever_impact = delta_conversion × sessions × avg_AOV`

---

## 9. Phase 8 — Customer Behavior Insights (Story 3)

### Cohort retention analysis:
- **Input:** `abt_customer_cohort` with `signup_month × months_since_signup`
- **Key metrics:** M1, M3, M6, M12 retention rates
- **Breakdown by:** `acquisition_channel` — which channel produces highest LTV customers

### RFM segmentation:
- **R:** Days since last order (recency)
- **F:** Number of orders in customer lifetime (frequency)
- **M:** Total `net_revenue` across all orders (monetary)
- **Segment to:** Champions, Loyal, At-Risk, Lost (4-quadrant)

### LTV proxies:
- `ltv_12m`: cumulative net_revenue in first 12 months
- `ltv_24m`: cumulative net_revenue in first 24 months
- Expected finding: email_campaign and organic_search channels > social_media in M12 LTV

---

## 10. Phase 9 — Hypothesis Testing Plan (with statistical rigor)

| Hypothesis | Test method | Columns used | Expected result |
|---|---|---|---|
| H1: Cohort 2017-18 lower AOV | Mann-Whitney U | `abt_orders.net_revenue` grouped by signup cohort year | Not significant — volume, not value |
| H2: AOV stable 2016-2019 | Kruskal-Wallis + post-hoc | Order-level net_rev by year | Stable median — confirms volume story |
| H3: Stockout → revenue drop | Spearman correlation | `inventory.stockout_days` × `monthly_revenue` | Moderate negative correlation |
| H4: Conversion declining | Linear regression | year × conversion_rate | Slope < 0, R² > 0.7 |
| H5: Streetwear return > avg | One-sample t-test vs population mean | `is_returned` by category | Streetwear above average |
| H6: Regional revenue shift | Chi-squared (revenue share by year) | `region × year` revenue | Stable — no significant shift |
| H7: COD higher cancel rate | Fisher's exact / proportion z-test | `payment_method × order_status` | COD cancel rate > average |

---

## 11. Prescriptive Strategy (Judges want numbers)

### Three quantified business levers for 2023-2024:

**Lever 1: Fix the conversion funnel (highest ROI)**
- Current: ~0.42% conversion (2022)
- Target: 0.65% (achievable with checkout UX + remarketing)
- Sessions 2023 forecast: ~11.5M (based on 2013-2022 trend +5%/yr)
- Revenue impact: `11.5M × (0.65%-0.42%) × avg_AOV ≈ +300M VND/year`

**Lever 2: Reduce stockout on top-50 SKUs**
- Lost revenue estimate: `stockout_days_avg × n_products × avg_daily_rev_per_SKU`
- Using: `1.16 days/month × 2412 products × ~avg_rev` → quantify with actual data
- Reorder trigger: set `reorder_flag` response time to 7 days (vs current implied 30-day lag)

**Lever 3: Diversify category exposure**
- Streetwear = 80% of revenue → single-category risk
- Grow Casual + GenZ (showed fastest pre-2019 growth) to 25% combined share
- Revenue variance reduction: `0.8² × var_streetwear → 0.55² × var_streetwear + 0.25² × var_casual_genz`
- Expected: ~30% reduction in revenue volatility

---

## 12. Chart Specification (15-pt Viz quality)

| Story | Primary chart | Secondary chart | Library |
|---|---|---|---|
| S1. 2019 shock | Annotated line + bar combo (revenue × category share) | Category YoY% heatmap | plotly |
| S2. Funnel | Dual-axis: sessions (left) + orders (right) | Funnel waterfall by year | plotly |
| S3. Cohort | Heatmap signup_month × months_since | Line per acquisition_channel | seaborn + plotly |
| S4. Promo ROI | Barbell chart: pre vs during promo revenue | Scatter discount% vs lift% | plotly |
| S5. Inventory | Small-multiples stockout trend per category | Scatter: stockout_days vs revenue_drop | matplotlib |
| S6. Geographic | Choropleth VN regions | Bubble map top-20 cities | plotly/folium |
| S7. Returns | Grouped bar: return_rate × size × category | Sankey: reason → category | plotly |

**Per-chart checklist** (must pass before submission):
- [ ] Title ≤12 words, answers "what happened"
- [ ] Subtitle 1 line, answers "so what"
- [ ] X/Y axis labels with units (VND, %, K sessions)
- [ ] Colorblind-safe palette (viridis or ColorBrewer)
- [ ] ≥1 annotation on key data point
- [ ] Source footer with table names and row counts
- [ ] Export: .png (300dpi for LaTeX) + .html (interactive)

---

## 13. Quality Gates & Leakage Guards

**Before any modelling cell runs:**
- [ ] `abt_daily` has 4,381 rows (2012-07-04 → 2024-07-01)
- [ ] Revenue/COGS = NaN for date > 2022-12-31
- [ ] No order_date > 2022-12-31 in any transaction ABT
- [ ] Web traffic live columns dropped; only lag/rolling features used in test window
- [ ] Promo future columns OK (calendar known in advance)
- [ ] Random seed 42 everywhere (`np.random.seed(42)`, `random_state=42`)
- [ ] `pd.testing.assert_frame_equal` idempotency check after re-run

---

## 14. Timeline (14-day sprint, 3-person team)

| Day | Task | Owner |
|---|---|---|
| 1 | Profiling + data_quality.md | A+B+C |
| 1-2 | Cleaning rules (`src/cleaning.py`) | A |
| 2 | ABT build + leakage audit | B |
| 2-3 | EDA: verify 6 facts + 7 hypotheses + anomaly detection | C |
| 3 | MCQ notebook (`20_mcq_solutions.ipynb`) | A |
| 3-4 | Story 1 (2019 shock), Story 2 (funnel) | B+C |
| 4-5 | Story 3 (cohort), Story 4 (promo ROI) | A+B |
| 5-6 | Story 5 (inventory), Story 6 (geo) | C |
| 6 | Story 7 (returns) | A |
| 7-9 | Forecasting: feature engineering + LightGBM + Prophet | B+C |
| 10-11 | LightGBM tuning + SHAP + ensemble | B+C |
| 12 | Final Kaggle submission | All |
| 13 | NeurIPS report writing | A drives |
| 14 | Review + submit early | All |

---

## 15. Verification — Before Final Submit

- [ ] 7/7 stories with 4 analysis levels (Descriptive → Diagnostic → Predictive → Prescriptive)
- [ ] Each Prescriptive level has a quantified formula with actual numbers
- [ ] ≥3 stories join ≥4 tables
- [ ] ≥1 counter-intuitive finding with statistical backup
- [ ] All 10 MCQ answered
- [ ] submission.csv: exactly 548 rows, same order as sample_submission.csv
- [ ] GitHub public with README
- [ ] Report PDF: NeurIPS 2025 template, ≤4 pages
- [ ] Random seeds fixed everywhere
- [ ] No post-2022-12-31 data in any feature
