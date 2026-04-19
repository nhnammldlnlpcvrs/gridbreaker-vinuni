# DATATHON 2026 — The Gridbreakers | Project Context

> **Purpose of this file:** Complete context dump for use with Claude Code / CLI / any agent to continue the project. Drop this file into your repo root and point the agent at it with: *"Read DATATHON_2026_CONTEXT.md then continue where we left off."*

---

## 0. Competition overview

- **Host:** VinTelligence (VinUni DS&AI Club)
- **Final round:** 23/05/2026, on-site at VinUni Hà Nội (≥1 member required)
- **Kaggle:** https://www.kaggle.com/competitions/datathon-2026-round-1
- **Report template:** NeurIPS 2025 LaTeX, max 4 pages (excluding refs/appendix)
- **Business domain:** VN fashion e-commerce, data 04/07/2012 → 31/12/2022
- **Test period:** 01/01/2023 → 01/07/2024 (548 days)

### Scoring (100 pts)

| Part | Points | Weight | Notes |
|---|---|---|---|
| 1 — MCQ | 20 | 20% | 10 × 2pt, no penalty for wrong answers |
| **2 — EDA + Visualization** | **60** | **60%** | ⭐ The part that decides winners |
| 3 — Forecasting | 20 | 20% | 12pt Kaggle + 8pt technical report |

### Part 2 rubric (60 pts)
- Viz quality: 15 pts
- **Analysis depth: 25 pts** (need 4 levels: Descriptive → Diagnostic → Predictive → Prescriptive)
- Business insight: 15 pts
- Creativity / storytelling: 5 pts

### Part 3 disqualification triggers
- Using test-period `Revenue`/`COGS` as features
- Using external data outside provided files
- Not attaching reproducible source code

---

## 1. Dataset (15 CSV files in `/mnt/user-data/uploads/`)

### Layer: Master
| File | Rows | Key cols |
|---|---|---|
| `products.csv` | 2,412 | product_id, category, segment, size, color, price, cogs |
| `customers.csv` | 121,930 | customer_id, zip, signup_date, gender, age_group, acquisition_channel |
| `promotions.csv` | 49 | promo_id, promo_type, discount_value, start_date, end_date, stackable_flag |
| `geography.csv` | 39,948 | zip, city, region, district |

### Layer: Transaction
| File | Rows | Key cols |
|---|---|---|
| `orders.csv` | 646,945 | order_id, order_date, customer_id, zip, order_status, payment_method, device_type, order_source |
| `order_items.csv` | 714,669 | order_id, product_id, quantity, unit_price, discount_amount, promo_id, promo_id_2 |
| `payments.csv` | 646,945 | order_id, payment_method, payment_value, installments (1:1 with orders) |
| `shipments.csv` | 566,067 | order_id, ship_date, delivery_date, shipping_fee (shipped/delivered/returned only) |
| `returns.csv` | 39,939 | return_id, order_id, product_id, return_date, return_reason, refund_amount |
| `reviews.csv` | 113,551 | review_id, order_id, product_id, customer_id, rating, review_title |

### Layer: Analytical
| File | Rows | Notes |
|---|---|---|
| `sales.csv` | 3,833 | Date, Revenue, COGS (daily aggregate — TRAIN) |
| `sample_submission.csv` | 548 | Date, Revenue, COGS — format + order for Kaggle submission |

### Layer: Operational
| File | Rows | Key cols |
|---|---|---|
| `inventory.csv` | 60,247 | snapshot_date (month-end), product_id, stock_on_hand, stockout_days, fill_rate, sell_through_rate, stockout/overstock/reorder flags |
| `web_traffic.csv` | 3,652 | date, sessions, unique_visitors, page_views, bounce_rate, avg_session_duration_sec, traffic_source |

### Order status distribution
```
delivered    516,716   cancelled    59,462
returned      36,142   shipped      13,773
paid          13,577   created       7,275
```

### Cardinality rules
- orders ↔ payments = 1:1
- orders ↔ shipments = 1:0/1 (only shipped/delivered/returned)
- orders ↔ returns = 1:0..N (only returned)
- orders ↔ reviews = 1:0..N (only delivered, ~20%)
- order_items ↔ promotions = N:0/1
- products ↔ inventory = 1:N (1 row/product/month)

---

## 2. Key data facts discovered (CRITICAL — these are the insights)

### 2.1 Annual revenue trend (NOT monotonic growth — the baseline's fatal flaw)
```
year    Revenue(VND)    YoY%    Margin%
2012    741M            —       20.8   (partial: Jul-Dec)
2013    1,657M          +124    11.5
2014    1,872M          +13     15.9
2015    1,890M          +1      11.9
2016    2,105M          +11     15.4   ← PEAK
2017    1,911M          -9      11.3
2018    1,850M          -3      16.6
2019    1,137M          -39     11.6   ← SHOCK
2020    1,054M          -7      16.0
2021    1,043M          -1       9.8
2022    1,170M          +12     12.8
```

### 2.2 The 2019 shock breakdown (**gold for Part 2 Diagnostic**)

Revenue YoY % by category:
```
category     2016   2017   2018   2019   2020
Streetwear   +16%   -11%    0%   -40%    -5%   ← 80% of biz, slow decay then crash
Outdoor       -3%    -9%   -29%  -30%   -17%   ← 4-year monotonic decline
Casual       +17%   +13%   +35%  -38%   -16%   ← Bubble then pop
GenZ          +4%    -2%   +50%  -56%   -15%   ← Bubble then pop
```

### 2.3 Counter-signals (proves shock was NOT demand collapse)
- **Web traffic TRENDS UP** every year 2013-2022 (18K → 30K sessions/day avg)
- **Customer signups TREND UP** every year (957 → 21,103/year)
- **Bounce rate FLAT** (~0.4-0.5%)
- **Margins stable** (no price war)

→ **Diagnosis:** Traffic up + conversion down → volume problem, not demand/price. Possible causes to test: cohort quality deterioration, AOV shift to cheaper SKUs, stockouts on hero products.

### 2.4 Seasonality
- **Peak months:** Apr-Jun (avg ~6.5M/day) — summer
- **Low months:** Nov-Jan (avg ~2.5M/day) — unusual for VN since Tết is typically peak
- **Day-of-week:** Wed highest (4.68M), Sat lowest (3.91M) — B2C-like, not retail-weekend
- **Day of month:** no strong pattern

### 2.5 How `sales.csv` is computed (verified experimentally)
Reconstruction from `order_items × products` shows:
- Using ALL order statuses → ~9% deviation
- Using delivered only → ~24% deviation
→ **`sales.csv` = SUM over all orders (including cancelled, returned)**, not just delivered. This matters when building aggregate features.

### 2.6 Promotion calendar is synthetic
- Exactly 50 promos, alternating 6-4-6-4 promo/year pattern
- Discount values alternate 20.8% / 15%
- **No Vietnamese holiday awareness in the data** — so adding Tết features yourself is high-value

---

## 3. Baseline notebook critique (`baseline.ipynb`)

### Bugs / weaknesses deliberately left in
1. **`TEST_FILE` undefined** → crashes at cell 2 (forces you to read carefully)
2. **Geometric mean YoY over 2013-2022** → diluted by 2018-2019 crash → 2023 forecast will be too low
3. **Base level = 2022 annual mean** → 2022 is near bottom, unrepresentative
4. **Ignores 14 of 15 files** (no web, promos, orders, inventory)
5. **No day-of-week effect**, only day-of-year
6. **No outlier handling** (peak 20M vs median 3.6M = huge skew)
7. **Uses MAPE** — but Kaggle scores by MAE, RMSE, R²

---

## 4. Plan by Part

### Part 1 — MCQ (Target: 18-20/20)

Ship one notebook `mcq_solutions.ipynb` with 10 code cells, one per question.

**Tricky cues:**
- **Q1 (median inter-order gap):** `orders.sort_values(['customer_id','order_date']).groupby('customer_id')['order_date'].diff().dt.days.median()`
- **Q3 (return reason for Streetwear):** MUST join `returns × products on product_id`, filter category = Streetwear, then `return_reason.value_counts()`
- **Q7 (revenue by region):** `sales.csv` has no region → reconstruct from `orders × order_items × customers × geography` (5-way join, use `order_items.quantity * unit_price - discount_amount`)
- **Q9 (return rate by size):** numerator = rows in `returns` joined to products, denominator = rows in `order_items` joined to products (NOT quantity)
- **Q5, Q6, Q8, Q10:** straightforward groupby; watch null handling in Q6

Strategy: answer all 10 (no wrong-answer penalty).

### Part 2 — EDA + Visualization (Target: 48-55/60) ⭐

**Framework: 5-7 stories × 4 analysis levels**

Each story = Descriptive + Diagnostic + Predictive + Prescriptive + ≥1 chart + quantified business recommendation.

#### Story 1 — The 2019 shock deep-dive (strongest, builds on §2.2 above)
- D: line chart yearly revenue
- Dx: breakdown by category × year heatmap
- P: what if pattern recurs in 2023-2024?
- Px: which 1-2 categories to diversify into to reduce single-cat exposure risk

#### Story 2 — Traffic vs conversion funnel
- Per-year chart: sessions vs unique_visitors vs orders vs revenue (normalized)
- Diagnose the 2019 divergence: traffic ↑ but revenue ↓ → conversion drop
- Predict 2023-2024 if conversion continues to decline
- Prescribe: invest in conversion tools, not acquisition

#### Story 3 — Cohort retention
- Heatmap: signup_month × months_since_signup
- Diagnose cohorts by acquisition_channel — which channels produce high-LTV users
- Predict 2023 LTV by cohort
- Prescribe: reallocate acquisition budget

#### Story 4 — Promo ROI
- Incremental revenue lift (promo vs matched non-promo days)
- Diagnose `stackable_flag` effect
- Identify top-3 / bottom-3 promos by margin impact
- Prescribe: revised promo calendar

#### Story 5 — Inventory health
- Stockout trends per category over time
- Lost-revenue estimate: stockout_days × avg daily sales/product
- Predict Q1-Q2 2023 demand vs current stock
- Prescribe: reorder policy

#### Story 6 (bonus for creativity) — Geographic heatmap VN
- Join `orders → customers → geography` → revenue per zip/region
- Map VN with choropleth (Plotly/folium)
- Under-served regions recommendation

#### Story 7 (optional) — Returns diagnostics
- Rate by size/color/category, correlate with review rating and delivery time
- Discontinue / QC-improve recommendations

**Creativity (5 pts) ingredients:**
- Cross-table analyses (≥3 tables joined in one insight)
- At least one counter-intuitive finding with data backup
- Interactive Plotly charts (2-3 of them)
- Consistent color palette, proper titles/axes/units/annotations

### Part 3 — Forecasting (Target: 15-17/20)

**Two parallel tracks.**

#### Track A — Quick baselines (to get Kaggle submissions early)
1. Fix `baseline.ipynb` (define `TEST_FILE`, use weighted recent-years trend instead of geometric mean)
2. **TimesFM zero-shot** (Google foundation model) — no training needed
3. **Prophet** with Vietnamese holidays hardcoded (Tết dates 2013-2024, 30/4, 2/9, 1/5)

#### Track B — ML main (LightGBM + features) — the scoring track

**Targets:** `log1p(Revenue)`, `log1p(COGS)` — separate models.

**Features to engineer (all leakage-safe):**

*Calendar* (all safe):
- year, month, day, dow, doy, week, quarter
- is_weekend, is_month_start, is_month_end, is_quarter_end
- cyclical encodings: sin/cos of month, dow, doy
- days_since_2012 (linear trend)

*Vietnamese holidays:*
- `is_tet_window` (±7d around day-1 of Tết, lunar dates hardcoded)
- `days_to_tet` (signed, clipped ±60)
- `is_fixed_holiday` (1/1, 30/4, 1/5, 2/9)

*Lags (≥28 days to avoid recursive forecasting):*
- `rev_lag_{28, 91, 182, 365, 730}`
- `cogs_lag_{28, 91, 182, 365, 730}`
- `yoy_ratio = rev_lag_365 / rev_lag_730`

*Rolling (shift-28 first, then roll):*
- `rev_roll_mean_{7, 28, 91, 365}`
- `rev_roll_std/min/max` for same windows

*Web traffic:*
- Aggregate daily across all traffic_source: total_sessions, total_visitors, total_pageviews, avg_bounce, avg_session_sec, n_sources
- Then lag 28 and 365 + rolling 28
- **Drop raw live columns** (NaN for test dates → leak-avoidance)

*Promotions:*
- `n_active_promos`, `max_active_discount`, `mean_active_discount`
- `any_percentage_promo`, `any_fixed_promo`
- `days_to_next_promo` (promo calendar is planned in advance → OK to use future dates)

*Inventory:*
- Monthly aggregates (total_stock, total_stockout_days, mean_fill_rate, pct_stockout_products, pct_overstock_products, mean_sell_through)
- Lag ≥28 days (attach each date to most recent snapshot ≥28 days old)

**CV strategy:** `TimeSeriesSplit(n_splits=5)` — expanding window. Never shuffle.

**LightGBM hyperparams (safe defaults):**
```python
LGB_PARAMS = dict(
    objective="regression", metric="rmse",
    learning_rate=0.03, num_leaves=63,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
    n_estimators=3000, verbose=-1, random_state=42,
)
```
Early stopping 100 rounds.

**Explainability (8 pts report):**
- SHAP `TreeExplainer` (exact for LGBM)
- Beeswarm plot (global importance)
- Waterfall for a couple of notable test-set predictions
- Scatter/dependence for top-3 features
- Translate findings into business language (not "feature X has high gain" but "last-week traffic predicts next week revenue because it reflects pipeline")

**Ensemble (final submission):**
- 0.5 × LightGBM + 0.3 × Prophet + 0.2 × TimesFM (tune weights on OOF)

### Disqualification guards (CRITICAL — re-check before submit)
- ❌ Never use `orders`/`order_items`/`shipments` with `order_date > 2022-12-31`
- ❌ No external data (no Fed rate, no weather, no VN CPI)
- ❌ No random shuffle in CV
- ❌ Set random seed everywhere (`np.random.seed(42)`, `random_state=42` in sklearn/LGBM)

### Submission format
- `submission.csv`: columns `Date,Revenue,COGS`
- Exactly 548 rows, same order as `sample_submission.csv`
- Dates as `YYYY-MM-DD` strings

---

## 5. K-Dense-AI skills absorbed (relevance to this project)

Source: https://github.com/K-Dense-AI/scientific-agent-skills (134 skills, mostly bio/chem but 7 relevant)

| Skill | Used for | Key principle absorbed |
|---|---|---|
| `exploratory-data-analysis` | Part 2 | 6-section report structure per story: metadata → basic info → format details → analysis → findings → recommendations |
| `statistical-analysis` | Part 2 diagnostic | Posterior Predictive Checks, ROPE (Region of Practical Equivalence), prior sensitivity; use effect sizes + credible intervals, not just p-values |
| `scientific-visualization` | Part 2 charts | Publication-quality, colorblind-safe palettes, consistent styling |
| `aeon` | Part 3 baseline | MiniRocket (<1s train, 0.95+ benchmark accuracy); HIVECOTE ensemble for SOTA |
| `TimesFM` | Part 3 zero-shot | No training needed for univariate series |
| `statsmodels` | Part 3 | ARIMA/SARIMAX for classical trend+seasonality |
| `SHAP` | Part 3 explainability | TreeExplainer exact+fast for LightGBM; beeswarm + waterfall + scatter = full explainability stack |
| `scientific-writing` | Report | IMRAD (Intro–Methods–Results–Discussion) structure |

---

## 6. Timeline — 14 days, 3-person team

```
Day 1-2    All          EDA full exploration + MCQ + env setup
Day 3-6    A            Part 2 stories 1-3 (build data foundation)
           B            Part 2 stories 4-5
           C            Part 2 geographic viz + diagnostic deep-dive
Day 7-9    B,C          Part 3 Track A baselines submitted to Kaggle
                        Track B feature engineering pipeline
Day 10-11  B,C          LightGBM tuning + SHAP + ensemble
           A            Story polishing + charts finalized
Day 12     All          Final Kaggle submission variants
Day 13     All          NeurIPS report writing (A drives, B+C supply methods/results)
Day 14     All          Review + final submit (submit EARLY to avoid Kaggle queue)
```

---

## 7. Final-submission checklist

- [ ] Kaggle `submission.csv` uploaded — 548 rows, exact `sample_submission.csv` order
- [ ] GitHub repo **public** (or access granted to organizers) with `README.md`
- [ ] Report PDF — NeurIPS 2025 template, ≤4 pages (excluding refs/appendix)
- [ ] Report includes: Part 2 viz + analysis, Part 3 methods + results, SHAP figure, GitHub link
- [ ] Random seeds fixed (`seed=42` everywhere)
- [ ] All 10 MCQ answered in submission form
- [ ] Student ID photos of ALL members uploaded
- [ ] ≥1 team member confirmed for 23/05/2026 on-site final at VinUni Hà Nội

---

## 8. Files in this workspace

- `/mnt/user-data/uploads/` — 15 CSV + `Đề-thi-Vòng-1.pdf` + `baseline.ipynb`
- `/home/claude/DATATHON_2026_CONTEXT.md` — this file

---

## 9. How to continue with Claude Code / CLI

Drop this file into your repo root. Suggested first prompts:

1. **"Read DATATHON_2026_CONTEXT.md. Then build `mcq_solutions.ipynb` answering all 10 MCQ questions from Part 1. Use pandas only, one code cell per question, print the answer clearly."**

2. **"Read DATATHON_2026_CONTEXT.md. Then implement the Part 3 LightGBM pipeline from §4 as `forecasting.ipynb`: feature engineering → TimeSeriesSplit CV → log1p targets → SHAP analysis → submission.csv. Follow the leakage guards in §4."**

3. **"Read DATATHON_2026_CONTEXT.md. Then build Part 2 Story 1 (the 2019 shock deep-dive) as a standalone notebook with all 4 analysis levels and 2-3 Plotly charts."**

4. **"Read DATATHON_2026_CONTEXT.md. Then draft the 4-page NeurIPS report in LaTeX with section skeletons and figure placeholders."**
