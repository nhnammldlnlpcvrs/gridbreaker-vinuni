# EDA & Preprocessing Plan — Datathon 2026 (The Gridbreakers)

> **Scope:** Toàn bộ pipeline EDA + Preprocessing chung, phục vụ cho cả Part 1 (MCQ), Part 2 (Visualization + Analysis 60pts), Part 3 (Forecasting 20pts).
> **Nguyên tắc vàng:** EDA và Preprocessing phải **leakage-safe** theo mốc cutoff `2022-12-31` (test period: `2023-01-01 → 2024-07-01`).

---

## 0. Trả lời câu hỏi chiến lược: Tách EDA/Preprocessing khỏi Visualization có hiệu quả không?

**Có, nhưng chỉ khi tách đúng lớp.** Đề xuất cấu trúc 3 lớp (không phải 2):

| Lớp | Mục đích | Output | Ai dùng |
|---|---|---|---|
| **L1 — EDA exploratory** | Khám phá, đặt giả thuyết, tìm data quality issues. Biểu đồ throw-away, ít polish | `notebooks/01_eda_exploratory.ipynb` + `reports/data_quality.md` | Nội bộ team |
| **L2 — Preprocessing / Data mart** | Làm sạch, join, tạo bảng phân tích tái sử dụng (analytical base tables — ABT) | `data/processed/*.parquet` + `src/preprocessing/*.py` | Cả Part 2 & Part 3 |
| **L3 — Storytelling visualization** | Chart publication-quality cho report & Kaggle notebook | `notebooks/02_part2_story_{1..7}.ipynb` | Giám khảo |

**Vì sao 3 lớp hiệu quả hơn 2:**
1. **Tránh lặp join 5-bảng:** Part 2 Story 1,2,6 và Part 3 features đều cần `orders × order_items × products × customers × geography`. Nếu không có ABT, mỗi notebook join lại → chậm + bug không đồng bộ.
2. **Leakage firewall:** Lớp L2 cưỡng chế cutoff `2022-12-31` một chỗ duy nhất. Part 3 không bị tự do lấy cột sai.
3. **EDA không nên đẹp:** L1 dùng matplotlib default, nhanh. Polish chart ở L3 chỉ cho 5-7 story cuối.
4. **Parallel hoá team 3 người:** Sau khi L2 xong, A/B/C làm story song song mà không đụng pipeline.

**Rủi ro nếu tách:** Dễ over-engineer ABT, tốn 1-2 ngày. Giảm thiểu bằng cách **chỉ build 3 ABT bắt buộc** (xem §4) — không build tất cả joinable tables.

**Kết luận:** Tách, nhưng L1 chỉ làm 1.5 ngày; L2 làm 1.5 ngày; L3 mới là phần đầu tư chính.

---

## 1. Repository layout đề xuất

```
gridbreaker-vinuni/
├── dataset/                         # raw CSV (read-only, không bao giờ ghi)
│   ├── Master/                      # products, customers, promotions, geography
│   ├── Transaction/                 # orders, order_items, payments, shipments, returns, reviews
│   ├── Operational/                 # inventory, web_traffic
│   ├── Analytical/                  # sales (daily aggregate, train target)
│   └── sample_submission.csv
├── data/
│   ├── interim/                     # bảng làm sạch từng file (parquet)
│   └── processed/                   # ABT cho phân tích & modelling
├── src/
│   ├── io.py                        # loaders với dtype schema
│   ├── cleaning.py                  # rule chuẩn hoá mỗi bảng
│   ├── joining.py                   # các hàm join chuẩn (tránh lặp code)
│   ├── features/
│   │   ├── calendar.py              # holidays, Tết lunar, cyclical
│   │   ├── daily_agg.py             # cho Part 3
│   │   └── cohort.py                # cho Part 2 Story 3
│   └── viz/style.py                 # palette, rcParams
├── notebooks/
│   ├── 00_data_profiling.ipynb      # L1
│   ├── 01_eda_exploratory.ipynb     # L1
│   ├── 10_build_abt.ipynb           # L2 (gọi src/)
│   ├── 20_mcq_solutions.ipynb       # Part 1
│   ├── 21_story_{1..7}.ipynb        # L3 / Part 2
│   └── 30_forecasting.ipynb         # Part 3
├── reports/
│   ├── data_quality.md              # findings L1
│   └── figures/                     # PNG export cho LaTeX
└── eda_plan.md                      # (this file)
```

---

## 2. Phase 1 — Data Profiling (Ngày 1, ~4h)

**Mục tiêu:** Hiểu shape, dtype, null, duplicate, phân phối từng cột của **15 file**. Không vẽ nhiều, không join nhiều.

### 2.1 Per-file profiling checklist (15 file × 6 mục = 90 checks)

Với mỗi file chạy:
1. **Metadata:** rows, cols, memory, đọc bằng dtype gợi ý (tránh `object` cho số)
2. **Schema check vs context §1:** cột khớp? thừa/thiếu?
3. **Missing:** `%null` từng cột, pattern missing (MCAR/MAR nhận định nhanh)
4. **Duplicates:** trên primary key (order_id, product_id…) — phải = 0
5. **Distribution quick:** `describe()` cho numeric, `value_counts(dropna=False).head(20)` cho categorical
6. **Date sanity:** min/max, gap, spike; so sánh với window `2012-07-04 → 2022-12-31`

### 2.2 Cross-file cardinality verification (từ context §1)

```
orders       ↔ payments   = 1:1          (assert)
orders       ↔ shipments  = 1:0/1        (chỉ status shipped/delivered/returned)
orders       ↔ returns    = 1:0..N       (chỉ status returned)
orders       ↔ reviews    = 1:0..N       (chỉ status delivered, ~20%)
order_items  ↔ promotions = N:0/1        (promo_id + promo_id_2)
products     ↔ inventory  = 1:N          (1 row/product/month)
customers    ↔ geography  = N:1 qua zip
```
Mỗi rule → 1 assert + log số dòng vi phạm vào `reports/data_quality.md`.

### 2.3 Output Phase 1
- `notebooks/00_data_profiling.ipynb`
- `reports/data_quality.md` gồm: 15 bảng tóm tắt + danh sách issues được ưu tiên xử lý (P0/P1/P2)

---

## 3. Phase 2 — Data Cleaning Rules (Ngày 1-2, ~6h)

**Triết lý:** Mỗi rule = 1 hàm thuần trong `src/cleaning.py`, có test nhẹ. Không sửa raw.

### 3.1 Cleaning rules đã biết trước từ context

| Bảng | Rule | Lý do |
|---|---|---|
| `orders` | Parse `order_date` sang datetime; chuẩn hoá `order_status` lowercase | §1 có 6 trạng thái |
| `order_items` | Tính `gross_revenue = quantity * unit_price`; `net_revenue = gross - discount_amount` | Dùng nhiều lần |
| `order_items` | `promo_id`, `promo_id_2` → null-safe join | N:0/1 |
| `payments` | Assert 1:1 với orders | §1 |
| `shipments` | Chỉ có cho shipped/delivered/returned — giữ nguyên, KHÔNG fill | Biết trước |
| `returns` | `return_date ≥ order_date`? check | Sanity |
| `reviews` | Rating ∈ [1,5]? drop/flag outlier | Sanity |
| `customers` | `age_group`, `gender` chuẩn hoá chữ; `signup_date` → datetime | |
| `products` | `price ≥ cogs`? margin-negative rows → flag | Có thể là error hoặc chiến lược |
| `promotions` | `start_date ≤ end_date`; mở rộng sang bảng daily `promo_active` | Dùng cho Part 3 |
| `inventory` | `snapshot_date` = last day of month (assert); 1 row/(product, month) | Biết trước |
| `web_traffic` | Sum/aggregate theo `date` để lấy daily total (nhiều traffic_source/ngày) | Dùng cho Part 3 |
| `sales` | Là label train; **chỉ dùng đến `2022-12-31`** | Leakage |
| `geography` | Dedupe theo `zip` (39,948 rows — có thể 1 zip nhiều district?) | Kiểm tra |

### 3.2 Outlier strategy (global)
- **KHÔNG remove** ở lớp cleaning. Chỉ **flag** cột `is_outlier_*` bằng IQR hoặc z>4.
- Ở modelling mới quyết định winsorize/log-transform (tuỳ feature).

### 3.3 Timezone & locale
- Tất cả date giả định local VN (no tz). Chốt 1 lần ở `io.py`.

---

## 4. Phase 3 — Build Analytical Base Tables (Ngày 2, ~6h)

**Chỉ 3 ABT bắt buộc** — đừng vượt quá.

### 4.1 `abt_daily.parquet` — backbone cho Part 3 & Story 2
Grain: 1 row / ngày (`2012-07-04 → 2024-07-01`).
Cột gốc (train window dùng raw, test window để NaN các cột bị leakage):
- `date`, `Revenue`, `COGS` (từ sales, NaN cho test window)
- `n_orders`, `n_delivered`, `n_cancelled`, `n_returned`
- `n_items`, `total_quantity`, `gross_revenue_recon`, `discount_amount_sum`
- Web: `sessions_total`, `visitors_total`, `pageviews_total`, `bounce_mean`, `session_sec_mean`, `n_sources`
- Promo: `n_active_promos`, `max_discount_active`, `mean_discount_active`, `any_pct_promo`, `any_fixed_promo`, `days_to_next_promo`
- Calendar: `dow`, `dom`, `month`, `quarter`, `year`, `is_weekend`, `is_tet_window`, `days_to_tet`, `is_fixed_holiday`

**Leakage guard:** tất cả cột từ orders/items/shipments chỉ fill đến `2022-12-31`. Cột calendar + promo (biết trước) fill toàn bộ. Web traffic có 3,652 rows (~10 năm) → kiểm tra có phủ test window không; nếu không → NaN → lagged features only.

### 4.2 `abt_orders_enriched.parquet` — backbone cho Part 2 & MCQ
Grain: 1 row / order_item (714,669 rows). 5-way join sẵn:
```
order_items
  ⟕ orders            (order_date, status, zip, payment_method, device_type, order_source)
  ⟕ products          (category, segment, size, color, price, cogs)
  ⟕ customers         (age_group, gender, acquisition_channel, signup_date)
  ⟕ geography on zip  (city, region, district)
```
Cột derived: `gross_rev`, `net_rev`, `item_margin`, `is_returned`, `days_to_delivery`, `has_review`, `rating` (fill NaN).

Dùng cho: MCQ Q3, Q7, Q9; Story 1, 6, 7.

### 4.3 `abt_customer_cohort.parquet` — cho Story 3
Grain: 1 row / (customer_id, months_since_signup).
Cột: `signup_month`, `acquisition_channel`, `orders_in_month`, `revenue_in_month`, `cum_revenue`, `is_active`.

### 4.4 Storage
- Format: **parquet + pyarrow** (nhanh, giữ dtype).
- Partition: `abt_orders_enriched` partition theo `year(order_date)` để load nhanh khi filter.

---

## 5. Phase 4 — Exploratory EDA (Ngày 2-3, ~8h)

**Notebook:** `01_eda_exploratory.ipynb`. Mục tiêu **tìm insight**, không làm đẹp.

### 5.1 Re-verify context facts (§2 của DATATHON file)
Không tin mù context — verify lại để tự tin khi viết report:
- [ ] Annual revenue 2012-2022 khớp §2.1
- [ ] 2019 shock khớp §2.2 (category × year heatmap)
- [ ] Traffic up while revenue down (§2.3)
- [ ] Seasonality Apr-Jun peak, Wed > Sat (§2.4)
- [ ] `sales.csv` = SUM over all order statuses (§2.5 — reconstruction test)
- [ ] Promo pattern 6-4-6-4, 20.8%/15% alternating (§2.6)

### 5.2 Hypothesis list (mỗi story 2-3 giả thuyết, test nhanh)
- **H1:** 2019 drop do cohort 2017-2018 chất lượng thấp (retention 2nd-order cohort)
- **H2:** 2019 drop do AOV giảm (shift sang SKU rẻ), không phải đơn giảm
- **H3:** 2019 drop do stockout hero SKU (cần inventory coverage)
- **H4:** Web conversion = orders/sessions giảm mỗi năm sau 2016
- **H5:** Streetwear return rate cao hơn mặt bằng → ảnh hưởng margin
- **H6:** Vùng miền (region) có revenue share thay đổi — shift from HN→HCM?
- **H7:** Promo stackable tăng lift nhưng margin bị bào

Mỗi H → 1 cell quick-test (không chart đẹp). Đánh dấu H nào thành story chính.

### 5.3 Data oddities cần tài liệu
- Bounce rate 0.4-0.5% là bất thường (bình thường 30-60%) → có thể unit là fraction-of-fraction; note trong report.
- 59,462 cancelled + 7,275 created + 13,577 paid + 13,773 shipped = 94,087 orders chưa delivered nhưng vẫn tính vào sales.csv (§2.5). Cần diễn giải.

---

## 6. Phase 5 — Feature Engineering for Part 3 (Ngày 5-6)

**Nguồn gốc:** `abt_daily.parquet`. Tất cả feature leakage-safe theo quy tắc §4 context.

### 6.1 Bảng feature (tất cả từ context §4 Part 3)
| Nhóm | Feature | Safe? | Note |
|---|---|---|---|
| Calendar | year, month, dow, doy, quarter, cyclical sin/cos | ✅ | |
| Calendar | days_since_2012 | ✅ | linear trend |
| Holiday VN | is_tet_window, days_to_tet, is_fixed_holiday | ✅ | hardcode lunar 2013-2024 |
| Lag | rev_lag_{28,91,182,365,730}, cogs_lag_* | ✅ | ≥28d |
| Lag | yoy_ratio = rev_lag_365 / rev_lag_730 | ✅ | |
| Roll | rev_roll_{mean,std,min,max}_{7,28,91,365} (shift-28 first) | ✅ | |
| Web | lag28 + lag365 + roll28 của sessions/visitors/pageviews/bounce | ✅ | drop live cols |
| Promo | n_active, max/mean_discount, days_to_next_promo | ✅ | promo calendar known |
| Inventory | monthly agg lag ≥28d | ✅ | attach latest snapshot ≥28d old |

### 6.2 Transform target
- `y_rev = log1p(Revenue)`, `y_cogs = log1p(COGS)` — 2 model riêng.
- Inverse ở predict time: `expm1(pred)`.

### 6.3 CV split
`TimeSeriesSplit(n_splits=5)` expanding, no shuffle, seed=42.

---

## 7. Phase 6 — Preprocessing for Visualization (Part 2)

Mỗi story cần pre-aggregate riêng để chart nhẹ (plotly chậm với >100k rows):

| Story | Input ABT | Pre-agg output | Key |
|---|---|---|---|
| 1. 2019 shock | `abt_orders_enriched` | `agg_year_category.parquet` | year, category → revenue |
| 2. Funnel | `abt_daily` | đã sẵn | |
| 3. Cohort | `abt_customer_cohort` | heatmap pivot | signup_month × months_since |
| 4. Promo ROI | `abt_orders_enriched` + promo calendar | `agg_promo_lift.parquet` | matched-day method |
| 5. Inventory | `inventory` + products | `agg_stockout_cat.parquet` | category, month |
| 6. Geo map | `abt_orders_enriched` + geography | `agg_region.parquet` | region, year |
| 7. Returns | `abt_orders_enriched` + returns | `agg_returns_dim.parquet` | size, color, category |

---

## 8. Quality Gates & Leakage Guards

Checklist tự động chạy trước khi dùng ABT cho modelling:
- [ ] `abt_daily` có 4,381 rows (2012-07-04 → 2024-07-01 inclusive)
- [ ] Revenue/COGS = NaN với date > `2022-12-31`
- [ ] Không feature nào pull từ `orders.order_date > 2022-12-31`
- [ ] `order_items`, `shipments`, `returns`, `reviews` filter date ≤ `2022-12-31` trước khi tạo feature cho test window
- [ ] Web traffic live cols đã drop, chỉ còn lag/roll
- [ ] Promo future cols OK (calendar known in advance)
- [ ] Random seed 42 ở mọi nơi
- [ ] `pd.testing` equality check sau re-run idempotent

---

## 9. Timeline đề xuất (cho team 3 người — map vào §6 context)

| Ngày | Task | Owner |
|---|---|---|
| 1 | Phase 1 profiling + viết data_quality.md | A+B+C chia 15 file |
| 1-2 | Phase 2 cleaning rules (`src/cleaning.py`) | A |
| 2 | Phase 3 ABT build (`src/joining.py`, notebook 10) | B |
| 2-3 | Phase 4 EDA + verify facts + hypothesis test | C (A,B review) |
| 3 | MCQ notebook (dùng `abt_orders_enriched`) | A |
| 3-4 | Phase 6 pre-agg per-story | B |
| 5-6 | Phase 5 feature engineering Part 3 | B+C |
| 7+ | Story polishing L3 + modelling | split theo §6 context |

---

## 10. Deliverables cuối Phase EDA/Preprocessing (trước khi sang viz/model)

1. ✅ `reports/data_quality.md` — issues P0/P1/P2
2. ✅ `data/processed/abt_daily.parquet`
3. ✅ `data/processed/abt_orders_enriched.parquet`
4. ✅ `data/processed/abt_customer_cohort.parquet`
5. ✅ `src/cleaning.py`, `src/joining.py`, `src/features/calendar.py` (có unit test nhẹ)
6. ✅ `notebooks/00_data_profiling.ipynb`, `01_eda_exploratory.ipynb`, `10_build_abt.ipynb`
7. ✅ Verified 6 context facts (§5.1 above) — có bằng chứng số
8. ✅ Hypothesis shortlist 5-7 stories để team Part 2 pick

---

## 11. Rủi ro & mitigation

| Rủi ro | Tác động | Mitigation |
|---|---|---|
| Over-engineer ABT | Mất 2+ ngày | Giới hạn 3 ABT, time-box 1.5 ngày |
| Leakage lọt vào Part 3 | Disqualify | Chốt cutoff một chỗ duy nhất trong `src/io.py`; assert ở ABT |
| Web traffic không phủ test window | Features NaN | Fallback: chỉ dùng lag365 (có value trong train) |
| Tết lunar sai năm | Feature noise | Hardcode từ lịch chính thức 2013-2024, unit test |
| Bounce rate unit lạ | Diễn giải sai | Note trong report; tránh làm kết luận tuyệt đối |
| Cohort table nặng | Notebook chậm | Partition theo year, dùng parquet |

---

## 12.A. Part 2 Rubric Coverage Audit (đối chiếu PDF Đề thi trang 10-14)

### 12.A.1 Ánh xạ rubric → deliverable

| Tiêu chí PDF | Điểm tối đa | Mức 13-15/21-25 yêu cầu | Deliverable cụ thể |
|---|---|---|---|
| Chất lượng trực quan hoá | 15 | "Tất cả biểu đồ đạt chuẩn, lựa chọn loại biểu đồ tối ưu cho từng insight" | §13 chart-type spec + §15 per-chart checklist |
| Chiều sâu phân tích | 25 | "Cả 4 cấp độ Descriptive→Diagnostic→Predictive→Prescriptive một cách nhất quán" trên **nhiều** phân tích | §14 per-story 4-level spec (≥5 story coverage) |
| Insight kinh doanh | 15 | "Đề xuất cụ thể, định lượng được, áp dụng được ngay" | §14 Prescriptive column có **công thức số** |
| Sáng tạo & kể chuyện | 5 | "Góc nhìn độc đáo, kết hợp nhiều nguồn dữ liệu, mạch trình bày thuyết phục" | §16 narrative arc + ≥1 counter-intuitive finding |

### 12.A.2 Các dòng dữ liệu PDF có mà plan cũ bỏ sót

- `promotions.applicable_category` (lọc được promo theo category — quan trọng Story 4)
- `promotions.promo_channel` (trùng với `orders.order_source`? → join để check)
- `promotions.min_order_value` (điều kiện áp dụng — ảnh hưởng promo effective rate)
- Công thức `discount_amount`: `percentage` → q×p×(d/100); `fixed` → q×d → **re-compute & assert khớp** cột `discount_amount` gốc (data quality check)
- `inventory.units_received`, `units_sold`, `days_of_supply` — dùng cho Story 5 Predictive (days_of_supply → stockout risk forecast)
- `returns.return_quantity` — Q9 MCQ nên thử cả 2 phương án (rows vs quantity-weighted)
- File test: PDF gọi là `sales_test.csv` (không công bố); train là `sales.csv` / `sales_train.csv` — note tên trong code
- Nullable cols: `customers.gender/age_group/acquisition_channel`, `orders.*` không null → filter null-safe khi groupby

---

## 13. Chart-type specification (cho tiêu chí 15đ Viz quality)

**Nguyên tắc:** mỗi insight một chart type tối ưu. Không dùng pie chart cho >5 slice; không dùng line cho categorical.

| Story | Chart chính | Chart phụ | Lý do chọn |
|---|---|---|---|
| 1. 2019 shock | Line chart yearly revenue + annotation (highlight 2019) | Heatmap category×year YoY% | Line cho trend, heatmap cho compare-across-dim |
| 2. Traffic vs conversion | Dual-axis line (sessions left, revenue right) normalized | Scatter sessions vs orders với year color | Dual-axis show divergence; scatter show correlation-shift |
| 3. Cohort retention | Cohort heatmap (signup_month × months_since) | Line overlay per acquisition_channel | Cohort = gold standard cho retention |
| 4. Promo ROI | Barbell/dumbbell chart (pre vs during promo revenue) | Scatter discount% vs lift% per promo | Matched-pair visualization |
| 5. Inventory health | Small-multiples line (stockout_days per category over time) | Waterfall lost-revenue contribution | Small multiples tránh 1 chart quá dày |
| 6. Geographic | Choropleth VN by region (Plotly) | Bubble map top 20 zip codes | Map cho geo, bubble cho concentration |
| 7. Returns diagnostic | Grouped bar return_rate × size × category | Sankey (return_reason → category) | Bar cho compare, Sankey cho flow |
| **Dashboard tổng** (in-report cover) | KPI strip + mini trend + category donut | — | 1 trang executive view |

**Reject list:** pie charts (>5 slices), 3D charts, rainbow palette cho ordinal data, stacked bar khi user cần đọc chính xác value.

---

## 14. Per-story 4-level analysis spec (cho tiêu chí 25đ Chiều sâu)

**Format:** mỗi story BẮT BUỘC có 4 sub-section — mỗi sub-section có ≥1 số cụ thể + method. Không viết văn suông.

### Story 1 — 2019 shock deep-dive
- **Descriptive:** Revenue 2016=2,105M (peak) → 2019=1,137M (-46%). Category breakdown bảng §2.2 context.
- **Diagnostic:** Test 3 hypotheses:
  - H1 Cohort quality: AOV của cohort 2017-2018 so với 2013-2016 (test `t-test` hoặc Bayesian ROPE)
  - H2 AOV shift: median(net_rev/order) theo year — giảm bao nhiêu?
  - H3 Stockout hero SKU: top-20 product 2016 revenue → stockout_days 2018-2019
- **Predictive:** Fit linear/piecewise trend 2020-2022 trên category share → projection 2023-2024. Quantify: "Streetwear share dự kiến giảm từ 80% → X% vào 2024 nếu xu hướng tiếp tục"
- **Prescriptive:** "Diversify vào top-2 category có CAGR dương (Casual/GenZ hồi phục 2020-2022). **Định lượng:** nếu reallocate 20% inventory budget → giảm single-category exposure từ 80% → 65%, expected revenue variance giảm ~W%"

### Story 2 — Traffic-conversion funnel
- **Descriptive:** Sessions 2013: 18K/day → 2022: 30K/day (+67%). Revenue 2016 peak → 2019 -46%. Conversion ratio rev/session giảm từ X → Y.
- **Diagnostic:** Bucket conversion theo device_type, order_source, traffic_source — cái nào drop nhiều nhất?
- **Predictive:** Fit conversion trend → nếu 2023 conversion tiếp tục giảm 2%/năm, revenue 2024 ~ Z (ngay cả khi traffic giữ +5%)
- **Prescriptive:** "Ngân sách acquisition 2023-2024 nên giảm N%, tái đầu tư vào conversion (A/B test, checkout UX). Expected ROI: nếu khôi phục conversion về mức 2016 → recovery revenue = M VND/năm"

### Story 3 — Cohort retention × acquisition channel
- **Descriptive:** Retention M3, M6, M12 per cohort. Heatmap signup_month × months_since.
- **Diagnostic:** Channel breakdown — acquisition_channel nào có LTV cao nhất? Test: `email_campaign` vs `paid_search` LTV@12m.
- **Predictive:** Fit retention curve (Weibull/BG-NBD) → forecast LTV 2023 cohort = P VND/customer.
- **Prescriptive:** "Reallocate Q% budget từ low-LTV channels (VD: social_media LTV=X) sang high-LTV (VD: organic_search LTV=Y). Expected incremental revenue: Z VND/năm"

### Story 4 — Promo ROI (dùng `applicable_category`, `min_order_value`, `promo_channel`)
- **Descriptive:** 49 promos, alternating 20.8%/15% discount, 6-4-6-4/năm (§2.6).
- **Diagnostic:** Matched-pair: revenue promo-days vs matched non-promo-days (DoW+month match). Break by `promo_channel`, `stackable_flag`. **Incremental lift** vs **cannibalization** (stolen from baseline).
- **Predictive:** Simulate 2023 promo calendar (keep pattern) → incremental revenue projection.
- **Prescriptive:** "Drop bottom-3 promos (incremental lift <K%). Shift budget sang top-3 (lift >L%). **Expected net gain:** = (top_3_lift − bottom_3_lift) × avg_daily_rev × promo_days = M VND/năm"

### Story 5 — Inventory health (dùng `days_of_supply`, `stockout_days`, `fill_rate`)
- **Descriptive:** Stockout_days tổng theo category×month. Pct overstock vs stockout.
- **Diagnostic:** Correlate stockout với revenue drop — category nào lose revenue vì stockout?
- **Predictive:** Extrapolate Q1-Q2 2023 demand (bootstrap từ 2020-2022) vs current stock_on_hand tháng 12/2022 → identify SKUs sẽ stockout trước 2023-03-31.
- **Prescriptive:** "Reorder list: N SKUs cần đặt thêm trước 2023-02-01. **Expected lost revenue avoided:** = stockout_days_forecast × avg_daily_sales_per_SKU = P VND"

### Story 6 — Geographic (choropleth VN)
- **Descriptive:** Revenue share theo region (Bắc/Trung/Nam) và top-10 city. YoY growth per region.
- **Diagnostic:** Under-served regions: high-traffic-zip nhưng low-order-zip (session÷order mismatch). Shipping_fee có cao bất thường không?
- **Predictive:** Regional CAGR → forecast 2024 regional share.
- **Prescriptive:** "Focus marketing 2023 vào top-3 under-served zip (high-traffic, low-conversion). **Định lượng:** nếu đạt conversion trung bình quốc gia → incremental Q VND/năm"

### Story 7 — Returns diagnostic
- **Descriptive:** Return rate theo size/color/category. Correlate với rating, delivery time.
- **Diagnostic:** Size "XL" của category nào hỏng nhất? Color-category combo nào trả nhiều?
- **Predictive:** Nếu trend tiếp tục, 2023 return volume = R units.
- **Prescriptive:** "Discontinue top-5 SKU return_rate >T%. **Expected margin save:** = return_volume × (unit_price − cogs) × fraction_non_resellable = S VND"

---

## 15. Per-chart production checklist (cho 15đ Viz quality)

Mỗi chart phải pass checklist sau trước khi vào report:
- [ ] Title ≤12 từ, trả lời "what happened"
- [ ] Subtitle 1 dòng ≤20 từ, trả lời "so what" (key insight)
- [ ] X-axis label + unit (VD: "Năm", "Doanh thu (triệu VND)")
- [ ] Y-axis label + unit, formatter (1e6 → "M")
- [ ] Legend khi ≥2 series, vị trí không che data
- [ ] Annotation ≥1 (mũi tên/text gắn trực tiếp vào point quan trọng)
- [ ] Data source footer: "Source: orders × order_items × products (n=714,669 rows)"
- [ ] Palette colorblind-safe (viridis / ColorBrewer categorical)
- [ ] Không quá 5 màu; ordinal dùng sequential, categorical dùng qualitative
- [ ] Font ≥10pt, không bị crop khi export PNG 300dpi
- [ ] Export cả `.png` (cho LaTeX) và `.html` (cho interactive version)
- [ ] Số liệu khớp với bảng tóm tắt trong section analysis (no mismatch)

`src/viz/style.py` phải set sẵn `rcParams` để mọi chart auto-compliant 70%.

---

## 16. Narrative arc & storytelling (cho 5đ Creativity)

**Meta-story:** *"The Gridbreakers tìm ra điểm vỡ của business 2019 và chỉ ra 3 đòn bẩy để hồi phục 2023-2024"*

**Cấu trúc report 4 trang NeurIPS:**
1. **Trang 1:** Story 1 (shock) + Story 2 (funnel) = "Chẩn đoán cái gì hỏng"
2. **Trang 2:** Story 3 (cohort) + Story 5 (inventory) = "Hai rễ cụ thể của vấn đề"
3. **Trang 3:** Story 4 (promo) + Story 6 (geo) = "Hai đòn bẩy hồi phục"
4. **Trang 4:** Part 3 (forecast + SHAP) = "Mô hình dự báo 2023-2024"

Story 7 → appendix.

**Counter-intuitive findings cần có ≥1** (quyết định điểm sáng tạo):
- Ứng viên A: "Wed revenue > Sat revenue" — ngược intuition retail
- Ứng viên B: "Traffic lên nhưng revenue xuống" — conversion mới là nút thắt
- Ứng viên C: "Tết KHÔNG phải peak của business này" (Apr-Jun mới peak §2.4)
- Ứng viên D: "Promo stackable làm margin giảm NHIỀU HƠN lift" (nếu data confirm)

**Multi-source joining badges** (mục tiêu ≥3 story có ≥4 bảng join):
- Story 1: orders × order_items × products × customers (4)
- Story 4: orders × order_items × promotions × products × customers (5)
- Story 6: orders × order_items × customers × geography × products (5)

---

## 17. Verification plan — Part 2 trước khi submit

Chạy checklist này trước deadline:
- [ ] 7/7 story có đủ 4 analysis level, mỗi level có số cụ thể
- [ ] 15/15 chart pass checklist §15
- [ ] ≥5 story có Prescriptive với công thức số (không viết "nên cân nhắc")
- [ ] ≥1 counter-intuitive finding có data backup
- [ ] ≥3 story join ≥4 bảng
- [ ] Report 4 trang không crop chart, font ≥10pt
- [ ] Narrative arc coherent đọc từ đầu đến cuối (test: nhờ 1 người ngoài đội đọc)
- [ ] Color palette nhất quán across 7 story (cùng màu cho cùng category)
- [ ] Mọi số liệu khớp giữa text và chart (kiểm kê lần cuối)

---

## 12. Trả lời trực tiếp câu hỏi của bạn

> **"Liệu việc tách ra làm EDA → Preprocessing rồi mới Visualization sau có hiệu quả không?"**

**Hiệu quả, với điều kiện:**
1. **Không làm EDA đẹp ở lớp đầu** — L1 chỉ để tìm insight, chart throw-away.
2. **Preprocessing phải đẻ ra ABT reusable** (3 bảng §4), không phải script rời.
3. **Time-box nghiêm:** L1+L2 tối đa 3 ngày / 14 ngày. Nếu vượt là dấu hiệu over-engineer.
4. **L3 viz kế thừa ABT** → mỗi story notebook chỉ ~50-100 dòng code, team song song được.

**Nếu gộp EDA+Viz:** Bạn sẽ join lại dataset 7 lần cho 7 story, bug không đồng bộ, chart đẹp nhưng số liệu khác nhau giữa các notebook — đây là cái bẫy phổ biến ở datathon.

**Nếu tách quá sâu (5-6 lớp):** Tốn thời gian infra hơn là insight — không đáng với 14 ngày.

→ **Kết luận: Tách 3 lớp L1/L2/L3 như trên là sweet spot.**
