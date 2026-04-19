# Data Quality Report
Generated: 2026-04-19 15:27

## Summary
Total issues: 4

## 🔴 P0 — CRITICAL — Blocks analysis (1 issues)

| Table | Issue |
|---|---|
| `orders×shipments` | Cardinality broken — extra=0 missing=564 |

## 🟡 P1 — Important (2 issues)

| Table | Issue |
|---|---|
| `shipments` | 564 eligible orders missing shipment record |
| `web_traffic` | web_traffic ends 2022-12-31, does NOT cover test period 2023-01-01→2024-07-01 → lag features only for Part 3 |

## 🟢 P2 — Minor / Notes (1 issues)

| Table | Issue |
|---|---|
| `web_traffic` | bounce_rate mean=0.0045 — likely stored as fraction (not %). Note in report: 0.005 = 0.5%, unusual for retail (normal: 30-60%) |

## Dataset overview

| File | Rows | Cols |
|---|---|---|
| products | 2,412 | 9 |
| customers | 121,930 | 7 |
| promotions | 50 | 11 |
| geography | 39,948 | 4 |
| orders | 646,945 | 8 |
| order_items | 714,669 | 9 |
| payments | 646,945 | 4 |
| shipments | 566,067 | 5 |
| returns | 39,939 | 7 |
| reviews | 113,551 | 7 |
| sales | 3,833 | 6 |
| inventory | 60,247 | 18 |
| web_traffic | 3,652 | 7 |

## Notes
- `bounce_rate` in web_traffic is stored as fraction (~0.004), NOT percent. Treat as-is.
- `sales.csv` reconstructs from ALL order statuses (not delivered-only). See §7.
- Promotion calendar is synthetic (6-4-6-4 promos/year, 20.8%/15% alternating).
- `web_traffic` may not cover test period (2023-2024) — use lag features only for Part 3.
- Nullable: `customers.gender`, `age_group`, `acquisition_channel` — use dropna=False in groupby.