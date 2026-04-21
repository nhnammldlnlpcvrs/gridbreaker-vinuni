"""
Per-table cleaning rules (eda_plan.md §3).
Each function is pure: takes a raw DataFrame, returns a cleaned copy.
Does NOT modify in-place. Does NOT write files.
Outliers are FLAGGED (is_outlier_*), never dropped.
"""
import pandas as pd
import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────────

def _flag_iqr_outliers(df: pd.DataFrame, col: str, k: float = 3.0) -> pd.Series:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    return ~df[col].between(q1 - k * iqr, q3 + k * iqr)


# ── Master layer ───────────────────────────────────────────────────────────────

def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Flag negative-margin rows (cogs >= price); don't drop — could be strategy
    out["margin_negative"] = out["cogs"] >= out["price"]
    out["margin_pct"] = (out["price"] - out["cogs"]) / out["price"] * 100
    # Normalise string cols
    for col in ["category", "segment", "size", "color"]:
        if col in out.columns:
            out[col] = out[col].str.strip().str.lower()
    return out


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Normalise nullable categoricals
    for col in ["gender", "age_group", "acquisition_channel"]:
        if col in out.columns:
            out[col] = out[col].str.strip().str.lower()
    # Ensure signup_date is datetime
    out["signup_date"] = pd.to_datetime(out["signup_date"], errors="coerce")
    # zip as string (leading zeros)
    out["zip"] = out["zip"].astype(str).str.zfill(5)
    return out


def clean_promotions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    out["end_date"]   = pd.to_datetime(out["end_date"],   errors="coerce")
    # Drop rows where start > end (data quality violation)
    bad = out["start_date"] > out["end_date"]
    if bad.sum() > 0:
        out = out[~bad].copy()
    out["promo_type"] = out["promo_type"].str.strip().str.lower()
    out["duration_days"] = (out["end_date"] - out["start_date"]).dt.days + 1
    return out


def clean_geography(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["zip"] = out["zip"].astype(str).str.zfill(5)
    for col in ["city", "region", "district"]:
        if col in out.columns:
            out[col] = out[col].str.strip()
    # Dedupe on zip — keep first occurrence
    out = out.drop_duplicates(subset=["zip"], keep="first").reset_index(drop=True)
    return out


# ── Transaction layer ──────────────────────────────────────────────────────────

def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["order_date"]   = pd.to_datetime(out["order_date"], errors="coerce")
    out["order_status"] = out["order_status"].str.strip().str.lower()
    out["zip"]          = out["zip"].astype(str).str.zfill(5)
    for col in ["payment_method", "device_type", "order_source"]:
        if col in out.columns:
            out[col] = out[col].str.strip().str.lower()
    return out


def clean_order_items(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Derived revenue columns (used many times downstream)
    out["gross_revenue"] = (out["quantity"] * out["unit_price"]).astype("float32")
    out["net_revenue"]   = (out["gross_revenue"] - out["discount_amount"]).astype("float32")
    # Flag rows where discount exceeds gross
    out["discount_exceeds_gross"] = out["net_revenue"] < 0
    # Outlier flags
    out["is_outlier_quantity"]       = _flag_iqr_outliers(out, "quantity")
    out["is_outlier_unit_price"]     = _flag_iqr_outliers(out, "unit_price")
    out["is_outlier_discount"]       = _flag_iqr_outliers(out, "discount_amount")
    return out


def clean_payments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["payment_method"] = out["payment_method"].str.strip().str.lower()
    out["is_outlier_payment_value"] = _flag_iqr_outliers(out, "payment_value")
    return out


def clean_shipments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ship_date"]     = pd.to_datetime(out["ship_date"],     errors="coerce")
    out["delivery_date"] = pd.to_datetime(out["delivery_date"], errors="coerce")
    out["delivery_days"] = (out["delivery_date"] - out["ship_date"]).dt.days
    # Do NOT fill missing shipments — only exists for shipped/delivered/returned
    out["is_outlier_delivery_days"] = _flag_iqr_outliers(
        out[out["delivery_days"].notna()].copy(), "delivery_days"
    ).reindex(out.index, fill_value=False)
    return out


def clean_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_date"]   = pd.to_datetime(out["return_date"], errors="coerce")
    out["return_reason"] = out["return_reason"].str.strip().str.lower()
    out["is_outlier_refund"] = _flag_iqr_outliers(out, "refund_amount")
    return out


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Drop reviews with rating outside [1, 5]
    out = out[out["rating"].between(1, 5)].copy().reset_index(drop=True)
    return out


# ── Analytical layer ───────────────────────────────────────────────────────────

def clean_sales(df: pd.DataFrame, cutoff: pd.Timestamp = pd.Timestamp("2022-12-31")) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.sort_values("Date").reset_index(drop=True)
    # Leakage guard — hard cutoff
    out = out[out["Date"] <= cutoff].copy()
    out["margin_pct"] = (out["Revenue"] - out["COGS"]) / out["Revenue"] * 100
    return out


# ── Operational layer ──────────────────────────────────────────────────────────

def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    out["is_outlier_stock_on_hand"] = _flag_iqr_outliers(out, "stock_on_hand")
    return out


def clean_web_traffic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"]           = pd.to_datetime(out["date"], errors="coerce")
    out["traffic_source"] = out["traffic_source"].str.strip().str.lower()
    # bounce_rate stored as fraction (~0.004) — keep as-is, document unit
    out["bounce_rate_pct"] = out["bounce_rate"] * 100  # for human readability
    # Daily aggregate (multiple rows per day, one per traffic_source)
    daily = (
        out.groupby("date")
        .agg(
            sessions_total=("sessions", "sum"),
            visitors_total=("unique_visitors", "sum"),
            pageviews_total=("page_views", "sum"),
            bounce_mean=("bounce_rate", "mean"),
            session_sec_mean=("avg_session_duration_sec", "mean"),
            n_sources=("traffic_source", "count"),
        )
        .reset_index()
    )
    return out, daily


# ── Promotion calendar expansion ───────────────────────────────────────────────

def build_promo_daily(promotions: pd.DataFrame, date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """Expand promotions into a daily table (one row per date)."""
    rows = []
    for _, promo in promotions.iterrows():
        days = pd.date_range(promo["start_date"], promo["end_date"], freq="D")
        for d in days:
            rows.append({
                "date": d,
                "promo_id": promo["promo_id"],
                "promo_type": promo.get("promo_type", None),
                "discount_value": promo.get("discount_value", None),
                "stackable_flag": promo.get("stackable_flag", None),
                "applicable_category": promo.get("applicable_category", None),
                "promo_channel": promo.get("promo_channel", None),
                "min_order_value": promo.get("min_order_value", None),
            })
    if not rows:
        return pd.DataFrame(columns=["date", "n_active_promos", "max_discount_active",
                                      "mean_discount_active", "any_pct_promo",
                                      "any_fixed_promo", "days_to_next_promo"])

    daily_promo_raw = pd.DataFrame(rows)
    daily_promo = (
        daily_promo_raw.groupby("date")
        .agg(
            n_active_promos=("promo_id", "count"),
            max_discount_active=("discount_value", "max"),
            mean_discount_active=("discount_value", "mean"),
            any_pct_promo=("promo_type", lambda x: (x == "percentage").any()),
            any_fixed_promo=("promo_type", lambda x: (x == "fixed").any()),
        )
        .reset_index()
    )

    # Fill full date range
    full = pd.DataFrame({"date": date_range})
    daily_promo = full.merge(daily_promo, on="date", how="left")
    daily_promo["n_active_promos"]    = daily_promo["n_active_promos"].fillna(0).astype("int16")
    daily_promo["any_pct_promo"]      = daily_promo["any_pct_promo"].fillna(False)
    daily_promo["any_fixed_promo"]    = daily_promo["any_fixed_promo"].fillna(False)

    # Days to next promo start
    next_promo_starts = sorted(promotions["start_date"].dropna().unique())
    def days_to_next(d):
        future = [s for s in next_promo_starts if s > d]
        return (future[0] - d).days if future else np.nan
    daily_promo["days_to_next_promo"] = daily_promo["date"].apply(days_to_next)

    return daily_promo
