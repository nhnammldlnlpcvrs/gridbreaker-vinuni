"""
Standard join functions for building ABTs (eda_plan.md §4).
All joins are leakage-safe — call only with cleaned DataFrames.
"""
import pandas as pd
from pathlib import Path

ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
INTERIM   = ROOT / "data" / "interim"

TRAIN_CUTOFF = pd.Timestamp("2022-12-31")


def build_orders_enriched(
    order_items: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
    geography: pd.DataFrame,
    returns: pd.DataFrame,
    reviews: pd.DataFrame,
    shipments: pd.DataFrame,
) -> pd.DataFrame:
    """
    5-way join → grain: 1 row per order_item (abt_orders_enriched).
    Filters to train window only (order_date <= TRAIN_CUTOFF).
    """
    # Base: order_items ← orders
    df = order_items.merge(
        orders[["order_id", "order_date", "order_status", "customer_id",
                "zip", "payment_method", "device_type", "order_source"]],
        on="order_id", how="left",
    )
    # Filter leakage
    df = df[df["order_date"] <= TRAIN_CUTOFF].copy()

    # ← products
    df = df.merge(
        products[["product_id", "category", "segment", "size", "color",
                  "price", "cogs", "margin_pct"]],
        on="product_id", how="left",
    )

    # ← customers
    df = df.merge(
        customers[["customer_id", "age_group", "gender",
                   "acquisition_channel", "signup_date"]],
        on="customer_id", how="left",
    )

    # ← geography (via zip from orders)
    df = df.merge(
        geography[["zip", "city", "region", "district"]],
        on="zip", how="left",
    )

    # Derived columns
    df["item_margin"]  = (df["net_revenue"] - df["quantity"] * df["cogs"]).astype("float32")
    df["is_returned"]  = df["order_id"].isin(set(returns["order_id"]))

    # Days to delivery from shipments
    ship_sub = shipments[["order_id", "delivery_days"]].drop_duplicates("order_id")
    df = df.merge(ship_sub, on="order_id", how="left")

    # Reviews (fill NaN for unreviewed)
    rev_sub = reviews[["order_id", "product_id", "rating"]].drop_duplicates(
        subset=["order_id", "product_id"]
    )
    df = df.merge(rev_sub, on=["order_id", "product_id"], how="left")
    df["has_review"] = df["rating"].notna()

    return df.reset_index(drop=True)


def build_daily_abt(
    sales: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    web_traffic_daily: pd.DataFrame,
    promo_daily: pd.DataFrame,
    full_date_range: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Grain: 1 row per calendar day (2012-07-04 → 2024-07-01).
    Revenue/COGS filled only for train window; test window = NaN (leakage guard).
    """
    df = pd.DataFrame({"date": full_date_range})

    # Sales (train window only)
    sales_sub = sales[["Date", "Revenue", "COGS"]].rename(columns={"Date": "date"})
    df = df.merge(sales_sub, on="date", how="left")

    # Order counts from orders (train window only)
    orders_train = orders[orders["order_date"] <= TRAIN_CUTOFF].copy()
    order_daily = (
        orders_train.groupby("order_date")
        .agg(
            n_orders=("order_id", "count"),
            n_delivered=("order_status", lambda x: (x == "delivered").sum()),
            n_cancelled=("order_status", lambda x: (x == "cancelled").sum()),
            n_returned=("order_status", lambda x: (x == "returned").sum()),
        )
        .reset_index()
        .rename(columns={"order_date": "date"})
    )
    df = df.merge(order_daily, on="date", how="left")

    # Order items aggregates (train window only)
    oi_train = order_items.merge(
        orders_train[["order_id", "order_date"]], on="order_id", how="inner"
    )
    items_daily = (
        oi_train.groupby("order_date")
        .agg(
            n_items=("order_id", "count"),
            total_quantity=("quantity", "sum"),
            gross_revenue_recon=("gross_revenue", "sum"),
            discount_amount_sum=("discount_amount", "sum"),
        )
        .reset_index()
        .rename(columns={"order_date": "date"})
    )
    df = df.merge(items_daily, on="date", how="left")

    # Web traffic
    df = df.merge(web_traffic_daily, on="date", how="left")

    # Promo calendar
    df = df.merge(promo_daily, on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_customer_cohort(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Grain: 1 row per (customer_id, months_since_signup).
    Train window only.
    """
    orders_train = orders[orders["order_date"] <= TRAIN_CUTOFF].copy()

    # Revenue per order
    rev_per_order = order_items.groupby("order_id").agg(
        order_revenue=("net_revenue", "sum")
    ).reset_index()

    merged = orders_train.merge(rev_per_order, on="order_id", how="left")
    merged = merged.merge(
        customers[["customer_id", "signup_date", "acquisition_channel"]],
        on="customer_id", how="left",
    )
    merged = merged.dropna(subset=["signup_date"])

    merged["signup_month"] = merged["signup_date"].dt.to_period("M")
    merged["order_month"]  = merged["order_date"].dt.to_period("M")
    merged["months_since_signup"] = (
        merged["order_month"] - merged["signup_month"]
    ).apply(lambda x: x.n if hasattr(x, "n") else None)

    cohort = (
        merged.groupby(["customer_id", "signup_month", "months_since_signup", "acquisition_channel"])
        .agg(
            orders_in_month=("order_id", "count"),
            revenue_in_month=("order_revenue", "sum"),
        )
        .reset_index()
    )
    cohort = cohort[cohort["months_since_signup"] >= 0].copy()

    cohort["cum_revenue"] = (
        cohort.sort_values("months_since_signup")
        .groupby("customer_id")["revenue_in_month"]
        .cumsum()
    )
    cohort["is_active"] = cohort["orders_in_month"] > 0

    return cohort.reset_index(drop=True)
