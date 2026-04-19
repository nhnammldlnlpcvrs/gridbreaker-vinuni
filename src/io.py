"""
Centralized data loaders with explicit dtype schemas.
All dates assumed VN local time (no tz-awareness).
TRAIN_CUTOFF is the single source of truth for leakage guard.
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "dataset"

MASTER      = DATA_RAW / "Master"
TRANSACTION = DATA_RAW / "Transaction"
ANALYTICAL  = DATA_RAW / "Analytical"
OPERATIONAL = DATA_RAW / "Operational"

TRAIN_CUTOFF = pd.Timestamp("2022-12-31")


def _read(path: Path, parse_dates: list = None, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_products() -> pd.DataFrame:
    return _read(
        MASTER / "products.csv",
        dtype={"product_id": "int32", "price": "float32", "cogs": "float32"},
    )


def load_customers() -> pd.DataFrame:
    return _read(
        MASTER / "customers.csv",
        parse_dates=["signup_date"],
        dtype={"customer_id": "int32", "zip": "str"},
    )


def load_promotions() -> pd.DataFrame:
    return _read(
        MASTER / "promotions.csv",
        parse_dates=["start_date", "end_date"],
    )


def load_geography() -> pd.DataFrame:
    return _read(
        MASTER / "geography.csv",
        dtype={"zip": "str"},
    )


def load_orders() -> pd.DataFrame:
    return _read(
        TRANSACTION / "orders.csv",
        parse_dates=["order_date"],
        dtype={"order_id": "int32", "customer_id": "int32", "zip": "str"},
    )


def load_order_items() -> pd.DataFrame:
    return _read(
        TRANSACTION / "order_items.csv",
        dtype={
            "order_id": "int32",
            "product_id": "int32",
            "quantity": "int16",
            "unit_price": "float32",
            "discount_amount": "float32",
        },
    )


def load_payments() -> pd.DataFrame:
    return _read(
        TRANSACTION / "payments.csv",
        dtype={"order_id": "int32", "payment_value": "float32", "installments": "int8"},
    )


def load_shipments() -> pd.DataFrame:
    return _read(
        TRANSACTION / "shipments.csv",
        parse_dates=["ship_date", "delivery_date"],
        dtype={"order_id": "int32", "shipping_fee": "float32"},
    )


def load_returns() -> pd.DataFrame:
    return _read(
        TRANSACTION / "returns.csv",
        parse_dates=["return_date"],
        dtype={
            "return_id": "int32",
            "order_id": "int32",
            "product_id": "int32",
            "return_quantity": "int16",
            "refund_amount": "float32",
        },
    )


def load_reviews() -> pd.DataFrame:
    return _read(
        TRANSACTION / "reviews.csv",
        dtype={
            "review_id": "int32",
            "order_id": "int32",
            "product_id": "int32",
            "customer_id": "int32",
            "rating": "float32",
        },
    )


def load_sales(cutoff: bool = True) -> pd.DataFrame:
    df = _read(ANALYTICAL / "sales.csv", parse_dates=["Date"])
    if cutoff:
        df = df[df["Date"] <= TRAIN_CUTOFF].copy()
    return df


def load_inventory() -> pd.DataFrame:
    return _read(
        OPERATIONAL / "inventory.csv",
        parse_dates=["snapshot_date"],
        dtype={
            "product_id": "int32",
            "stock_on_hand": "int32",
            "stockout_days": "int16",
            "fill_rate": "float32",
            "sell_through_rate": "float32",
            "days_of_supply": "float32",
        },
    )


def load_web_traffic() -> pd.DataFrame:
    return _read(
        OPERATIONAL / "web_traffic.csv",
        parse_dates=["date"],
        dtype={
            "sessions": "int32",
            "unique_visitors": "int32",
            "page_views": "int32",
            "bounce_rate": "float32",
            "avg_session_duration_sec": "float32",
        },
    )
