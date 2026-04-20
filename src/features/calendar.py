"""
Calendar & VN holiday features (eda_plan.md §6.1).
All features are leakage-safe (derived from calendar knowledge alone).
Tết dates hardcoded from official VN lunar calendar 2013–2025.
"""
import numpy as np
import pandas as pd


# Tết Nguyên Đán — first day of lunar new year (Gregorian equiv.)
TET_DATES = {
    2013: "2013-02-10",
    2014: "2014-01-31",
    2015: "2015-02-19",
    2016: "2016-02-08",
    2017: "2017-01-28",
    2018: "2018-02-16",
    2019: "2019-02-05",
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
    2025: "2025-01-29",
}

# Fixed public holidays (month-day)
FIXED_HOLIDAYS = [
    (1,  1),   # New Year
    (4,  30),  # Reunification Day
    (5,  1),   # International Labour Day
    (9,  2),   # National Day
]


def _tet_windows(years: list) -> set:
    """Return set of Timestamps in Tết window (±7 days around Tết day)."""
    dates = set()
    for yr in years:
        if yr in TET_DATES:
            center = pd.Timestamp(TET_DATES[yr])
            for offset in range(-7, 8):
                dates.add(center + pd.Timedelta(days=offset))
    return dates


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar features to a daily DataFrame."""
    out = df.copy()
    d = out[date_col]

    out["year"]       = d.dt.year.astype("int16")
    out["month"]      = d.dt.month.astype("int8")
    out["quarter"]    = d.dt.quarter.astype("int8")
    out["dow"]        = d.dt.dayofweek.astype("int8")   # 0=Mon
    out["dom"]        = d.dt.day.astype("int8")
    out["doy"]        = d.dt.dayofyear.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")

    # Days since epoch (linear trend proxy)
    epoch = pd.Timestamp("2012-07-04")
    out["days_since_start"] = (d - epoch).dt.days.astype("int32")

    # Cyclical encoding
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12).astype("float32")
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12).astype("float32")
    out["dow_sin"]   = np.sin(2 * np.pi * out["dow"] / 7).astype("float32")
    out["dow_cos"]   = np.cos(2 * np.pi * out["dow"] / 7).astype("float32")

    # Fixed VN public holidays
    out["is_fixed_holiday"] = d.apply(
        lambda x: any(x.month == m and x.day == dy for m, dy in FIXED_HOLIDAYS)
    ).astype("int8")

    # Tết window
    years = list(range(2012, 2026))
    tet_window_set = _tet_windows(years)
    out["is_tet_window"] = d.isin(tet_window_set).astype("int8")

    # Days to next Tết
    tet_timestamps = sorted([pd.Timestamp(v) for v in TET_DATES.values()])
    def _days_to_tet(x):
        future = [t for t in tet_timestamps if t > x]
        return (future[0] - x).days if future else 999
    out["days_to_tet"] = d.apply(_days_to_tet).astype("int16")

    return out


def add_lag_roll_features(
    df: pd.DataFrame,
    target_cols: list,
    lags: list = None,
    roll_windows: list = None,
    min_shift: int = 28,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Add lag and rolling window features (leakage-safe via min_shift).
    All lags are >= min_shift days to respect the forecasting horizon.
    """
    if lags is None:
        lags = [28, 91, 182, 365, 730]
    if roll_windows is None:
        roll_windows = [7, 28, 91, 365]

    out = df.copy().sort_values(date_col).reset_index(drop=True)

    for col in target_cols:
        if col not in out.columns:
            continue
        s = out[col]

        # Lag features
        for lag in lags:
            if lag < min_shift:
                continue
            out[f"{col}_lag{lag}"] = s.shift(lag).astype("float32")

        # YoY ratio
        if 365 in lags and 730 in lags:
            lag365 = s.shift(365)
            lag730 = s.shift(730)
            out[f"{col}_yoy_ratio"] = (lag365 / lag730.replace(0, np.nan)).astype("float32")

        # Rolling features (shifted by min_shift first to avoid leakage)
        s_shifted = s.shift(min_shift)
        for w in roll_windows:
            roll = s_shifted.rolling(w, min_periods=max(1, w // 2))
            out[f"{col}_roll{w}_mean"] = roll.mean().astype("float32")
            out[f"{col}_roll{w}_std"]  = roll.std().astype("float32")
            out[f"{col}_roll{w}_min"]  = roll.min().astype("float32")
            out[f"{col}_roll{w}_max"]  = roll.max().astype("float32")

    return out
