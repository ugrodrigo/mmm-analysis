"""
mmm/preprocessing.py
--------------------
Data loading, cleaning, and feature engineering for the MMM pipeline.
All functions are pure (no side effects) and return a new DataFrame.
"""

import numpy as np
import pandas as pd
import holidays

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMESERIES_ID = "596eef7c71f933d820d0e485935d0e8f"

GOOGLE_SPEND_COLS = [
    "google_pmax_spend",
    "google_paid_search_spend",
    "google_display_spend",
    "google_video_spend",
]

META_SPEND_COLS = [
    "meta_facebook_spend",
    "meta_instagram_spend",
    "meta_other_spend",
]

PAID_CHANNEL_COLS = GOOGLE_SPEND_COLS + META_SPEND_COLS

# Non-paid traffic channels used as control variables
CONTROL_COLS = [
    "direct_clicks",
    "branded_search_clicks",
    "organic_search_clicks",
    "email_clicks",
    "referral_clicks",
    "all_other_clicks",
]

# Columns that represent the target KPI
TARGET_COL = "all_purchases"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_timeseries(
    path: str,
    timeseries_id: str = TIMESERIES_ID,
) -> pd.DataFrame:
    """Load the CSV and return only the rows for the given timeseries_id.

    The date column is parsed and the DataFrame is sorted chronologically.
    Column names are lowercased.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df["date_day"] = pd.to_datetime(df["date_day"])
    df = (
        df[df["mmm_timeseries_id"] == timeseries_id]
        .sort_values("date_day")
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def fill_missing_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN spend/clicks/impressions with 0.

    A missing value means the channel was not active on that day, not that
    the measurement is unknown.  Zero is the correct imputation.
    """
    spend_like_cols = [
        c for c in df.columns
        if any(c.endswith(s) for s in ("_spend", "_clicks", "_impressions"))
    ]
    df = df.copy()
    df[spend_like_cols] = df[spend_like_cols].fillna(0.0)
    return df


def fill_missing_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill small gaps in non-paid traffic columns, then fill remaining NaN with 0."""
    df = df.copy()
    existing_controls = [c for c in CONTROL_COLS if c in df.columns]
    df[existing_controls] = (
        df[existing_controls]
        .ffill()
        .fillna(0.0)
    )
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def aggregate_platform_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Create total_google_spend and total_meta_spend from individual channel cols."""
    df = df.copy()
    existing_google = [c for c in GOOGLE_SPEND_COLS if c in df.columns]
    existing_meta = [c for c in META_SPEND_COLS if c in df.columns]
    df["total_google_spend"] = df[existing_google].sum(axis=1)
    df["total_meta_spend"] = df[existing_meta].sum(axis=1)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features derived from date_day."""
    df = df.copy()
    df["day_of_week"] = df["date_day"].dt.dayofweek          # 0=Mon … 6=Sun
    df["week_of_year"] = df["date_day"].dt.isocalendar().week.astype(int)
    df["month"] = df["date_day"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_holiday_flags(df: pd.DataFrame, country: str = "US") -> pd.DataFrame:
    """Add a binary is_holiday column for public holidays in the given country."""
    df = df.copy()
    years = df["date_day"].dt.year.unique().tolist()
    country_holidays = holidays.country_holidays(country, years=years)
    # NOTE: compare on datetime.date — `holidays` is keyed by date objects,
    # so matching Timestamps directly silently yields all-zeros.
    df["is_holiday"] = df["date_day"].dt.date.isin(country_holidays).astype(int)
    return df


def log_scale_spend(
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply log1p to spend columns to reduce skew and improve model conditioning.

    A new column ``<col>_log`` is created for each transformed column so the
    originals are preserved for interpretability.
    """
    df = df.copy()
    if cols is None:
        cols = ["total_google_spend", "total_meta_spend"]
    for col in cols:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col])
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    log_spend: bool = True,
) -> pd.DataFrame:
    """Apply the full preprocessing pipeline in the correct order.

    Steps:
      1. fill_missing_spend
      2. fill_missing_controls
      3. aggregate_platform_spend
      4. add_temporal_features
      5. add_holiday_flags
      6. log_scale_spend  (if log_spend=True)

    Returns a clean DataFrame ready for modelling.
    """
    df = fill_missing_spend(df)
    df = fill_missing_controls(df)
    df = aggregate_platform_spend(df)
    df = add_temporal_features(df)
    df = add_holiday_flags(df)
    if log_spend:
        df = log_scale_spend(df)
    return df


# ---------------------------------------------------------------------------
# Train / holdout split
# ---------------------------------------------------------------------------

def train_holdout_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically into train and holdout sets.

    Default 80/20 → ~407 train rows, ~102 holdout rows.
    """
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
