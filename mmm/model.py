"""
mmm/model.py
------------
PyMC-Marketing MMM model definition.

Exports
-------
CHANNEL_COLS        : default paid-media feature columns
MODEL_CONTROL_COLS  : default control variable columns
prepare_model_data  : standardize controls, return (X, y, scaler)
build_mmm           : create an unfitted MMM instance

Usage (from a notebook):
    from mmm.model import build_mmm, prepare_model_data, CHANNEL_COLS, MODEL_CONTROL_COLS

    X_train, y_train, scaler   = prepare_model_data(train)
    X_holdout, y_holdout, _    = prepare_model_data(holdout, scaler=scaler)
    active_controls = [c for c in MODEL_CONTROL_COLS if c in X_train.columns]

    mmm = build_mmm(control_columns=active_controls)
    idata = mmm.fit(X=X_train, y=y_train, draws=1000, tune=500, chains=2,
                    target_accept=0.90, random_seed=42)
"""

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

# ---------------------------------------------------------------------------
# Column sets
# ---------------------------------------------------------------------------

# Max-normalized platform spend in [0, 1] — normalizing makes saturation_lam
# identifiable from data. Log-scaled inputs (values 5–8) cause lam to be
# unidentified because logistic_saturation(6, lam>0.3) ≈ 1 for any lam.
CHANNEL_COLS = [
    "total_google_spend_norm",
    "total_meta_spend_norm",
]

# Raw channel spend columns used for normalization
_CHANNEL_COLS_RAW = [
    "total_google_spend",
    "total_meta_spend",
]

# Control columns after standardization (scaled suffix) + binary flags
MODEL_CONTROL_COLS = [
    "direct_clicks_scaled",
    "branded_search_clicks_scaled",
    "organic_search_clicks_scaled",
    "email_clicks_scaled",
    "referral_clicks_scaled",
    "all_other_clicks_scaled",
    "is_weekend",
    "is_holiday",
]

# Raw traffic column names — standardized inside prepare_model_data()
_TRAFFIC_COLS_RAW = [
    "direct_clicks",
    "branded_search_clicks",
    "organic_search_clicks",
    "email_clicks",
    "referral_clicks",
    "all_other_clicks",
]


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_model_data(df, scaler=None):
    """Standardize continuous control columns and return (X, y, scaler).

    Parameters
    ----------
    df     : pd.DataFrame
        Clean DataFrame produced by ``preprocessing.prepare_features()``.
    scaler : dict or None
        Pass ``None`` for training data — scalers are fitted and returned.
        Pass the training scaler dict when preparing the holdout set to
        prevent data leakage.  Shape: {"traffic": StandardScaler, "channel_max": dict}

    Returns
    -------
    X      : pd.DataFrame   feature matrix (date + channels + controls)
    y      : pd.Series      target variable (all_purchases)
    scaler : dict           fitted scalers
    """
    from sklearn.preprocessing import StandardScaler

    df = df.copy()

    # ── Traffic controls: StandardScaler ─────────────────────────────────
    existing_raw = [c for c in _TRAFFIC_COLS_RAW if c in df.columns]
    if scaler is None:
        traffic_scaler = StandardScaler()
        df[[f"{c}_scaled" for c in existing_raw]] = traffic_scaler.fit_transform(
            df[existing_raw]
        )
        # Channel spend: max-normalize to [0, 1] using training-set max
        channel_max = {}
        for raw_col in _CHANNEL_COLS_RAW:
            if raw_col in df.columns:
                col_max = df[raw_col].max()
                channel_max[raw_col] = col_max if col_max > 0 else 1.0
                norm_col = raw_col.replace("total_", "total_").replace("_spend", "_spend_norm")
                df[norm_col] = df[raw_col] / channel_max[raw_col]
        scaler = {"traffic": traffic_scaler, "channel_max": channel_max}
    else:
        df[[f"{c}_scaled" for c in existing_raw]] = scaler["traffic"].transform(
            df[existing_raw]
        )
        for raw_col, col_max in scaler["channel_max"].items():
            if raw_col in df.columns:
                norm_col = raw_col.replace("total_", "total_").replace("_spend", "_spend_norm")
                df[norm_col] = df[raw_col] / col_max

    active_controls = [c for c in MODEL_CONTROL_COLS if c in df.columns]
    feature_cols = ["date_day"] + CHANNEL_COLS + active_controls
    X = df[feature_cols].copy()
    y = df["all_purchases"].copy()
    return X, y, scaler


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_mmm(
    channel_columns=None,
    control_columns=None,
    date_column="date_day",
    adstock_max_lag=14,
    yearly_seasonality=1,
):
    """Create and return an unfitted PyMC-Marketing MMM instance.

    Parameters
    ----------
    channel_columns   : list[str] or None
        Paid-media feature columns (log-scaled spend totals).
        Defaults to CHANNEL_COLS.
    control_columns   : list[str] or None
        Control variable columns (standardized traffic + binary flags).
        Pass the active subset filtered from MODEL_CONTROL_COLS.
    date_column       : str
        Name of the date column in X (default: "date_day").
    adstock_max_lag   : int
        Maximum carry-over window in days (default: 14).
    yearly_seasonality: int
        Number of yearly Fourier pairs.  1 → sin + cos (2 terms).
        Keep <= 2 for datasets shorter than 2 years.
    """
    if channel_columns is None:
        channel_columns = CHANNEL_COLS

    adstock = GeometricAdstock(l_max=adstock_max_lag)
    saturation = LogisticSaturation()

    kwargs = dict(
        channel_columns=channel_columns,
        date_column=date_column,
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=yearly_seasonality,
    )
    if control_columns:
        kwargs["control_columns"] = control_columns

    return MMM(**kwargs)
