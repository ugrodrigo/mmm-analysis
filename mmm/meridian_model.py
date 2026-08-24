"""
mmm/meridian_model.py
---------------------
Google Meridian counterpart to ``mmm/model.py`` (PyMC-Marketing).

Same KPI, same two platform channels, same controls and same train/holdout
split as the PyMC run, so the two models can be compared directly.

Differences that are inherent to Meridian (and therefore deliberate):
  * media execution is measured in **impressions**, spend is used only for ROI
    (PyMC-Marketing used max-normalised spend as the execution variable);
  * saturation is **Hill**, not logistic;
  * the baseline is a **time-varying knot spline**, not Fourier seasonality;
  * priors are set on **ROI**, not on channel coefficients.

Exports
-------
MERIDIAN_CHANNELS   : channel names
build_meridian_frame: clean CSV -> flat DataFrame in Meridian's layout
load_input_data     : flat DataFrame -> meridian InputData
build_model         : InputData (+ holdout mask) -> unfitted Meridian model
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MERIDIAN_CHANNELS = ["Google", "Meta"]

# Sub-channels rolled up into the two platform totals (same split as
# preprocessing.GOOGLE_SPEND_COLS / META_SPEND_COLS).
_PLATFORM_PARTS = {
    "Google": ["google_pmax", "google_paid_search", "google_display", "google_video"],
    "Meta": ["meta_facebook", "meta_instagram", "meta_other"],
}

# Same controls as MODEL_CONTROL_COLS; Meridian standardises them internally,
# so the raw columns are passed through.
MERIDIAN_CONTROLS = [
    "direct_clicks",
    "branded_search_clicks",
    "organic_search_clicks",
    "email_clicks",
    "referral_clicks",
    "all_other_clicks",
    "is_weekend",
    "is_holiday",
]

TARGET_COL = "all_purchases"
REVENUE_COL = "all_purchases_original_price"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_meridian_frame(df: pd.DataFrame, execution: str = "impressions") -> pd.DataFrame:
    """Reshape the clean daily frame into the flat layout Meridian expects.

    One row per (geo, time).  This is a national model, so a single constant
    geo is used.  ``revenue_per_kpi`` is the daily average selling price: it
    converts the purchases KPI into revenue so Meridian's ROI is a normal
    revenue ROAS and its default ROI prior stays meaningful.

    ``execution`` selects the media execution variable: ``"impressions"``
    (Meridian's convention) or ``"spend"`` (what the PyMC model used).
    """
    if execution not in ("impressions", "spend"):
        raise ValueError(f"execution must be 'impressions' or 'spend', got {execution!r}")
    out = pd.DataFrame(
        {
            "geo": "national",
            "time": pd.to_datetime(df["date_day"]).dt.strftime("%Y-%m-%d"),
            "conversions": df[TARGET_COL].astype(float).values,
        }
    )

    for channel, parts in _PLATFORM_PARTS.items():
        for kind in ("impressions", "spend"):
            cols = [f"{p}_{kind}" for p in parts if f"{p}_{kind}" in df.columns]
            out[f"{kind}_{channel}"] = df[cols].fillna(0).sum(axis=1).astype(float).values

    # Media execution copy.  Meridian normally measures execution in
    # impressions; ``execution="spend"`` mirrors it from spend instead, which
    # is what the PyMC model used and makes the two directly comparable.
    for channel in _PLATFORM_PARTS:
        src = "impressions" if execution == "impressions" else "spend"
        out[f"exec_{channel}"] = out[f"{src}_{channel}"].values

    # Average selling price per day; fall back to the period mean on days with
    # no purchases so the series never contains 0 or NaN.
    asp = (df[REVENUE_COL] / df[TARGET_COL].replace(0, np.nan)).astype(float)
    out["revenue_per_conversion"] = asp.fillna(asp.mean()).values

    for c in MERIDIAN_CONTROLS:
        if c in df.columns:
            out[c] = df[c].astype(float).values

    return out


def load_input_data(frame: pd.DataFrame):
    """Build a Meridian ``InputData`` object from :func:`build_meridian_frame`."""
    from meridian.data import load

    # Meridian rejects controls with no time variation in a national model.
    # (``is_holiday`` is constant in the current clean CSV — see
    # preprocessing.add_holiday_flags.)
    controls = [
        c for c in MERIDIAN_CONTROLS
        if c in frame.columns and frame[c].nunique() > 1
    ]
    dropped = [c for c in MERIDIAN_CONTROLS if c in frame.columns and c not in controls]
    if dropped:
        print(f"[data] dropping constant controls: {dropped}")
    coord_to_columns = load.CoordToColumns(
        time="time",
        geo="geo",
        controls=controls,
        kpi="conversions",
        revenue_per_kpi="revenue_per_conversion",
        media=[f"exec_{c}" for c in MERIDIAN_CHANNELS],
        media_spend=[f"spend_{c}" for c in MERIDIAN_CHANNELS],
    )
    loader = load.DataFrameDataLoader(
        df=frame,
        kpi_type="non_revenue",
        coord_to_columns=coord_to_columns,
        media_to_channel={f"exec_{c}": c for c in MERIDIAN_CHANNELS},
        media_spend_to_channel={f"spend_{c}": c for c in MERIDIAN_CHANNELS},
    )
    return loader.load()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(input_data, holdout_id=None, max_lag: int = 14, knots: int | None = 20):
    """Create an unfitted Meridian model.

    max_lag=14 matches the PyMC geometric adstock window (14 days).
    ``knots`` controls the flexibility of the time-varying baseline; the
    default of ~20 over 509 daily observations is roughly one knot per month,
    which plays the role of PyMC's trend + yearly Fourier term.
    ``holdout_id`` is a boolean mask of time periods to exclude from the
    likelihood — Meridian's native equivalent of fitting on the train split.
    """
    from meridian.model import model, spec

    kwargs = {"max_lag": max_lag}
    if knots is not None:
        kwargs["knots"] = knots
    if holdout_id is not None:
        kwargs["holdout_id"] = np.asarray(holdout_id, dtype=bool)

    return model.Meridian(input_data=input_data, model_spec=spec.ModelSpec(**kwargs))
