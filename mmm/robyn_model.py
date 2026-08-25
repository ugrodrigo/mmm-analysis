"""
mmm/robyn_model.py
------------------
Meta Robyn counterpart to ``mmm/model.py`` (PyMC-Marketing) and
``mmm/meridian_model.py`` (Google Meridian).

Robyn is not Bayesian.  It fits ridge regression via R's glmnet (reached through
rpy2) and tunes the adstock/saturation hyperparameters with Nevergrad, optimising
two objectives at once:

  * NRMSE          — prediction error
  * DECOMP.RSSD    — distance between each channel's effect share and its
                     spend share

That second objective is the reason Robyn is worth running here: it encodes a
structural assumption (effect should look like spend) in a completely different
place from PyMC (prior on coefficients) or Meridian (prior on ROI).

Requires R with the glmnet package.  Call :func:`setup_r_env` before importing
anything from ``robyn``.

Exports
-------
ROBYN_CHANNELS      : spend column names used as paid media
setup_r_env         : point rpy2 at the local R installation
build_robyn_frame   : clean CSV -> flat DataFrame in Robyn's layout
build_spec          : MMMData + Hyperparameters for the shared train window
"""

from __future__ import annotations

import os
import glob

import pandas as pd

# Robyn takes spend columns directly as the media variables.  The PyMC model
# used spend as its execution variable too, so this matches it; the Meridian
# sensitivity run showed the impressions-vs-spend choice does not flip the
# channel ranking either way.
ROBYN_CHANNELS = ["total_google_spend", "total_meta_spend"]

# Same controls as the other two models.  ``is_holiday`` is excluded: it is
# constant (all zeros) in the current clean CSV and a zero-variance regressor
# breaks the ridge fit.
ROBYN_CONTEXT = [
    "direct_clicks",
    "branded_search_clicks",
    "organic_search_clicks",
    "email_clicks",
    "referral_clicks",
    "all_other_clicks",
    "is_weekend",
]

TARGET_COL = "all_purchases"
DATE_COL = "date_day"

# Shared train window — the 407-day split used by both other models.
TRAIN_START = "2022-07-29"
TRAIN_END = "2023-09-08"


def setup_r_env(r_home: str | None = None) -> str:
    """Point rpy2 at the local R install.  Must run before importing ``robyn``.

    On Windows rpy2 needs R_HOME set, R's binary directory on PATH so package
    DLLs resolve their dependencies, and R_LIBS_USER so per-user packages
    (glmnet) are findable.
    """
    if r_home is None:
        r_home = os.environ.get("R_HOME")
    if r_home is None:
        candidates = sorted(glob.glob(r"C:\Program Files\R\R-*"), reverse=True)
        if not candidates:
            raise RuntimeError(
                "R not found. Install it (winget install --id RProject.R) or pass r_home."
            )
        r_home = candidates[0]

    os.environ["R_HOME"] = r_home
    for sub in ("bin\\x64", "bin"):
        d = os.path.join(r_home, sub)
        if os.path.isdir(d):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            break

    if "R_LIBS_USER" not in os.environ:
        ver = os.path.basename(r_home).replace("R-", "")
        major_minor = ".".join(ver.split(".")[:2])
        user_lib = os.path.join(
            os.environ.get("LOCALAPPDATA", ""), "R", "win-library", major_minor
        )
        if os.path.isdir(user_lib):
            os.environ["R_LIBS_USER"] = user_lib

    return r_home


def build_robyn_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Flat daily frame in the layout Robyn expects: date, KPI, spend, controls."""
    cols = [DATE_COL, TARGET_COL] + ROBYN_CHANNELS + [
        c for c in ROBYN_CONTEXT if c in df.columns
    ]
    out = df[cols].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    for c in cols[1:]:
        out[c] = out[c].astype(float)
    return out.reset_index(drop=True)


def build_spec(frame: pd.DataFrame):
    """Return (MMMData, HolidaysData, Hyperparameters) for the train window.

    The modelling window is fixed to the same 407 days the other two models
    trained on, so Robyn's decomposition covers exactly the same period as
    their attribution numbers.
    """
    from robyn.data.entities.mmmdata import MMMData
    from robyn.data.entities.holidays_data import HolidaysData
    from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
    from robyn.data.entities.enums import (
        DependentVarType,
        AdstockType,
        ProphetVariableType,
        ProphetSigns,
        PaidMediaSigns,
        ContextSigns,
    )

    context = [c for c in ROBYN_CONTEXT if c in frame.columns and frame[c].nunique() > 1]
    dropped = [c for c in ROBYN_CONTEXT if c in frame.columns and c not in context]
    if dropped:
        print(f"[data] dropping constant controls: {dropped}")

    spec = MMMData.MMMDataSpec(
        dep_var=TARGET_COL,
        dep_var_type=DependentVarType.CONVERSION,
        date_var=DATE_COL,
        window_start=pd.Timestamp(TRAIN_START),
        window_end=pd.Timestamp(TRAIN_END),
        paid_media_spends=ROBYN_CHANNELS,
        paid_media_vars=ROBYN_CHANNELS,
        paid_media_signs=[PaidMediaSigns.POSITIVE] * len(ROBYN_CHANNELS),
        context_vars=context,
        context_signs=[ContextSigns.DEFAULT] * len(context),
        # Must be [] not None: Robyn concatenates organic_vars into the
        # regressor list without a None guard.
        organic_vars=[],
        organic_signs=[],
        # Daily data. Robyn defaults to weekly, so both must be set explicitly.
        day_interval=1,
        interval_type="day",
    )
    mmm_data = MMMData(data=frame, mmmdata_spec=spec)

    holidays = HolidaysData(
        dt_holidays=_load_holidays(),
        prophet_vars=[ProphetVariableType.TREND, ProphetVariableType.SEASON],
        prophet_signs=[ProphetSigns.DEFAULT, ProphetSigns.DEFAULT],
        prophet_country="US",
    )

    # Geometric adstock. theta bounds are set for DAILY decay: PyMC's
    # Beta(1, 3) prior over a 14-day window implies a daily retention around
    # 0.25, so [0, 0.6] brackets it generously. alphas/gammas are Robyn's
    # documented Hill ranges.
    hyper = Hyperparameters(
        hyperparameters={
            ch: ChannelHyperparameters(
                thetas=[0.0, 0.6], alphas=[0.5, 3.0], gammas=[0.3, 1.0]
            )
            for ch in ROBYN_CHANNELS
        },
        adstock=AdstockType.GEOMETRIC,
        train_size=[0.5, 0.8],
    )
    return mmm_data, holidays, hyper


def _load_holidays() -> pd.DataFrame:
    """Robyn ships a Prophet holiday table; fall back to an empty US frame."""
    import robyn

    path = os.path.join(
        os.path.dirname(robyn.__file__), "tutorials", "resources", "dt_prophet_holidays.csv"
    )
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["ds", "holiday", "country", "year"])
