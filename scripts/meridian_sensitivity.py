"""
scripts/meridian_sensitivity.py
-------------------------------
The headline Meridian fit ranks Google above Meta; the PyMC fit ranked Meta
3.3x above Google.  This script refits Meridian under the two specification
differences that could produce that reversal, so the cause can be attributed:

  spend-exec  : media execution measured in spend, as PyMC did (isolates the
                impressions-vs-spend difference; Meta's CPM is ~2x Google's)
  few-knots   : a stiffer baseline (8 knots instead of 20), to check the
                result is not an artefact of the flexible time-varying
                baseline soaking up media signal / the divergences

Writes data/mmm_meridian_sensitivity.csv.

    .venv-meridian/Scripts/python.exe scripts/meridian_sensitivity.py
"""

import os
import sys
import time
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmm.meridian_model import (
    MERIDIAN_CHANNELS,
    build_meridian_frame,
    load_input_data,
    build_model,
)
from run_meridian import N_TRAIN, scalar  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "mmm_data_clean.csv")
OUT = os.path.join(ROOT, "data", "mmm_meridian_sensitivity.csv")

VARIANTS = [
    # name, execution variable, knots
    ("spend-exec", "spend", 20),
    ("few-knots", "impressions", 8),
]


def run(name, execution, knots, df):
    import arviz as az
    from meridian.analysis import analyzer

    print(f"\n=== {name}: execution={execution} knots={knots} ===")
    frame = build_meridian_frame(df, execution=execution)
    data = load_input_data(frame)
    holdout = np.zeros(len(frame), dtype=bool)
    holdout[N_TRAIN:] = True
    mmm = build_model(data, holdout_id=holdout, knots=knots)

    mmm.sample_prior(500, seed=42)
    t0 = time.time()
    mmm.sample_posterior(n_chains=4, n_adapt=2000, n_burnin=500, n_keep=500, seed=42)
    print(f"  sampled in {time.time() - t0:,.0f}s")

    idata = mmm.inference_data
    rhat = az.rhat(idata, var_names=["roi_m", "alpha_m", "ec_m", "sigma"])
    max_rhat = float(max(np.nanmax(v.values) for v in rhat.data_vars.values()))
    div = (
        int(np.sum(idata.trace.diverging.values))
        if "trace" in idata and "diverging" in idata.trace
        else 0
    )

    an = analyzer.Analyzer(mmm)
    times = list(frame["time"])
    eva = an.expected_vs_actual_data(
        aggregate_geos=True, aggregate_times=False, use_kpi=True
    )
    pred = np.asarray(eva.expected.sel(metric="mean").values).squeeze()[N_TRAIN:]
    act = np.asarray(eva.actual.values).squeeze()[N_TRAIN:]
    r2 = float(1 - np.sum((act - pred) ** 2) / np.sum((act - act.mean()) ** 2))
    wape = float(np.sum(np.abs(act - pred)) / np.sum(np.abs(act)) * 100)

    sm = an.summary_metrics(selected_times=times[:N_TRAIN], aggregate_geos=True)
    post = sm.sel(distribution="posterior")
    train_purch = float(df.iloc[:N_TRAIN]["all_purchases"].sum())

    rows = []
    ppd = {}
    for ch in MERIDIAN_CHANNELS:
        c = post.sel(channel=ch)
        spend = scalar(c.spend, "mean")
        ppd[ch] = 1.0 / scalar(c.cpik, "mean")
        attr = ppd[ch] * spend
        rows.append(
            dict(
                variant=name,
                channel=ch,
                spend=round(spend, 0),
                purch_per_dollar=round(ppd[ch], 5),
                attr_purchases=round(attr, 1),
                share_pct=round(attr / train_purch * 100, 2),
                roi_revenue=round(scalar(c.roi, "mean"), 3),
                holdout_r2=round(r2, 4),
                holdout_wape=round(wape, 2),
                max_rhat=round(max_rhat, 4),
                divergences=div,
            )
        )
        print(
            f"  {ch}: purch/$={ppd[ch]:.5f} share={attr / train_purch * 100:.2f}%"
        )
    winner = max(ppd, key=ppd.get)
    print(
        f"  -> more efficient: {winner} ({ppd[winner] / min(ppd.values()):.2f}x)  "
        f"holdout R2={r2:.3f} rhat={max_rhat:.4f} div={div}"
    )
    return rows


def main():
    df = pd.read_csv(CLEAN, parse_dates=["date_day"]).sort_values("date_day")
    all_rows = []
    for name, execution, knots in VARIANTS:
        all_rows += run(name, execution, knots, df)
    pd.DataFrame(all_rows).to_csv(OUT, index=False)
    print(f"\n[done] -> {OUT}")


if __name__ == "__main__":
    main()
