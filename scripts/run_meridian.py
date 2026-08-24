"""
scripts/run_meridian.py
-----------------------
Fit the Google Meridian MMM on the same data and the same train/holdout split
as the PyMC-Marketing model, and write results to data/mmm_meridian_results.csv.

    .venv-meridian/Scripts/python.exe scripts/run_meridian.py          # fit + analyse
    .venv-meridian/Scripts/python.exe scripts/run_meridian.py --reuse  # analyse saved fit
"""

import os
import sys
import json
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "mmm_data_clean.csv")
OUT = os.path.join(ROOT, "data", "mmm_meridian_results.csv")
MODEL_PATH = os.path.join(ROOT, "data", "mmm_meridian_model.pkl")
N_TRAIN = 407  # same split as the PyMC run (80/20 of 509 days)


def scalar(da, metric=None):
    """Read a scalar out of a summary_metrics DataArray.

    Deterministic quantities (spend, impressions) carry no ``metric`` dimension;
    posterior quantities do.  This tolerates both.
    """
    if metric is not None and "metric" in getattr(da, "dims", ()):
        da = da.sel(metric=metric)
    return float(np.asarray(da.values).squeeze())


def fit(frame, reuse=False):
    from meridian.model import model as mmodel

    if reuse and os.path.exists(MODEL_PATH):
        print(f"[fit] reusing {MODEL_PATH}")
        return mmodel.load_mmm(MODEL_PATH)

    data = load_input_data(frame)
    holdout = np.zeros(len(frame), dtype=bool)
    holdout[N_TRAIN:] = True
    mmm = build_model(data, holdout_id=holdout)

    print("[fit] sampling prior ...")
    mmm.sample_prior(500, seed=42)
    print("[fit] sampling posterior (4 chains x 500 draws, n_adapt=2000) ...")
    t0 = time.time()
    mmm.sample_posterior(n_chains=4, n_adapt=2000, n_burnin=500, n_keep=500, seed=42)
    print(f"[fit] done in {time.time() - t0:,.0f}s")
    mmodel.save_mmm(mmm, MODEL_PATH)
    print(f"[fit] saved -> {MODEL_PATH}")
    return mmm


def main(reuse=False):
    df = pd.read_csv(CLEAN, parse_dates=["date_day"]).sort_values("date_day")
    frame = build_meridian_frame(df)
    n_times = len(frame)
    print(f"[data] {n_times} days, train={N_TRAIN}, holdout={n_times - N_TRAIN}")

    mmm = fit(frame, reuse=reuse)

    import arviz as az
    from meridian.analysis import analyzer

    idata = mmm.inference_data
    rhat = az.rhat(idata, var_names=["roi_m", "alpha_m", "ec_m", "slope_m", "sigma"])
    max_rhat = float(max(np.nanmax(v.values) for v in rhat.data_vars.values()))
    div = (
        int(np.sum(idata.trace.diverging.values))
        if "trace" in idata and "diverging" in idata.trace
        else 0
    )
    print(f"[diag] max r-hat={max_rhat:.4f} divergences={div}")

    an = analyzer.Analyzer(mmm)
    times = list(frame["time"])
    train_times = times[:N_TRAIN]

    # ---- fit quality, in purchases (use_kpi=True) ------------------------
    eva = an.expected_vs_actual_data(
        aggregate_geos=True, aggregate_times=False, use_kpi=True
    )
    pred_all = np.asarray(eva.expected.sel(metric="mean").values).squeeze()
    act_all = np.asarray(eva.actual.values).squeeze()
    pred_k, act_k = pred_all[N_TRAIN:], act_all[N_TRAIN:]
    resid = act_k - pred_k
    mape = float(np.mean(np.abs(resid / act_k)) * 100)
    wape = float(np.sum(np.abs(resid)) / np.sum(np.abs(act_k)) * 100)
    r2 = float(1 - np.sum(resid**2) / np.sum((act_k - act_k.mean()) ** 2))
    tr_p, tr_a = pred_all[:N_TRAIN], act_all[:N_TRAIN]
    tr_r2 = float(1 - np.sum((tr_a - tr_p) ** 2) / np.sum((tr_a - tr_a.mean()) ** 2))
    print(f"[holdout] MAPE={mape:.1f}% WAPE={wape:.1f}% R2={r2:.3f} | train R2={tr_r2:.3f}")

    # ---- attribution over the training window ----------------------------
    sm = an.summary_metrics(selected_times=train_times, aggregate_geos=True)
    post = sm.sel(distribution="posterior")
    train_purch = float(df.iloc[:N_TRAIN]["all_purchases"].sum())

    rows = [
        ("Model Quality", "Holdout MAPE (%)", round(mape, 2)),
        ("Model Quality", "Holdout WAPE (%)", round(wape, 2)),
        ("Model Quality", "Holdout R2", round(r2, 4)),
        ("Model Quality", "Train R2", round(tr_r2, 4)),
        ("Model Quality", "R-hat (max)", round(max_rhat, 4)),
        ("Model Quality", "Divergences", div),
    ]

    summary = {}
    for ch in MERIDIAN_CHANNELS:
        c = post.sel(channel=ch)
        spend = scalar(c.spend, "mean")
        cpik = scalar(c.cpik, "mean")    # cost per incremental KPI
        roi_rev = scalar(c.roi, "mean")  # revenue ROAS
        purch_per_dollar = 1.0 / cpik
        attr_purch = purch_per_dollar * spend
        share = attr_purch / train_purch * 100
        summary[ch] = dict(
            spend=spend,
            cpik=cpik,
            roi_revenue=roi_rev,
            roi_ci=[scalar(c.roi, "ci_lo"), scalar(c.roi, "ci_hi")],
            purch_per_dollar=purch_per_dollar,
            attr_purchases=attr_purch,
            share_pct=share,
            pct_of_contribution=scalar(c.pct_of_contribution, "mean"),
        )
        rows += [
            ("Attribution", f"{ch} purchases (attr.)", round(attr_purch, 1)),
            ("Attribution", f"{ch} share (%)", round(share, 2)),
            ("Attribution", f"{ch} ROAS (purch/$)", round(purch_per_dollar, 5)),
            ("Attribution", f"{ch} ROI (revenue $/$)", round(roi_rev, 3)),
        ]
        print(
            f"[attr] {ch}: spend=${spend:,.0f} purch/$={purch_per_dollar:.5f} "
            f"ROI={roi_rev:.2f} share={share:.2f}%"
        )

    # ---- budget optimisation --------------------------------------------
    from meridian.analysis import optimizer

    opt = optimizer.BudgetOptimizer(mmm)
    # Meridian's default +/-30% band, then the same 10%-200%-of-current bounds
    # the PyMC run used, so the two optimisations are comparable.
    for label, lo, hi, tag in [
        ("default +/-30%", 0.3, 0.3, " [30%]"),
        ("PyMC bounds", 0.9, 1.0, ""),
    ]:
        try:
            res = opt.optimize(
                selected_times=(train_times[0], train_times[-1]),
                spend_constraint_lower=lo,
                spend_constraint_upper=hi,
                use_kpi=True,
            )
        except Exception as e:
            print(f"[opt/{label}] skipped: {type(e).__name__}: {e}")
            continue
        nonopt, optd = res.nonoptimized_data, res.optimized_data
        base = scalar(nonopt.incremental_outcome.sum(dim="channel"), "mean")
        new = scalar(optd.incremental_outcome.sum(dim="channel"), "mean")
        lift = (new / base - 1) * 100
        for ch in MERIDIAN_CHANNELS:
            cur = scalar(nonopt.spend.sel(channel=ch))
            new_s = scalar(optd.spend.sel(channel=ch))
            rows += [
                ("Optimisation", f"Current {ch} spend ($){tag}", round(cur, 0)),
                ("Optimisation", f"Optimal {ch} spend ($){tag}", round(new_s, 0)),
            ]
            summary.setdefault(ch, {}).update(
                {("cur_spend" + tag).strip(): cur, ("opt_spend" + tag).strip(): new_s}
            )
        rows.append(("Optimisation", f"Optimisation lift (%){tag}", round(lift, 2)))
        print(
            f"[opt/{label}] lift={lift:.2f}%  "
            + "  ".join(
                f"{ch}=${scalar(optd.spend.sel(channel=ch)):,.0f}"
                for ch in MERIDIAN_CHANNELS
            )
        )

    pd.DataFrame(rows, columns=["section", "metric", "value"]).to_csv(OUT, index=False)
    with open(os.path.join(ROOT, "data", "mmm_meridian_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main(reuse="--reuse" in sys.argv)
