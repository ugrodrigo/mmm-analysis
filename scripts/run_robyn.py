"""
scripts/run_robyn.py
--------------------
Fit Meta Robyn on the same data and the same 407-day train window as the
PyMC-Marketing and Meridian models, and write data/mmm_robyn_results.csv.

Requires R + glmnet (reached via rpy2).

    .venv-robyn/Scripts/python.exe scripts/run_robyn.py            # full run
    .venv-robyn/Scripts/python.exe scripts/run_robyn.py --smoke    # tiny run, API check
"""

import os
import sys
import json
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmm.robyn_model import (  # noqa: E402
    ROBYN_CHANNELS,
    TRAIN_START,
    TRAIN_END,
    setup_r_env,
    build_robyn_frame,
    build_spec,
)

R_HOME = setup_r_env()  # must run before importing robyn

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "mmm_data_clean.csv")
OUT = os.path.join(ROOT, "data", "mmm_robyn_results.csv")
WORKDIR = os.path.join(ROOT, "data", "robyn_out")

PRETTY = {"total_google_spend": "Google", "total_meta_spend": "Meta"}
ASP = 182.02  # train-window average selling price, USD per purchase


def main(smoke=False):
    print(f"[env] R_HOME={R_HOME}")
    df = pd.read_csv(CLEAN, parse_dates=["date_day"]).sort_values("date_day")
    frame = build_robyn_frame(df)
    print(f"[data] {len(frame)} days; window {TRAIN_START} -> {TRAIN_END}")

    from robyn.robyn import Robyn
    from robyn.modeling.entities.modelrun_trials_config import TrialsConfig

    mmm_data, holidays, hyper = build_spec(frame)

    r = Robyn(working_dir=WORKDIR, console_log_level="WARNING")
    r.initialize(mmm_data=mmm_data, holidays_data=holidays, hyperparameters=hyper)
    r.feature_engineering(display_plots=False)

    trials, iters = (1, 20) if smoke else (5, 400)
    print(f"[fit] Nevergrad search: {trials} trials x {iters} iterations")
    t0 = time.time()
    r.train_models(
        trials_config=TrialsConfig(trials=trials, iterations=iters),
        ts_validation=False,
        rssd_zero_penalty=True,
        cores=1,  # rpy2 is not fork-safe; keep the R bridge single-threaded
        display_plots=False,
        export_plots=False,
    )
    print(f"[fit] done in {time.time() - t0:,.0f}s")

    # NOTE: r.evaluate_models() is not usable in robynpy 0.3.6 -- after Pareto
    # optimisation it calls plot_data_generator.robyn_immcarr(), which raises
    # KeyError: 'dep_var'. That is a plotting-data bug, not a modelling one, so
    # the Pareto stages are run directly here and the plot step is skipped.
    from robyn.modeling.pareto.pareto_optimizer import ParetoOptimizer

    po = ParetoOptimizer(
        mmm_data=r.mmm_data,
        model_outputs=r.model_outputs,
        hyperparameter=r.hyperparameters,
        featurized_mmm_data=r.featurized_mmm_data,
        holidays_data=r.holidays_data,
    )
    agg = po.data_aggregator.aggregate_model_data(False)
    agg["result_hyp_param"] = po._compute_pareto_fronts(agg, "auto", 0.1)
    pdata = po.prepare_pareto_data(agg, "auto", 100, False)
    # Response curves populate spend_share / effect_share / roi_total / cpa_total
    # on decomp_spend_dist. This is the last stage before the plotting bug.
    pdata = po.response_curve_calculator.compute_response_curves(pdata, agg)

    decomp = pdata.decomp_spend_dist
    hyp = pdata.result_hyp_param
    sol_col = "sol_id" if "sol_id" in decomp.columns else "solID"

    if smoke:
        print(f"\n[smoke] decomp_spend_dist cols: {list(decomp.columns)}")
        print(f"[smoke] result_hyp_param cols: {list(hyp.columns)}")
        print(f"[smoke] shapes: {decomp.shape} {hyp.shape}")
        return

    sols = decomp[sol_col].unique()
    print(f"[pareto] {len(sols)} Pareto-optimal solutions")

    # Robyn returns a Pareto front, not one model. Report the median across
    # solutions plus the spread, since picking one is a human judgement call.
    rows, summary = [], {}
    train_purch = float(
        df[(df.date_day >= TRAIN_START) & (df.date_day <= TRAIN_END)]["all_purchases"].sum()
    )

    # With dep_var_type=CONVERSION Robyn reports cost-per-acquisition rather
    # than ROI, so purchases-per-dollar comes from 1/cpa_total in that case.
    eff_col = "effect_share" if "effect_share" in decomp.columns else "xDecompPerc"
    roi_col = next(
        (c for c in ("roi_total", "roi_mean", "cpa_total") if c in decomp.columns), None
    )
    if roi_col is None:
        raise RuntimeError(f"no ROI/CPA column in x_decomp_agg: {list(decomp.columns)}")
    invert = roi_col.startswith("cpa")
    print(f"[cols] effect={eff_col} efficiency={roi_col}{' (inverted)' if invert else ''}")

    per_sol = {}
    for ch in ROBYN_CHANNELS:
        sub = decomp[decomp["rn"] == ch]
        eff_share = sub.groupby(sol_col)[eff_col].first().astype(float)
        eff = sub.groupby(sol_col)[roi_col].first().astype(float)
        roas = (1.0 / eff.replace(0, np.nan)) if invert else eff
        per_sol[ch] = dict(eff=eff_share, roas=roas)

    spend_tot = {
        ch: float(
            df[(df.date_day >= TRAIN_START) & (df.date_day <= TRAIN_END)][ch].sum()
        )
        for ch in ROBYN_CHANNELS
    }
    spend_all = sum(spend_tot.values())

    for ch in ROBYN_CHANNELS:
        name = PRETTY[ch]
        eff = per_sol[ch]["eff"]
        roas = per_sol[ch]["roas"].dropna()
        eff_med = float(eff.median()) * 100
        roas_med = float(roas.median())
        attr_purch = eff_med / 100 * train_purch
        spend_share = spend_tot[ch] / spend_all * 100
        summary[name] = dict(
            spend=spend_tot[ch],
            spend_share_pct=spend_share,
            effect_share_pct=eff_med,
            effect_share_range=[float(eff.min()) * 100, float(eff.max()) * 100],
            purch_per_dollar=roas_med,
            revenue_roas=roas_med * ASP,
            roas_range=[float(roas.min()), float(roas.max())],
            attr_purchases=attr_purch,
            n_solutions=int(len(eff)),
        )
        rows += [
            ("Attribution", f"{name} purchases (attr.)", round(attr_purch, 1)),
            ("Attribution", f"{name} share (%)", round(eff_med, 2)),
            ("Attribution", f"{name} spend share (%)", round(spend_share, 2)),
            ("Attribution", f"{name} ROAS (purch/$)", round(roas_med, 5)),
            ("Attribution", f"{name} ROI (revenue $/$)", round(roas_med * ASP, 3)),
        ]
        print(
            f"[attr] {name}: effect share {eff_med:.2f}% (spend share {spend_share:.2f}%)  "
            f"purch/$={roas_med:.5f}  revenue ROAS={roas_med * ASP:.2f}x"
        )

    for metric, label in (("rsq_train", "Train R2"), ("nrmse", "NRMSE"), ("decomp.rssd", "DECOMP.RSSD")):
        if metric in hyp.columns:
            v = float(pd.to_numeric(hyp[metric], errors="coerce").median())
            rows.insert(0, ("Model Quality", label, round(v, 4)))
            print(f"[fit] median {label}: {v:.4f}")
    rows.insert(0, ("Model Quality", "Pareto solutions", len(sols)))

    pd.DataFrame(rows, columns=["section", "metric", "value"]).to_csv(OUT, index=False)
    with open(os.path.join(ROOT, "data", "mmm_robyn_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
