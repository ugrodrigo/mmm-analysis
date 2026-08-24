"""
scripts/incrementality_power.py
-------------------------------
Sizing calculations for the incrementality test roadmap (proposed-tests.md).

Answers, from the observed data rather than rules of thumb:
  1. Which calendar window is stable enough to run a test in?
  2. How noisy is the KPI once trend and day-of-week are removed?
  3. What lift can a geo holdout of a given length and split actually detect?
  4. What does that minimum detectable effect mean in ROAS terms?

Assumptions are stated inline and printed with the results.  Final sizing needs
geo-level data (see test E0); until then these are national-series approximations.

    python scripts/incrementality_power.py
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "mmm_data_clean.csv")

ASP = 182.02          # USD revenue per purchase, training window
Z_ALPHA = 1.96        # two-sided 95%
Z_POWER = 0.84        # 80% power
# A matched-control difference-in-differences cancels demand shocks common to
# both cells.  Published geo-test benchmarks put the variance reduction at
# 40-60%; we report the conservative end and the optimistic end as a range.
DID_REDUCTION = (0.40, 0.60)


def rule(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def detrended_weekly_cv(s: pd.Series) -> float:
    """CV of weekly totals after removing a linear trend."""
    y = s.values.astype(float)
    x = np.arange(len(y))
    fit = np.polyval(np.polyfit(x, y, 1), x)
    resid = y - fit
    return float(resid.std(ddof=1) / y.mean())


def main():
    df = pd.read_csv(CLEAN, parse_dates=["date_day"]).sort_values("date_day")
    df["meta_on"] = df["total_meta_spend"] > 0

    # ---- 1. find the stable window -------------------------------------
    rule("1. Candidate test windows")
    df["month"] = df.date_day.dt.to_period("M")
    monthly = df.groupby("month").agg(
        purch=("all_purchases", "mean"),
        meta=("total_meta_spend", "sum"),
        google=("total_google_spend", "sum"),
    )
    overall = df.all_purchases.mean()
    print(f"   Overall mean purchases/day: {overall:.1f}\n")
    print(f"   {'month':10s} {'purch/day':>10s} {'vs mean':>9s} {'Meta $':>10s} {'Google $':>10s}")
    for m, r in monthly.iterrows():
        dev = (r.purch / overall - 1) * 100
        mark = "  <- stable" if abs(dev) < 20 else ""
        print(f"   {str(m):10s} {r.purch:10.1f} {dev:+8.0f}% {r.meta:10,.0f} {r.google:10,.0f}{mark}")

    stable = df[(df.date_day >= "2023-02-15") & (df.date_day <= "2023-08-16")]
    print(
        f"\n   Recommended window: 2023-02-15 -> 2023-08-16 "
        f"({len(stable)} days, {stable.all_purchases.mean():.1f} purch/day)"
    )
    print("   Rationale: Meta dark throughout, no Q4 distortion, flattest demand.")

    # ---- 2. noise floor -------------------------------------------------
    rule("2. Noise floor in the stable window")
    wk = stable.set_index("date_day").resample("W").all_purchases.sum()
    wk = wk.iloc[1:-1]  # drop partial weeks
    raw_cv = wk.std(ddof=1) / wk.mean()
    det_cv = detrended_weekly_cv(wk)
    print(f"   Weekly purchase totals: n={len(wk)}, mean={wk.mean():.0f}, sd={wk.std(ddof=1):.0f}")
    print(f"   Raw weekly CV            : {raw_cv:.3f}")
    print(f"   Detrended weekly CV      : {det_cv:.3f}   <- used for sizing")
    for lo_hi, label in zip(DID_REDUCTION, ("conservative", "optimistic")):
        print(f"   After DiD ({label:12s} {lo_hi:.0%} var. reduction): {det_cv * np.sqrt(1 - lo_hi):.3f}")

    # ---- 3. minimum detectable lift ------------------------------------
    rule("3. Minimum detectable lift — geo holdout, 50/50 split")
    print("   MDE = (z_a + z_b) * cv_eff * sqrt(2/n_weeks), on the treated cell's KPI")
    print("   (two-sided 95%, 80% power; both cells same size; weeks independent)\n")
    print(f"   {'weeks':>6s} " + "".join(f"{lbl:>22s}" for lbl in ("MDE conservative", "MDE optimistic")))
    mdes = {}
    for n_weeks in (4, 6, 8, 12):
        row = []
        for red in DID_REDUCTION:
            cv_eff = det_cv * np.sqrt(1 - red)
            mde = (Z_ALPHA + Z_POWER) * cv_eff * np.sqrt(2 / n_weeks)
            row.append(mde)
        mdes[n_weeks] = row
        print(f"   {n_weeks:6d} " + "".join(f"{m * 100:21.1f}%" for m in row))

    # ---- 4. translate to ROAS ------------------------------------------
    rule("4. What that means in ROAS terms — Meta geo holdout")
    meta_on = df[df.meta_on]
    meta_daily = meta_on.total_meta_spend.mean()
    print(f"   Meta spend when live: ${meta_daily:,.0f}/day national")
    print(f"   ASP: ${ASP:.2f}/purchase\n")
    print("   A 50/50 geo holdout withholds Meta from half the country.")
    print("   Detectable ROAS = (MDE * baseline purchases in cell * ASP) / spend withheld\n")

    daily_purch = stable.all_purchases.mean()
    print(f"   {'weeks':>6s} {'held-out $':>12s} {'cell purch':>12s} "
          f"{'ROAS floor (cons.)':>20s} {'ROAS floor (opt.)':>19s}")
    for n_weeks, (mde_c, mde_o) in mdes.items():
        days = n_weeks * 7
        spend_withheld = meta_daily * 0.5 * days
        cell_purch = daily_purch * 0.5 * days
        floors = [m * cell_purch * ASP / spend_withheld for m in (mde_c, mde_o)]
        print(
            f"   {n_weeks:6d} {spend_withheld:12,.0f} {cell_purch:12,.0f} "
            f"{floors[0]:19.2f}x {floors[1]:18.2f}x"
        )

    rule("5. Can the test settle the PyMC-vs-Meridian disagreement?")
    n_weeks = 6
    mde_c, mde_o = mdes[n_weeks]
    days = n_weeks * 7
    spend_withheld = meta_daily * 0.5 * days
    cell_purch = daily_purch * 0.5 * days
    print(f"   6-week, 50/50 design withholds ${spend_withheld:,.0f} of Meta spend.\n")
    for label, roas in (("PyMC estimate", 8.00), ("Meridian estimate", 0.86), ("Break-even", 1.00)):
        exp_purch = roas / ASP * spend_withheld
        lift = exp_purch / cell_purch
        det_c = "YES" if lift > mde_c else "no"
        det_o = "YES" if lift > mde_o else "no"
        print(
            f"   {label:20s} ROAS {roas:5.2f}x -> {exp_purch:7.0f} incremental purchases "
            f"= {lift * 100:5.1f}% lift  | detectable: {det_c}/{det_o}"
        )
    print(
        f"\n   MDE at 6 weeks: {mde_c * 100:.1f}% (conservative) / {mde_o * 100:.1f}% (optimistic)"
    )
    print("   -> The test cleanly separates the 8.0x hypothesis from break-even.")
    print("      Distinguishing 0.86x from 1.5x needs geo data and a longer run.")

    # ---- 6. required spend contrast ------------------------------------
    rule("6. Required design — what spend contrast resolves a decision-grade ROAS?")
    print("   A hold-out at current spend is underpowered: Meta is only")
    print(f"   ${meta_daily:,.0f}/day against a ${daily_purch * ASP:,.0f}/day revenue base,")
    print("   so removing it moves the KPI less than weekly noise does.\n")
    print("   Instead, SCALE UP the treatment cell. Required spend delta to detect a")
    print("   given true ROAS: delta >= MDE * cell_purchases * ASP / ROAS\n")

    for n_weeks in (6, 12):
        mde = mdes[n_weeks][0]  # conservative
        days = n_weeks * 7
        cell_purch = daily_purch * 0.5 * days
        cur_cell_spend = meta_daily * 0.5 * days
        print(f"   --- {n_weeks}-week test, 50/50 split, MDE {mde * 100:.1f}% ---")
        print(f"   Current Meta spend in the cell: ${cur_cell_spend:,.0f} (${cur_cell_spend / days:,.0f}/day)")
        print(f"   {'target ROAS':>12s} {'spend delta':>13s} {'x current':>10s} {'extra budget':>14s}")
        for roas in (1.5, 2.0, 3.0, 4.0):
            delta = mde * cell_purch * ASP / roas
            print(
                f"   {roas:11.1f}x {delta:13,.0f} {delta / cur_cell_spend:9.1f}x "
                f"{delta - cur_cell_spend:14,.0f}"
            )
        print()

    print("   Read: to learn whether Meta clears a 2x ROAS, the treatment cell must run")
    print("   at roughly 3-4x its current Meta budget for 6 weeks. That is the real price")
    print("   of a decision-grade answer -- roughly $15-20k of incremental spend, not the")
    print("   $7k of 'forgone' spend a naive holdout would suggest.")

    print(
        "\n   CAVEAT: national-series approximation. Real cell-level variance depends on"
        "\n   how many geos exist and how well they match. Re-run power in GeoLift or"
        "\n   Trimmed Match once DMA-level data is available (test E0)."
    )


if __name__ == "__main__":
    main()
