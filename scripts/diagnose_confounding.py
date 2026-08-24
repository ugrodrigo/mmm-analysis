"""
scripts/diagnose_confounding.py
-------------------------------
Cheap diagnostics that motivate the proposed test plan (see proposed-tests.md).

Neither model can identify the Google-vs-Meta ranking.  This script checks the
data for the specific mechanisms that would explain why, so the test plan is
grounded in measured numbers rather than speculation:

  1. Meta's on/off flighting  — is "Meta on" confounded with promo or seasonality?
  2. Traffic controls         — are they post-treatment mediators of paid media?
  3. Repeat purchases         — how much of the KPI cannot respond to ads?
  4. Noise floor              — how much does weekly aggregation help?
  5. Channel collinearity     — can Google and Meta be separated at all?

    python scripts/diagnose_confounding.py
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(ROOT, "data", "mmm_data_clean.csv")
N_TRAIN = 407

TRAFFIC = [
    "direct_clicks",
    "branded_search_clicks",
    "organic_search_clicks",
    "email_clicks",
    "referral_clicks",
    "all_other_clicks",
]


def rule(title):
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main():
    df = pd.read_csv(CLEAN, parse_dates=["date_day"]).sort_values("date_day")
    df["discount_rate"] = (
        df["all_purchases_gross_discount"] / df["all_purchases_original_price"]
    )
    tr = df.iloc[:N_TRAIN].copy()
    tr["meta_on"] = tr["total_meta_spend"] > 0

    # -- 1. Is "Meta on" confounded? --------------------------------------
    rule("1. Meta flighting: what else differs between on and off days?")
    on, off = tr[tr.meta_on], tr[~tr.meta_on]
    print(f"   Meta ON: {len(on)} days   Meta OFF: {len(off)} days")
    for col, label in [
        ("all_purchases", "purchases/day"),
        ("discount_rate", "discount rate"),
        ("total_google_spend", "Google spend/day"),
        ("direct_clicks", "direct clicks/day"),
        ("organic_search_clicks", "organic clicks/day"),
        ("email_clicks", "email clicks/day"),
    ]:
        a, b = on[col].mean(), off[col].mean()
        lift = (a / b - 1) * 100 if b else float("nan")
        print(f"   {label:22s} ON {a:10.3f}   OFF {b:10.3f}   {lift:+7.1f}%")

    # naive difference vs difference after removing high-discount days
    naive = (on.all_purchases.mean() / off.all_purchases.mean() - 1) * 100
    med = tr.discount_rate.median()
    lo = tr[tr.discount_rate <= med]
    lo_on, lo_off = lo[lo.meta_on], lo[~lo.meta_on]
    adj = (lo_on.all_purchases.mean() / lo_off.all_purchases.mean() - 1) * 100
    print(f"\n   Naive Meta-on purchase lift          : {naive:+.1f}%")
    print(f"   Same, restricted to low-discount days: {adj:+.1f}%")
    print(f"   -> {naive - adj:+.1f}pp of the raw Meta signal co-moves with promo depth")

    # how bursty is the flighting
    runs = (tr.meta_on != tr.meta_on.shift()).cumsum()
    on_runs = tr[tr.meta_on].groupby(runs).size()
    print(
        f"\n   Meta on-periods: {len(on_runs)} bursts, "
        f"median {on_runs.median():.0f} days, max {on_runs.max():.0f} days"
    )

    # -- 2. Are the traffic controls mediators? ---------------------------
    rule("2. Traffic controls vs paid media (post-treatment risk)")
    print("   Correlation of each control with same-day paid spend:")
    print(f"   {'control':24s} {'Google':>9s} {'Meta':>9s}")
    for c in TRAFFIC:
        g = tr[c].corr(tr.total_google_spend)
        m = tr[c].corr(tr.total_meta_spend)
        flag = "  <-- mediator risk" if max(abs(g), abs(m)) > 0.4 else ""
        print(f"   {c:24s} {g:9.3f} {m:9.3f}{flag}")
    print(
        "\n   A control that responds to advertising is a mediator: conditioning on"
        "\n   it removes the very effect the model is trying to measure."
    )

    # -- 3. How much of the KPI can advertising move? ---------------------
    rule("3. KPI composition")
    first, allp = tr.first_purchases.sum(), tr.all_purchases.sum()
    print(f"   first_purchases : {first:>8,.0f}  ({first / allp * 100:.1f}% of all)")
    print(f"   repeat (implied): {allp - first:>8,.0f}  ({(1 - first / allp) * 100:.1f}%)")
    print(
        f"\n   Correlation with Google spend  — all: {tr.all_purchases.corr(tr.total_google_spend):.3f}"
        f"   first: {tr.first_purchases.corr(tr.total_google_spend):.3f}"
    )
    print(
        f"   Correlation with Meta spend    — all: {tr.all_purchases.corr(tr.total_meta_spend):.3f}"
        f"   first: {tr.first_purchases.corr(tr.total_meta_spend):.3f}"
    )

    # -- 4. Noise floor: daily vs weekly ----------------------------------
    rule("4. Noise floor — daily vs weekly aggregation")
    wk = (
        tr.set_index("date_day")
        .resample("W")[["all_purchases", "total_google_spend", "total_meta_spend"]]
        .sum()
    )
    d_cv = tr.all_purchases.std() / tr.all_purchases.mean()
    w_cv = wk.all_purchases.std() / wk.all_purchases.mean()
    print(f"   Daily  KPI coefficient of variation: {d_cv:.3f}  (n={len(tr)})")
    print(f"   Weekly KPI coefficient of variation: {w_cv:.3f}  (n={len(wk)})")
    print(f"   -> weekly cuts relative noise {(1 - w_cv / d_cv) * 100:.0f}%")
    print(f"   Weekly Meta zero-spend weeks: {(wk.total_meta_spend == 0).sum()} of {len(wk)}")
    print(
        f"   Weekly corr(purchases, Meta spend): {wk.all_purchases.corr(wk.total_meta_spend):.3f}"
        f"   Google: {wk.all_purchases.corr(wk.total_google_spend):.3f}"
    )

    # -- 5. Can the two channels be separated? ----------------------------
    rule("5. Channel separability")
    r = tr.total_google_spend.corr(tr.total_meta_spend)
    print(f"   corr(Google spend, Meta spend), daily : {r:.3f}")
    print(f"   corr(Google spend, Meta spend), weekly: {wk.total_google_spend.corr(wk.total_meta_spend):.3f}")
    print(f"   Variance inflation from collinearity  : {1 / (1 - r**2):.2f}x")
    print(
        f"\n   Google spend range: ${tr.total_google_spend.min():,.0f}-${tr.total_google_spend.max():,.0f}/day"
        f"  (CV {tr.total_google_spend.std() / tr.total_google_spend.mean():.2f})"
    )
    print(
        f"   Meta spend range  : ${tr.total_meta_spend.min():,.0f}-${tr.total_meta_spend.max():,.0f}/day"
        f"  (CV {tr.total_meta_spend.std() / tr.total_meta_spend.mean():.2f})"
    )


if __name__ == "__main__":
    main()
