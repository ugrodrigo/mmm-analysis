"""
scripts/compare_models.py
-------------------------
Join the PyMC-Marketing, Meridian and Robyn result files into one table.

    .venv-meridian/Scripts/python.exe scripts/compare_models.py
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ASP = 182.02  # train-period average selling price, USD per purchase

NA = float("nan")


def load(path):
    p = os.path.join(DATA, path)
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    return dict(zip(df["metric"], df["value"]))


def main():
    p = load("mmm_results_summary.csv")     # PyMC-Marketing
    m = load("mmm_meridian_results.csv")    # Meridian
    b = load("mmm_robyn_results.csv")       # Robyn

    def roas(d, ch, key="ROAS (purch/$)"):
        v = d.get(f"{ch} {key}")
        return v * ASP if v is not None else NA

    rows = [
        # --- fit quality -------------------------------------------------
        ("Model quality", "Holdout R²", p.get("Holdout R²", NA), m.get("Holdout R2", NA), NA),
        ("Model quality", "Holdout WAPE (%)", p.get("Holdout WAPE (%)", NA), m.get("Holdout WAPE (%)", NA), NA),
        ("Model quality", "Train R²", NA, m.get("Train R2", NA), b.get("Train R2", NA)),
        ("Model quality", "Max R-hat", 1.006, m.get("R-hat (max)", NA), NA),
        ("Model quality", "Divergences", 0, m.get("Divergences", NA), NA),
        ("Model quality", "Pareto solutions", NA, NA, b.get("Pareto solutions", NA)),

        # --- attribution -------------------------------------------------
        ("Attribution", "Google share (%)", p.get("Google share (%)", NA),
         m.get("Google share (%)", NA), b.get("Google share (%)", NA)),
        ("Attribution", "Meta share (%)", p.get("Meta share (%)", NA),
         m.get("Meta share (%)", NA), b.get("Meta share (%)", NA)),
        ("Attribution", "Total paid share (%)",
         round(p.get("Google share (%)", 0) + p.get("Meta share (%)", 0), 2),
         round(m.get("Google share (%)", 0) + m.get("Meta share (%)", 0), 2),
         round(b.get("Google share (%)", 0) + b.get("Meta share (%)", 0), 2)),

        # --- efficiency ---------------------------------------------------
        ("Efficiency", "Google ROAS (purch/$)", p.get("Google ROAS (purch/$)", NA),
         m.get("Google ROAS (purch/$)", NA), b.get("Google ROAS (purch/$)", NA)),
        ("Efficiency", "Meta ROAS (purch/$)", p.get("Meta ROAS (purch/$)", NA),
         m.get("Meta ROAS (purch/$)", NA), b.get("Meta ROAS (purch/$)", NA)),
        ("Efficiency", "Google revenue ROAS (x)", round(roas(p, "Google"), 2),
         round(roas(m, "Google"), 2), round(roas(b, "Google"), 2)),
        ("Efficiency", "Meta revenue ROAS (x)", round(roas(p, "Meta"), 2),
         round(roas(m, "Meta"), 2), round(roas(b, "Meta"), 2)),
    ]

    out = pd.DataFrame(rows, columns=["section", "metric", "pymc", "meridian", "robyn"])
    path = os.path.join(DATA, "mmm_model_comparison.csv")
    out.to_csv(path, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 10):
        print(out.to_string(index=False, na_rep="—"))

    # --- verdict line -----------------------------------------------------
    print("\nWhich channel each model ranks more efficient (revenue ROAS):")
    for name, d in (("PyMC-Marketing", p), ("Meridian", m), ("Robyn", b)):
        g, mt = roas(d, "Google"), roas(d, "Meta")
        if g != g or mt != mt:  # NaN check
            continue
        win, ratio = ("Meta", mt / g) if mt > g else ("Google", g / mt)
        print(f"  {name:16s} {win:7s} by {ratio:.2f}x   (Google {g:.2f}x, Meta {mt:.2f}x)")

    print(f"\n[done] -> {path}")


if __name__ == "__main__":
    main()
