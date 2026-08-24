"""
scripts/compare_models.py
-------------------------
Join the PyMC-Marketing and Meridian result files into one comparison table.

    .venv-meridian/Scripts/python.exe scripts/compare_models.py
"""

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
ASP = 182.02  # train-period average selling price, USD per purchase


def load(path):
    df = pd.read_csv(os.path.join(DATA, path))
    return dict(zip(df["metric"], df["value"]))


def main():
    p = load("mmm_results_summary.csv")
    m = load("mmm_meridian_results.csv")

    rows = [
        ("Model quality", "Holdout R²", p["Holdout R²"], m["Holdout R2"]),
        ("Model quality", "Holdout WAPE (%)", p["Holdout WAPE (%)"], m["Holdout WAPE (%)"]),
        ("Model quality", "Holdout MAPE (%)", p["Holdout MAPE (%)"], m["Holdout MAPE (%)"]),
        ("Model quality", "Max R-hat", 1.006, m["R-hat (max)"]),
        ("Model quality", "Divergences", 0, m["Divergences"]),
        ("Attribution", "Google share (%)", p["Google share (%)"], m["Google share (%)"]),
        ("Attribution", "Meta share (%)", p["Meta share (%)"], m["Meta share (%)"]),
        (
            "Attribution",
            "Total paid share (%)",
            round(p["Google share (%)"] + p["Meta share (%)"], 2),
            round(m["Google share (%)"] + m["Meta share (%)"], 2),
        ),
        ("Efficiency", "Google ROAS (purch/$)", p["Google ROAS (purch/$)"], m["Google ROAS (purch/$)"]),
        ("Efficiency", "Meta ROAS (purch/$)", p["Meta ROAS (purch/$)"], m["Meta ROAS (purch/$)"]),
        (
            "Efficiency",
            "Google revenue ROAS (x)",
            round(p["Google ROAS (purch/$)"] * ASP, 2),
            round(m["Google ROAS (purch/$)"] * ASP, 2),
        ),
        (
            "Efficiency",
            "Meta revenue ROAS (x)",
            round(p["Meta ROAS (purch/$)"] * ASP, 2),
            round(m["Meta ROAS (purch/$)"] * ASP, 2),
        ),
        ("Optimisation", "Optimal Google spend ($)", p["Optimal Google spend ($)"], m["Optimal Google spend ($)"]),
        ("Optimisation", "Optimal Meta spend ($)", p["Optimal Meta spend ($)"], m["Optimal Meta spend ($)"]),
        ("Optimisation", "Lift (%)", p["Optimisation lift (%)"], m["Optimisation lift (%)"]),
    ]

    out = pd.DataFrame(rows, columns=["section", "metric", "pymc", "meridian"])
    out["ratio"] = (out["meridian"] / out["pymc"]).round(3)
    path = os.path.join(DATA, "mmm_model_comparison.csv")
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"\n[done] -> {path}")


if __name__ == "__main__":
    main()
