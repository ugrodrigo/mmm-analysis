# Marketing Mix Modeling (MMM) — Bayesian Attribution & Budget Optimization

A full end-to-end **Bayesian Marketing Mix Model** built from scratch in Python, applied to a real eCommerce brand in the Beauty & Fitness / Hair Care vertical (US market). The project demonstrates how to quantify the incremental contribution of paid media channels to sales and use that to recommend a data-driven budget allocation.

---

## What is Marketing Mix Modeling?

Marketing Mix Modeling is a statistical technique used by marketing teams to measure the effectiveness of each advertising channel (Google, Meta, TV, etc.) on business outcomes. Unlike last-click attribution, MMM accounts for lagged effects, diminishing returns, and organic demand — making it a more accurate foundation for budget decisions.

---

## Project Outcomes

| Metric | Result |
|---|---|
| Holdout R² | 0.59 |
| Holdout WAPE | 52.5% |
| Google ROAS | 0.013 purchases / $ |
| Meta ROAS | 0.044 purchases / $ |
| Budget reallocation lift | **+31%** in channel-attributable purchases |

**Key finding:** Meta is ~3.3× more efficient per dollar than Google in this period. Shifting from a 77%/23% to a 53%/47% Google/Meta split — with the same total budget — is projected to yield a 31% increase in paid-media-driven purchases.

---

## Technical Stack

| Layer | Tools |
|---|---|
| Bayesian inference | [PyMC-Marketing](https://www.pymc-marketing.io/) v0.19.4 · PyMC · ArviZ |
| Sampling backend | **numpyro (JAX/XLA)** — reduces MCMC from hours to ~50 s on CPU |
| Adstock | Geometric decay (`l_max = 14 days`) |
| Saturation | Logistic saturation |
| Optimization | `scipy.optimize.minimize` (SLSQP) |
| Data & EDA | pandas · NumPy · statsmodels |
| Visualisation | matplotlib · ArviZ |

---

## Repository Structure

```
mmm-analysis/
├── data/
│   ├── mmm_data.csv                  # Raw dataset
│   ├── mmm_data_clean.csv            # Preprocessed (509 rows × 59 cols)
│   ├── mmm_trace.nc                  # Saved posterior (4 chains × 500 draws)
│   ├── mmm_scaler.pkl                # Fitted scaler (no leakage)
│   ├── mmm_results_summary.csv       # Key metrics & ROAS table
│   └── mmm_budget_scenarios.csv      # Efficient frontier scenarios
├── mmm/
│   ├── preprocessing.py              # Data loading, cleaning, feature engineering
│   ├── model.py                      # PyMC-Marketing model factory
│   ├── transformations.py            # Adstock & saturation math
│   └── __init__.py
└── notebooks/
    ├── 01-eda.ipynb                  # Phase 1 — Exploratory data analysis
    ├── 02-model.ipynb                # Phase 2 — Model training & diagnostics
    └── 03-results.ipynb              # Phase 3 — Contributions, ROAS & optimization
```

---

## Methodology

### Phase 1 — EDA & Preprocessing
- Loaded and validated a multi-brand, multi-region weekly eCommerce dataset
- Aggregated Google and Meta spend across sub-channels
- Engineered temporal features (day-of-week, holidays, yearly seasonality)
- Applied max-normalisation to paid channels (critical for identifiable saturation priors)
- 80/20 chronological train/holdout split — no random shuffling to respect time order

### Phase 2 — Bayesian Model Training
- Built a hierarchical model using `pymc-marketing.MMM` with:
  - **Geometric adstock** — captures carry-over effects up to 14 days after exposure
  - **Logistic saturation** — models diminishing returns on spend
  - 8 organic traffic controls (direct, branded search, organic search, email, referral, etc.)
- Sampled with **4 chains × 500 draws** using the numpyro NUTS sampler (JAX backend)
- Convergence diagnostics: max R-hat **1.006**, ESS > 1000, **0 divergences**

### Phase 3 — Results & Reporting
- Posterior predictive checks on train and holdout sets
- Contribution decomposition: Google 10%, Meta 10%, Baseline 80%
- ROAS calculation using raw USD spend as denominator
- Budget optimisation via calibrated saturation response curves
- Scenario analysis across 0.5×–2× total budget levels

---

## Notable Engineering Decisions

- **numpyro backend**: PyMC's default PyTensor sampler requires a C++ compiler. On Windows without g++, sampling took >3.5 hours and crashed with multiprocessing errors. Switching to `nuts_sampler="numpyro"` (JAX/XLA) reduced sampling to ~50 seconds with no code changes to the model.
- **Max-normalised channel inputs**: Log-scaled spend values (e.g. 5–8) caused logistic saturation to saturate immediately for any `lam > 0.3`, making the posterior flat and producing 23 divergences. Normalising each channel to [0, 1] resolved this entirely.
- **No data leakage**: The scaler is fitted on training data only and serialised to disk; holdout transformation uses the saved scaler object.

---

## Project Blueprint

The full planning document — dataset schema, modelling decisions, tradeoffs, and what changed during implementation — is in [mmm-project-plan.md](mmm-project-plan.md).

---

## Dataset

Source: [Multi-Region MMM Dataset for Several eCommerce Brands](https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841) — published on figshare.

