# Marketing Mix Modeling — Three Frameworks, and Why They Disagree

An end-to-end **Marketing Mix Modeling** study on a real eCommerce brand in the Beauty &
Fitness / Hair Care vertical (US market), fitted independently in **three** MMM frameworks:
PyMC-Marketing, Google Meridian, and Meta Robyn.

The frameworks return three different answers on which channel to fund. The bulk of this
project is diagnosing *why*, establishing that the disagreement is a property of the data
rather than of the tools, and designing the in-market experiment that would actually settle
it.

---

## What is Marketing Mix Modeling?

Marketing Mix Modeling is a statistical technique used by marketing teams to measure the
effectiveness of each advertising channel (Google, Meta, TV, etc.) on business outcomes.
Unlike last-click attribution, MMM accounts for lagged effects, diminishing returns, and
organic demand, making it a more accurate foundation for budget decisions.

---

## Headline Result

Same 509 days, same KPI, same channels, same 407/102 train/holdout split:

| | PyMC-Marketing | Google Meridian | Meta Robyn |
|---|---|---|---|
| Inference | Bayesian MCMC | Bayesian MCMC | Ridge + Nevergrad |
| Assumption anchored on | coefficients | ROI | spend share |
| Holdout R² | 0.591 | **0.779** | n/a |
| Holdout WAPE | 52.5% | **32.2%** | n/a |
| Google revenue ROAS | 2.43× | 1.41× | 1.47× |
| Meta revenue ROAS | **8.00×** | **0.86×** | **2.22×** |
| Total paid contribution | 20.4% | 7.0% | 8.7% |
| Recommendation | shift → Meta | shift → Google | shift → Meta |

**Key finding: the channel ranking is not identified by this dataset.** Meta's estimated
ROAS spans 0.86×–8.00× depending only on which framework you use, and each number traces to
where that framework constrains media effects — not to evidence in the data.

The cause is in the media plan. Meta was never flighted in a way that permits measurement:
it ran two long campaigns covering both Q4 peaks and went dark through the entire summer
trough, so "Meta is on" correlates −0.675 with linear time. How much credit Meta receives is
decided by how flexible the model's baseline is:

| Baseline flexibility | Meta revenue ROAS |
|---|---|
| PyMC — intercept + 1 Fourier pair | 8.00× |
| Meridian — 8 knots | 1.00× |
| Meridian — 20 knots | 0.86× |

A stiff baseline hands Christmas to Meta; a flexible one keeps it. Neither is "right",
because the data contains no variation that separates them.

---

## Technical Stack

| Layer | Tools |
|---|---|
| Bayesian MMM | [PyMC-Marketing](https://www.pymc-marketing.io/) 0.19.4 · PyMC · ArviZ |
| Bayesian MMM | [Google Meridian](https://developers.google.com/meridian) 1.8.0 · TensorFlow Probability |
| Regularised MMM | [Meta Robyn](https://facebookexperimental.github.io/Robyn/) (`robynpy` 0.3.6) · R 4.6.1 · glmnet · Nevergrad |
| Sampling backend | **numpyro (JAX/XLA)** for PyMC, reduces MCMC from hours to ~50 s on CPU |
| Adstock | Geometric decay (`l_max = 14 days`) across all three |
| Saturation | Logistic (PyMC) · Hill (Meridian, Robyn) |
| Optimisation | `scipy.optimize` SLSQP · Meridian `BudgetOptimizer` |
| Data & EDA | pandas · NumPy · statsmodels |
| Visualisation | matplotlib · ArviZ |

Each framework runs in its own virtual environment — their dependency stacks are mutually
incompatible (Meridian needs TensorFlow; Robyn needs R via `rpy2` and pins `pandas<3`).

---

## Repository Structure

```
mmm-analysis/
├── data/
│   ├── mmm_data.csv                      # Raw dataset
│   ├── mmm_data_clean.csv                # Preprocessed (509 rows × 59 cols)
│   ├── mmm_results_summary.csv           # PyMC metrics & ROAS
│   ├── mmm_budget_scenarios.csv          # PyMC efficient frontier
│   ├── mmm_meridian_results.csv          # Meridian metrics & ROAS
│   ├── mmm_meridian_sensitivity.csv      # Meridian spec sensitivity runs
│   ├── mmm_robyn_results.csv             # Robyn Pareto-front summary
│   └── mmm_model_comparison.csv          # Three-way comparison table
├── mmm/
│   ├── preprocessing.py                  # Loading, cleaning, feature engineering
│   ├── transformations.py                # Adstock & saturation math
│   ├── model.py                          # PyMC-Marketing model factory
│   ├── meridian_model.py                 # Meridian data builder & model factory
│   └── robyn_model.py                    # Robyn spec & R environment setup
├── notebooks/
│   ├── 01-eda.ipynb                      # Phase 1 — Exploratory data analysis
│   ├── 02-model.ipynb                    # Phase 2 — PyMC training & diagnostics
│   └── 03-results.ipynb                  # Phase 3 — Contributions, ROAS & optimisation
├── scripts/
│   ├── run_meridian.py                   # Meridian fit, diagnostics, attribution
│   ├── meridian_sensitivity.py           # Spend-execution and knot-count refits
│   ├── run_robyn.py                      # Robyn Nevergrad search & Pareto front
│   ├── compare_models.py                 # Joins all three result sets
│   ├── diagnose_confounding.py           # Why the models disagree
│   └── incrementality_power.py           # Experiment sizing / power analysis
├── proposed-tests.md                     # In-market incrementality test roadmap
└── mmm-project-plan.md                   # Planning doc, schema, design decisions
```

---

## Methodology

### Phase 1 — EDA & Preprocessing
- Loaded and validated a multi-brand, multi-region **daily** eCommerce dataset
- Aggregated Google and Meta spend across sub-channels
- Engineered temporal features (day-of-week, holidays, yearly seasonality)
- Applied max-normalisation to paid channels (critical for identifiable saturation priors)
- 80/20 chronological train/holdout split, no random shuffling to respect time order

### Phase 2 — Bayesian Model Training (PyMC-Marketing)
- **Geometric adstock** for carry-over up to 14 days, **logistic saturation** for
  diminishing returns
- Organic traffic controls (direct, branded search, organic search, email, referral)
- 4 chains × 500 draws via numpyro NUTS; max R-hat **1.006**, ESS > 1000, **0 divergences**

### Phase 3 — Results & Reporting
- Posterior predictive checks, contribution decomposition, ROAS, budget optimisation
- Scenario analysis across 0.5×–2× total budget levels

### Phase 4 — Cross-Framework Comparison
- Refitted the same specification in **Meridian** (ROI-parameterised priors, Hill
  saturation, knot-spline baseline) and **Robyn** (ridge regression, Nevergrad
  hyperparameter search, Pareto front)
- Sensitivity runs isolating each candidate cause: impressions vs spend as the execution
  variable, and baseline flexibility. Neither restored PyMC's ranking
- Diagnosed the identification failure and quantified it

### Phase 5 — Experiment Design
- Confounding diagnostics: Meta's flighting, promotional overlap, post-treatment controls
- Power analysis sizing a geo experiment that could actually resolve the disagreement
- Full roadmap in [proposed-tests.md](proposed-tests.md)

---

## Notable Engineering & Analytical Findings

- **The disagreement is structural, not a bug.** PyMC priors channel *coefficients*, leaving
  the implied return unconstrained. Meridian priors *ROI* directly (`LogNormal(0.2, 0.9)`,
  90% interval 0.28×–5.37×) — PyMC's 8.00× is off the end of that distribution. Robyn priors
  nothing but *penalises* deviation of effect share from spend share. Three assumptions,
  three answers.
- **Meridian's media estimates are largely prior-driven here.** Google's ROI posterior
  standard deviation shrank only **5%** from its prior; Meta's 52%. The data has very little
  to say.
- **Robyn returns a front, not an answer.** Across its 130 Pareto-optimal solutions Meta's
  ROAS spans 1.19×–5.87×, all equally valid on Robyn's own criteria. Its reported ranking is
  a small wobble around "effect follows spend".
- **Four of six traffic controls are post-treatment.** `direct_clicks` correlates 0.58 with
  Meta spend, `referral` 0.53, `email` 0.47, `organic_search` 0.42. Conditioning on
  variables that respond to advertising biases media effects toward zero. Counter to
  expectation, `branded_search` is the *cleanest* control here (r ≈ 0.00).
- **`is_holiday` was silently all-zeros.** `add_holiday_flags` compared `Timestamp` values
  against the date-keyed `holidays` package; pandas 2.x cast between them silently, **pandas
  3.0 removed that cast**, and the project pins 3.0.3. Fixed by comparing on `.dt.date`.
- **Power analysis changed the recommended experiment.** The obvious test — hold Meta out of
  half the country — has a ROAS floor of 6.59× over 6 weeks, so it would return a null result
  almost regardless of the truth. The plan is a **scale-up** instead, at ~$22k incremental.
- **numpyro backend**: PyMC's default PyTensor sampler needs a C++ compiler. On Windows
  without g++, sampling took >3.5 hours and crashed with multiprocessing errors.
  `nuts_sampler="numpyro"` reduced it to ~50 seconds with no model changes.
- **Max-normalised channel inputs**: log-scaled spend (values 5–8) saturated the logistic
  curve for any `lam > 0.3`, flattening the posterior and producing 23 divergences.
  Normalising each channel to [0, 1] resolved it.

---

## Reproducing

```bash
# PyMC-Marketing (notebooks)
pip install pymc-marketing arviz pandas numpy scipy matplotlib seaborn scikit-learn \
            statsmodels holidays openpyxl plotly

# Google Meridian
python -m venv .venv-meridian
.venv-meridian/Scripts/python.exe -m pip install google-meridian holidays
.venv-meridian/Scripts/python.exe scripts/run_meridian.py
.venv-meridian/Scripts/python.exe scripts/meridian_sensitivity.py

# Meta Robyn — needs R + glmnet; robynpy fits via glmnet through rpy2
winget install --id RProject.R
Rscript -e "install.packages('glmnet', lib=Sys.getenv('R_LIBS_USER'))"
python -m venv .venv-robyn
.venv-robyn/Scripts/python.exe -m pip install robynpy --no-deps
.venv-robyn/Scripts/python.exe -m pip install rpy2 "pandas<3" nevergrad prophet nlopt \
            ipython scikit-learn lmfit plotnine
.venv-robyn/Scripts/python.exe scripts/run_robyn.py

# Comparison and experiment sizing
.venv-meridian/Scripts/python.exe scripts/compare_models.py
python scripts/diagnose_confounding.py
python scripts/incrementality_power.py
```

> `robynpy` 0.3.6 does not install cleanly as published — it pins an `rpy2` version with no
> Windows wheel, calls the pandas-3-removed `DataFrame.applymap`, and crashes in
> `evaluate_models()`. The workarounds are documented in
> [scripts/run_robyn.py](scripts/run_robyn.py).

---

## Project Blueprint

The full planning document, dataset schema, modelling decisions, and what changed during
implementation is in [mmm-project-plan.md](mmm-project-plan.md). The in-market test roadmap
is in [proposed-tests.md](proposed-tests.md).

---

## Dataset

Source: [Multi-Region MMM Dataset for Several eCommerce Brands](https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841), published on figshare.
