# MMM Analysis — Project Plan

**Timeseries ID:** `596eef7c71f933d820d0e485935d0e8f`  
**Brand profile:** Beauty & Fitness / Hair Care | US market | USD  
**Data range:** 2022-07-29 → 2023-12-19 (509 daily observations)  
**Active marketing sources:** Google, Meta

---

## Project Structure

```
mmm-analysis/
├── notebooks/
│   ├── 01-eda.ipynb              # Phase 1: EDA, preprocessing, validation
│   ├── 02-model.ipynb            # Phase 2: adstock/saturation, model fit, diagnostics
│   └── 03-results.ipynb          # Phase 3: contributions, ROAS, optimization, reporting
├── mmm/
│   ├── __init__.py
│   ├── preprocessing.py          # filtering, imputation, feature engineering
│   ├── transformations.py        # adstock + saturation implementations
│   └── model.py                  # PyMC-Marketing model definition
├── data/
│   ├── mmm_data.csv
│   ├── mmm_data_dictionary.xlsx
│   ├── mmm_data_clean.csv            # output of Phase 1
│   ├── mmm_trace.nc                  # saved posterior (4 chains × 500 draws)
│   ├── mmm_scaler.pkl                # fitted scaler (no leakage)
│   ├── mmm_results_summary.csv       # key metrics & ROAS table
│   └── mmm_budget_scenarios.csv      # efficient frontier scenarios
├── mmm-project-plan.md               # (gitignored — internal planning doc)
└── readme.md
```

**Rule:** if a function is used in more than one notebook, it goes into `mmm/`. Notebooks stay readable and import from the module.

---

## Dataset Overview

### Identifiers & Metadata

| Column | Description |
|--------|-------------|
| `mmm_timeseries_id` | Unique identifier for this MMM timeseries |
| `organisation_id` | Anonymous eCommerce brand identifier |
| `organisation_vertical` | Top-level product category (Beauty & Fitness) |
| `organisation_subvertical` | Sub-category (Hair Care) |
| `organisation_marketing_sources` | Active ad platforms (Google, Meta) |
| `organisation_primary_territory_name` | Primary territory (US) |
| `territory_name` | Territory scope (All Territories) |
| `date_day` | Observation date |
| `currency_code` | Currency (USD) |

### Target / KPI Variables

| Column | Description |
|--------|-------------|
| `all_purchases` | **Primary KPI** — total web purchases (all customers) |
| `all_purchases_units` | Total units purchased |
| `all_purchases_original_price` | Total merchandise value before discount |
| `all_purchases_gross_discount` | Total discount value (⚠ potential data leakage — use with care) |
| `first_purchases` | New customer acquisitions |
| `first_purchases_units` | Units purchased by new customers |
| `first_purchases_original_price` | New customer merchandise value |
| `first_purchases_gross_discount` | New customer discount value |

### Paid Media Channels — Spend

| Column | Active | Notes |
|--------|--------|-------|
| `google_pmax_spend` | ✅ (3 nulls) | Primary Google channel |
| `google_paid_search_spend` | ✅ (32 nulls) | Non-branded paid search |
| `meta_facebook_spend` | ✅ (191 nulls) | |
| `google_display_spend` | ⚠ sparse (441 nulls) | |
| `meta_instagram_spend` | ⚠ sparse (452 nulls) | |
| `meta_other_spend` | ⚠ sparse (454 nulls) | |
| `google_video_spend` | ⚠ sparse (474 nulls) | |
| `google_shopping_spend` | ❌ no data | |
| `tiktok_spend` | ❌ no data | Not in marketing sources |

### Paid Media Channels — Clicks & Impressions

Same structure as spend for each channel above (suffixed `_clicks` and `_impressions`).

### Non-Paid Traffic Channels

| Column | Active | Notes |
|--------|--------|-------|
| `direct_clicks` | ✅ (5 nulls) | |
| `organic_search_clicks` | ✅ (5 nulls) | |
| `branded_search_clicks` | ✅ (78 nulls) | Spend not tracked |
| `email_clicks` | ✅ (5 nulls) | Spend not tracked |
| `referral_clicks` | ✅ (5 nulls) | |
| `all_other_clicks` | ✅ | Catch-all |

---

## Implementation Plan

### Phase 1 — EDA & Preprocessing
**Files:** `notebooks/01-eda.ipynb` · reusable logic → `mmm/preprocessing.py`

**Step 1 — Exploratory Data Analysis**
- [x] Load CSV and filter to target `mmm_timeseries_id`
- [x] Check date continuity (no gaps in the 509-day series)
- [x] Plot `all_purchases` over time — identify trend, seasonality, anomalies
- [x] Plot per-channel spend over time (Google PMax, Paid Search, Meta Facebook)
- [x] Correlation heatmap: spend channels vs `all_purchases`
- [x] Distribution plots for KPI and spend variables (check for skew, outliers)

**Step 2 — Data Preprocessing** → `mmm/preprocessing.py`
- [x] Fill missing spend/clicks/impressions with `0` (absence of spend, not measurement error)
- [x] Impute small gaps in non-paid traffic (5 nulls) via forward-fill or `0`
- [x] Aggregate sparse channels into platform totals:
  - `total_google_spend` = pmax + paid_search + display + video
  - `total_meta_spend` = facebook + instagram + other
- [x] Temporal features: `day_of_week`, `week_of_year`, `month`, `is_weekend`
- [x] US public holiday flags (Black Friday, Christmas, Labor Day, etc.)
- [x] **Max-normalise** each channel to [0, 1] (replaces log-scaling — required for logistic saturation identifiability; log-scaled inputs caused 23 divergences in early runs)

---

### Phase 2 — Modelling
**Files:** `notebooks/02-model.ipynb` · `mmm/transformations.py` · `mmm/model.py`

**Step 3 — Adstock & Saturation Transformations** → `mmm/transformations.py`
- [x] Geometric adstock decay: `x*_t = x_t + λ · x*_(t-1)`, λ ∈ [0, 1]; `l_max = 14 days`
- [x] **Logistic saturation**: `f(x) = x / (x + λ)` *(Hill saturation was planned but logistic was used — better fit for normalised inputs)*
- [x] Both as parameterised functions so PyMC estimates parameters jointly with the model

**Step 4 — Model Design** → `mmm/model.py`

Framework: **PyMC-Marketing** (Bayesian, full posterior)

Model structure:
```
all_purchases ~ baseline
              + Σ β_i · saturation(adstock(channel_spend_i))   # paid media
              + Σ β_j · non_paid_traffic_j                     # organic/control
              + seasonality (Fourier terms or day-of-week)
              + trend
              + ε
```

Prior choices (as implemented):
- Adstock decay `α` per channel: `Beta(1, 3)` — biased toward short carry-over
- Saturation `λ` per channel: `Gamma(alpha=3, beta=1)` — logistic saturation half-saturation point
- Channel coefficients `β`: `HalfNormal(sigma=2)` — positive-only
- Baseline intercept: `Normal` (PyMC-Marketing default)
- Noise `σ`: `HalfNormal`
- Likelihood: `Normal`

**Step 5 — Model Training**
- [x] Train/holdout split: first 407 rows (80%) train · last 102 rows (20%) holdout
- [x] Fit with MCMC NUTS: `draws=500`, `tune=1000`, `chains=4`
  - Sampler: **numpyro (JAX/XLA backend)** — required on Windows without a C++ compiler; reduced sampling time from >3.5 hours to ~50 seconds
- [x] Convergence achieved: max R-hat **1.006**, ESS > 1000, **0 divergences**

---

### Phase 3 — Results & Reporting
**Files:** `notebooks/03-results.ipynb`

**Step 6 — Model Validation & Diagnostics**
- [x] Predicted vs actual `all_purchases` plot (in-sample)
- [x] Holdout metrics: MAPE 80.3%, WAPE 52.5%, R² 0.591
- [x] Posterior predictive check (PPC) with `arviz`
- [x] Residual analysis: ACF confirms no significant autocorrelation
- [ ] Prior sensitivity check — *not implemented; convergence diagnostics (R-hat, ESS, 0 divergences) gave sufficient confidence; natural next step for a production deployment*

**Step 7 — Contribution & Attribution Analysis**
- [x] Decompose `all_purchases`: baseline + paid media per channel + organic/control
  - Google: 10.1% | Meta: 10.3% | Baseline + controls: ~80%
- [x] ROAS per channel (unit: purchases / USD — multiply by ASP for revenue ROAS)
  - Google: 0.013 purch/$ | Meta: 0.044 purch/$ (Meta ~3.3× more efficient)
- [x] Built-in `plot_components_contributions` stacked chart
- [x] Posterior contribution share HDI plot

**Step 8 — Budget Optimization**
- [x] Response curves per channel (built-in `plot_direct_contribution_curves`)
- [x] Constrained optimisation via `scipy.optimize.minimize` (SLSQP) with calibrated saturation response
  - Unconstrained: shift 100% to Meta (+82% lift) — corner solution driven by ROAS differential
  - Practical bounds (±10–200% per channel): Google 52.9% / Meta 47.1% → **+31% lift**
- [x] Efficient frontier: budget allocation vs incremental purchases (0.5×–2× budget)
- [x] Scenario table across 7 budget levels

**Step 9 — Reporting**
- [x] Model metrics summary (MAPE, R², WAPE)
- [x] Channel contribution table (% of total purchases + ROAS)
- [x] Budget recommendation table
- [x] Key findings and caveats
- [x] Saved: `data/mmm_results_summary.csv` and `data/mmm_budget_scenarios.csv`

---

## Dependencies

### Python Packages

```text
# Core
pandas==3.0.3
numpy==2.4.6
scipy==1.17.1

# Visualization
matplotlib==3.10.9
seaborn==0.13.2
plotly==6.7.0          # optional, for interactive charts

# Bayesian MMM
pymc-marketing==0.19.4  # includes PyMC + adstock/saturation utilities
arviz==0.23.4           # MCMC diagnostics and plots

# Sampling backend (required on Windows without C++ compiler)
jax[cpu]               # JAX CPU build
numpyro                # numpyro NUTS sampler

# Feature engineering & stats
scikit-learn==1.8.0    # scaling, metrics
statsmodels==0.14.6    # ACF/PACF plots

# Holiday calendar
holidays==0.97         # US public holidays

# Serialisation
joblib==1.5.3           # scaler persistence

# Excel support (for data dictionary)
openpyxl==3.1.5
```

Install with:

```bash
pip install pymc-marketing arviz pandas numpy scipy matplotlib seaborn scikit-learn statsmodels holidays openpyxl plotly
```

### Environment

- Python 3.14.3 (developed and tested on); 3.10+ should work
- Virtual environment: `.venv/`
- Jupyter / VS Code Notebooks for interactive development
- **No C++ compiler required** — numpyro sampler bypasses PyTensor C compilation

---

## Key Caveats & Data Notes

1. **TikTok** has no spend data for this timeseries — exclude from model.
2. **Google Shopping** has no spend data — exclude from model.
3. **Sparse channels** (`google_display`, `google_video`, `meta_instagram`, `meta_other`) may be aggregated into platform totals to avoid identification issues.
4. **Gross discount columns** contain potential data leakage — do not use as a predictor; use only as context.
5. **Branded search clicks** are tracked but spend is not — this channel captures organic brand demand and should be used as a control variable, not a media variable.
6. **Currency** is USD for all observations in this timeseries — no conversion needed.
7. The dataset covers **509 days** (> 449 minimum threshold), giving sufficient data for seasonal patterns and model fitting.
