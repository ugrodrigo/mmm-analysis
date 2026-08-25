# Proposed Incrementality Tests

**Brand:** Beauty & Fitness / Hair Care · US · USD · timeseries `596eef7c71f933d820d0e485935d0e8f`
**Written:** 2026-08-21
**Sizing:** `scripts/incrementality_power.py` — every budget and duration below is derived from it
**Context:** `scripts/diagnose_confounding.py`, plus the PyMC / Meridian / Robyn comparison
**Updated:** 2026-08-25 — Meta Robyn added as a third framework

---

## Why we need to test in market

Three MMMs — spanning two Bayesian frameworks and one regularised-regression
framework — fit to the same 509 days give three different answers on which channel to fund:

| | PyMC-Marketing | Meridian | Robyn |
|---|---|---|---|
| Method | Bayesian MCMC | Bayesian MCMC | Ridge + Nevergrad |
| Assumption anchored on | coefficients | ROI | spend share |
| Google revenue ROAS | 2.43× | 1.41× | 1.47× |
| Meta revenue ROAS | **8.00×** | **0.86×** | **2.22×** |
| Total paid contribution | 20.4% | 7.0% | 8.7% |
| Recommendation | Move budget → Meta | Move budget → Google | Move budget → Meta |

The disagreement is not a modelling artefact. Refitting Meridian on spend instead of
impressions, and with three different baseline settings, never restored PyMC's ranking —
and adding a third framework with a genuinely different estimator produced a **third
answer rather than a tiebreak**. Each model's number traces to where that framework
constrains media effects, not to the data.

Robyn makes this unusually explicit. Its second objective, DECOMP.RSSD, pulls each
channel's effect share toward its spend share, and it binds: among media only, Robyn splits
effect **68.4% Google / 31.6% Meta** against a spend split of **76.5% / 23.5%**. The ROAS
ratio it reports (1.51×) is exactly the ratio of those two deviations. More telling still,
across its **130 Pareto-optimal solutions** Meta's ROAS spans **1.19× to 5.87×** and
Google's **0.48× to 3.17×** — every one equally defensible on Robyn's own criteria. The
method does not narrow the answer; a human choosing a solution does.

Nor is "two of three rank Meta higher" a vote worth counting: PyMC and Robyn agree on
direction for unrelated reasons — an unconstrained coefficient in one case, an 8-point
effect-share drift in the other. Neither is evidence about incrementality.

The cause is in the media plan itself:

**Meta was never flighted in a way that permits measurement.** It ran as two long
campaigns, live for both Q4 peaks and dark through the entire summer trough:

| Regime | Period | Days | Purchases/day |
|---|---|---|---|
| ON | 2022-07-29 → 2023-01-18 | 174 | 62.2 |
| OFF | 2023-02-15 → 2023-08-16 | 183 | 44.4 |
| ON | 2023-08-17 → 2023-12-19 | 125 | 66.2 |

"Meta is on" is nearly the same variable as "demand is seasonally high" (correlation with
linear time: −0.675). No observational model can separate the two, and the estimate you get
depends entirely on how flexible you let the baseline be — 8.00× with a stiff baseline,
0.86× with a flexible one. **This is a media-plan problem and only in-market testing fixes it.**

Three frameworks is where this line of work stops paying. A fourth model would produce a
fourth answer for a fourth reason. **The budget for further modelling should be redirected
to the tests below.**

---

## The uncomfortable finding from sizing

The obvious test — hold Meta out of half the country and see what happens — **does not work
at current budgets.**

Meta spends $322/day nationally against a **$8,076/day revenue base**. Removing it moves the
KPI less than ordinary weekly noise does. After matched-control adjustment, weekly noise is
16–21%; a 6-week 50/50 holdout can only detect a lift of **26.3%**, which corresponds to a
ROAS floor of **6.6×**.

| Test length | ROAS floor detectable (50/50 holdout at current spend) |
|---|---|
| 4 weeks | 8.07× |
| 6 weeks | 6.59× |
| 8 weeks | 5.71× |
| 12 weeks | 4.66× |

So a naive holdout would refute PyMC's 8.00× and tell you nothing else. It could not
separate Meridian's 0.86× from Robyn's 2.22× — and that 0.86×–2.22× band is precisely
where the budget decision lives. The three models' own disagreement defines the resolution
the test must achieve: **anything coarser than ~1.5× is not decision-grade.**

**The fix is to scale up, not to hold out.** Required incremental spend in the treatment
cell to resolve a given true ROAS over 6 weeks:

| Target ROAS to resolve | Spend needed in cell | × current | Extra budget |
|---|---|---|---|
| 4.0× | $11,156 | 1.6× | $4,384 |
| 3.0× | $14,875 | 2.2× | $8,103 |
| **2.0×** | **$22,312** | **3.3×** | **$15,540** |
| 1.5× | $29,749 | 4.4× | $22,977 |

**A decision-grade answer on Meta costs roughly $15–20k of incremental spend, not the $7k of
forgone spend a holdout implies.** That is the single most important number in this document,
and it should be the basis of the budget conversation.

---

## E0. Acquire geo-level data — prerequisite, start immediately

**Not a test — the thing that makes every test below possible.**

Everything above is a national-series approximation. Proper power analysis, geo matching, and
the tests themselves all require the KPI and media series split by DMA or state. Today the
models run on `n_geos = 1`.

- **Ask:** DMA-level (or state-level) daily purchases, revenue, and per-channel spend /
  impressions / clicks, for the full history.
- **Feasibility:** Google Ads and Meta both report at this grain natively; this is a data
  request, not new instrumentation. The web analytics KPI split is the piece to confirm.
- **Cost:** zero media. Lead time is the constraint, which is why it starts on day 1.
- **Secondary benefit:** geo variation alone would materially improve the MMM even before any
  experiment runs, since Meridian's hierarchical pooling is currently switched off entirely.

**Until E0 lands, re-run `scripts/incrementality_power.py` assumptions in Meta's GeoLift or
Google's Trimmed Match to get true cell-level variance.**

---

## E1. Meta scale-up geo test — the priority test

**Question.** Does Meta deliver a ROAS above or below break-even, and is it anywhere near
the 8.00× the current PyMC model claims?

**Why first.** Largest model disagreement — 0.86×, 2.22× and 8.00× across the three
frameworks, a 9× span — worst confounding, and the smaller of the two budgets, so it is the
cheapest place to buy certainty. It is also the number driving a recommendation to move
~$49k into Meta.

**Design.**
- Matched-market geo split, treatment vs control, assigned on pre-period purchase volume,
  trend, and Meta reach.
- **Treatment cell runs Meta at 3–4× current budget; control cell holds Meta at current
  level.** This is a scale-up, not a blackout — it creates the spend contrast that the
  power analysis says is required, and it does not risk revenue by going dark.
- Google spend held constant and equal across both cells.
- **6 weeks minimum**, 12 preferred.

**Timing.** Run between **mid-February and mid-August**. The monthly profile shows Nov/Dec at
+142%/+64% against the annual mean — testing across Q4 would reintroduce exactly the
seasonal confound the test exists to break. May–July is the flattest stretch (−7%, −3%, −16%).

**Read-out.** Difference-in-differences on cell-level purchases, converted to incremental
ROAS. Pre-register the analysis before launch.

**Budget.** ~$15–20k incremental over the test window to resolve a 2.0× threshold.

**Decision criteria.** The three modelled estimates give the test explicit hypotheses to
falsify — pre-register these before launch:
- Interval **excludes 8.00×** → retire the PyMC allocation recommendation; it is a
  seasonality artefact. (Expected: 8.00× sits above the top of Robyn's entire Pareto front
  and outside Meridian's prior interval, so this is the most likely outcome.)
- Interval **above break-even** → Meta is a real channel, and Meridian's 0.86× is its prior
  showing through. Feed the tested estimate in as a calibrated ROI prior and re-run allocation.
- Interval **below break-even** → Robyn's 2.22× is spend-share anchoring, not measurement.
  Cut Meta rather than holding flat.
- Interval **spanning break-even** → Meta does not currently justify scale; hold budget flat
  and revisit with creative or audience changes rather than spend changes.

**What it unlocks.** Meridian accepts a calibrated ROI prior directly, and Robyn accepts
experimental results as calibration input too. This is what all three models are missing:
Meridian's default `LogNormal(0.2, 0.9)` prior is supplying most of its Meta answer, and
Robyn's spend-share penalty is supplying most of its own. One lift test replaces both
assumptions with a measurement, and collapses Robyn's 130-solution Pareto front to the
subset consistent with the tested value.

**Practical note.** Check whether the Meta account qualifies for **Meta Conversion Lift**
first. Meta's own user-level randomisation is more powerful than a geo split at this budget
and carries no media cost, though it measures Meta-attributed conversions rather than total
business outcome — so treat it as complementary to, not a replacement for, a geo test.

---

## E2. Google PMax incrementality test

**Question.** How much of PMax's attributed conversion volume is genuinely incremental, and
how much is demand that would have converted through organic or direct anyway?

**Why.** PMax is the single largest line in the plan and the least transparent. It buys across
Search, Shopping, Display and YouTube inventory with no channel-level control, and it is
structurally prone to harvesting existing demand. Google is the one place the three models
roughly agree — 2.43×, 1.41× and 1.47×, a 1.7× spread against Meta's 9× — but agreement
between three models that share the same confounded data is not validation, and Google is
the larger budget at risk. Note also that Robyn's Google figure spans 0.48×–3.17× across its
Pareto front, so even that apparent agreement is narrower than it looks.

**Design.**
- Google Ads **geo experiment** (native split-testing in the platform), or a matched-market
  holdout if PMax campaigns cannot be geo-split cleanly.
- Reduce PMax budget by 50–70% in the treatment cell, hold everything else constant.
- 6–8 weeks, same seasonal window rules as E1.

**A holdout is the right shape here, unlike E1** — Google spends $460/day, materially more
than Meta, so removing a large share of it produces a spend contrast big enough to measure
without needing incremental budget.

**Read-out.** Incremental purchases, plus **cannibalisation check**: track organic search,
branded search and direct clicks in both cells. If PMax is harvesting rather than creating
demand, the treatment cell's paid decline will be partly offset by an organic rise.

**Worth knowing before you start:** in the current data `branded_search_clicks` correlates
**0.028 with Google spend and 0.005 with Meta spend** — essentially zero. That is weak
evidence *against* heavy brand cannibalisation, and it makes the test more likely to show
genuine incrementality. Worth confirming rather than assuming.

**Budget.** Net negative — you spend less during the test. The cost is forgone conversions in
the treatment cell if PMax turns out to be highly incremental.

**Decision criteria.** If incremental ROAS lands materially below the modelled 2.43×, rebase
the Google plan and re-run the optimiser with the tested value.

---

## E3. Email × Meta interaction test

**Question.** Is the purchase lift currently credited to Meta actually coming from email
campaigns running at the same time?

**Why.** The two are co-flighted. Email clicks run **+119% higher on Meta-on days**
(248.7/day vs 113.5/day), and email clicks correlate 0.473 with Meta spend. When two channels
switch on together, no model can apportion the credit between them — and the current MMM has
email as a *control*, which means it is actively competing with Meta for the same variance.

**Design.** 2×2 factorial across geo cells over 6–8 weeks:

| | Email on | Email off/reduced |
|---|---|---|
| **Meta scaled** | cell A | cell B |
| **Meta current** | cell C | cell D |

If four cells are impractical at this list size, run the simpler version: stagger email and
Meta campaign calendars so they no longer start and stop together, then re-fit the MMM after
one quarter of decorrelated data.

**Read-out.** Main effects of each channel plus the interaction term.

**Budget.** Marginal — mostly campaign-calendar discipline rather than incremental spend.

**Decision criteria.** A large positive interaction means the channels should be planned and
budgeted jointly, and the MMM must model email as a treatment rather than a control. A null
interaction means they can be optimised independently and email should stay a control.

**Lower priority than E1/E2** but very cheap, and it fixes a structural problem in how the
media calendar is built rather than just measuring one.

---

## E4. Always-on vs flighted structure test

**Question.** Does Meta's burst pattern cost performance versus steady always-on delivery at
the same total budget?

**Why.** This test exists to fix the measurement problem permanently, not just to answer a
media question. As long as Meta runs in long seasonal blocks, **no future MMM will be able to
measure it either** — the confound rebuilds itself every year. Randomised or staggered
delivery creates the identifying variation that observational modelling needs.

**Design.** Split geos into always-on versus flighted delivery at matched total spend. Run for
a full quarter. Randomise the flighting schedule rather than aligning it to the business
calendar — the randomisation is the entire point.

**Read-out.** Total purchases per cell at equal spend, plus the width of the posterior ROAS
interval when the MMM is refitted on the new data.

**Budget.** Net zero — same total spend, different distribution.

**Decision criteria.** Adopt whichever delivery pattern wins on efficiency. Either way, the
resulting decorrelated spend data permanently improves every subsequent MMM refresh.

**Strategic note.** If only one thing on this list is adopted long-term, make it this. It is
the difference between an MMM that measures and an MMM that assumes — and the three-way
disagreement documented above is what "an MMM that assumes" looks like in practice.

---

## E5. Sparse-channel triage

**Question.** Are Google Display, Google Video, Instagram and Meta Other doing anything at all?

**Why.** These four channels are too sparse to model — 441, 474, 452 and 454 nulls
respectively out of 509 days — so they were aggregated into platform totals and have never
been measured separately. They may be quiet waste or genuinely useful; nobody knows.

**Design.** Rather than a formal test, run a **structured on/off rotation**: turn each channel
fully on in a randomised subset of geos for 4 weeks, fully off elsewhere, one channel at a
time or in an orthogonal (Latin-square) schedule.

**Read-out.** Whether each channel produces a detectable lift at its current budget. Given the
power constraints above, expect most of these to be **statistically indistinguishable from
zero** — which is itself a decision: consolidate the budget into channels large enough to
measure.

**Budget.** Neutral to negative.

**Priority.** Last. Do it only after E1 and E2 have settled the main channels.

---

## Sequencing and cost

| Timing | Action | Incremental media cost |
|---|---|---|
| Day 1 | **E0** — request geo-level data | $0 |
| Day 1 | Request the promotion calendar (see caveat below) | $0 |
| Weeks 1–2 | Re-run power in GeoLift/Trimmed Match on real geo data; pre-register E1 | $0 |
| Weeks 3–4 | Check Meta Conversion Lift eligibility; design cells for E1 | $0 |
| **Feb–Aug window** | **E1 — Meta scale-up geo test, 6–12 weeks** | **$15–20k** |
| Same window, parallel geos | **E2 — PMax holdout, 6–8 weeks** | net negative |
| Following quarter | E3 email × Meta, E4 always-on vs flighted | ~$0 |
| After E1/E2 read out | Refit Meridian with calibrated ROI priors; re-run allocation | $0 |
| Later | E5 sparse-channel triage | neutral |

**Total incremental media budget for a decision-grade answer on both major channels:
roughly $15–20k**, since E2 runs net-negative and offsets part of E1.

---

## Two caveats to carry into the tests

**Promotions are an uncontrolled variable.** Meta-on days show +33.5% more purchases than
Meta-off days, but restricting to below-median-discount days turns that into −11.6% — a sign
flip. The discount figure available in this dataset is computed from realised purchases, so
it is post-outcome and that −11.6% is *not* a causal estimate; it is a flag. Request the
**planned promotion calendar** and ensure test cells are balanced on promotional exposure,
or a promo spike in one cell will masquerade as a media effect.

**Test windows must avoid Q4.** November and December run +142% and +64% above the annual
mean. Any test spanning them will have its effect swamped, and the seasonal confound that
motivated this entire plan will simply reassert itself inside the experiment.

---

## Modelling fixes — tracked separately

Not incrementality tests, but they gate whether test results can be read correctly. Recorded
here so they are not lost:

- **`is_holiday` is all-zeros** for all 509 days. `preprocessing.add_holiday_flags` compared
  `Timestamp` values against the date-keyed `holidays` package; pandas 2.x cast silently,
  **pandas 3.0 removed that cast**, and the project pins 3.0.3. Fixed in
  `mmm/preprocessing.py`; regenerate `data/mmm_data_clean.csv` to pick it up (18 days match).
- **Four of six traffic controls are post-treatment** — `direct_clicks` (r = 0.58 with Meta),
  `referral_clicks` (0.53), `email_clicks` (0.47), `organic_search_clicks` (0.42). Controlling
  for variables that respond to advertising biases media effects toward zero. `branded_search`
  and `all_other` are clean.
- **Meridian's headline fit has not converged** — R-hat 1.029, 60 divergences. The 8-knot
  variant is clean (1.012, 19). No number should be published from the unconverged fit.
- **Robyn returns a front, not an answer** — 130 Pareto-optimal solutions, all equally valid
  on its own criteria, spanning 1.19×–5.87× for Meta. The figures quoted here are front
  medians. Any single-solution Robyn number quoted elsewhere should be treated as a choice,
  not a result.
- **`robynpy` 0.3.6 does not run as published** — it requires R + glmnet via rpy2, pins an
  `rpy2` version with no Windows wheel, calls the pandas-3-removed `DataFrame.applymap`, and
  crashes in `evaluate_models()` (`KeyError: 'dep_var'` in the plot-data generator). The
  workarounds are documented in `scripts/run_robyn.py`; budget setup time if anyone re-runs it.
