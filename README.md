# P2-ETF-STOCHASTIC-BIFURCATION

Does a market approach a loss of stability the way a dynamical system near a
bifurcation does — and does that show up early enough, **and beyond plain
volatility**, to be useful?

## The theory

Near a fold (saddle-node), transcritical or Hopf bifurcation the dominant
eigenvalue of the linearized dynamics approaches the stability boundary, so
recovery from perturbations slows ("critical slowing down"). For a noisy state
`dx = -λx dt + σ dW` the lag-1 autocorrelation is `exp(-λΔt) → 1` and the
variance `σ²/(2λ) → ∞` as `λ → 0`. A Hopf approach instead has a **complex**
eigenvalue pair whose modulus tends to 1 (a growing oscillation). Fold and
transcritical share the real-eigenvalue signature, so early-warning statistics
**cannot tell them apart**; the engine reports the distinction that *is*
identifiable: real-root class vs complex-pair class.

References: Scheffer et al. 2009 (*Nature* 461:53, "Early-warning signals for
critical transitions"); Dakos et al. 2012 (*PLoS ONE*, "Methods for detecting
early warnings of critical transitions in time series illustrated using
simulated ecological data"); Kritzman, Li, Page & Rigobon 2011 (the
standardized-"shift" construction used for the warning). Evidence in financial
markets is **mixed**: Gatfaoui, Nagot & de Peretti (2016) tested lag-1
autocorrelation, variance and skewness on the Global Financial Crisis across
eleven stock markets and found variance and skewness of cumulative returns
preceded it while autocorrelation of log-indices and cumulative returns carried
no useful information.

## What it estimates

For each ETF, the **state** is `x = log price − causal EMA(log price)`. Over a
trailing window it measures lag-1 autocorrelation, log-variance, |skewness| and
Sarle's bimodality coefficient, and fits an **OLS AR(2)** (the discrete-time
linear-stability model). Its characteristic roots are the local eigenvalues:
dominant modulus `ρ` (margin = `1 − ρ`, the distance to the boundary), angle
`θ`, and whether the pair is complex (Hopf class) or real (fold/transcritical
class).

**The warning** is the *level* of the autocorrelation and of the log-variance
relative to each series' own trailing 504-day history (a standardized "shift"):
`ews_score = (z_ac1 + z_var)/2`, **active** when `ews_score ≥ 1.5`.

## How the design got here (all on simulation, none on real data)

Four first-draft choices failed their own tests and were replaced. They are
listed because each is a trap other implementations can fall into.

| first draft | what simulation showed | replaced by |
|---|---|---|
| Warning = *trend* (correlation with time) of the rolling statistics over 126 days | AUC 0.59 / 0.65 / 0.54 (fold / Hopf / transcritical) at 30 days before the bifurcation — the drift in a rolling statistic (~0.1 in log-variance per 126 days) is far below its sampling noise | *Level* z-scores: 0.70 / 0.82 / 0.61 |
| Warning z-scored a second time against a window that contains the approach itself | share of approach days active fell from 22% / 43% / 16% to 12% / 11% / 13% at an identical ~8% null rate | threshold on the score directly |
| Scored only the *moment of crossing* | a persistent state fires once then stays elevated, so crossings throw away the signal (Hopf: AUC 0.85 but only 11% "detected") | score the warning as an **active state** with a circular-shift chance level |
| Time-to-bifurcation extrapolated from the margin's trend | reported on ~30% of true fold approaches, typically off by 1.5–2× (4× at 50 days out), and on 17–24% of days of **pure noise**, 64–71% of those saying "under 150 days" | **removed** rather than shipped |

## The three tests (each against volatility)

Rising variance of a detrended price is largely just volatility, so every test
is built to ask whether the warning adds anything *beyond* it.

1. **Lead test.** Event rate (a "collapse": the next 20 days' return ≤ −2 ×
   trailing 63-day vol × √20) while the warning is active, against the base rate
   and a chance level from **circularly shifting each ETF's warning flags**
   (preserving their persistence). *Volatility-controlled*: active and inactive
   days are compared only **within 50 volatility-state strata** (deciles of
   vol21/vol63 × quintiles of the 504-day volatility z-score), combined by
   Mantel–Haenszel. This was added after the scorecard passed this test on pure
   synthetic noise: events are defined relative to *trailing* volatility, which
   lags a surge, so right afterwards many days look like large moves while a slow
   variance warning is still active — and any slow volatility proxy "wins".
   Coarse strata (quartiles × terciles) left most of that in place; the finer ones
   do not.
2. **Risk ranking.** Do the warning features rank *which ETFs will suffer the
   worst forward drawdown* (maximum adverse excursion over 5 or 20 days) beyond
   `vol21, vol63, dd63, mom21, mom63`? One **primary** configuration is fixed in
   advance and carries the verdict; the other 47 configurations (spans, windows,
   horizons, training windows, class-neutral or not) are robustness only.
3. **Overlay.** On the equal-weight universe, does cutting exposure on the
   warning beat a plain volatility target? Exposure is decided with information
   through day *t* and applied to day *t+1*; costs are 15 bps round trip on
   exposure traded; a circular block bootstrap (21-day blocks) gives the
   probability the overlay improves on the volatility target in Sharpe and in
   maximum drawdown.

**Scorecard:** *No evidence it adds beyond volatility* (0 tests pass) /
*Weak* (1) / *Partial* (2) / *Evidence on all three* (3). The verdict uses one
pre-registered test per universe, not the ~50 lead tests the dashboard also lists.

## What is validated, concretely

**Estimators, against known answers:** AR(1) lag-1 autocorrelation 0.895 for a
true 0.900; an AR(2) with a complex pair `(1.2, −0.5)` recovered as
`p1 = 1.198, p2 = −0.500`, modulus 0.7066 (true 0.7071), angle 0.557 (true
0.5586), complex in 100% of windows; real roots `(1.3, −0.35)` as modulus 0.9117
(true 0.9192) and complex in 0%; skewness and kurtosis identical to scipy;
forward drawdown identical to brute force; the detrended state is causal (editing
the future changes nothing before it).

**Detection in simulation** (150 runs per scenario, through the same
detrending as real data; detrending span 126 / window 126):

| scenario | warning active in the 55 days before the bifurcation | AUC of the score 30 days before | oracle AUC (no detrending) |
|---|---|---|---|
| fold (saddle-node) | **29%** of days | 0.67–0.69 | 0.70 |
| Hopf | **41%** | 0.84–0.86 | 0.87 |
| transcritical | **13%** | 0.51–0.54 | 0.66 |
| *null: i.i.d. / GARCH / stable mean-reverting* | *8.1% / 6.4% / 7.1%* | — | — |

* **Volatility clustering does not inflate the warning** (GARCH null 7.0% active
  overall, no higher than i.i.d. or the stable null).
* **Type discrimination:** the AR(2) fit calls a Hopf approach "complex" in 100%
  of windows when the cycle is 10 days, 91% at 20 days, and **0% at 40 days**
  (the root angle is too small to resolve from a double real root); on fold runs
  it says "complex" 0% of the time. So the complex-pair label is only meaningful
  for cycles of roughly 10–20 days.
* **Transcritical is the weak case:** detrending removes much of its slow
  dynamics (oracle 0.66 vs 0.52), and 23% of runs tip over before the
  bifurcation (14% for fold) — noise can push a system over a shrinking barrier
  early, a real limit on any early warning.
* A shorter configuration (span 63 / window 63) is clearly weaker (active 19% /
  27% / 8% against a null of 4–5%).

**Lead test, calibrated** (collapse events): on 25 datasets of 43 ETFs × 2,500
days with shared volatility regimes the controlled test gave p < 0.05 in **4%**
(mean relative risk 0.99); it recovers a planted 2.5× effect as 2.28× (p = 0.005).
**Volatility-spike events are not reliably controlled** (20% false positives,
mean RR 1.13 — a variance-based warning responding to volatility events is partly
circular), so they are shown for reference and **excluded from the verdict**.

**Risk ablation:** a planted effect on an EWS feature is detected (dIC t = +18.9);
a planted effect on a *baseline* feature is not credited to the warning (dIC t =
+1.3).

**Overlay mechanics:** exposures are causal; costs checked by hand
(`0.5×0.01 − 0.5×7.5 bps = 0.004625`); Sharpe and drawdown identical to manual
calculation; the bootstrap gives 0.00 for identical series, 1.00 for a genuinely
higher-mean series and 0.67 for two unrelated equal ones.

**The whole scorecard on 24 noise datasets** (43 ETFs, 2,500 days): lead test
passed 17% (about 10% across both calibration runs; nominal is 5%), risk test 0%,
overlay test 0%, **no dataset passed two tests**. So one passing test is weak
evidence, and two or more are needed before the verdict means much.

## Honest limits

* The theory assumes a slowly drifting control parameter and stationary noise.
  Real prices have neither exactly; the warning is a statement about recent
  dynamics of a detrended price, not a forecast of a crash.
* Detrending a random walk with an EMA induces autocorrelation of its own, so the
  *level* of `ac1` is not interpretable — only its level **relative to its own
  past**. The detrending span is a grid axis for that reason.
* A warning can only resolve slowing on timescales shorter than its window; when
  the relaxation time approaches the window the statistics cannot see it.
* Fold and transcritical cannot be separated by these statistics.
* The design choices above were made on **synthetic** data, so the real-data
  tests are not contaminated by them — but nothing here says the warning works on
  real markets. The first real run is the actual test; read the scorecard, and the
  volatility-controlled lead relative risk, before looking at any single ETF.
* All thresholds (`WARNING_SCORE = 1.5`, the 20-day event horizon, 2σ events, the
  overlay percentiles 80/95) were fixed in advance, not tuned on real data. This
  is not financial advice.

## Repo structure

```
config.py          universes, asset-class tags, state/window grid, the warning, event
                   definitions, the pre-registered primary configuration, overlay rules
data_manager.py     master parquet from HF (same as sibling repos)
ews_model.py        detrending, rolling statistics, AR(2) stability fit, the warning,
                   volatility strata, the controlled lead test, risk ablation, overlay,
                   block bootstrap
validation.py       simulated fold / Hopf / transcritical approaches vs i.i.d., GARCH and
                   stable nulls; oracle comparison; Hopf-period sweep
trainer.py          per-universe analysis, the three tests, scorecard, today's monitor,
                   validation, JSON
push_results.py     HfApi.upload_file to the results dataset
us_calendar.py       trading-day helpers
streamlit_app.py     Stability Monitor / Do the warnings add anything? / Simulation validation
.github/workflows/daily_run.yml   weekday cron after the US close (GitHub Actions)
```

## Running (GitHub Actions)

The workflow runs `python trainer.py` with the `HF_TOKEN` secret and pushes
`bifurcation_results_YYYY-MM-DD.json` to `P2SAMAPA/p2-etf-stochastic-bifurcation-results`
(created on first run if it does not exist). The simulation validation re-runs each
time (about 10 seconds) and ships in the JSON, so the dashboard always shows what the
method can and cannot detect. For a local dashboard: `streamlit run streamlit_app.py`.
