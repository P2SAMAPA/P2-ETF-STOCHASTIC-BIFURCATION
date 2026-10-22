"""
config.py  —  Configuration for P2 STOCHASTIC-BIFURCATION

Does a market approach a loss of stability the way a dynamical system near a
bifurcation does — and does that show up early enough, and beyond plain
volatility, to be useful?

Theory (critical slowing down). Near a fold (saddle-node), transcritical or
Hopf bifurcation the dominant eigenvalue of the linearized dynamics approaches
the stability boundary, so recovery from perturbations slows. For a noisy
state x with dx = -lambda*x*dt + sigma*dW the lag-1 autocorrelation is
exp(-lambda*dt) -> 1 and the variance sigma^2/(2*lambda) -> infinity as
lambda -> 0; asymmetric potentials add rising skewness ("flickering") and, in
a bimodal landscape, rising bimodality. A Hopf approach instead shows a
COMPLEX eigenvalue pair whose modulus -> 1 (a growing oscillation at a
characteristic frequency), which is visible as a complex dominant root of a
fitted AR(2). Fold and transcritical share the real-eigenvalue signature, so
early-warning statistics alone CANNOT tell them apart; the engine therefore
reports the identifiable distinction — real-root class vs complex-pair class.

References: Scheffer et al. 2009 (Nature 461:53, "Early-warning signals for
critical transitions"); Dakos et al. 2012 (PLoS ONE, "Methods for detecting
early warnings of critical transitions in time series illustrated using
simulated ecological data"); Guttal & Jayaprakash 2008 (Ecology Letters,
"Changing skewness"). Evidence in financial markets is MIXED: Gatfaoui,
Nagot & de Peretti (2016) tested lag-1 autocorrelation, variance and skewness
on the Global Financial Crisis across eleven stock markets and found variance
and skewness of cumulative returns preceded it while autocorrelation of
log-indices and cumulative returns carried no useful information.
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-etf-stochastic-bifurcation-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
}

# Hand-labeled static asset-class tags: used for the class-neutral variant of
# the risk-ranking test and for dashboard coloring. Not a model input.
ASSET_CLASS_MAP = {
    "TLT": "Fixed Income", "VCIT": "Fixed Income", "LQD": "Fixed Income", "HYG": "Fixed Income",
    "VNQ": "Real Estate", "XLRE": "Real Estate",
    "GLD": "Commodity", "SLV": "Commodity", "GDX": "Commodity", "XME": "Commodity", "URA": "Commodity",
    "SPY": "Equity-Broad", "QQQ": "Equity-Broad", "IWM": "Equity-Broad", "IWD": "Equity-Broad",
    "IWO": "Equity-Broad", "IWF": "Equity-Broad", "IWR": "Equity-Broad", "VO": "Equity-Broad",
    "VB": "Equity-Broad", "VIG": "Equity-Broad", "VEA": "Equity-Broad", "QUAL": "Equity-Broad",
    "SPYG": "Equity-Broad",
    "XLK": "Equity-Sector", "XLF": "Equity-Sector", "XLE": "Equity-Sector", "XLV": "Equity-Sector",
    "XLI": "Equity-Sector", "VGT": "Equity-Sector", "VDE": "Equity-Sector", "XLC": "Equity-Sector",
    "IBB": "Equity-Sector", "XLY": "Equity-Sector", "XLP": "Equity-Sector", "XLU": "Equity-Sector",
    "XSD": "Equity-Sector", "SOXX": "Equity-Sector", "SMH": "Equity-Sector", "XBI": "Equity-Sector",
    "XLB": "Equity-Sector", "VUG": "Equity-Sector", "VTV": "Equity-Sector",
}
MIN_CLASS_SIZE = 3

# ---------------------------------------------------------------------------
# The state variable and the early-warning statistics.
#
# State: x_t = log price - causal EMA(log price, TREND_SPAN): the deviation
# from the slow trend (the same price-only state used by the delay-agent
# engine). Detrending is a known source of ARTIFACTS in early-warning work (a
# filter induces autocorrelation of its own), so the span is a grid axis and
# every conclusion is judged against nulls that go through the SAME filter.
#
# Per ETF, per day, over the trailing EWS_WINDOW days of x:
#   ac1       lag-1 autocorrelation                       (slowing down)
#   logvar    log variance                                (variance rise)
#   skew_abs  |skewness|                                  (reported, not used: no signal in simulation)
#   bimod     Sarle bimodality coefficient                (a P-bifurcation cue)
#   ar2 fit   OLS AR(2): dominant root modulus rho, angle theta, complex?
#             margin = 1 - rho  (distance to the stability boundary)
# The WARNING is the level of ac1 and of logvar relative to each series' own
# trailing Z_WINDOW history (a standardized "shift", the Kritzman-et-al. 2011
# construction):   ews_score = (z_ac1 + z_var)/2.  The warning is ews_score >= WARNING_SCORE;
# the score is deliberately not standardized a second time (see ews_model).
# This replaced a first design built on the TREND of the rolling statistics
# over a 126-day window, which simulation showed to be underpowered (see
# ews_model.ews_from_state): the drift in a rolling statistic over so short a
# window is far below its sampling noise.
# ---------------------------------------------------------------------------
TREND_SPANS = [63, 126, 252]
EWS_WINDOWS = [63, 126]
Z_WINDOW = 504
Z_MIN = 252

# ---------------------------------------------------------------------------
# What a warning is tested against (fixed in advance; nothing here is tuned).
#   collapse : the next EVENT_HORIZON days' return <= -EVENT_K * trailing
#              63-day daily vol * sqrt(EVENT_HORIZON)       (saddle-node story)
#   vol_spike: the next EVENT_HORIZON days' realized vol >= VOL_SPIKE_RATIO *
#              the trailing 63-day vol                        (Hopf story)
# A warning is ACTIVE while ews_score >= WARNING_SCORE. Slowing down is a persistent
# state, so it is scored as one: the pooled event rate on active days versus
# the base rate (relative risk), with the chance level and p-value from
# circularly shifting each ETF's active flags (preserving their persistence).
# An earlier version scored only the moment of crossing; simulation showed
# that throws away exactly the persistent signal being sought.
# ---------------------------------------------------------------------------
EVENT_HORIZON = 20
EVENT_K = 2.0
VOL_SPIKE_RATIO = 1.5
WARNING_SCORE = 1.5   # ~8% of days on simulated nulls (null sd of the score is ~1.1)
N_SHIFTS = 200

# ---------------------------------------------------------------------------
# Test 2 — does the EWS family add risk information BEYOND volatility?
# Target: forward H-day maximum adverse excursion (the worst drawdown from
# today within H days), cross-sectionally standardized. Pooled ridge on
#   baseline: vol21, vol63, dd63, mom21, mom63   (what volatility persistence
#             and momentum already know)
#   ews     : baseline + ac1, z_ac1, z_var, z_margin, ews_score, margin,
#             complex-pair strength, bimodality
# refit at each date on a trailing WINDOW; non-overlapping test dates.
# PRIMARY is fixed in advance and carries the verdict (no selection); the rest
# of the grid is shown as robustness only. The class-neutral variant de-means
# features and target within asset class AND scores against de-meaned
# outcomes (the evaluation fix carried over from the spectral-flow engine).
# ---------------------------------------------------------------------------
RISK_HORIZONS = [5, 20]
RISK_WINDOWS = [504, 1008]
RISK_NEUTRAL_OPTIONS = [False, True]
RIDGE_LAMBDA = 0.01
MIN_TRAIN_DAYS = 60
SELECTION_FRACTION = 0.70
MIN_HOLDOUT_PERIODS = 30
IC_T_SUPPORTED = 1.65
PRIMARY = {"trend_span": 126, "ews_window": 126, "horizon": 20, "window": 504, "neutralize": False}

# ---------------------------------------------------------------------------
# Test 3 — does a warning-driven overlay beat a plain volatility target?
# Applied to the equal-weight universe. Exposure is decided with information
# up to day t and applied to day t+1's return.
#   buy-hold : exposure 1
#   vol-target: exposure = min(1, median trailing vol / current 21d vol)
#   ews      : 0.5 when the universe-average ews_score is at/above its causal
#              expanding OVERLAY_Q1 percentile, 0 above OVERLAY_Q2
#   ews+vt   : product of the two exposures
# Costs: TRADING_COST_BPS round trip per unit of exposure traded. A circular
# block bootstrap (BOOT_BLOCK days) gives the probability the overlay improves
# on vol-targeting in Sharpe and in maximum drawdown.
# ---------------------------------------------------------------------------
OVERLAY_Q1 = 0.80
OVERLAY_Q2 = 0.95
TRADING_COST_BPS = 15
BOOT_BLOCK = 21
BOOT_REPS = 1000

# Validation by simulation (run every time, results shown in the dashboard):
# does the machinery detect a simulated fold / Hopf / transcritical approach,
# and how often does it cry wolf on i.i.d., volatility-clustered (GARCH) and
# mean-reverting-but-stable nulls that go through the same detrending?
SIM_REPS = 60
SERIES_STEP = 3
