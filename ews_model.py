"""
ews_model.py  —  Early-warning statistics of an approaching bifurcation

What is estimated
-----------------
For each ETF the state is x_t = log price - causal EMA(log price). Over a
trailing window the engine measures the classic critical-slowing-down
statistics (lag-1 autocorrelation, variance, skewness, bimodality) and fits an
OLS AR(2) — the discrete-time linear-stability model x_t = p1*x_{t-1} +
p2*x_{t-2} + e. Its characteristic roots ARE the eigenvalues of the local
dynamics:

  * a real dominant root r -> +1            fold / transcritical approach
                                            (indistinguishable from EWS alone)
  * a complex pair with modulus rho -> 1    Hopf-class approach (a growing
                                            oscillation at angle theta)

margin = 1 - rho is the distance to the stability boundary. The WARNING is the
level of the autocorrelation and the log-variance relative to each ETF's own
trailing history (`ews_score`). No
time-to-bifurcation estimate is produced: an extrapolation from the margin's
trend was tested in simulation and removed (see ews_from_state).

Three tests of whether any of it is useful (each against a volatility
baseline, because rising variance of x is, in large part, just volatility)
---------------------------------------------------------------------------
1. Lead test: is a collapse / volatility spike more likely while a warning is
   active than chance (circular-shift permutation, per ETF) — and more than
   while a plain volatility trigger is active?
2. Risk-ranking ablation: do the EWS features predict which ETFs will have the
   worst forward drawdown BEYOND vol21/vol63/drawdown/momentum?
3. Overlay: does de-risking on a warning beat a plain volatility target?

Honest scope
------------
* The theory assumes a slowly drifting control parameter and a stationary
  noise level. Real prices have volatility clustering, which raises variance
  and the autocorrelation of the detrended state with no bifurcation at all;
  validation.py measures exactly how often that produces false warnings.
* Detrending a random walk with an EMA induces autocorrelation of its own, so
  levels of ac1 are not interpretable; only changes relative to the series'
  own past are used.
* Fold and transcritical cannot be told apart from these statistics.
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Optional, Tuple

import config

# ---------------------------------------------------------------------------
# Basic series
# ---------------------------------------------------------------------------
def compute_forward_returns(returns: np.ndarray, horizon: int) -> np.ndarray:
    """fwd[i] = sum of returns over days i+1 .. i+horizon (NaN where not fully observed)."""
    n, T = returns.shape
    cum = np.concatenate([np.zeros((1, T)), np.cumsum(returns, axis=0)], axis=0)
    fwd = np.full((n, T), np.nan)
    if n - horizon > 0:
        fwd[: n - horizon] = cum[horizon + 1: n + 1] - cum[1: n - horizon + 1]
    return fwd


def forward_mae(returns: np.ndarray, horizon: int) -> np.ndarray:
    """Forward maximum adverse excursion: worst cumulative loss from today within the next H days (>= 0)."""
    n, T = returns.shape
    cum = np.concatenate([np.zeros((1, T)), np.cumsum(returns, axis=0)], axis=0)
    out = np.full((n, T), np.nan)
    m = n - horizon
    if m > 0:
        worst = np.zeros((m, T))
        for k in range(1, horizon + 1):
            worst = np.minimum(worst, cum[1 + k: m + 1 + k] - cum[1: m + 1])
        out[:m] = -worst
    return out


def roll_std(a: np.ndarray, w: int) -> np.ndarray:
    return pd.DataFrame(a).rolling(w, min_periods=w).std().values


def detrended_state(log_prices: np.ndarray, span: int) -> np.ndarray:
    """x = log price - causal EMA(log price, span); the first `span` rows are NaN (filter burn-in)."""
    trend = pd.DataFrame(log_prices).ewm(span=span, adjust=False).mean().values
    x = log_prices - trend
    x[:span] = np.nan
    return x


# ---------------------------------------------------------------------------
# Early-warning statistics
# ---------------------------------------------------------------------------
def rolling_ews(x: np.ndarray, W: int) -> Dict[str, np.ndarray]:
    """Per-day, per-column statistics of x over the trailing W days (all arrays (n,T))."""
    df = pd.DataFrame(x)
    x1, x2 = df.shift(1), df.shift(2)

    def r(s):
        return s.rolling(W, min_periods=W)

    c00, c11, c22 = r(df).var().values, r(x1).var().values, r(x2).var().values
    c01, c02, c12 = r(df).cov(x1).values, r(df).cov(x2).values, r(x1).cov(x2).values
    skew = r(df).skew().values
    kurt = r(df).kurt().values

    with np.errstate(invalid="ignore", divide="ignore"):
        ac1 = c01 / np.sqrt(c00 * c11)
        det = c11 * c22 - c12 ** 2
        p1 = (c01 * c22 - c02 * c12) / det
        p2 = (c02 * c11 - c01 * c12) / det
        disc = p1 ** 2 + 4 * p2
        complex_pair = disc < 0
        sq = np.sqrt(np.where(disc >= 0, disc, 0.0))
        r_plus, r_minus = (p1 + sq) / 2, (p1 - sq) / 2
        real_dom = np.where(np.abs(r_plus) >= np.abs(r_minus), r_plus, r_minus)
        rho = np.where(complex_pair, np.sqrt(np.maximum(-p2, 0.0)), np.abs(real_dom))
        theta = np.where(complex_pair, np.arccos(np.clip(p1 / (2 * np.maximum(rho, 1e-12)), -1, 1)), np.nan)
        margin = np.clip(1.0 - rho, -0.2, 1.0)
        ok = np.isfinite(rho)
        margin = np.where(ok, margin, np.nan)
        bimod = (skew ** 2 + 1.0) / (kurt + 3.0 * (W - 1) ** 2 / ((W - 2) * (W - 3)))
        logvar = np.log(np.maximum(c00, 1e-18))
        innov = np.maximum(c00 - c01 ** 2 / c11, 1e-18)
        logsig = 0.5 * np.log(innov)
    return {"ac1": ac1, "logvar": logvar, "skew_abs": np.abs(skew), "bimod": bimod, "p1": p1, "p2": p2,
            "rho": np.where(ok, rho, np.nan), "theta": theta, "complex": np.where(ok, complex_pair, False),
            "real_dom": np.where(ok, real_dom, np.nan), "margin": margin, "logsig": logsig}


def rolling_z(a: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    df = pd.DataFrame(a)
    mu = df.rolling(window, min_periods=min_periods).mean()
    sd = df.rolling(window, min_periods=min_periods).std()
    return ((df - mu) / sd.where(sd > 1e-12)).values


def ews_features(log_prices: np.ndarray, trend_span: int, W: int) -> Dict[str, np.ndarray]:
    """All early-warning statistics from a LOG PRICE: causal EMA detrending, then ews_from_state."""
    return ews_from_state(detrended_state(log_prices, trend_span), W)


def ews_from_state(x: np.ndarray, W: int) -> Dict[str, np.ndarray]:
    """
    Statistics from an already-defined state series x, and the warning built from them.

    The warning is the LEVEL of the autocorrelation and of the log-variance relative to each series' own
    trailing history (a standardized "shift", as in Kritzman et al. 2011 for the absorption ratio):
        z_ac1, z_var  = rolling z-scores of ac1 and logvar over the trailing Z_WINDOW days
        ews_score     = (z_ac1 + z_var) / 2          (null sd ~1.1; the warning is ews_score >= WARNING_SCORE)
    The score is deliberately NOT standardized a second time: doing so (a z-score of the score against a window
    that contains the approach itself) cut the share of approach days with an active warning from 22% / 43% /
    16% to 12% / 11% / 13% (fold / Hopf / transcritical) at an identical ~8% null rate.
    Two earlier designs were tested on simulated fold, Hopf and transcritical approaches and replaced:
      * the TREND (windowed correlation with time) of the rolling statistics over 126 days — AUCs of
        0.59 / 0.65 / 0.54 at 30 days before the bifurcation versus 0.70 / 0.82 / 0.61 for the level
        z-scores, because the drift in a rolling statistic (~0.1 in log-variance per 126 days) is far
        below its sampling noise over so short a window;
      * a time-to-bifurcation extrapolation from the margin's trend — reported on ~30% of true fold
        approaches, typically off by 1.5-2x, and on 17-24% of days of pure noise (64-71% of those saying
        "under 150 days"). It was removed rather than shipped.
    Skewness carried no signal (AUC ~0.5) and is reported but not used.
    """
    s = rolling_ews(x, W)
    z = lambda a: rolling_z(a, config.Z_WINDOW, config.Z_MIN)
    z_ac1, z_var, z_margin = z(s["ac1"]), z(s["logvar"]), -z(s["margin"])
    score = (z_ac1 + z_var) / 2.0
    out = dict(s)
    out.update({"x": x, "z_ac1": z_ac1, "z_var": z_var, "z_margin": z_margin, "ews_score": score,
                "hopf_strength": np.where(s["complex"], s["rho"], 0.0)})
    return out


def baseline_features(returns: np.ndarray, log_prices: np.ndarray) -> Dict[str, np.ndarray]:
    """What volatility persistence and momentum already know: vol21, vol63, dd63, mom21, mom63."""
    cum = pd.DataFrame(returns).rolling
    vol21 = np.log(np.maximum(roll_std(returns, 21), 1e-9))
    vol63 = np.log(np.maximum(roll_std(returns, 63), 1e-9))
    dd63 = log_prices - pd.DataFrame(log_prices).rolling(63, min_periods=63).max().values
    mom21 = pd.DataFrame(returns).rolling(21, min_periods=21).sum().values
    mom63 = pd.DataFrame(returns).rolling(63, min_periods=63).sum().values
    return {"vol21": vol21, "vol63": vol63, "dd63": dd63, "mom21": mom21, "mom63": mom63}


BASE_NAMES = ["vol21", "vol63", "dd63", "mom21", "mom63"]
EWS_NAMES = ["ac1", "z_ac1", "z_var", "z_margin", "ews_score", "margin", "hopf_strength", "bimod"]
FEATURE_NAMES = BASE_NAMES + EWS_NAMES
N_BASE = len(BASE_NAMES)


# ---------------------------------------------------------------------------
# Events and the pooled lead test
# ---------------------------------------------------------------------------
def event_flags(returns: np.ndarray, horizon: int, k: float, vol_ratio: float) -> Dict[str, np.ndarray]:
    """
    collapse  : forward H-day return <= -k * trailing 63d daily vol * sqrt(H)
    vol_spike : forward H-day realized vol >= vol_ratio * trailing 63d daily vol
    NaN-safe: flags are False where the forward window or the trailing vol is unavailable;
    `valid` marks where an event could be evaluated.
    """
    vol63 = roll_std(returns, 63)
    fwd = compute_forward_returns(returns, horizon)
    rv = np.sqrt(compute_forward_returns(returns ** 2, horizon) / horizon)
    valid = np.isfinite(fwd) & np.isfinite(vol63)
    with np.errstate(invalid="ignore"):
        collapse = valid & (fwd <= -k * vol63 * np.sqrt(horizon))
        spike = valid & (rv >= vol_ratio * vol63)
    return {"collapse": collapse, "vol_spike": spike, "valid": valid}


def warning_active(score: np.ndarray, thr: float = None) -> np.ndarray:
    """A warning is ACTIVE while the score is at or above thr (NaN counts as inactive)."""
    thr = config.WARNING_SCORE if thr is None else thr
    return np.nan_to_num(score, nan=-9.0) >= thr


def vol_strata(vol21: np.ndarray, vol63: np.ndarray, vol_level_z: np.ndarray, n_ratio: int = 10,
               n_level: int = 5) -> np.ndarray:
    """
    Volatility-state strata, known at day t: deciles of vol21/vol63 (how far the trailing-63d normalizer
    LAGS a volatility change) x quintiles of the 504-day volatility z-score (the regime). The artifact
    lives in the extreme tail of that ratio — the first weeks after a surge — so coarse bins (four
    quartiles x three terciles) left most of it in place; deciles x quintiles = 50 strata do not.
    Pooled quantiles over the universe; -1 where unavailable. Events here are defined relative to
    trailing 63-day volatility, so right after a surge the normalizer is stale and many days look like
    large moves — any slow volatility-like warning is then 'active' exactly when events are likely.
    Comparing active and inactive days only WITHIN these strata removes that artifact.
    """
    ratio = np.exp(vol21 - vol63)
    ok = np.isfinite(ratio) & np.isfinite(vol_level_z)
    out = np.full(ratio.shape, -1, dtype=np.int64)
    if ok.sum() < 100:
        return out
    qr = np.quantile(ratio[ok], np.arange(1, n_ratio) / n_ratio)
    qz = np.quantile(vol_level_z[ok], np.arange(1, n_level) / n_level)
    out[ok] = n_level * np.searchsorted(qr, ratio[ok]) + np.searchsorted(qz, vol_level_z[ok])
    return out


def _mh_rr(A, N1, E, N):
    """Mantel-Haenszel relative risk (active vs inactive) from per-stratum counts."""
    N0, B = N - N1, E - A
    with np.errstate(invalid="ignore", divide="ignore"):
        num = np.nansum(np.where(N > 0, A * N0 / N, 0.0))
        den = np.nansum(np.where(N > 0, B * N1 / N, 0.0))
    return float(num / den) if den > 0 else float("nan")


def pooled_state_test(active: np.ndarray, event: np.ndarray, valid: np.ndarray, strata: np.ndarray = None,
                      n_shifts: int = None, seed: int = 0) -> Dict:
    """
    Event rate while a warning is ACTIVE versus inactive, pooled over ETFs. `event[i]` means "an event starts
    within the next EVENT_HORIZON days after day i", so a warning active on day i is a genuine forecast.
    * Naive relative risk: event rate on active days / base rate.
    * Volatility-controlled relative risk (when `strata` is given): the Mantel-Haenszel combination of
      active-vs-inactive risk ratios WITHIN volatility-state strata — the number the verdict uses.
    Chance level and p-value: circularly shift each ETF's active flags by an independent random amount (this
    preserves the flags' persistence and the events' clustering, which a binomial test would not) and recompute
    the same statistic. p = P(shifted statistic >= observed).
    """
    n_shifts = config.N_SHIFTS if n_shifts is None else n_shifts
    n, T = active.shape
    K = int(strata.max()) + 1 if strata is not None and strata.max() >= 0 else 1
    segs = []
    for j in range(T):
        idx = np.where(valid[:, j])[0]
        if len(idx) < 100:
            continue
        lo, hi = idx[0], idx[-1] + 1
        st = strata[lo:hi, j] if strata is not None else np.zeros(hi - lo, dtype=np.int64)
        good = st >= 0
        st = np.where(good, st, K)                      # an extra bin, dropped below
        segs.append((active[lo:hi, j].astype(np.float64), event[lo:hi, j].astype(np.float64), st))
    if not segs:
        return {"available": False}
    n_val = int(sum(len(a) for a, _, _ in segs))
    n_act = float(sum(a.sum() for a, _, _ in segs))
    n_ev = float(sum(e.sum() for _, e, _ in segs))
    base = {"available": True, "n_valid_days": n_val, "n_active_days": int(n_act), "n_event_days": int(n_ev),
            "active_share": n_act / n_val, "rate_base": n_ev / n_val if n_val else None}
    if n_act < 10 or n_ev < 10:
        base.update({"rate_active": None, "relative_risk_naive": None, "relative_risk": None, "rate_chance": None,
                     "p_value": None})
        return base

    E = sum(np.bincount(st, weights=e, minlength=K + 1) for _, e, st in segs)[:K]
    N = sum(np.bincount(st, minlength=K + 1).astype(float) for _, _, st in segs)[:K]

    def stat(rolled):
        A = np.zeros(K); N1 = np.zeros(K)
        for (a0, e, st), a in zip(segs, rolled):
            A += np.bincount(st, weights=a * e, minlength=K + 1)[:K]
            N1 += np.bincount(st, weights=a, minlength=K + 1)[:K]
        return A, N1

    A0, N10 = stat([s[0] for s in segs])
    obs_mh = _mh_rr(A0, N10, E, N)
    obs_naive = float(A0.sum() / N10.sum() / (n_ev / n_val)) if N10.sum() > 0 else float("nan")
    use = obs_mh if strata is not None else obs_naive
    rng = np.random.default_rng(seed)
    null = np.zeros(n_shifts)
    for k in range(n_shifts):
        A, N1 = stat([np.roll(s[0], int(rng.integers(0, len(s[0])))) for s in segs])
        null[k] = _mh_rr(A, N1, E, N) if strata is not None else float(A.sum() / max(N1.sum(), 1) / (n_ev / n_val))
    null = null[np.isfinite(null)]
    base.update({"rate_active": float(A0.sum() / max(N10.sum(), 1)), "relative_risk_naive": obs_naive,
                 "relative_risk": obs_mh if strata is not None else obs_naive,
                 "rate_chance": float(np.mean(null)) if len(null) else None,
                 "p_value": float((1 + np.sum(null >= use)) / (1 + len(null))) if len(null) and np.isfinite(use) else None,
                 "controlled": strata is not None})
    return base


# ---------------------------------------------------------------------------
# Class-neutral standardization (carried over from the sibling engines)
# ---------------------------------------------------------------------------
def group_ids_from_map(tickers: List[str], class_map: Dict[str, str], min_size: int) -> np.ndarray:
    labels = [class_map.get(t) for t in tickers]
    counts = Counter(l for l in labels if l)
    uniq = sorted(l for l, c in counts.items() if c >= min_size)
    idx = {l: i for i, l in enumerate(uniq)}
    return np.array([idx.get(l, -1) if l else -1 for l in labels])


def demean_by_group(a: np.ndarray, gids: np.ndarray) -> np.ndarray:
    out = a.astype(float).copy()
    with np.errstate(invalid="ignore"):
        gmean = np.nanmean(a, axis=1, keepdims=True)
        for g in np.unique(gids):
            m = gids == g
            if g < 0:
                out[:, m] = a[:, m] - gmean if a.ndim == 2 else a[:, m, :] - gmean
            else:
                cm = np.nanmean(a[:, m] if a.ndim == 2 else a[:, m, :], axis=1, keepdims=True)
                if a.ndim == 2:
                    out[:, m] = a[:, m] - cm
                else:
                    out[:, m, :] = a[:, m, :] - cm
    out[~np.isfinite(a)] = np.nan
    return out


def cs_zscore(a: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(a, axis=1, keepdims=True)
        sd = np.nanstd(a, axis=1, keepdims=True)
        out = np.where(sd > 1e-9, (a - mu) / np.where(sd > 1e-9, sd, 1.0), 0.0)
    out[~np.isfinite(a)] = np.nan
    return out


def standardize(a: np.ndarray, gids: Optional[np.ndarray]) -> np.ndarray:
    return cs_zscore(a if gids is None else demean_by_group(a, gids))


# ---------------------------------------------------------------------------
# Test 2: risk-ranking ablation
# ---------------------------------------------------------------------------
def build_cross_products(Fz: np.ndarray, Yz: np.ndarray) -> Dict:
    n, T, p = Fz.shape
    feat_ok = np.isfinite(Fz).all(axis=(1, 2))
    ok = feat_ok & np.isfinite(Yz).all(axis=1)
    F0 = np.where(ok[:, None, None], np.nan_to_num(Fz), 0.0)
    Y0 = np.where(ok[:, None], np.nan_to_num(Yz), 0.0)
    XtX = np.einsum("dtp,dtq->dpq", F0, F0)
    Xty = np.einsum("dtp,dt->dp", F0, Y0)
    return {"cXtX": np.concatenate([np.zeros((1, p, p)), np.cumsum(XtX, axis=0)], axis=0),
            "cXty": np.concatenate([np.zeros((1, p)), np.cumsum(Xty, axis=0)], axis=0),
            "cN": np.concatenate([[0], np.cumsum(ok.astype(int))]), "feat_ok": feat_ok}


def ridge_solve(XtX: np.ndarray, Xty: np.ndarray, lam_rel: float) -> np.ndarray:
    p = XtX.shape[0]
    return np.linalg.solve(XtX + (lam_rel * np.trace(XtX) / p + 1e-12) * np.eye(p), Xty)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def tstat(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0


def risk_walk_forward(Fz: np.ndarray, cp: Dict, target_eval: np.ndarray, horizon: int, window: int) -> Optional[Dict]:
    """
    Rank-IC of the baseline model and the baseline+EWS model against the forward-risk target on
    non-overlapping dates. The fit at row i uses rows <= i - horizon, whose targets have realized
    by day i (no look-ahead).
    """
    n = Fz.shape[0]
    base_idx = list(range(N_BASE))
    rows, ic_e, ic_b = [], [], []
    i = 0
    while i <= n - 1 - horizon:
        a, b = max(0, i - window), i - horizon
        if b < a or not cp["feat_ok"][i]:
            i += 1 if b < a else horizon
            continue
        if cp["cN"][b + 1] - cp["cN"][a] < config.MIN_TRAIN_DAYS:
            i += horizon
            continue
        y = target_eval[i]
        if not np.isfinite(y).all():
            i += horizon
            continue
        XtX, Xty = cp["cXtX"][b + 1] - cp["cXtX"][a], cp["cXty"][b + 1] - cp["cXty"][a]
        try:
            be = ridge_solve(XtX, Xty, config.RIDGE_LAMBDA)
            bb = ridge_solve(XtX[np.ix_(base_idx, base_idx)], Xty[base_idx], config.RIDGE_LAMBDA)
        except np.linalg.LinAlgError:
            i += horizon
            continue
        f = Fz[i]
        rows.append(i)
        ic_e.append(spearman(f @ be, y))
        ic_b.append(spearman(f[:, base_idx] @ bb, y))
        i += horizon
    if len(rows) < 10:
        return None
    return {"rows": np.array(rows), "ic_e": np.array(ic_e), "ic_b": np.array(ic_b)}


def risk_summary(wf: Dict, lo: int, hi: int) -> Dict:
    ic_e, ic_b = wf["ic_e"][lo:hi], wf["ic_b"][lo:hi]
    n = len(ic_e)
    if n < 3:
        return {"n": int(n)}
    return {"n": int(n), "ic_ews": float(ic_e.mean()), "ic_ews_t": tstat(ic_e), "ic_base": float(ic_b.mean()),
            "ic_base_t": tstat(ic_b), "d_ic": float((ic_e - ic_b).mean()), "d_ic_t": tstat(ic_e - ic_b)}


# ---------------------------------------------------------------------------
# Test 3: overlay vs volatility target
# ---------------------------------------------------------------------------
def overlay_exposures(agg_returns: np.ndarray, agg_score: np.ndarray) -> Dict[str, np.ndarray]:
    """Exposure decided with information up to day t (applied to day t+1). Warm-up days are fully invested."""
    s = pd.Series(agg_returns)
    vol21 = s.rolling(21, min_periods=21).std()
    med = vol21.expanding(min_periods=126).median()
    vt = np.clip((med / vol21).values, 0.0, 1.0)
    vt = np.where(np.isfinite(vt), vt, 1.0)
    sc = pd.Series(agg_score)
    q1 = sc.expanding(min_periods=252).quantile(config.OVERLAY_Q1).shift(1)     # causal: excludes today's value
    q2 = sc.expanding(min_periods=252).quantile(config.OVERLAY_Q2).shift(1)
    with np.errstate(invalid="ignore"):
        e_ews = np.where(sc.values >= q2.values, 0.0, np.where(sc.values >= q1.values, 0.5, 1.0))
    e_ews = np.where(np.isfinite(sc.values) & np.isfinite(q1.values), e_ews, 1.0)
    return {"buy_hold": np.ones_like(vt), "vol_target": vt, "ews": e_ews, "ews_vt": vt * e_ews}


def strategy_returns(exposure: np.ndarray, agg_returns: np.ndarray, cost_bps: float) -> np.ndarray:
    """return_{t+1} = e_t * r_{t+1} - |e_t - e_{t-1}| * cost_bps/2 per unit traded."""
    r_next = np.concatenate([agg_returns[1:], [np.nan]])
    turn = np.abs(np.diff(exposure, prepend=exposure[0]))
    return exposure * r_next - turn * cost_bps / 2e4


def perf_stats(r: np.ndarray, exposure: np.ndarray) -> Dict:
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return {}
    eq = np.exp(np.cumsum(r))
    dd = (eq / np.maximum.accumulate(eq) - 1.0).min()
    ann_ret, ann_vol = r.mean() * 252, r.std(ddof=1) * np.sqrt(252)
    return {"n_days": int(len(r)), "ann_return": float(ann_ret), "ann_vol": float(ann_vol),
            "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else 0.0, "max_drawdown": float(dd),
            "calmar": float(ann_ret / abs(dd)) if dd < 0 else None,
            "avg_exposure": float(np.nanmean(exposure)), "annual_turnover": float(np.nanmean(np.abs(np.diff(exposure))) * 252)}


def _sharpe_dd(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sh = R.mean(axis=1) * 252 / (R.std(axis=1, ddof=1) * np.sqrt(252) + 1e-12)
    eq = np.exp(np.cumsum(R, axis=1))
    dd = (eq / np.maximum.accumulate(eq, axis=1) - 1.0).min(axis=1)
    return sh, dd


def block_bootstrap_compare(r_new: np.ndarray, r_base: np.ndarray, block: int, reps: int, seed: int = 0) -> Dict:
    """Circular block bootstrap of the PAIRED difference (new - base) in Sharpe and in max drawdown."""
    ok = np.isfinite(r_new) & np.isfinite(r_base)
    a, b = r_new[ok], r_base[ok]
    m = len(a)
    if m < 2 * block:
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(m / block))
    starts = rng.integers(0, m, size=(reps, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % m
    idx = idx.reshape(reps, -1)[:, :m]
    sh_a, dd_a = _sharpe_dd(a[idx])
    sh_b, dd_b = _sharpe_dd(b[idx])
    d_sh, d_dd = sh_a - sh_b, dd_a - dd_b            # max drawdown is negative: larger = shallower = better
    return {"d_sharpe_mean": float(d_sh.mean()), "d_sharpe_lo": float(np.percentile(d_sh, 5)),
            "d_sharpe_hi": float(np.percentile(d_sh, 95)), "p_sharpe_better": float((d_sh > 0).mean()),
            "d_maxdd_mean": float(d_dd.mean()), "d_maxdd_lo": float(np.percentile(d_dd, 5)),
            "d_maxdd_hi": float(np.percentile(d_dd, 95)), "p_maxdd_better": float((d_dd > 0).mean())}
