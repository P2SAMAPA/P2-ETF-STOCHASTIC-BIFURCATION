"""
validation.py  —  Does the machinery detect what it claims to detect?

Every scenario produces a LOG PRICE (a slow random-walk trend plus a state x
with the stated dynamics) and goes through exactly the same pipeline as real
data: causal EMA detrending -> rolling statistics -> trend -> composite ->
level z-scores -> composite score -> active warning. So whatever the detrending filter does to
the statistics is in the validation too. An "oracle" run (the state itself, no
trend, no detrending) shows what the statistics could do if the filter cost
nothing.

Approach scenarios (a control parameter drifts to the bifurcation at T_CRIT):
  fold          du = kappa*(r(t) - u^2) dt + sigma dW,  r: 1 -> 0
  hopf          dz = ((mu(t) + i*omega) z - |z|^2 z) dt + sigma dW_c, mu: -0.1 -> 0
  transcritical du = kappa*(mu(t) u - u^2) dt + sigma dW, mu: -1 -> 0
Null scenarios (nothing is approaching anything):
  iid     i.i.d. returns (a random walk)
  garch   GARCH(1,1) returns — volatility clustering, the main imposter
  ou      a STABLE mean-reverting state with constant parameters (the correct
          null for the theory)

All approach tests are anchored at the bifurcation point T_CRIT (where the
theory applies), not at the later moment the state visibly collapses; runs that
tip over before T_CRIT-30 (noise can push a system over a shrinking barrier
before the bifurcation proper — a real limit on early warning) are excluded and
counted.

Reported: AUC of the composite score at T_CRIT-30 against nulls; the share of
days with an ACTIVE warning in [T_CRIT-60, T_CRIT-5] against a comparable
window of each null; the overall null active share; and the share of
near-bifurcation days on which the AR(2) fit says "complex pair" (type
discrimination), swept over the Hopf oscillation period.
"""

import numpy as np
from typing import Dict, List

import config
import ews_model as em

N_DAYS = 1500
T_START = 400          # the control parameter starts drifting here
T_CRIT = 1150          # ... and reaches the bifurcation here
CRASH_X = -0.30        # a state below this has visibly collapsed
TREND_SD = 0.003
SPAN_WINDOW_GRID = [(126, 126), (63, 63), (252, 126)]
HOPF_PERIODS = [10, 20, 40]


def _trend(n, reps, rng):
    return np.cumsum(rng.normal(0, TREND_SD, (n, reps)), axis=0)


def _ramp(t, lo, hi):
    """Linear control-parameter path: lo at T_START, hi at T_CRIT (continues past it)."""
    return lo + (hi - lo) * max(t - T_START, 0) / (T_CRIT - T_START)


def sim_fold(n, reps, rng, r0=1.0, kappa=0.05, sigma=0.08, s=0.15, sub=4):
    u = np.full(reps, np.sqrt(r0)); X = np.zeros((n, reps)); dt = 1.0 / sub
    for t in range(n):
        r = _ramp(t, r0, 0.0)
        for _ in range(sub):
            u = np.maximum(u + kappa * (r - u * u) * dt + sigma * np.sqrt(dt) * rng.normal(size=reps), -6.0)
        X[t] = s * u
    return X


def sim_hopf(n, reps, rng, mu0=0.10, period=40.0, sigma=0.012, sub=2):
    # Exponential integrator: the linear part is propagated EXACTLY. (Plain explicit Euler has a per-step
    # modulus |1 + (mu + i*omega)*dt| that exceeds 1 for short periods — 1.0006 at a 10-day period even
    # with mu = -0.1 — so the scheme itself manufactures growth and the bifurcation is no longer where it is stated.)
    omega = 2 * np.pi / period
    z = np.zeros(reps, dtype=complex); X = np.zeros((n, reps)); dt = 1.0 / sub
    for t in range(n):
        mu = _ramp(t, -mu0, 0.0)
        prop = np.exp((mu + 1j * omega) * dt)
        for _ in range(sub):
            noise = sigma * np.sqrt(dt) * (rng.normal(size=reps) + 1j * rng.normal(size=reps))
            z = prop * z - (np.abs(z) ** 2) * z * dt + noise
        X[t] = z.real
    return X


def sim_transcritical(n, reps, rng, kappa=0.05, sigma=0.02, s=0.15, sub=4):
    u = np.zeros(reps); X = np.zeros((n, reps)); dt = 1.0 / sub
    for t in range(n):
        mu = _ramp(t, -1.0, 0.0)
        for _ in range(sub):
            u = np.maximum(u + kappa * (mu * u - u * u) * dt + sigma * np.sqrt(dt) * rng.normal(size=reps), -6.0)
        X[t] = s * u
    return X


def sim_ou(n, reps, rng, lam=0.10, sd=0.03):
    phi = np.exp(-lam); sig = sd * np.sqrt(1 - phi ** 2)
    x = np.zeros(reps); X = np.zeros((n, reps))
    for t in range(n):
        x = phi * x + sig * rng.normal(size=reps)
        X[t] = x
    return X


def sim_iid_logprice(n, reps, rng, sd=0.01):
    return np.cumsum(rng.normal(0, sd, (n, reps)), axis=0)


def sim_garch_logprice(n, reps, rng, omega=2e-6, alpha=0.08, beta=0.90):
    var = np.full(reps, omega / (1 - alpha - beta)); r = np.zeros((n, reps))
    for t in range(n):
        e = np.sqrt(var) * rng.normal(size=reps)
        r[t] = e
        var = omega + alpha * e * e + beta * var
    return np.cumsum(r, axis=0)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(random positive > random negative), ties half — via ranks (Mann-Whitney), so memory stays O(n)."""
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _tipped_early(X: np.ndarray) -> np.ndarray:
    """Runs whose state visibly collapsed before T_CRIT - 30 (excluded from the approach statistics)."""
    t = np.arange(X.shape[0])[:, None]
    return ((X < CRASH_X) & (t > T_START) & (t < T_CRIT - 30)).any(axis=0)


def build_scenarios(reps: int, seed: int, hopf_period: float = 40.0) -> Dict[str, Dict]:
    rng = np.random.default_rng(seed)
    n = N_DAYS
    out = {}
    for name, X in [("fold", sim_fold(n, reps, rng)), ("hopf", sim_hopf(n, reps, rng, period=hopf_period)),
                    ("transcritical", sim_transcritical(n, reps, rng))]:
        out[name] = {"logp": _trend(n, reps, rng) + X, "state": X, "excluded": _tipped_early(X)}
    ou_state = sim_ou(n, reps, rng)
    out["iid"] = {"logp": sim_iid_logprice(n, reps, rng), "state": None, "excluded": np.zeros(reps, bool)}
    out["garch"] = {"logp": sim_garch_logprice(n, reps, rng), "state": None, "excluded": np.zeros(reps, bool)}
    out["ou"] = {"logp": _trend(n, reps, rng) + ou_state, "state": ou_state, "excluded": np.zeros(reps, bool)}
    return out


def _approach_metrics(feats, d):
    keep = np.where(~d["excluded"])[0]
    act = em.warning_active(feats["ews_score"])
    det = [float(act[T_CRIT - 60: T_CRIT - 5, j].mean()) for j in keep]
    score = feats["ews_score"][T_CRIT - 30, keep]
    near = np.nanmean(feats["complex"][T_CRIT - 150: T_CRIT - 5][:, keep].astype(float), axis=0)
    return {"n_usable": int(len(keep)), "n_total": int(len(d["excluded"])),
            "early_tip_share": float(d["excluded"].mean()),
            "active_share": float(np.mean(det)) if len(keep) else None,
            "mean_score": float(np.nanmean(score)) if len(keep) else None,
            "complex_share_near": float(np.nanmean(near)) if len(keep) else None,
            "_score": score}


def run_validation(reps: int = None, seed: int = 7, configs: List = None) -> Dict:
    reps = config.SIM_REPS if reps is None else reps
    configs = SPAN_WINDOW_GRID if configs is None else configs
    sc = build_scenarios(reps, seed)
    rng = np.random.default_rng(seed + 1)
    window_ends = rng.integers(T_CRIT - 350, T_CRIT - 50, size=40)
    results = {}
    for span, W in configs:
        feats = {nm: em.ews_features(d["logp"], span, W) for nm, d in sc.items()}
        null = {}
        for nm in ["iid", "garch", "ou"]:
            act = em.warning_active(feats[nm]["ews_score"])
            wr = np.mean([act[e - 60: e - 5].mean() for e in window_ends])
            lo, hi = T_START + 300, T_CRIT
            null[nm] = {"active_share": float(wr), "active_share_overall": float(act[lo:hi].mean()),
                        "complex_share": float(np.nanmean(feats[nm]["complex"][lo:hi].astype(float))),
                        "_score": feats[nm]["ews_score"][T_CRIT - 350: T_CRIT - 50].ravel()}
        approach = {}
        for nm in ["fold", "hopf", "transcritical"]:
            m = _approach_metrics(feats[nm], sc[nm])
            for nn in null:
                m["auc_vs_" + nn] = auc(m["_score"], null[nn]["_score"])
            m.pop("_score")
            approach[nm] = m
        for nn in null:
            null[nn].pop("_score")
        # oracle: the state itself (no trend, no detrending), against a stable-OU state
        oracle_feats = {nm: em.ews_from_state(d["state"], W) for nm, d in sc.items() if d["state"] is not None}
        oracle = {}
        for nm in ["fold", "hopf", "transcritical"]:
            m = _approach_metrics(oracle_feats[nm], sc[nm])
            m["auc_vs_ou"] = auc(m["_score"], oracle_feats["ou"]["ews_score"][T_CRIT - 350: T_CRIT - 50].ravel())
            m.pop("_score")
            oracle[nm] = m
        results["span%d_win%d" % (span, W)] = {"approach": approach, "null": null, "oracle": oracle,
                                               "span": span, "window": W}
    # type discrimination vs oscillation period (primary detrending only)
    span, W = configs[0]
    types = {}
    for period in HOPF_PERIODS:
        scp = build_scenarios(reps, seed + 100 + period, hopf_period=float(period))
        f = em.ews_features(scp["hopf"]["logp"], span, W)
        keep = ~scp["hopf"]["excluded"]
        cs = np.nanmean(f["complex"][T_CRIT - 150: T_CRIT - 5][:, keep].astype(float))
        f2 = em.ews_features(scp["fold"]["logp"], span, W)
        keep2 = ~scp["fold"]["excluded"]
        fs = np.nanmean(f2["complex"][T_CRIT - 150: T_CRIT - 5][:, keep2].astype(float))
        types[str(period)] = {"hopf_complex_share": float(cs), "fold_complex_share": float(fs)}
    return {"reps": reps, "n_days": N_DAYS, "t_start": T_START, "t_crit": T_CRIT, "results": results,
            "type_discrimination": types}
