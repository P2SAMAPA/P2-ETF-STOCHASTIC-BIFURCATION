"""
trainer.py  —  P2 STOCHASTIC-BIFURCATION

Per ETF and per day: early-warning statistics of critical slowing down (lag-1
autocorrelation, variance, skewness, bimodality) and an AR(2) stability fit
whose dominant root says how close the local dynamics are to a bifurcation and
of which kind (real root -> fold/transcritical class; complex pair -> Hopf
class). The warning is the composite `ews_score`, active at >= WARNING_SCORE.

Three tests of whether the warning is worth anything, each against volatility,
because rising variance of a detrended price is largely just volatility:

  1. LEAD   — event rate (collapse / volatility spike) while the warning is
              active, vs the base rate and vs a circular-shift chance level,
              side by side with the same test for a plain volatility trigger.
  2. RISK   — do the EWS features rank which ETFs will suffer the worst
              forward drawdown BEYOND vol21/vol63/drawdown/momentum? One
              PRIMARY configuration is fixed in advance and carries the
              verdict; the whole grid is shown as robustness.
  3. OVERLAY— does de-risking on the universe-average warning beat a plain
              volatility target (circular block bootstrap of the paired
              difference in Sharpe and maximum drawdown)?

Simulation validation (fold / Hopf / transcritical approaches vs i.i.d., GARCH
and stable-OU nulls, through the same detrending) is re-run every time and
shipped in the results so the dashboard can show what the machinery can and
cannot detect.
"""

import os
import sys
import zlib
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import ews_model as em
import validation as val
from data_manager import load_master_data, validate_data

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _clean(d):
    if isinstance(d, dict):
        return {k: _clean(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_clean(v) for v in d]
    if isinstance(d, (float, np.floating)):
        return _f(d)
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.bool_,)):
        return bool(d)
    return d


def bif_class(score: float, complex_pair: bool) -> str:
    if not np.isfinite(score) or score < config.WARNING_SCORE:
        return "stable"
    return "complex pair (Hopf class)" if complex_pair else "real root (fold / transcritical class)"


def run_lead_tests(returns, ews, vol_level_z, ev_by_kind, is_primary, strata):
    """
    State-based lead tests, volatility-controlled (Mantel-Haenszel within 50 volatility-state strata):
    the composite, and for the primary config its components and a plain volatility trigger for reference.
    """
    out = {}
    triggers = {"ews": ews["ews_score"]}
    if is_primary:
        triggers.update({"z_var_only": ews["z_var"], "z_ac1_only": ews["z_ac1"], "vol_level": vol_level_z})
    for kind, ev in ev_by_kind.items():
        for tname, series in triggers.items():
            valid = ev["valid"] & np.isfinite(series)
            out["%s|%s" % (tname, kind)] = em.pooled_state_test(em.warning_active(series), ev[kind], valid, strata,
                                                                  seed=zlib.crc32((tname + kind).encode()) % 10000)
    return out


def analyze_universe(name: str, tickers: List[str], prices_df: pd.DataFrame) -> Optional[Dict]:
    available = [t for t in tickers if t in prices_df.columns]
    if not available:
        return None
    px_df = prices_df[available]
    px_df = px_df[~px_df.isna().any(axis=1)]
    if len(px_df) < 900:
        logger.warning(f"Not enough data for {name}")
        return None

    values = px_df.values
    log_prices = np.log(values)[1:]
    returns = np.diff(np.log(values), axis=0)
    dates = [d.strftime("%Y-%m-%d") for d in px_df.index[1:]]
    n, T = returns.shape
    gids = em.group_ids_from_map(available, config.ASSET_CLASS_MAP, config.MIN_CLASS_SIZE)

    base = em.baseline_features(returns, log_prices)
    vol_level_z = em.rolling_z(base["vol21"], config.Z_WINDOW, config.Z_MIN)
    strata = em.vol_strata(base["vol21"], base["vol63"], vol_level_z)
    ev = em.event_flags(returns, config.EVENT_HORIZON, config.EVENT_K, config.VOL_SPIKE_RATIO)
    ev_by_kind = {"collapse": {"valid": ev["valid"], "collapse": ev["collapse"]},
                  "vol_spike": {"valid": ev["valid"], "vol_spike": ev["vol_spike"]}}
    mae = {h: em.forward_mae(returns, h) for h in config.RISK_HORIZONS}

    lead, risk_grid, ews_store = {}, [], {}
    primary_cfg = (config.PRIMARY["trend_span"], config.PRIMARY["ews_window"])
    for span in config.TREND_SPANS:
        for W in config.EWS_WINDOWS:
            key = "span%d_win%d" % (span, W)
            ews = em.ews_features(log_prices, span, W)
            ews_store[(span, W)] = ews
            lead[key] = run_lead_tests(returns, ews, vol_level_z, ev_by_kind, (span, W) == primary_cfg, strata)

            feat_raw = [base[k] for k in em.BASE_NAMES] + [ews[k] for k in em.EWS_NAMES]
            for neut in config.RISK_NEUTRAL_OPTIONS:
                g_arg = gids if neut else None
                Fz = np.stack([em.standardize(a, g_arg) for a in feat_raw], axis=2)
                for h in config.RISK_HORIZONS:
                    cp = em.build_cross_products(Fz, em.standardize(mae[h], g_arg))
                    target_eval = em.demean_by_group(mae[h], gids) if neut else mae[h]
                    for window in config.RISK_WINDOWS:
                        wf = em.risk_walk_forward(Fz, cp, target_eval, h, window)
                        if wf is None:
                            continue
                        m = len(wf["rows"])
                        split = int(m * config.SELECTION_FRACTION)
                        risk_grid.append({"trend_span": span, "ews_window": W, "horizon": h, "window": window,
                                          "neutralize": neut, "n_periods": m,
                                          "full": em.risk_summary(wf, 0, m), "holdout": em.risk_summary(wf, split, m)})
        logger.info(f"  [{name}] trend span {span}: done")

    # ---- the pre-registered primary risk cell and the grid-wide picture
    pc = config.PRIMARY
    prim = [g for g in risk_grid if all(g[k] == pc[k] for k in pc)]
    primary = prim[0] if prim else None
    hold = [g["holdout"] for g in risk_grid if g["holdout"].get("n", 0) >= config.MIN_HOLDOUT_PERIODS]
    risk_summary = {
        "n_cells": len(risk_grid), "n_cells_holdout": len(hold),
        "share_d_ic_positive": float(np.mean([h["d_ic"] > 0 for h in hold])) if hold else None,
        "mean_d_ic": float(np.mean([h["d_ic"] for h in hold])) if hold else None,
        "mean_d_ic_t": float(np.mean([h["d_ic_t"] for h in hold])) if hold else None,
        "share_t_above_165": float(np.mean([h["d_ic_t"] >= config.IC_T_SUPPORTED for h in hold])) if hold else None,
    }

    # ---- overlay on the equal-weight universe, driven by the primary configuration's average warning
    ews_p = ews_store[primary_cfg]
    agg_ret = returns.mean(axis=1)
    agg_score = np.nanmean(ews_p["ews_score"], axis=1)
    expo = em.overlay_exposures(agg_ret, agg_score)
    start = int(np.argmax(np.isfinite(agg_score))) + 252
    rets = {k: em.strategy_returns(v, agg_ret, config.TRADING_COST_BPS) for k, v in expo.items()}
    split_d = start + int((n - 1 - start) * config.SELECTION_FRACTION)
    overlay = {"start_date": dates[start], "holdout_start_date": dates[split_d], "strategies": {}, "bootstrap": {}}
    for k in expo:
        overlay["strategies"][k] = {"full": em.perf_stats(rets[k][start:n - 1], expo[k][start:n - 1]),
                                    "holdout": em.perf_stats(rets[k][split_d:n - 1], expo[k][split_d:n - 1])}
    for a, b in [("ews", "vol_target"), ("ews_vt", "vol_target"), ("ews_vt", "buy_hold")]:
        overlay["bootstrap"]["%s_vs_%s" % (a, b)] = em.block_bootstrap_compare(
            rets[a][split_d:n - 1], rets[b][split_d:n - 1], config.BOOT_BLOCK, config.BOOT_REPS)
    ci = np.arange(0, n, config.SERIES_STEP)
    ci = ci[np.isfinite(agg_score[ci])]
    overlay["series"] = {"dates": [dates[i] for i in ci], "agg_score": [_f(agg_score[i]) for i in ci],
                         "share_active": [_f(np.nanmean(em.warning_active(ews_p["ews_score"][i]))) for i in ci],
                         "exposure_ews": [_f(expo["ews"][i]) for i in ci], "exposure_vt": [_f(expo["vol_target"][i]) for i in ci]}

    # ---- today's monitor
    last = n - 1
    snap = []
    for j, t in enumerate(available):
        sc = ews_p["ews_score"][last, j]
        snap.append({"ticker": t, "ews_score": sc, "z_ac1": ews_p["z_ac1"][last, j], "z_var": ews_p["z_var"][last, j],
                     "ac1": ews_p["ac1"][last, j], "margin": ews_p["margin"][last, j], "rho": ews_p["rho"][last, j],
                     "theta": ews_p["theta"][last, j], "bimod": ews_p["bimod"][last, j],
                     "vol_z": vol_level_z[last, j], "asset_class": config.ASSET_CLASS_MAP.get(t),
                     "class": bif_class(sc, bool(ews_p["complex"][last, j])),
                     "active": bool(np.isfinite(sc) and sc >= config.WARNING_SCORE)})
    snap.sort(key=lambda r: -(r["ews_score"] if np.isfinite(r["ews_score"]) else -99))

    # ---- scorecard
    lp = lead["span%d_win%d" % primary_cfg]
    def rr(tr, kind):
        t = lp.get("%s|%s" % (tr, kind), {})
        return t.get("relative_risk"), t.get("p_value")
    lead_rows = {}
    for kind in ["collapse", "vol_spike"]:
        r_e, p_e = rr("ews", kind); r_v, p_v = rr("vol_level", kind)
        lead_rows[kind] = {"rr_ews": r_e, "p_ews": p_e, "rr_vol": r_v, "p_vol": p_v}
    # The verdict rests on COLLAPSE events only: calibrated at the pipeline's scale (4% false positives, mean
    # controlled RR 0.99 on regime-volatility noise). Volatility-spike events are NOT reliably controlled
    # (20% false positives, mean RR 1.13) — a variance-based warning responding to volatility events is
    # partly circular — so they are shown for reference and excluded from the verdict.
    lc = lead_rows["collapse"]
    lead_pass = bool(lc["rr_ews"] is not None and lc["p_ews"] is not None and lc["rr_ews"] >= 1.2 and lc["p_ews"] < 0.05)
    risk_pass = bool(primary is not None and primary["holdout"].get("n", 0) >= config.MIN_HOLDOUT_PERIODS
                     and primary["holdout"]["d_ic"] > 0 and primary["holdout"]["d_ic_t"] >= config.IC_T_SUPPORTED)
    bs = overlay["bootstrap"].get("ews_vt_vs_vol_target", {})
    overlay_pass = bool(bs and bs.get("p_sharpe_better", 0) >= 0.90 and bs.get("p_maxdd_better", 0) >= 0.90)
    npass = int(lead_pass) + int(risk_pass) + int(overlay_pass)
    scorecard = {"lead_collapse_vol_controlled": lead_pass, "risk_adds_beyond_vol": risk_pass, "overlay_beats_vol_target": overlay_pass,
                 "n_pass": npass, "lead_rows": lead_rows,
                 "verdict": ["No evidence it adds beyond volatility", "Weak: one test passes", "Partial evidence: two tests pass",
                             "Evidence on all three tests"][npass]}

    n_lead_tests = sum(1 for k in lead for t in lead[k].values() if t.get("p_value") is not None)
    return {"tickers": available, "n_days": n, "last_date": dates[-1], "snapshot": snap, "lead": lead,
            "risk_grid": risk_grid, "risk_primary": primary, "risk_summary": risk_summary, "overlay": overlay,
            "scorecard": scorecard, "n_lead_tests": n_lead_tests, "class_groups": {t: int(g) for t, g in zip(available, gids)}}


def run_trainer() -> Dict:
    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {"run_date": run_date, "algorithm": "STOCHASTIC-BIFURCATION", "universes": {}, "universe_meta": {},
               "settings": {"warning_score": config.WARNING_SCORE, "event_horizon": config.EVENT_HORIZON,
                            "event_k": config.EVENT_K, "vol_spike_ratio": config.VOL_SPIKE_RATIO, "primary": config.PRIMARY,
                            "trading_cost_bps": config.TRADING_COST_BPS, "trend_spans": config.TREND_SPANS,
                            "ews_windows": config.EWS_WINDOWS, "z_window": config.Z_WINDOW,
                            "feature_names": em.FEATURE_NAMES}}
    total_tests = 0
    for name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {name}...")
        out = analyze_universe(name, tickers, prices_df)
        if out is None:
            continue
        total_tests += out["n_lead_tests"]
        results["universes"][name] = _clean({k: v for k, v in out.items() if k not in ("tickers", "n_days", "last_date", "class_groups")})
        results["universe_meta"][name] = {"tickers": out["tickers"], "n_days": out["n_days"], "last_date": out["last_date"]}
        sc = out["scorecard"]
        pr = out["risk_primary"]
        logger.info(f"  scorecard: {sc['verdict']} | lead={sc['lead_collapse_vol_controlled']} risk={sc['risk_adds_beyond_vol']} overlay={sc['overlay_beats_vol_target']}")
        if pr:
            h = pr["holdout"]
            logger.info(f"  primary risk cell holdout: n={h.get('n')} IC(ews)={h.get('ic_ews', 0):+.3f} vs IC(base)={h.get('ic_base', 0):+.3f}, dIC t={h.get('d_ic_t', 0):+.2f}")
        active = sum(1 for r in out["snapshot"] if r["active"])
        logger.info(f"  today: {active}/{len(out['snapshot'])} ETFs with an active warning")
    results["settings"]["n_lead_tests"] = total_tests

    logger.info("\n🧪 Running simulation validation...")
    results["validation"] = _clean(val.run_validation(reps=config.SIM_REPS))

    output_path = f"bifurcation_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, separators=(",", ":"), default=str)
    logger.info(f"\n💾 Saved: {output_path}")
    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")
    return results


if __name__ == "__main__":
    run_trainer()
