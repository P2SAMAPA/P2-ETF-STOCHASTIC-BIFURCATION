import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import glob
import os
import plotly.graph_objects as go

st.set_page_config(page_title="P2 Stochastic-Bifurcation", page_icon="🌋", layout="wide",
                   initial_sidebar_state="collapsed")

PRIMARY = "#b45309"
INK = "#0f172a"
SUBTLE = "#64748b"
CARD_BORDER = "#e2e8f0"
PAGE_BG = "#f8fafc"
GOOD, WARN, BAD = "#16a34a", "#d97706", "#64748b"
CLASS_COLORS = {"Fixed Income": "#0e7490", "Real Estate": "#7c3aed", "Commodity": "#b45309",
                "Equity-Broad": "#be123c", "Equity-Sector": "#15803d"}

CSS = """
<style>
    .stApp { background-color: __PAGE_BG__; }
    #MainMenu, footer {visibility: hidden;}
    .app-title { font-size: 1.9rem; font-weight: 800; color: __INK__; margin: 0; }
    .app-subtitle { color: __SUBTLE__; font-size: 0.95rem; margin-bottom: 1.1rem; }
    .universe-heading { font-size: 1.15rem; font-weight: 700; color: __INK__; margin: 0 0 0.15rem 0; }
    .kpi-card { background: #ffffff; border: 1px solid __BORDER__; border-radius: 12px; padding: 0.8rem 1rem; text-align: center; }
    .kpi-value { font-size: 1.3rem; font-weight: 800; color: __INK__; }
    .kpi-label { font-size: 0.7rem; color: __SUBTLE__; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.15rem; }
    .verdict { background: #ffffff; border: 1px solid __BORDER__; border-left: 6px solid var(--accent);
        border-radius: 10px; padding: 0.8rem 1.1rem; font-size: 0.9rem; color: __INK__; margin: 0.5rem 0 0.9rem 0; }
    .badge { display: inline-block; font-size: 0.75rem; font-weight: 800; text-transform: uppercase;
        letter-spacing: 0.05em; padding: 0.15rem 0.6rem; border-radius: 999px; color: white; background: var(--accent); margin-right: 0.5rem; }
    .note { font-size: 0.82rem; color: __SUBTLE__; margin: 0.4rem 0 0.9rem 0; }
    div[data-testid="stDataFrame"] { border: 1px solid __BORDER__; border-radius: 10px; overflow: hidden; }
</style>
"""
CSS = (CSS.replace("__PAGE_BG__", PAGE_BG).replace("__INK__", INK).replace("__SUBTLE__", SUBTLE)
       .replace("__BORDER__", CARD_BORDER))
st.markdown(CSS, unsafe_allow_html=True)

RESULTS_REPO = "P2SAMAPA/p2-etf-stochastic-bifurcation-results"


def _find_latest_hf_result(repo_id):
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo_id, repo_type="dataset")
        names = sorted(f for f in files if f.startswith("bifurcation_results_") and f.endswith(".json"))
        if not names:
            return None, "No bifurcation_results_*.json files found in " + repo_id + "."
        return names[-1], None
    except Exception as e:
        return None, "Could not list files in " + repo_id + ": " + str(e)


@st.cache_data(show_spinner="Loading results...")
def load_data(_cache_key=""):
    notes = []
    local = glob.glob("bifurcation_results_*.json")
    if local:
        latest = sorted(local)[-1]
        try:
            with open(latest, "r") as f:
                return json.load(f), ["Loaded local file: " + latest]
        except Exception as e:
            notes.append("Found local file " + latest + " but couldn't parse it: " + str(e))
    name, err = _find_latest_hf_result(RESULTS_REPO)
    if err:
        notes.append(err)
    else:
        try:
            url = "https://huggingface.co/datasets/" + RESULTS_REPO + "/resolve/main/" + name
            headers = {}
            token = os.environ.get("HF_TOKEN")
            if token:
                headers["Authorization"] = "Bearer " + token
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json(), ["Loaded from HF dataset: " + name]
            notes.append("Fetching " + name + " returned HTTP " + str(resp.status_code) + ".")
        except Exception as e:
            notes.append("Fetching " + name + " failed: " + str(e))
    return None, notes


def kpi(label, value):
    return ('<div class="kpi-card"><div class="kpi-value">' + value + '</div>'
            '<div class="kpi-label">' + label + '</div></div>')


def fnum(x, fmt="%.2f", na="n/a"):
    return na if x is None else (fmt % x)


def check(ok):
    return "✅" if ok else "❌"


def render_agg_series(overlay, thr, key):
    s = (overlay or {}).get("series")
    if not s or not s.get("dates"):
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["dates"], y=s["agg_score"], mode="lines", name="Universe-average warning score",
                             line=dict(color=PRIMARY, width=1.8)))
    fig.add_trace(go.Scatter(x=s["dates"], y=[v * 4 if v is not None else None for v in s["share_active"]], mode="lines",
                             name="Share of ETFs with an active warning (x4, for scale)", line=dict(color="#94a3b8", width=1.2)))
    fig.add_hline(y=thr, line_dash="dash", line_color="#334155")
    fig.update_layout(height=290, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      yaxis=dict(title="Score", gridcolor="#eef2f7"), xaxis=dict(gridcolor="#eef2f7"), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, key="agg_" + key)


def render_exposure(overlay, key):
    s = (overlay or {}).get("series")
    if not s or not s.get("dates"):
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["dates"], y=s["exposure_vt"], mode="lines", name="Volatility target",
                             line=dict(color="#0e7490", width=1.4)))
    fig.add_trace(go.Scatter(x=s["dates"], y=s["exposure_ews"], mode="lines", name="Warning overlay",
                             line=dict(color=PRIMARY, width=1.6)))
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
                      yaxis=dict(title="Exposure", range=[-0.05, 1.05], gridcolor="#eef2f7"), xaxis=dict(gridcolor="#eef2f7"),
                      hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, key="expo_" + key)


def lead_table(lead_cfg):
    rows = []
    labels = {"ews": "Warning composite", "z_var_only": "Variance component only", "z_ac1_only": "Autocorrelation component only",
              "vol_level": "Plain volatility trigger (21d vol z-score)"}
    for k, t in lead_cfg.items():
        if not t.get("available") or t.get("relative_risk") is None:
            continue
        trig, kind = k.split("|")
        rows.append({"Trigger": labels.get(trig, trig),
                     "Event": "collapse" if kind == "collapse" else "volatility spike (not reliably controlled)",
                     "Active share": t["active_share"], "Event rate when active": t["rate_active"], "Base rate": t["rate_base"],
                     "Chance (shifted)": t["rate_chance"], "Relative risk (naive)": t.get("relative_risk_naive"),
                     "Relative risk (vol-controlled)": t["relative_risk"], "p-value": t["p_value"]})
    return pd.DataFrame(rows)


def main():
    if "cache_bust" not in st.session_state:
        st.session_state.cache_bust = 0
    head, refresh = st.columns([5, 1])
    with head:
        st.markdown('<div class="app-title">🌋 P2 Stochastic-Bifurcation</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Early-warning statistics of critical slowing down and an AR(2) stability fit. '
                    'Tested against volatility, because rising variance of a detrended price is largely just volatility.</div>',
                    unsafe_allow_html=True)
    with refresh:
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.cache_bust += 1
            st.rerun()

    data, notes = load_data(_cache_key=str(st.session_state.cache_bust))
    if not data:
        st.error("No results found.")
        with st.expander("Why? (click for details)", expanded=True):
            for n in notes or ["No diagnostic information was returned."]:
                st.write("• " + n)
            st.write("Run `python trainer.py` to create a local `bifurcation_results_*.json`, or check the HF results dataset and HF_TOKEN.")
        return

    st.caption("🕒 Results generated: **" + str(data.get("run_date", "Unknown")) + "**")
    if notes:
        st.caption("📡 " + notes[0])
    stt = data.get("settings", {})
    thr = stt.get("warning_score", 1.5)

    tab1, tab2, tab3 = st.tabs(["🌡️ Stability Monitor", "⚖️ Do the warnings add anything?", "🧪 Simulation validation"])

    with tab1:
        st.markdown('<div class="note">Each ETF\'s state is its log price minus a slow trend. A warning is active when the '
                    'autocorrelation and variance of that state are well above their own past (score >= ' + str(thr)
                    + '). A warning is a statement about recent dynamics, not a forecast of a crash; the next tab shows '
                    'whether it carries any information beyond volatility, and the third shows what the method can and '
                    'cannot detect even in ideal simulations.</div>', unsafe_allow_html=True)
        for uni, u in data.get("universes", {}).items():
            snap = u["snapshot"]
            sc = u["scorecard"]
            accent = [BAD, WARN, WARN, GOOD][sc["n_pass"]]
            active = [r for r in snap if r["active"]]
            st.markdown('<div class="universe-heading">' + uni.replace("_", " ").title() + '</div>', unsafe_allow_html=True)
            st.markdown('<div class="verdict" style="--accent: ' + accent + ';"><span class="badge">' + sc["verdict"] + '</span>'
                        + str(len(active)) + ' of ' + str(len(snap)) + ' ETFs have an active warning today. The warning '
                        'passes ' + str(sc["n_pass"]) + ' of 3 tests against volatility (see the next tab).</div>', unsafe_allow_html=True)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("###### Universe-average warning over time")
                render_agg_series(u["overlay"], thr, "mon_" + uni)
            with c2:
                st.markdown("###### ETFs ranked by warning score today")
                rows = [{"ETF": r["ticker"], "Score": r["ews_score"], "Class": r["class"], "Autocorr z": r["z_ac1"],
                         "Variance z": r["z_var"], "Margin (1-rho)": r["margin"], "Volatility z": r["vol_z"]} for r in snap[:12]]
                st.dataframe(pd.DataFrame(rows).style.format({"Score": "{:+.2f}", "Autocorr z": "{:+.2f}", "Variance z": "{:+.2f}",
                                                              "Margin (1-rho)": "{:.2f}", "Volatility z": "{:+.2f}"}, na_rep="n/a"),
                             use_container_width=True, hide_index=True)
                st.caption("Class: 'real root' = the AR(2) dominant root is real (fold / transcritical approach — early-warning "
                           "statistics cannot tell these apart); 'complex pair' = Hopf-class. Simulation shows the complex-pair "
                           "label is only reliable for oscillation periods of roughly 10-20 days.")
            st.markdown("<hr style='margin: 1.4rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

    with tab2:
        n_tests = stt.get("n_lead_tests", 0)
        st.markdown('<div class="note">Three tests, each against volatility. Pass rates on pure noise (24 datasets): the lead '
                    'test about 10-17%, the risk test 0%, the overlay test 0%, and no dataset passed two — so a single pass is '
                    'weak evidence and two or more are needed before it means much. ' + str(n_tests) + ' lead tests are run in '
                    'total; the verdict uses one pre-registered test per universe.</div>', unsafe_allow_html=True)
        for uni, u in data.get("universes", {}).items():
            sc = u["scorecard"]
            st.markdown('<div class="universe-heading">' + uni.replace("_", " ").title() + '</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            lc = sc["lead_rows"].get("collapse", {})
            with cols[0]:
                st.markdown(kpi(check(sc["lead_collapse_vol_controlled"]) + " Lead (collapse, vol-controlled)",
                                "RR " + fnum(lc.get("rr_ews")) + ", p=" + fnum(lc.get("p_ews"), "%.3f")), unsafe_allow_html=True)
            ph = (u.get("risk_primary") or {}).get("holdout", {})
            with cols[1]:
                st.markdown(kpi(check(sc["risk_adds_beyond_vol"]) + " Risk ranking beyond volatility",
                                "dIC " + fnum(ph.get("d_ic"), "%+.3f") + ", t=" + fnum(ph.get("d_ic_t"), "%+.2f")), unsafe_allow_html=True)
            bs = u["overlay"]["bootstrap"].get("ews_vt_vs_vol_target", {})
            with cols[2]:
                st.markdown(kpi(check(sc["overlay_beats_vol_target"]) + " Overlay vs volatility target",
                                "P(Sharpe) " + fnum(bs.get("p_sharpe_better")) + ", P(DD) " + fnum(bs.get("p_maxdd_better"))), unsafe_allow_html=True)

            st.markdown("###### 1. Lead test: do collapses cluster while the warning is active? (primary configuration)")
            st.caption("Active-day event rate against the base rate and against a chance level from circularly shifting each ETF's "
                       "warning flags. 'Vol-controlled' compares active and inactive days only within 50 volatility-state strata "
                       "(Mantel-Haenszel), which is what removes the fact that events are defined relative to lagging trailing "
                       "volatility. Volatility-spike events are shown for reference only: a variance-based warning responding to "
                       "volatility events is partly circular, and simulation showed that control to be unreliable for them.")
            df = lead_table(u["lead"].get("span%d_win%d" % (stt["primary"]["trend_span"], stt["primary"]["ews_window"]), {}))
            if not df.empty:
                st.dataframe(df.style.format({"Active share": "{:.1%}", "Event rate when active": "{:.1%}", "Base rate": "{:.1%}",
                                              "Chance (shifted)": "{:.1%}", "Relative risk (naive)": "{:.2f}",
                                              "Relative risk (vol-controlled)": "{:.2f}", "p-value": "{:.3f}"}, na_rep="n/a"),
                             use_container_width=True, hide_index=True)

            st.markdown("###### 2. Risk ranking: do the warning features rank forward drawdown beyond volatility and momentum?")
            rs = u["risk_summary"]
            pr = u.get("risk_primary")
            if pr:
                h = pr["holdout"]
                st.caption("Primary configuration (fixed in advance): holdout " + str(h.get("n")) + " periods, rank-IC of the "
                           "baseline-plus-warning model " + fnum(h.get("ic_ews"), "%+.3f") + " vs baseline alone "
                           + fnum(h.get("ic_base"), "%+.3f") + " (difference t=" + fnum(h.get("d_ic_t"), "%+.2f")
                           + "). Across all " + str(rs["n_cells_holdout"]) + " configurations the warning features improved the IC in "
                           + ("%.0f" % (100 * (rs.get("share_d_ic_positive") or 0))) + "% of cells (mean t "
                           + fnum(rs.get("mean_d_ic_t"), "%+.2f") + "). Adding irrelevant features lowers IC slightly, so a mean "
                           "slightly below zero is the no-information result.")
            grid = u["risk_grid"]
            if grid:
                gdf = pd.DataFrame([{"Trend span": g["trend_span"], "Window": g["ews_window"], "Horizon": g["horizon"],
                                     "Train window": g["window"], "Class-neutral": g["neutralize"],
                                     "Hold n": g["holdout"].get("n"), "IC (baseline + warning)": g["holdout"].get("ic_ews"),
                                     "IC (baseline)": g["holdout"].get("ic_base"), "dIC": g["holdout"].get("d_ic"),
                                     "dIC t": g["holdout"].get("d_ic_t")} for g in grid])
                with st.expander("All " + str(len(gdf)) + " configurations (holdout)"):
                    st.dataframe(gdf.style.format({"IC (baseline + warning)": "{:+.3f}", "IC (baseline)": "{:+.3f}",
                                                   "dIC": "{:+.3f}", "dIC t": "{:+.2f}"}, na_rep="n/a"),
                                 use_container_width=True, hide_index=True)

            st.markdown("###### 3. Overlay: does de-risking on the warning beat a plain volatility target?")
            ov = u["overlay"]
            names = {"buy_hold": "Buy and hold", "vol_target": "Volatility target", "ews": "Warning overlay",
                     "ews_vt": "Warning overlay x volatility target"}
            rows = []
            for k, v in ov["strategies"].items():
                h = v.get("holdout", {})
                rows.append({"Strategy": names.get(k, k), "Ann. return": h.get("ann_return"), "Ann. vol": h.get("ann_vol"),
                             "Sharpe": h.get("sharpe"), "Max drawdown": h.get("max_drawdown"), "Avg exposure": h.get("avg_exposure"),
                             "Annual turnover": h.get("annual_turnover")})
            st.dataframe(pd.DataFrame(rows).style.format({"Ann. return": "{:.1%}", "Ann. vol": "{:.1%}", "Sharpe": "{:.2f}",
                                                          "Max drawdown": "{:.1%}", "Avg exposure": "{:.2f}",
                                                          "Annual turnover": "{:.1f}"}, na_rep="n/a"),
                         use_container_width=True, hide_index=True)
            brow = []
            for k, b in ov["bootstrap"].items():
                if b:
                    a, c = k.split("_vs_")
                    brow.append({"Comparison": names.get(a, a) + " vs " + names.get(c, c),
                                 "Sharpe difference": b["d_sharpe_mean"], "90% interval": "[" + ("%+.2f" % b["d_sharpe_lo"]) + ", " + ("%+.2f" % b["d_sharpe_hi"]) + "]",
                                 "P(Sharpe better)": b["p_sharpe_better"], "Max-DD difference": b["d_maxdd_mean"],
                                 "P(drawdown shallower)": b["p_maxdd_better"]})
            if brow:
                st.dataframe(pd.DataFrame(brow).style.format({"Sharpe difference": "{:+.2f}", "P(Sharpe better)": "{:.2f}",
                                                              "Max-DD difference": "{:+.1%}", "P(drawdown shallower)": "{:.2f}"}),
                             use_container_width=True, hide_index=True)
            st.caption("Holdout from " + str(ov.get("holdout_start_date")) + ", costs " + str(stt.get("trading_cost_bps", 15))
                       + " bps round trip on exposure traded, exposure set with information through day t and applied to day t+1. "
                       "The overlay cuts exposure to 50% when the universe-average score is at or above its causal 80th percentile "
                       "and to 0% above the 95th.")
            render_exposure(ov, "ov_" + uni)
            st.markdown("<hr style='margin: 1.6rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

    with tab3:
        v = data.get("validation") or {}
        st.markdown('<div class="note">Re-run every time. Synthetic log prices (a random-walk trend plus a state with the stated '
                    'dynamics) go through exactly the same detrending and statistics as real data. A control parameter drifts to '
                    'a fold, Hopf or transcritical bifurcation at day ' + str(v.get("t_crit", "?")) + '; nulls are i.i.d. returns, '
                    'GARCH volatility clustering, and a stable mean-reverting state. "Active share" is the share of days with an '
                    'active warning in the 55 days before the bifurcation; AUC is how well the score 30 days before separates an '
                    'approaching system from the null (0.5 = no better than chance).</div>', unsafe_allow_html=True)
        res = v.get("results", {})
        rows = []
        for key, r in res.items():
            for nm, e in r["approach"].items():
                o = r["oracle"][nm]
                rows.append({"Detrending span / window": str(r["span"]) + " / " + str(r["window"]), "Scenario": nm,
                             "Runs used": str(e["n_usable"]) + "/" + str(e["n_total"]), "Tipped early": e["early_tip_share"],
                             "Active share": e["active_share"], "AUC vs i.i.d.": e["auc_vs_iid"], "AUC vs GARCH": e["auc_vs_garch"],
                             "AUC vs stable": e["auc_vs_ou"], "Oracle AUC (no detrending)": o["auc_vs_ou"]})
        if rows:
            st.markdown("###### Detection of simulated approaches")
            st.dataframe(pd.DataFrame(rows).style.format({"Tipped early": "{:.0%}", "Active share": "{:.0%}", "AUC vs i.i.d.": "{:.2f}",
                                                          "AUC vs GARCH": "{:.2f}", "AUC vs stable": "{:.2f}",
                                                          "Oracle AUC (no detrending)": "{:.2f}"}, na_rep="n/a"),
                         use_container_width=True, hide_index=True)
        nrows = []
        for key, r in res.items():
            for nm, e in r["null"].items():
                nrows.append({"Detrending span / window": str(r["span"]) + " / " + str(r["window"]), "Null": nm,
                              "Active share in comparable windows": e["active_share"], "Active share overall": e["active_share_overall"]})
        if nrows:
            st.markdown("###### False warnings on nulls (what 'active' means when nothing is approaching)")
            st.dataframe(pd.DataFrame(nrows).style.format({"Active share in comparable windows": "{:.1%}", "Active share overall": "{:.1%}"}),
                         use_container_width=True, hide_index=True)
        td = v.get("type_discrimination", {})
        if td:
            st.markdown("###### Can the AR(2) fit tell a Hopf approach from a fold?")
            st.dataframe(pd.DataFrame([{"Oscillation period (days)": p, "Complex pair, Hopf runs": t["hopf_complex_share"],
                                        "Complex pair, fold runs": t["fold_complex_share"]} for p, t in td.items()]
                                      ).style.format({"Complex pair, Hopf runs": "{:.0%}", "Complex pair, fold runs": "{:.0%}"}, na_rep="n/a"),
                         use_container_width=True, hide_index=True)
            st.caption("The complex-pair label works for cycles of roughly 10-20 days and fails at 40 days, where the root angle is too "
                       "small to resolve from a double real root. Fold and transcritical share the real-root signature and cannot be "
                       "separated by these statistics.")


if __name__ == "__main__":
    main()
