"""
ui/lab_models.py
================

Extended Quant Lab panels — the models that were listed as "available · not
yet in UI" in the Lab catalog, now built into real panels.

This module holds the panel renderers; `ui/lab.py` mounts them as a third Lab
view ("Extended models"). Keeping them here keeps `lab.py` thin and isolates
this larger surface.

Two groups
----------
PRICE-BASED (run on the active ticker's loaded price Series, no extra input):
    - Omori            : aftershock-decay of volatility after a large move
    - Lyapunov         : maximal Lyapunov exponent (sensitivity / chaos)
    - Ergodicity       : ensemble- vs time-average growth gap
    - Wavelet regimes  : multi-scale (period) power decomposition
    - Lempel-Ziv       : complexity / compressibility regime
    - Sandpile         : self-organized-criticality avalanche power law
    - Kelly            : growth-optimal sizing from the name's own return/vol
    - Poisson jumps    : jump detection + arrival intensity

OPTIONS (need parameters the data layer doesn't fetch -> MANUAL INPUT panels):
    - Black-Scholes    : European price + greeks
    - SABR             : stochastic-vol implied-vol smile point
    - Longstaff-Schwartz : American option price via LSM Monte Carlo

VERIFIED SIGNATURES (read from the quant modules):
    compute_omori(prices) -> {available, reason, ...}
    compute_lyapunov(prices) -> {available, reason, lyapunov?, ...}  (needs >=200 bars)
    analyze_ergodicity_from_prices(prices) -> {available, reason, ...}  (needs >=60 bars)
    compute_wavelet_regimes(prices) -> {available, reason, periods, global_power, z_score, ...}
    compute_lz_complexity(prices) -> {available, reason, current_complexity, lz_normalized, regime_label, median, ...}
    run_sandpile(prices, ...) -> {available, reason, sizes, binned_centers, binned_freqs, tau, fit_succeeded, ...}
    kelly_report(capital, expected_return, vol, risk_free_rate=0, fractional=..., max_drawdown=0.20) -> dict
    detect_jumps(prices, z_threshold=4.0) -> {n_jumps, jump_indices, jump_sizes, intensity, scale_mad, note}
    bs_price(S, K, T, r, sigma, option_type="call", q=0.0) -> float
    bs_greeks(S, K, T, r, sigma, option_type="call", q=0.0) -> dict
    sabr_implied_vol(F, K, T, alpha, beta, rho, nu=None) -> float
    price_american_option(S0, K, r, sigma, T, option_type="put", n_paths=5000, n_steps=50, ...) -> dict

Every panel degrades to a clear message when the module is missing or the data
is insufficient (the modules return {available: False, reason: ...}); panels
never raise into the page.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import components as C

logger = logging.getLogger("ary_quant.ui.lab_models")


# ======================================================================
# Soft imports (package or flat layout)
# ======================================================================
def _imp(modname: str, attr: str) -> Any:
    for path in (f"quant.{modname}", modname):
        try:
            return getattr(__import__(path, fromlist=[attr]), attr)
        except Exception:
            continue
    logger.warning("lab_models: %s.%s unavailable", modname, attr)
    return None


_compute_omori = _imp("omori", "compute_omori")
_compute_lyapunov = _imp("lyapunov", "compute_lyapunov")
_analyze_ergodicity = _imp("ergodicity", "analyze_ergodicity_from_prices")
_compute_wavelet = _imp("wavelet_regimes", "compute_wavelet_regimes")
_compute_lz = _imp("lempel_ziv", "compute_lz_complexity")
_run_sandpile = _imp("sandpile", "run_sandpile")
_kelly_report = _imp("kelly", "kelly_report")
_detect_jumps = _imp("poisson", "detect_jumps")
_bs_price = _imp("black_scholes", "bs_price")
_bs_greeks = _imp("black_scholes", "bs_greeks")
_sabr_iv = _imp("sabr", "sabr_implied_vol")
_price_american = _imp("longstaff_schwartz", "price_american_option")


# ======================================================================
# Helpers
# ======================================================================
def _series(prices: pd.DataFrame, min_len: int = 2) -> Optional[pd.Series]:
    if not isinstance(prices, pd.DataFrame) or prices.empty or "close" not in prices.columns:
        return None
    s = prices["close"].dropna()
    return s if len(s) >= min_len else None


def _unavailable(res: dict, what: str) -> bool:
    """Render the module's reason if it returned available=False. Returns True
    if unavailable (caller should stop)."""
    if not isinstance(res, dict):
        st.warning(f"{what}: unexpected result type.")
        return True
    if not res.get("available", True):
        st.info(f"{what}: {res.get('reason', 'not available for this data.')}")
        return True
    return False


def _last_price(prices: pd.DataFrame) -> Optional[float]:
    s = _series(prices)
    return float(s.iloc[-1]) if s is not None else None


def _annual_vol(prices: pd.DataFrame) -> Optional[float]:
    s = _series(prices, 20)
    if s is None:
        return None
    r = np.log(s / s.shift(1)).dropna()
    return float(r.std() * np.sqrt(252)) if len(r) else None


def _annual_ret(prices: pd.DataFrame) -> Optional[float]:
    s = _series(prices, 20)
    if s is None:
        return None
    r = np.log(s / s.shift(1)).dropna()
    return float(r.mean() * 252) if len(r) else None


# ======================================================================
# PRICE-BASED PANELS
# ======================================================================
def _panel_omori(prices: pd.DataFrame) -> None:
    st.markdown("#### Omori law — volatility aftershocks")
    st.caption("After a large price shock, volatility decays like aftershocks "
               "following an earthquake. A slow decay (small p) means turbulence "
               "lingers; fast decay means the market settles quickly.")
    if _compute_omori is None:
        st.warning("omori module unavailable.")
        return
    s = _series(prices, 60)
    if s is None:
        st.info("Need ≥ 60 price bars.")
        return
    res = _compute_omori(s)
    if _unavailable(res, "Omori"):
        return
    cols = st.columns(3)
    cd = res.get("crash_date")
    cr = res.get("crash_return")
    cols[0].metric("Shock date", str(cd)[:10] if cd else "—")
    cols[1].metric("Shock return",
                   C.fmt_pct(cr) if C._is_num(cr) else "—")
    rates = res.get("rates_data") or {}
    # The decay exponent p lives in the top-level 'fit' block, not inside
    # rates_data. Read it from there.
    fit_block = res.get("fit") or {}
    p = fit_block.get("p")
    cols[2].metric("Decay p", f"{p:.2f}" if C._is_num(p) and p > 0 else "—",
                   help="Omori decay exponent; higher = faster calming. "
                        f"({res.get('decay_label', '')})")
    aft = res.get("aftershocks")
    if isinstance(aft, pd.DataFrame) and not aft.empty and "t_days" in aft.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=aft["t_days"], y=aft.get("abs_return"),
                                 mode="markers", marker=dict(size=5, color="#f59e0b"),
                                 name="|return| after shock"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="days since shock", yaxis_title="|return|",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def _panel_lyapunov(prices: pd.DataFrame) -> None:
    st.markdown("#### Lyapunov exponent — sensitivity / chaos")
    st.caption("Measures how fast nearby trajectories diverge. A positive "
               "exponent indicates chaotic dynamics (tiny differences blow up, "
               "limiting predictability); near zero suggests more stable, "
               "forecastable behavior.")
    if _compute_lyapunov is None:
        st.warning("lyapunov module unavailable.")
        return
    s = _series(prices, 200)
    if s is None:
        st.info("Need ≥ 200 price bars.")
        return
    res = _compute_lyapunov(s)
    if _unavailable(res, "Lyapunov"):
        return
    lyap = res.get("lyapunov") or res.get("lyapunov_exponent") or res.get("max_lyapunov")
    cols = st.columns(2)
    cols[0].metric("Max Lyapunov λ",
                   f"{lyap:.4f}" if C._is_num(lyap) else "—")
    if C._is_num(lyap):
        verdict = ("chaotic / low predictability" if lyap > 0.01
                   else "near-neutral" if lyap > -0.01
                   else "stable / mean-reverting")
        cols[1].metric("Interpretation", verdict)


def _panel_ergodicity(prices: pd.DataFrame) -> None:
    st.markdown("#### Ergodicity — ensemble vs time average")
    st.caption("The arithmetic (ensemble) average return overstates what a "
               "single investor actually compounds over time. The gap between "
               "them is the 'ergodicity cost' of volatility — bigger gap means "
               "volatility is quietly eroding your compound growth.")
    if _analyze_ergodicity is None:
        st.warning("ergodicity module unavailable.")
        return
    s = _series(prices, 60)
    if s is None:
        st.info("Need ≥ 60 price bars.")
        return
    res = _analyze_ergodicity(s)
    if _unavailable(res, "Ergodicity"):
        return
    cols = st.columns(3)
    ens = res.get("ensemble_average") or res.get("ensemble_avg") or res.get("arithmetic_mean")
    tim = res.get("time_average") or res.get("time_avg") or res.get("geometric_mean")
    gap = res.get("ergodicity_gap") or res.get("gap")
    cols[0].metric("Ensemble avg", C.fmt_pct(ens) if C._is_num(ens) else "—")
    cols[1].metric("Time avg", C.fmt_pct(tim) if C._is_num(tim) else "—")
    cols[2].metric("Gap", C.fmt_pct(gap) if C._is_num(gap) else "—",
                   help="Ensemble minus time average — the volatility drag.")


def _panel_wavelet(prices: pd.DataFrame) -> None:
    st.markdown("#### Wavelet regimes — multi-scale power")
    st.caption("Decomposes the series into cycles of different periods (days to "
               "months) and shows where the energy concentrates. A spike at a "
               "given period means that cycle length is currently dominant.")
    if _compute_wavelet is None:
        st.warning("wavelet_regimes module unavailable.")
        return
    s = _series(prices, 64)
    if s is None:
        st.info("Need ≥ 64 price bars.")
        return
    res = _compute_wavelet(s)
    if _unavailable(res, "Wavelet"):
        return
    periods = res.get("periods")
    gpower = res.get("global_power")
    if periods is not None and gpower is not None:
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.asarray(periods), y=np.asarray(gpower),
                                     mode="lines", fill="tozeroy",
                                     line=dict(color="#8b5cf6", width=1.6),
                                     fillcolor="rgba(139,92,246,0.15)"))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="period (bars)", yaxis_title="global power",
                              xaxis_type="log",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:  # pragma: no cover
            st.caption(f"Could not plot wavelet power: {e}")
    z = res.get("z_score")
    if C._is_num(z):
        st.metric("Current power z-score", f"{z:+.2f}σ",
                  help="How unusual current oscillation energy is vs history.")


def _panel_lempel_ziv(prices: pd.DataFrame) -> None:
    st.markdown("#### Lempel-Ziv — complexity / compressibility")
    st.caption("Treats the up/down return sequence as a string and measures how "
               "compressible it is. Low complexity = repetitive, more "
               "predictable; high complexity = random-like, harder to forecast.")
    if _compute_lz is None:
        st.warning("lempel_ziv module unavailable.")
        return
    s = _series(prices, 60)
    if s is None:
        st.info("Need ≥ 60 price bars.")
        return
    res = _compute_lz(s)
    if _unavailable(res, "Lempel-Ziv"):
        return
    cols = st.columns(3)
    # lz_normalized is the full time-series array; current_complexity is the
    # latest scalar value. Show the scalar. There is no 'median' key in the
    # return, so we show lz_raw (latest raw complexity) instead.
    cur = res.get("current_complexity")
    label = res.get("regime_label")
    cols[0].metric("LZ complexity (latest)",
                   f"{cur:.3f}" if C._is_num(cur) else "—",
                   help="Normalized Lempel-Ziv complexity of the latest window "
                        "(0 = perfectly repetitive, 1 = random-like).")
    raw = res.get("lz_raw")
    # lz_raw may also be an array; only show if scalar.
    cols[1].metric("LZ raw", f"{raw:.0f}" if C._is_num(raw) else "—",
                   help="Raw LZ phrase count (latest).")
    cols[2].metric("Regime", str(label) if label else "—")


def _panel_sandpile(prices: pd.DataFrame) -> None:
    st.markdown("#### Sandpile — self-organized criticality")
    st.caption("Drives a sandpile model with the stock's daily stress; avalanche "
               "sizes follow a power law if the market is in a critical state "
               "(small shocks can occasionally cascade into large ones). The "
               "exponent τ characterizes that distribution.")
    if _run_sandpile is None:
        st.warning("sandpile module unavailable.")
        return
    s = _series(prices, 60)
    if s is None:
        st.info("Need ≥ 60 price bars.")
        return
    with st.spinner("Running sandpile simulation…"):
        res = _run_sandpile(s)
    if _unavailable(res, "Sandpile"):
        return
    cols = st.columns(2)
    # tau and fit_succeeded live inside the 'power_law_fit' sub-dict; the
    # binned distribution lives inside 'avalanche_distribution'.
    pl_fit = res.get("power_law_fit") or {}
    dist = res.get("avalanche_distribution") or {}
    tau = pl_fit.get("tau")
    cols[0].metric("Power-law τ",
                   f"{tau:.2f}" if C._is_num(tau) and tau > 0 else "—",
                   help="Avalanche-size exponent; ~1-1.5 is typical of SOC.")
    cols[1].metric("Fit", "succeeded" if pl_fit.get("fit_succeeded") else "weak/failed")
    centers = dist.get("binned_centers")
    freqs = dist.get("binned_freqs")
    if centers is not None and freqs is not None:
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.asarray(centers), y=np.asarray(freqs),
                                     mode="markers", marker=dict(size=7, color="#f59e0b")))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="avalanche size", yaxis_title="frequency",
                              xaxis_type="log", yaxis_type="log",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Log-log axes: a straight line is the power-law signature.")
        except Exception as e:  # pragma: no cover
            st.caption(f"Could not plot avalanche distribution: {e}")


def _panel_kelly(prices: pd.DataFrame) -> None:
    st.markdown("#### Kelly sizing — from this name's own return/vol")
    st.caption("Growth-optimal bet size implied by the stock's own historical "
               "drift and volatility. Shown at fractional Kelly (the module's "
               "default multiplier) with a drawdown cap — full Kelly is too "
               "aggressive for single-name equity bets.")
    if _kelly_report is None:
        st.warning("kelly module unavailable.")
        return
    er = _annual_ret(prices)
    vol = _annual_vol(prices)
    if not (C._is_num(er) and C._is_num(vol) and vol > 0):
        st.info("Need price history to estimate return and volatility.")
        return
    c1, c2 = st.columns(2)
    capital = c1.number_input("Capital ($)", min_value=1000.0, value=100_000.0,
                              step=1000.0, key="lab_kelly_cap")
    maxdd = c2.slider("Max drawdown cap", 0.05, 0.50, 0.20, 0.05,
                      key="lab_kelly_dd")
    res = _kelly_report(capital, er, vol, risk_free_rate=0.0, max_drawdown=maxdd)
    if not isinstance(res, dict):
        st.warning("Kelly report returned unexpected type.")
        return
    cols = st.columns(4)
    cols[0].metric("Est. ann. return", C.fmt_pct(er))
    cols[1].metric("Est. ann. vol", C.fmt_pct(vol))
    frac = res.get("fraction") or res.get("recommended_percent")
    cols[2].metric("Kelly fraction", C.fmt_pct(frac) if C._is_num(frac) else "—")
    dollars = res.get("recommended_dollar_size")
    cols[3].metric("≈ Position", C.fmt_big(dollars) if C._is_num(dollars) else "—")


def _panel_poisson(prices: pd.DataFrame) -> None:
    st.markdown("#### Poisson jumps — jump detection & intensity")
    st.caption("Flags returns too large to be normal diffusion (price 'jumps') "
               "and estimates their arrival rate. A rising intensity means the "
               "name is gapping more often than usual.")
    if _detect_jumps is None:
        st.warning("poisson module unavailable.")
        return
    s = _series(prices, 60)
    if s is None:
        st.info("Need ≥ 60 price bars.")
        return
    z_thr = st.slider("Jump z-threshold", 2.0, 6.0, 4.0, 0.5, key="lab_poisson_z",
                      help="How many MAD-sigmas a return must exceed to count as "
                           "a jump.")
    res = _detect_jumps(s, z_threshold=z_thr)
    if not isinstance(res, dict):
        st.warning("Jump detection returned unexpected type.")
        return
    cols = st.columns(3)
    n = res.get("n_jumps")
    intensity = res.get("intensity")
    cols[0].metric("Jumps detected", int(n) if C._is_num(n) else "—")
    cols[1].metric("Intensity", f"{intensity:.4f}" if C._is_num(intensity) else "—",
                   help="Jumps per observation.")
    mad = res.get("scale_mad")
    cols[2].metric("MAD σ", f"{mad:.4f}" if C._is_num(mad) else "—")
    # Plot price with jumps marked.
    idxs = res.get("jump_indices")
    if idxs is not None and len(np.asarray(idxs)) and s is not None:
        try:
            arr = np.asarray(idxs, dtype=int)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines",
                                     line=dict(color="#64748b", width=1.2),
                                     name="price"))
            jump_x = s.index[arr]
            jump_y = s.values[arr]
            fig.add_trace(go.Scatter(x=jump_x, y=jump_y, mode="markers",
                                     marker=dict(size=9, color="#dc2626", symbol="x"),
                                     name="jump"))
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:  # pragma: no cover
            st.caption(f"Could not plot jumps: {e}")


# ======================================================================
# OPTIONS PANELS (manual input)
# ======================================================================
def _options_disclaimer() -> None:
    st.caption("⚠ Manual inputs: this build has no options-chain feed, so enter "
               "the contract parameters yourself. Spot is pre-filled from the "
               "active ticker's last price where possible.")


def _panel_black_scholes(prices: pd.DataFrame) -> None:
    st.markdown("#### Black-Scholes — European price & greeks")
    _options_disclaimer()
    if _bs_price is None:
        st.warning("black_scholes module unavailable.")
        return
    spot_default = _last_price(prices) or 100.0
    c1, c2, c3 = st.columns(3)
    S = c1.number_input("Spot (S)", min_value=0.01, value=float(spot_default),
                        step=1.0, key="bs_S")
    K = c2.number_input("Strike (K)", min_value=0.01, value=float(round(spot_default)),
                        step=1.0, key="bs_K")
    T = c3.number_input("Time to expiry (yrs)", min_value=0.001, value=0.25,
                        step=0.05, key="bs_T")
    c4, c5, c6 = st.columns(3)
    r = c4.number_input("Risk-free rate", min_value=-0.05, value=0.045,
                        step=0.005, format="%.3f", key="bs_r")
    sigma = c5.number_input("Volatility (σ)", min_value=0.001, value=0.30,
                            step=0.05, key="bs_sigma")
    otype = c6.selectbox("Type", ["call", "put"], key="bs_type")

    try:
        price = _bs_price(S, K, T, r, sigma, option_type=otype)
        st.metric(f"{otype.title()} price", C.fmt_money(float(price)))
    except Exception as e:
        st.error(f"Pricing failed: {e}")
        return

    if _bs_greeks is not None:
        try:
            g = _bs_greeks(S, K, T, r, sigma, option_type=otype)
            # bs_greeks returns either a dict or a tuple
            # (price, delta, gamma, vega, theta, rho). Handle both.
            greek_vals = {}
            if isinstance(g, dict):
                greek_vals = g
            elif isinstance(g, (tuple, list)) and len(g) >= 6:
                names = ("price", "delta", "gamma", "vega", "theta", "rho")
                greek_vals = dict(zip(names, g))
            if greek_vals:
                gcols = st.columns(5)
                for col, key in zip(gcols, ("delta", "gamma", "vega", "theta", "rho")):
                    v = greek_vals.get(key)
                    col.metric(key.title(), f"{float(v):.4f}" if C._is_num(v) else "—")
        except Exception as e:  # pragma: no cover
            st.caption(f"Greeks unavailable: {e}")


def _panel_sabr(prices: pd.DataFrame) -> None:
    st.markdown("#### SABR — stochastic-vol implied vol")
    _options_disclaimer()
    if _sabr_iv is None:
        st.warning("sabr module unavailable.")
        return
    fwd_default = _last_price(prices) or 100.0
    c1, c2, c3 = st.columns(3)
    F = c1.number_input("Forward (F)", min_value=0.01, value=float(fwd_default),
                        step=1.0, key="sabr_F")
    K = c2.number_input("Strike (K)", min_value=0.01, value=float(round(fwd_default)),
                        step=1.0, key="sabr_K")
    T = c3.number_input("Expiry (yrs)", min_value=0.001, value=0.5, step=0.05,
                        key="sabr_T")
    c4, c5, c6, c7 = st.columns(4)
    alpha = c4.number_input("α (alpha)", min_value=0.001, value=0.20, step=0.01,
                            key="sabr_alpha", help="Overall vol level.")
    beta = c5.number_input("β (beta)", min_value=0.0, max_value=1.0, value=0.5,
                           step=0.05, key="sabr_beta", help="0=normal, 1=lognormal.")
    rho = c6.number_input("ρ (rho)", min_value=-0.999, max_value=0.999, value=-0.3,
                          step=0.05, key="sabr_rho", help="Spot/vol correlation (skew).")
    nu = c7.number_input("ν (nu)", min_value=0.0, value=0.4, step=0.05,
                         key="sabr_nu", help="Vol-of-vol (smile curvature).")
    try:
        iv = _sabr_iv(F, K, T, alpha, beta, rho, nu)
        st.metric("SABR implied vol", C.fmt_pct(float(iv)))
        st.caption("Vary strike to trace the smile; ρ controls skew, ν controls "
                   "curvature.")
    except Exception as e:
        st.error(f"SABR vol failed: {e}")


def _panel_longstaff(prices: pd.DataFrame) -> None:
    st.markdown("#### Longstaff-Schwartz — American option (LSM)")
    _options_disclaimer()
    if _price_american is None:
        st.warning("longstaff_schwartz module unavailable.")
        return
    spot_default = _last_price(prices) or 100.0
    c1, c2, c3 = st.columns(3)
    S0 = c1.number_input("Spot (S₀)", min_value=0.01, value=float(spot_default),
                         step=1.0, key="lsm_S")
    K = c2.number_input("Strike (K)", min_value=0.01, value=float(round(spot_default)),
                        step=1.0, key="lsm_K")
    T = c3.number_input("Expiry (yrs)", min_value=0.01, value=1.0, step=0.25,
                        key="lsm_T")
    c4, c5, c6 = st.columns(3)
    r = c4.number_input("Risk-free rate", min_value=-0.05, value=0.045,
                        step=0.005, format="%.3f", key="lsm_r")
    sigma = c5.number_input("Volatility (σ)", min_value=0.001, value=0.30,
                            step=0.05, key="lsm_sigma")
    otype = c6.selectbox("Type", ["put", "call"], key="lsm_type")
    n_paths = st.select_slider("Monte Carlo paths", [1000, 2500, 5000, 10000],
                               value=5000, key="lsm_paths",
                               help="More paths = more accurate, slower.")
    if st.button("Price American option", key="lsm_run"):
        with st.spinner(f"Running LSM with {n_paths} paths…"):
            try:
                res = _price_american(S0, K, r, sigma, T, option_type=otype,
                                      n_paths=int(n_paths))
            except Exception as e:
                st.error(f"LSM pricing failed: {e}")
                return
        if isinstance(res, dict):
            cols = st.columns(3)
            price = res.get("american_price")
            se = res.get("american_se")
            euro = res.get("european_price")
            cols[0].metric("American price",
                           C.fmt_money(float(price)) if C._is_num(price) else "—")
            cols[1].metric("Std error", f"{se:.4f}" if C._is_num(se) else "—")
            if C._is_num(euro):
                cols[2].metric("vs European", C.fmt_money(float(euro)),
                               help="Early-exercise premium = American − European.")
            st.caption("LSM estimates the early-exercise value via regression on "
                       "Monte Carlo paths. The American price should be ≥ the "
                       "European.")


# ======================================================================
# REGISTRY + PUBLIC ENTRY
# ======================================================================
# Panel registry: label -> (renderer, group). Group drives the selectbox
# grouping; renderer takes the active price DataFrame.
_PANELS: dict[str, tuple[Callable[[pd.DataFrame], None], str]] = {
    "Omori (vol aftershocks)": (_panel_omori, "Price-based"),
    "Lyapunov (chaos)": (_panel_lyapunov, "Price-based"),
    "Ergodicity (vol drag)": (_panel_ergodicity, "Price-based"),
    "Wavelet regimes": (_panel_wavelet, "Price-based"),
    "Lempel-Ziv (complexity)": (_panel_lempel_ziv, "Price-based"),
    "Sandpile (criticality)": (_panel_sandpile, "Price-based"),
    "Kelly sizing": (_panel_kelly, "Price-based"),
    "Poisson jumps": (_panel_poisson, "Price-based"),
    "Black-Scholes (European)": (_panel_black_scholes, "Options (manual)"),
    "SABR (smile)": (_panel_sabr, "Options (manual)"),
    "Longstaff-Schwartz (American)": (_panel_longstaff, "Options (manual)"),
}


def render_extended_models(ticker: str, prices: pd.DataFrame) -> None:
    """Render the extended-models view: pick a model, see its panel.

    Mounted by ui/lab.py as a third Lab view. The active ticker's price frame
    drives the price-based panels and pre-fills spot for the options panels.
    """
    st.markdown(f"### Extended models — {ticker}")
    st.caption("The full quant library, now in the UI. Price-based models run "
               "on the loaded price window; options models take manual contract "
               "inputs (no options feed in this build).")

    # Group the options for a cleaner selectbox.
    labels = list(_PANELS.keys())
    choice = st.selectbox("Model", labels, key="lab_ext_model")
    st.markdown("---")
    renderer, _group = _PANELS[choice]
    try:
        renderer(prices)
    except Exception as e:  # pragma: no cover - last-resort guard
        st.error(f"Panel '{choice}' failed: {e}")


__all__ = ["render_extended_models"]

# D:\Ary Fund\ui\lab_models.py