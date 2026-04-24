"""
Quant Playground tab.

A second tab in the dashboard that runs the ``quant/`` probabilistic
models on whatever ticker and lookback window are already selected in
the sidebar. Charts are in the same Plotly style as the main price chart.

Models shown:
    - HMM regime (2 or 3 state, probabilistic)
    - Hurst exponent (R/S) with rolling H
    - Ornstein-Uhlenbeck mean-reversion fit
    - GBM Monte Carlo forward simulation

Each sub-section fails softly: if data is missing, the model errors, or
an optional dep (hmmlearn) is absent, the user sees a clear message
instead of a stack trace.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Backend — soft-imported so this module still loads if the quant pkg
# is in a broken state during a refactor.
_QUANT_ERRORS: list[str] = []
try:
    from quant.regime_hmm import fit_hmm_regime
except Exception as e:  # pragma: no cover
    fit_hmm_regime = None  # type: ignore[assignment]
    _QUANT_ERRORS.append(f"quant.regime_hmm unavailable: {e}")

try:
    from quant.hurst import hurst_exponent, rolling_hurst
except Exception as e:  # pragma: no cover
    hurst_exponent = rolling_hurst = None  # type: ignore[assignment]
    _QUANT_ERRORS.append(f"quant.hurst unavailable: {e}")

try:
    from quant.ou_process import fit_ou_process
except Exception as e:  # pragma: no cover
    fit_ou_process = None  # type: ignore[assignment]
    _QUANT_ERRORS.append(f"quant.ou_process unavailable: {e}")

try:
    from quant.gbm import gbm_from_prices
except Exception as e:  # pragma: no cover
    gbm_from_prices = None  # type: ignore[assignment]
    _QUANT_ERRORS.append(f"quant.gbm unavailable: {e}")


# Shared palette (matches app.py)
_COLOR_BULL = "#16a34a"
_COLOR_BEAR = "#dc2626"
_COLOR_NEUTRAL = "#9ca3af"
_COLOR_ACCENT = "#8b5cf6"
_COLOR_AMBER = "#f59e0b"
_COLOR_BLUE = "#3b82f6"

_STATE_COLOR_BY_LABEL = {
    "bearish": _COLOR_BEAR,
    "crisis": "#7f1d1d",
    "neutral": _COLOR_NEUTRAL,
    "bullish": _COLOR_BULL,
}


def _rgba(hex_color: str, alpha: float) -> str:
    """#rrggbb -> 'rgba(r,g,b,a)'. Used for stacked fill colours."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _contiguous_runs(series: pd.Series) -> list[tuple[Any, Any, int]]:
    """Return list of (start_timestamp, end_timestamp, state) runs."""
    if series.empty:
        return []
    runs = []
    values = series.values
    idx = series.index
    run_start = 0
    for i in range(1, len(values)):
        if values[i] != values[run_start]:
            runs.append((idx[run_start], idx[i - 1], int(values[run_start])))
            run_start = i
    runs.append((idx[run_start], idx[-1], int(values[run_start])))
    return runs


# ======================================================================
# HMM section
# ======================================================================
def _render_hmm(ticker: str, prices: pd.DataFrame) -> None:
    st.markdown("### 🎭 Hidden Markov Regime")
    st.caption(
        "Probabilistic state classifier fit on log returns. Complements the "
        "rule-based `regime.py` — treat disagreement between the two as a signal."
    )

    if fit_hmm_regime is None:
        st.warning("HMM module not available.")
        return
    if prices.empty or "close" not in prices.columns:
        st.info(f"No price history for {ticker}.")
        return

    n_states = st.radio(
        "Number of states", [2, 3], index=0, horizontal=True, key="hmm_nstates"
    )

    with st.spinner("Fitting HMM…"):
        result = fit_hmm_regime(prices["close"], n_states=int(n_states))

    if not result["available"]:
        st.warning(result["reason"])
        return

    labels = result["state_labels"]
    current_label = result["current_label"]
    current_prob = result["current_probability"]

    # Top-line summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Regime", current_label.title())
    c2.metric("Confidence", f"{current_prob:.1%}")
    c3.metric("# States", n_states)

    # Per-state stats
    st.markdown("**State characteristics** (daily, sorted by mean return)")
    stats_df = pd.DataFrame(
        {
            "Label": labels,
            "Mean return": [f"{m:+.4f}" for m in result["state_means"]],
            "Daily vol": [f"{v:.4f}" for v in result["state_vols"]],
            "Ann. vol": [f"{v * np.sqrt(252):.2%}" for v in result["state_vols"]],
        }
    )
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Chart: price with state background shading + state probability stack
    state_series = result["_series"]["states"]
    prob_df = result["_series"]["probabilities"]
    price_sub = prices.loc[prices.index.intersection(state_series.index), "close"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.6, 0.4],
        subplot_titles=("Price with HMM state shading", "State posterior probabilities"),
    )

    # Price line
    fig.add_trace(
        go.Scatter(
            x=price_sub.index, y=price_sub.values, mode="lines",
            name="Close", line=dict(color="#1f2937", width=2),
        ),
        row=1, col=1,
    )

    # Shading by state: draw colored background rects for contiguous runs
    # of the same state. vrect is the cleanest way to do this in Plotly.
    runs = _contiguous_runs(state_series)
    for start_idx, end_idx, state in runs:
        label = labels[int(state)]
        color = _STATE_COLOR_BY_LABEL.get(label, _COLOR_NEUTRAL)
        fig.add_vrect(
            x0=start_idx, x1=end_idx,
            fillcolor=color, opacity=0.12, line_width=0,
            row=1, col=1,
        )

    # Probability stack (area chart, one trace per state)
    for i, label in enumerate(labels):
        col = f"p_{label}"
        color = _STATE_COLOR_BY_LABEL.get(label, _COLOR_NEUTRAL)
        fig.add_trace(
            go.Scatter(
                x=prob_df.index, y=prob_df[col],
                mode="lines", name=label.title(),
                stackgroup="probs",  # stack to sum = 1
                line=dict(width=0.5, color=color),
                fillcolor=_rgba(color, 0.6),
            ),
            row=2, col=1,
        )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="P(state)", range=[0, 1], row=2, col=1)
    fig.update_layout(
        height=550,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Transition matrix", expanded=False):
        trans_df = pd.DataFrame(
            result["transition_matrix"],
            index=[f"from {l}" for l in labels],
            columns=[f"to {l}" for l in labels],
        )
        st.dataframe(
            trans_df.style.format("{:.3f}").background_gradient(cmap="Blues"),
            use_container_width=True,
        )


# ======================================================================
# Hurst section
# ======================================================================
def _render_hurst(ticker: str, prices: pd.DataFrame) -> None:
    st.markdown("### 📐 Hurst Exponent (R/S)")
    st.caption(
        "H < 0.5 → mean-reverting · H ≈ 0.5 → random walk · H > 0.5 → trending. "
        "Helpful for choosing momentum vs mean-reversion overlays."
    )

    if hurst_exponent is None:
        st.warning("Hurst module not available.")
        return
    if prices.empty or "close" not in prices.columns:
        st.info(f"No price history for {ticker}.")
        return

    with st.spinner("Computing R/S…"):
        result = hurst_exponent(prices["close"])

    if not result["available"]:
        st.warning(result["reason"])
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Hurst (H)", f"{result['hurst']:.3f}")
    c2.metric("Interpretation", result["interpretation"].title())
    c3.metric("# returns", result["n_returns"])

    # Log-log R/S plot
    lags = result["_series"]["lags"]
    rs = result["_series"]["rs"]
    fit = result["_series"]["fit_line"]
    mask = np.isfinite(rs) & (rs > 0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=lags[mask], y=rs[mask],
            mode="markers", name="R/S (empirical)",
            marker=dict(size=9, color=_COLOR_ACCENT),
        )
    )
    if fit:
        fig.add_trace(
            go.Scatter(
                x=fit["x"], y=fit["y"], mode="lines",
                name=f"Fit (slope = H = {result['hurst']:.3f})",
                line=dict(color=_COLOR_AMBER, width=2, dash="dash"),
            )
        )
    # Reference H = 0.5 line: passes through same intercept at lag=1
    if fit and len(fit["x"]) > 0:
        anchor_x = fit["x"][0]
        anchor_y = fit["y"][0]
        ref_y = anchor_y * (fit["x"] / anchor_x) ** 0.5
        fig.add_trace(
            go.Scatter(
                x=fit["x"], y=ref_y, mode="lines",
                name="Random walk (H=0.5)",
                line=dict(color=_COLOR_NEUTRAL, width=1.5, dash="dot"),
            )
        )

    fig.update_layout(
        title="Rescaled range vs chunk size",
        xaxis=dict(type="log", title="lag (log scale)"),
        yaxis=dict(type="log", title="R/S (log scale)"),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Rolling H
    show_rolling = st.checkbox("Show rolling Hurst", value=True, key="hurst_rolling")
    if show_rolling and rolling_hurst is not None:
        with st.spinner("Computing rolling Hurst…"):
            n = len(prices["close"])
            window = min(120, max(60, n // 3))
            rh = rolling_hurst(prices["close"], window=window, step=5, max_lag=60)
        if len(rh) == 0:
            st.info("Not enough data for rolling Hurst.")
        else:
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=rh.index, y=rh.values, mode="lines+markers",
                    name=f"Rolling H ({window}d)",
                    line=dict(color=_COLOR_ACCENT, width=1.8),
                    marker=dict(size=4),
                )
            )
            fig2.add_hline(
                y=0.5, line_dash="dot", line_color=_COLOR_NEUTRAL,
                annotation_text="H = 0.5 (random walk)", annotation_position="right",
            )
            fig2.add_hrect(
                y0=0.0, y1=0.45, fillcolor=_COLOR_BLUE, opacity=0.08, line_width=0,
                annotation_text="mean-reverting", annotation_position="bottom left",
            )
            fig2.add_hrect(
                y0=0.55, y1=1.0, fillcolor=_COLOR_AMBER, opacity=0.08, line_width=0,
                annotation_text="trending", annotation_position="top left",
            )
            fig2.update_layout(
                title=f"Rolling Hurst ({window}-bar window, 5-bar stride)",
                yaxis=dict(title="H", range=[0, 1]),
                height=320,
                margin=dict(l=20, r=20, t=50, b=20),
                hovermode="x unified",
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)


# ======================================================================
# OU section
# ======================================================================
def _render_ou(ticker: str, prices: pd.DataFrame) -> None:
    st.markdown("### 🪝 Ornstein-Uhlenbeck Mean Reversion")
    st.caption(
        "Fits `dX = θ(μ − X)dt + σdW` on log-price via AR(1). Half-life = ln(2)/θ; "
        "a short half-life relative to your horizon means the series is actually reverting."
    )

    if fit_ou_process is None:
        st.warning("OU module not available.")
        return
    if prices.empty or "close" not in prices.columns:
        st.info(f"No price history for {ticker}.")
        return

    band_sigmas = st.slider(
        "Band width (σ)", 1.0, 3.0, 2.0, step=0.25, key="ou_band_sigmas"
    )

    with st.spinner("Fitting OU…"):
        result = fit_ou_process(prices["close"], use_log=True, band_sigmas=band_sigmas)

    if not result["available"]:
        st.warning(result["reason"])
        return

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("θ (speed)", f"{result['theta']:.4f}")
    hl = result["half_life"]
    c2.metric("Half-life", f"{hl:.0f} bars" if np.isfinite(hl) else "∞")
    c3.metric("μ ($, geometric)", f"${np.exp(result['mu']):,.2f}")
    c4.metric("AR(1) R²", f"{result['r_squared']:.3f}")

    if not result["is_mean_reverting"]:
        st.info(
            "AR(1) coefficient doesn't imply mean reversion on this window — the "
            "series behaves more like a random walk / drifting process. The mean "
            "line and band below still visualize the fit, but treat the half-life "
            "with skepticism."
        )

    # Chart: price + mu + bands
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prices.index, y=prices["close"], mode="lines",
            name="Close", line=dict(color="#1f2937", width=2),
        )
    )
    mean_line = result["_series"]["mean_line"]
    upper = result["_series"]["upper_band"]
    lower = result["_series"]["lower_band"]
    fig.add_trace(
        go.Scatter(
            x=mean_line.index, y=mean_line.values, mode="lines",
            name="μ (long-run mean)",
            line=dict(color=_COLOR_AMBER, width=1.5, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=upper.index, y=upper.values, mode="lines",
            name=f"μ + {band_sigmas:g}σ",
            line=dict(color=_COLOR_NEUTRAL, width=1, dash="dot"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lower.index, y=lower.values, mode="lines",
            name=f"μ ± {band_sigmas:g}σ",
            line=dict(color=_COLOR_NEUTRAL, width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(156,163,175,0.10)",
        )
    )

    title = (
        f"OU fit on log-price · half-life = {hl:.0f} bars"
        if np.isfinite(hl) else
        "OU fit on log-price"
    )
    fig.update_layout(
        title=title,
        yaxis_title="Price",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residual distribution
    with st.expander("Residuals & diagnostics", expanded=False):
        resid = result["_series"]["residuals"]
        fig_r = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Residuals over time", "Residual distribution"),
        )
        fig_r.add_trace(
            go.Scatter(
                x=resid.index, y=resid.values, mode="lines",
                line=dict(color=_COLOR_ACCENT, width=1), showlegend=False,
            ),
            row=1, col=1,
        )
        fig_r.add_trace(
            go.Histogram(
                x=resid.values, nbinsx=40, marker_color=_COLOR_ACCENT,
                opacity=0.75, showlegend=False,
            ),
            row=1, col=2,
        )
        fig_r.update_layout(
            height=280, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified",
        )
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption(
            f"σ (instantaneous) = {result['sigma']:.4f}  ·  n = {result['n_obs']} obs"
        )


# ======================================================================
# GBM section
# ======================================================================
def _render_gbm(ticker: str, prices: pd.DataFrame) -> None:
    st.markdown("### 🎲 GBM Monte Carlo")
    st.caption(
        "Simulates forward price paths under GBM. Drift μ and vol σ are fit from "
        "your lookback window; override drift to zero for a pure volatility stress fan."
    )

    if gbm_from_prices is None:
        st.warning("GBM module not available.")
        return
    if prices.empty or "close" not in prices.columns:
        st.info(f"No price history for {ticker}.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        horizon = st.number_input(
            "Horizon (days)", min_value=10, max_value=504, value=60, step=5,
            key="gbm_horizon",
        )
    with c2:
        n_paths = st.number_input(
            "# paths", min_value=50, max_value=5000, value=500, step=50,
            key="gbm_npaths",
        )
    with c3:
        drift_mode = st.selectbox(
            "Drift", ["Estimated μ", "Zero drift", "Custom"], index=0, key="gbm_drift_mode",
        )
    with c4:
        custom_drift = st.number_input(
            "Custom μ (ann.)", min_value=-1.0, max_value=1.0, value=0.05, step=0.01,
            format="%.3f", disabled=(drift_mode != "Custom"), key="gbm_custom_drift",
        )

    drift_override = None
    if drift_mode == "Zero drift":
        drift_override = 0.0
    elif drift_mode == "Custom":
        drift_override = float(custom_drift)

    with st.spinner(f"Simulating {int(n_paths)} paths…"):
        result = gbm_from_prices(
            prices["close"],
            horizon_days=int(horizon),
            n_paths=int(n_paths),
            drift_override=drift_override,
            random_state=42,
        )

    if not result["available"]:
        st.warning(result["reason"])
        return

    # Top-line stats
    fit = result["fit"]
    t = result["terminal"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spot", f"${result['spot']:,.2f}")
    c2.metric("Fitted σ (ann.)", f"{fit['estimated_sigma']:.1%}")
    c3.metric("μ used (ann.)", f"{result['mu']:+.2%}")
    c4.metric("P(S_T > spot)", f"{t['prob_up']:.1%}")

    # Fan chart: light-grey sample paths + colored quantile bands + median
    paths = result["_series"]["paths"]
    days = result["_series"]["days"]
    qb = result["quantile_bands"]
    # Future dates so x-axis flows from today
    last_date = prices.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=len(days))

    fig = go.Figure()

    # Faded sample of paths (cap display to 120 to keep the browser happy)
    display_paths = paths if paths.shape[0] <= 120 else paths[::max(1, paths.shape[0] // 120)]
    for i in range(display_paths.shape[0]):
        fig.add_trace(
            go.Scatter(
                x=future_dates, y=display_paths[i], mode="lines",
                line=dict(color="rgba(156,163,175,0.15)", width=0.6),
                hoverinfo="skip", showlegend=False,
            )
        )

    # 5-95 band
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=qb["p95"], mode="lines",
            line=dict(width=0, color=_COLOR_ACCENT), showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=qb["p05"], mode="lines",
            line=dict(width=0, color=_COLOR_ACCENT),
            fill="tonexty", fillcolor=_rgba(_COLOR_ACCENT, 0.12),
            name="5–95% band",
        )
    )
    # 25-75 band
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=qb["p75"], mode="lines",
            line=dict(width=0, color=_COLOR_ACCENT), showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=qb["p25"], mode="lines",
            line=dict(width=0, color=_COLOR_ACCENT),
            fill="tonexty", fillcolor=_rgba(_COLOR_ACCENT, 0.25),
            name="25–75% band",
        )
    )
    # Median
    fig.add_trace(
        go.Scatter(
            x=future_dates, y=qb["p50"], mode="lines",
            line=dict(color=_COLOR_ACCENT, width=2),
            name="Median",
        )
    )
    # Spot reference line
    fig.add_hline(
        y=result["spot"], line_dash="dash", line_color=_COLOR_NEUTRAL,
        annotation_text=f"Spot ${result['spot']:,.2f}", annotation_position="right",
    )

    fig.update_layout(
        title=f"{ticker} — {int(horizon)}-day GBM fan ({int(n_paths)} paths)",
        yaxis_title="Price",
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Terminal distribution histogram
    fig_t = go.Figure()
    fig_t.add_trace(
        go.Histogram(
            x=paths[:, -1], nbinsx=60,
            marker_color=_COLOR_ACCENT, opacity=0.75, name="Terminal",
        )
    )
    fig_t.add_vline(
        x=result["spot"], line_dash="dash", line_color=_COLOR_NEUTRAL,
        annotation_text="Spot", annotation_position="top",
    )
    fig_t.add_vline(
        x=t["median"], line_dash="dot", line_color=_COLOR_BULL,
        annotation_text="Median", annotation_position="top",
    )
    fig_t.update_layout(
        title=f"Terminal price distribution (day {int(horizon)})",
        xaxis_title="Price at horizon", yaxis_title="Paths",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_t, use_container_width=True)

    # Terminal-VaR / ES panel (same sign convention as var_es.py)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("E[return]", f"{t['expected_return']:+.2%}")
    c2.metric("5th pct", f"${t['p05']:,.2f}")
    c3.metric("95th pct", f"${t['p95']:,.2f}")
    c4.metric(
        "Terminal VaR95 (log)", f"{t['var_95']:.3f}",
        help="5% worst-case terminal log-loss. Positive = loss, matching var_es.py.",
    )


# ======================================================================
# Top-level renderer
# ======================================================================
def render_playground_tab(ticker: str, prices: pd.DataFrame) -> None:
    """
    Render the Quant Playground tab.

    Called from app.main() inside a ``with tab_playground:`` block.
    Takes the same ticker + prices frame that the main tab uses so the
    two stay in sync with the sidebar.
    """
    if _QUANT_ERRORS:
        with st.expander("⚠️ Quant module warnings", expanded=False):
            for msg in _QUANT_ERRORS:
                st.warning(msg)

    st.markdown(f"## 🧪 Quant Playground — {ticker}")
    st.caption(
        "Probabilistic and stochastic models applied to the sidebar's ticker "
        "and lookback window. All inputs are anchored on your actual price "
        "data; nothing is hardcoded or synthetic."
    )

    model = st.radio(
        "Model",
        ["HMM Regime", "Hurst Exponent", "Ornstein-Uhlenbeck", "GBM Monte Carlo"],
        horizontal=True,
        key="playground_model",
    )
    st.markdown("---")

    if model == "HMM Regime":
        _render_hmm(ticker, prices)
    elif model == "Hurst Exponent":
        _render_hurst(ticker, prices)
    elif model == "Ornstein-Uhlenbeck":
        _render_ou(ticker, prices)
    elif model == "GBM Monte Carlo":
        _render_gbm(ticker, prices)