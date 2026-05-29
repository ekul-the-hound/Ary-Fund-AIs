"""
ui/metrics_section.py
=====================
Streamlit render module for the "Metrics" tab — operational telemetry
for the agent layer (token spend, latency percentiles, call volume,
success rate, cost).

Kept as a standalone module (mirroring ``data_point_analyzer_section``)
so ``app.py`` only needs a one-line defensive import + a tab call. All
data comes from ``data.metrics_db``; nothing here writes.

Charts:
* Token spend over time   — line, one trace per agent
* Latency P50/P90/P99     — line over daily buckets (overall)
* Call volume per agent   — bar
* Success rate per agent  — bar (0–100%)
* Cost over time          — line (daily buckets)

Everything degrades gracefully: an empty / missing metrics DB shows an
info banner rather than erroring.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from data import metrics_db
except Exception:  # pragma: no cover - defensive UI boundary
    metrics_db = None  # type: ignore[assignment]


def _metrics_df(since_days: int, db_path: Optional[str]) -> pd.DataFrame:
    """Pull recent rows into a DataFrame with a parsed timestamp + date."""
    if metrics_db is None:
        return pd.DataFrame()
    rows = metrics_db.get_metrics(since_days=since_days, limit=100_000,
                                  db_path=db_path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    # Coerce numerics defensively — NULLs become NaN, then 0 where it's a sum.
    for col in ("prompt_tokens", "completion_tokens", "total_tokens",
                "latency_ms", "cost_usd", "success"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _daily_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day P50/P90/P99 of latency_ms."""
    if df.empty or "latency_ms" not in df.columns:
        return pd.DataFrame()
    g = df.dropna(subset=["latency_ms"]).groupby("date")["latency_ms"]
    out = g.quantile([0.5, 0.9, 0.99]).unstack()
    out.columns = ["p50", "p90", "p99"]
    return out.reset_index()


def render_metrics_tab(
    config: Any = None,
    db_path: Optional[str] = None,
) -> None:
    """Render the full Metrics dashboard tab."""
    st.markdown("### 📈 Agent Telemetry")

    if metrics_db is None:
        st.error(
            "metrics_db module unavailable. Ensure `data/metrics_db.py` "
            "exists and imports cleanly."
        )
        return

    # Resolve db_path from config if not explicitly passed.
    if db_path is None and config is not None:
        db_path = getattr(config, "METRICS_DB_PATH", None)

    window = st.selectbox(
        "Time window",
        options=[1, 7, 14, 30, 90],
        index=1,
        format_func=lambda d: f"Last {d} day{'s' if d != 1 else ''}",
    )

    df = _metrics_df(window, db_path)
    if df.empty:
        st.info(
            "No telemetry recorded yet in this window. Run the pipeline "
            "(`python main.py --tickers AAPL`) or any agent call routed "
            "through `agent.metrics.instrumented_ask` to populate "
            "`metrics.db`."
        )
        return

    # ---- Headline KPIs -------------------------------------------------
    total_tokens = float(df["total_tokens"].fillna(0).sum())
    total_cost = float(df["cost_usd"].fillna(0).sum())
    n_calls = len(df)
    overall_success = (
        100.0 * df["success"].fillna(0).sum() / n_calls if n_calls else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LLM calls", f"{n_calls:,}")
    c2.metric("Tokens", f"{total_tokens:,.0f}")
    c3.metric("Cost (notional)", f"${total_cost:,.4f}")
    c4.metric("Success rate", f"{overall_success:.1f}%")

    st.markdown("---")

    # ---- Token spend over time, by agent -------------------------------
    st.markdown("#### Token spend over time")
    by_day_agent = (
        df.groupby(["date", "agent_name"], dropna=False)["total_tokens"]
        .sum()
        .reset_index()
    )
    fig_tok = go.Figure()
    for agent, grp in by_day_agent.groupby("agent_name", dropna=False):
        label = agent if agent is not None else "(unlabeled)"
        fig_tok.add_trace(go.Scatter(
            x=grp["date"], y=grp["total_tokens"],
            mode="lines+markers", name=str(label),
        ))
    fig_tok.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="tokens", xaxis_title="date",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_tok, use_container_width=True)

    # ---- Latency percentiles -------------------------------------------
    st.markdown("#### Latency (ms) — P50 / P90 / P99")
    pct = _daily_percentiles(df)
    if pct.empty:
        st.caption("No latency samples in this window.")
    else:
        fig_lat = go.Figure()
        for col, color in (("p50", None), ("p90", None), ("p99", None)):
            fig_lat.add_trace(go.Scatter(
                x=pct["date"], y=pct[col], mode="lines+markers", name=col.upper(),
            ))
        fig_lat.update_layout(
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="latency (ms)", xaxis_title="date",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    # ---- Call volume + success rate per agent --------------------------
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### Call volume per agent")
        vol = (
            df.assign(agent_name=df["agent_name"].fillna("(unlabeled)"))
            .groupby("agent_name")
            .size()
            .reset_index(name="calls")
            .sort_values("calls", ascending=False)
        )
        fig_vol = go.Figure(go.Bar(x=vol["agent_name"], y=vol["calls"]))
        fig_vol.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="calls",
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with colB:
        st.markdown("#### Success rate per agent")
        stats = metrics_db.get_success_rate_by_agent(window, db_path)
        if stats:
            agents = list(stats.keys())
            rates = [100.0 * stats[a]["rate"] for a in agents]
            fig_sr = go.Figure(go.Bar(x=agents, y=rates))
            fig_sr.update_layout(
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                yaxis_title="success %", yaxis_range=[0, 100],
            )
            st.plotly_chart(fig_sr, use_container_width=True)
        else:
            st.caption("No agent rows in this window.")

    # ---- Cost over time ------------------------------------------------
    st.markdown("#### Cost over time (notional)")
    by_day_cost = df.groupby("date")["cost_usd"].sum().reset_index()
    fig_cost = go.Figure(go.Scatter(
        x=by_day_cost["date"], y=by_day_cost["cost_usd"],
        mode="lines+markers", name="cost",
    ))
    fig_cost.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="USD", xaxis_title="date",
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    st.caption(
        "Cost is notional (local Ollama is free); set "
        "`config.METRICS_COST_PER_1K_TOKENS` to a hosted provider's price "
        "to model cloud-equivalent spend."
    )
