"""
data/global_risk_pulse.py
=========================
Market-wide composite "risk pulse" for the Ary Fund system.

Replaces the previous stub that called ``recompute_risk_scores("__GLOBAL__")``
and treated the market as a single ticker. The new pipeline:

  1. Selects a universe from configurable sources (watchlist, positions,
     full registry, etc.)
  2. Filters for staleness and minimum coverage
  3. Computes a weighting vector (equal / market-cap / liquidity / hybrid)
  4. Computes six independently-normalized subcomponents
     (volatility, breadth, correlation, concentration, dispersion,
     macro_regime)
  5. Aggregates into a single ``pulse_score`` ∈ [-1, +1]
  6. Returns a structured object with score, components, weights,
     coverage, confidence, timestamp, provenance, and diagnostics

Scale convention
----------------
**Positive = risk-off (stressed market). Negative = risk-on (calm).**

Every subcomponent is normalized to [-1, +1] before aggregation so the
final composite has the same interpretation. The convention matches
industry usage of "risk indices" like the St. Louis Fed Financial Stress
Index where positive = stress.

Subcomponents
-------------
- **volatility** : cross-sectional, weighted mean of 30-day realized vol
  (annualized), z-scored against the long-run baseline.
- **breadth** : ``-(2 * pct_above_50dma - 1)`` so high breadth → negative
  (risk-on); also blends in ``pct_positive_5d``.
- **correlation** : median pairwise correlation of daily returns over a
  60-day window; high pairwise correlation indicates systemic risk.
- **concentration** : ``2 * HHI - 1`` of the weighting vector; high
  concentration → +1.
- **dispersion** : cross-sectional std-dev of per-ticker composite risk
  scores; a *positive* coefficient by default (high dispersion → +) but
  configurable, since interpretation depends on regime.
- **macro_regime** : rule-based blend of VIX, 2s10s spread, recession
  probability, and financial-stress index, each mapped to [-1, +1].

The final ``pulse_score`` is a weighted sum with weights defined by
``PulseConfig.subcomponent_weights``. Defaults sum to 1.0; if a
subcomponent is missing, the remaining weights are renormalized so
present subcomponents still average to a well-defined number.

Performance
-----------
Designed for universes up to ~5,000 tickers. The price panel is loaded
in one batched SQL query; all subsequent math is vectorized via pandas
and numpy. The pairwise correlation matrix is the only O(N²) step and
is sub-sampled at ``correlation_subsample_cap`` tickers (default 500),
deterministically by ticker hash, to keep runtime bounded.

Backwards compatibility
-----------------------
The class method ``DerivedSignals.recompute_global_risk_pulse()`` now
delegates here. A separate convenience function ``global_pulse_score()``
returns just the float for callers that wanted a single number.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================


_DEFAULT_SUBCOMPONENT_WEIGHTS: dict[str, float] = {
    # Sum = 1.0. Override individual values via PulseConfig.
    "volatility":    0.25,
    "breadth":       0.20,
    "correlation":   0.15,
    "macro_regime":  0.20,
    "concentration": 0.10,
    "dispersion":    0.10,
}


@dataclass(frozen=True)
class PulseConfig:
    """Typed configuration for the global risk pulse.

    All fields have safe defaults. The dataclass is frozen so a config
    instance can be hashed, compared, and reused safely across runs.
    """

    # ---- Universe selection ---------------------------------------------
    # One of "tracked_equities" | "portfolio_positions" | "watchlist"
    #        | "all_registry_tickers" | "primary_listings"
    universe_source: str = "tracked_equities"
    only_primary_listing: bool = False
    sector_include: tuple[str, ...] = ()
    sector_exclude: tuple[str, ...] = ()
    liquidity_filter_adv_usd: float = 0.0   # min 30-day avg $ volume
    marketcap_filter_min: float = 0.0       # min market cap in USD
    max_staleness_days: int = 7
    min_coverage_pct: float = 0.60           # below → partial result

    # ---- Weighting ------------------------------------------------------
    # "equal" | "market_cap" | "liquidity" | "hybrid" | "sector_balanced"
    weighting: str = "hybrid"
    hybrid_mcap_weight: float = 0.6
    hybrid_liquidity_weight: float = 0.4

    # ---- Subcomponent weights -------------------------------------------
    # Renormalized to sum to 1.0 at compute time.
    subcomponent_weights: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_SUBCOMPONENT_WEIGHTS)
    )

    # ---- Subcomponent tunables ------------------------------------------
    vol_window_days: int = 30               # realized vol lookback
    vol_baseline_years: int = 3             # z-score reference window
    breadth_sma_days: int = 50              # SMA reference for breadth
    breadth_short_lookback: int = 5         # short-horizon returns for breadth
    correlation_window_days: int = 60       # return window for pairwise corr
    correlation_subsample_cap: int = 500    # max tickers in corr matrix

    # ---- Caller-facing alert thresholds ---------------------------------
    risk_off_threshold: float = 0.6         # pulse ≥ this → risk-off
    risk_on_threshold: float = -0.6         # pulse ≤ this → risk-on

    # ---- Misc -----------------------------------------------------------
    version: str = "recompute_global_risk_pulse:v1"

    @classmethod
    def from_overrides(cls, overrides: Optional[dict] = None) -> "PulseConfig":
        """Build a config from a partial dict; unspecified fields use defaults."""
        if not overrides:
            return cls()
        # Merge subcomponent_weights specially so callers can override one key
        sub = dict(_DEFAULT_SUBCOMPONENT_WEIGHTS)
        if "subcomponent_weights" in overrides:
            sub.update(overrides["subcomponent_weights"])
            overrides = {**overrides, "subcomponent_weights": sub}
        # Filter to known dataclass fields so unknown keys don't crash
        known = {f for f in cls.__dataclass_fields__.keys()}
        clean = {k: v for k, v in overrides.items() if k in known}
        return cls(**clean)


# =============================================================================
# Public entrypoint
# =============================================================================


def recompute_global_risk_pulse(
    universe: Optional[list[str]] = None,
    as_of: Optional[datetime] = None,
    config: Optional[Union[dict, PulseConfig]] = None,
    db_path: str = "data/hedgefund.db",
    registry: Any = None,
    portfolio: Any = None,
    persist: bool = True,
) -> dict:
    """Compute the market-wide risk pulse.

    Parameters
    ----------
    universe :
        Explicit ticker list. If ``None``, the universe is selected per
        ``config.universe_source``.
    as_of :
        Reference timestamp (UTC). Defaults to ``datetime.now(timezone.utc)``.
        Drives the "current row" of the price panel and freshness checks.
    config :
        ``PulseConfig`` instance or partial dict of overrides.
    db_path :
        SQLite path. Defaults to the canonical hedgefund.db.
    registry :
        Optional ``DataRegistry``. Constructed from ``db_path`` if omitted.
    portfolio :
        Optional ``PortfolioDB``. Constructed from ``db_path`` if omitted
        and the universe source needs it.
    persist :
        If True, write a row to ``global_risk_pulse_history`` for future
        diff-based diagnostics.

    Returns
    -------
    dict
        Structured object matching the schema documented in the module
        docstring (and in the README snippet shipped alongside).
    """
    cfg = config if isinstance(config, PulseConfig) else PulseConfig.from_overrides(config)
    as_of_dt = as_of or datetime.now(timezone.utc)
    if as_of_dt.tzinfo is None:
        as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)

    _ensure_history_table(db_path)

    # ---- 1. Universe selection -----------------------------------------
    universe_candidates, universe_meta = _select_universe(
        universe, cfg, db_path, registry, portfolio,
    )
    if not universe_candidates:
        return _empty_result(cfg, as_of_dt, reason="universe is empty")

    # ---- 2. Load price panel + freshness filter ------------------------
    panel, excluded = _load_price_panel(
        universe_candidates, cfg, db_path, as_of_dt,
    )
    included = list(panel.columns)
    if not included:
        return _empty_result(
            cfg, as_of_dt,
            reason="no included tickers after staleness filter",
            universe_meta=universe_meta, excluded=excluded,
        )

    # ---- 3. Compute weights --------------------------------------------
    weights, weight_meta = _compute_weights(included, cfg, db_path, registry)
    # Defensive: drop any tickers that didn't get a weight assigned
    weights = weights[weights.notna() & (weights > 0)]
    if weights.empty:
        return _empty_result(
            cfg, as_of_dt,
            reason="weighting produced no usable weights",
            universe_meta=universe_meta, excluded=excluded,
        )
    panel = panel[weights.index]

    # ---- 4. Per-ticker daily returns ------------------------------------
    returns = panel.pct_change()
    last_returns = returns.iloc[-1] if not returns.empty else pd.Series(dtype=float)

    # ---- 5. Subcomponents ----------------------------------------------
    subcomponents = {
        "volatility":    _compute_volatility(panel, returns, weights, cfg, db_path, registry),
        "breadth":       _compute_breadth(panel, returns, cfg),
        "correlation":   _compute_correlation(returns, cfg),
        "concentration": _compute_concentration(weights),
        "dispersion":    _compute_dispersion(included, db_path),
        "macro_regime":  _compute_macro_regime(registry, db_path),
    }

    # ---- 6. Aggregate --------------------------------------------------
    pulse_score, applied_weights = _aggregate(subcomponents, cfg.subcomponent_weights)

    # ---- 7. Coverage / confidence --------------------------------------
    coverage = _coverage_summary(
        universe_meta, included, excluded, db_path, panel,
    )
    confidence = _compute_confidence(coverage, subcomponents, len(weights))

    # Below-coverage runs return partial result with reduced confidence
    if coverage["coverage_pct"] < cfg.min_coverage_pct:
        confidence *= 0.5

    # ---- 8. Diagnostics ------------------------------------------------
    diagnostics = _build_diagnostics(
        panel=panel, returns=returns, last_returns=last_returns,
        weights=weights, subcomponents=subcomponents, db_path=db_path,
        cfg=cfg, pulse_score=pulse_score,
    )

    # ---- 9. Provenance / output ----------------------------------------
    out: dict = {
        "pulse_score": (
            None if pulse_score is None
            else float(np.clip(pulse_score, -1.0, 1.0))
        ),
        "scale": "[-1.0, +1.0]; positive = risk-off, negative = risk-on",
        "subcomponents": {k: _serialize_subcomponent(v) for k, v in subcomponents.items()},
        "weights": {
            "subcomponents": applied_weights,
            "universe_weighting": weight_meta,
        },
        "coverage": coverage,
        "confidence": round(float(confidence), 4),
        "timestamp_utc": as_of_dt.isoformat(),
        "provenance": {
            "computed_by": cfg.version,
            "db_path": db_path,
            "universe_source": cfg.universe_source,
            "weighting": cfg.weighting,
            "data_sources": sorted({
                "registry:price_history",
                "registry:data_points",
                "config:WATCHLIST" if cfg.universe_source != "all_registry_tickers" else "registry:tickers",
            }),
        },
        "thresholds": {
            "risk_off": cfg.risk_off_threshold,
            "risk_on": cfg.risk_on_threshold,
        },
        "diagnostics": diagnostics,
    }

    # ---- 10. Persist ----------------------------------------------------
    if persist and out["pulse_score"] is not None:
        try:
            _persist_history(db_path, out)
        except Exception:  # noqa: BLE001
            logger.exception("global_risk_pulse | history persist failed (non-fatal)")

    return out


def global_pulse_score(*args: Any, **kwargs: Any) -> Optional[float]:
    """Thin compatibility wrapper for callers that want a float, not a dict.

    Forwards every argument to :func:`recompute_global_risk_pulse`. Returns
    ``out["pulse_score"]`` (may be ``None`` if coverage was insufficient).
    """
    return recompute_global_risk_pulse(*args, **kwargs).get("pulse_score")


# =============================================================================
# Universe selection
# =============================================================================


def _select_universe(
    explicit: Optional[list[str]],
    cfg: PulseConfig,
    db_path: str,
    registry: Any,
    portfolio: Any,
) -> tuple[list[str], dict]:
    """Resolve the candidate universe.

    Returns ``(tickers, meta)`` where ``meta`` captures the source and
    raw count for diagnostics. Tickers are uppercased and de-duplicated;
    sector filters are applied here using the portfolio's stored sector
    where available (positions table) — registry doesn't currently have
    a canonical ticker.sector field, so unknown-sector tickers pass
    through unless ``sector_include`` is set.
    """
    meta: dict = {"source": "explicit" if explicit else cfg.universe_source}

    if explicit:
        tickers = [t.upper().strip() for t in explicit if t and t.strip()]
        meta["raw_count"] = len(tickers)
        return _apply_sector_filters(tickers, cfg, db_path), meta

    src = cfg.universe_source

    if src == "tracked_equities":
        positions = _safe_positions(db_path, portfolio)
        watchlist = _safe_watchlist(db_path, portfolio)
        tickers = sorted(set(p["ticker"].upper() for p in positions)
                         | set(w["ticker"].upper() for w in watchlist))
    elif src == "portfolio_positions":
        positions = _safe_positions(db_path, portfolio)
        tickers = sorted(set(p["ticker"].upper() for p in positions))
    elif src == "watchlist":
        watchlist = _safe_watchlist(db_path, portfolio)
        tickers = sorted(set(w["ticker"].upper() for w in watchlist))
    elif src in ("all_registry_tickers", "primary_listings"):
        tickers = _registry_tickers(db_path)
        if src == "primary_listings":
            # Filter out the obvious non-equities: tickers with "." or "-"
            # for share classes, plus tickers with non-alpha prefixes.
            tickers = [t for t in tickers if t.isalpha() and len(t) <= 5]
    else:
        logger.warning(
            "global_risk_pulse | unknown universe_source %r — falling back "
            "to tracked_equities", src,
        )
        positions = _safe_positions(db_path, portfolio)
        watchlist = _safe_watchlist(db_path, portfolio)
        tickers = sorted(set(p["ticker"].upper() for p in positions)
                         | set(w["ticker"].upper() for w in watchlist))

    meta["raw_count"] = len(tickers)
    return _apply_sector_filters(tickers, cfg, db_path), meta


def _safe_positions(db_path: str, portfolio: Any) -> list[dict]:
    if portfolio is not None and hasattr(portfolio, "get_positions"):
        try:
            return portfolio.get_positions() or []
        except Exception:  # noqa: BLE001
            pass
    # Direct SQL fallback — avoids importing PortfolioDB if not needed
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ticker, sector FROM positions"
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error:
        return []


def _safe_watchlist(db_path: str, portfolio: Any) -> list[dict]:
    if portfolio is not None and hasattr(portfolio, "get_watchlist"):
        try:
            return portfolio.get_watchlist() or []
        except Exception:  # noqa: BLE001
            pass
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ticker FROM watchlist"
            ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error:
        return []


def _registry_tickers(db_path: str) -> list[str]:
    """Distinct ticker-type entity_ids in the registry."""
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT entity_id FROM data_points "
                "WHERE entity_type = 'ticker' "
                "ORDER BY entity_id"
            ).fetchall()
        return [r[0].upper() for r in rows]
    except sqlite3.Error:
        return []


def _apply_sector_filters(
    tickers: list[str], cfg: PulseConfig, db_path: str,
) -> list[str]:
    """Apply ``sector_include`` / ``sector_exclude`` filters.

    Sectors come from the positions table; tickers without a stored
    sector pass the filter unless ``sector_include`` is non-empty (in
    which case unknown-sector tickers are dropped).
    """
    if not cfg.sector_include and not cfg.sector_exclude:
        return tickers
    sectors = _load_sectors(db_path)
    out: list[str] = []
    inc = set(s.lower() for s in cfg.sector_include)
    exc = set(s.lower() for s in cfg.sector_exclude)
    for t in tickers:
        s = (sectors.get(t) or "").lower()
        if inc and s not in inc:
            continue
        if exc and s in exc:
            continue
        out.append(t)
    return out


def _load_sectors(db_path: str) -> dict[str, str]:
    """Return ``{ticker: sector}`` from the positions table."""
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT ticker, sector FROM positions"
            ).fetchall()
        return {t.upper(): (s or "") for t, s in rows}
    except sqlite3.Error:
        return {}


# =============================================================================
# Price panel loading
# =============================================================================


def _load_price_panel(
    universe: list[str],
    cfg: PulseConfig,
    db_path: str,
    as_of_dt: datetime,
) -> tuple[pd.DataFrame, dict]:
    """Load a wide (date × ticker) price panel from ``price_history``.

    Filters tickers by:
      - Has any price data at all
      - Latest date ≥ as_of - max_staleness_days
      - Optionally min ADV (avg daily $ volume over last ~30 days)

    Returns (panel, excluded) where ``excluded`` is ``{ticker: reason}``.
    The panel is sorted by date ascending and columns by ticker.
    """
    excluded: dict[str, str] = {}
    if not universe:
        return pd.DataFrame(), excluded

    # We need enough history for vol baseline + correlation + breadth.
    # Pull ~3.5 years to support the volatility z-score window.
    days = max(cfg.vol_baseline_years * 366, 750)
    cutoff = (as_of_dt - timedelta(days=days)).strftime("%Y-%m-%d")
    staleness_cutoff = (
        as_of_dt - timedelta(days=cfg.max_staleness_days)
    ).strftime("%Y-%m-%d")

    # Single batched query — much faster than one query per ticker for N=5000
    placeholders = ",".join("?" for _ in universe)
    sql = (
        f"SELECT ticker, date, close, volume "
        f"FROM price_history "
        f"WHERE ticker IN ({placeholders}) AND date >= ? "
        f"ORDER BY ticker, date"
    )
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(
                sql, conn, params=[*universe, cutoff],
            )
    except sqlite3.Error as e:
        logger.error("global_risk_pulse | price load failed: %s", e)
        for t in universe:
            excluded[t] = "price_load_error"
        return pd.DataFrame(), excluded

    if df.empty:
        for t in universe:
            excluded[t] = "no_price_data"
        return pd.DataFrame(), excluded

    df["ticker"] = df["ticker"].str.upper()
    df["date"] = pd.to_datetime(df["date"])

    # Staleness check — drop tickers whose latest date is too old
    last_dates = df.groupby("ticker")["date"].max()
    stale_mask = last_dates < pd.to_datetime(staleness_cutoff)
    for t in last_dates[stale_mask].index:
        excluded[t] = f"stale_last={last_dates[t]:%Y-%m-%d}"
    fresh = last_dates[~stale_mask].index
    df = df[df["ticker"].isin(fresh)]

    # Also flag universe tickers with NO data at all
    seen = set(df["ticker"].unique())
    for t in universe:
        if t not in seen and t not in excluded:
            excluded[t] = "no_price_data"

    if df.empty:
        return pd.DataFrame(), excluded

    # Pivot to date × ticker close panel; keep volume aside for ADV filter
    close_panel = df.pivot(index="date", columns="ticker", values="close").sort_index()
    if cfg.liquidity_filter_adv_usd > 0:
        vol_panel = df.pivot(index="date", columns="ticker", values="volume").sort_index()
        # 30-day dollar ADV
        adv = (close_panel * vol_panel).tail(30).mean()
        liq_fail = adv[adv < cfg.liquidity_filter_adv_usd].index
        for t in liq_fail:
            excluded[t] = f"low_adv={adv[t]:.0f}"
        close_panel = close_panel.drop(columns=liq_fail)

    return close_panel, excluded


# =============================================================================
# Weighting
# =============================================================================


def _compute_weights(
    tickers: list[str],
    cfg: PulseConfig,
    db_path: str,
    registry: Any,
) -> tuple[pd.Series, dict]:
    """Build the universe weighting vector.

    Returns (weights, meta). ``weights`` is a pandas Series indexed by
    ticker, summing to 1.0. ``meta`` documents the method used and any
    fallbacks (e.g., if market_cap data was missing for half the
    universe, hybrid falls back to equal-weight for those).
    """
    method = cfg.weighting
    meta: dict = {"method": method}

    if method == "equal" or not tickers:
        w = pd.Series(1.0 / len(tickers), index=tickers) if tickers else pd.Series(dtype=float)
        return w, meta

    mcap = _load_market_caps(tickers, db_path, registry)
    liq = _load_liquidity(tickers, db_path)

    if method == "market_cap":
        w = _normalize_weights(mcap)
        meta["missing_mcap"] = int((mcap <= 0).sum())
    elif method == "liquidity":
        w = _normalize_weights(liq)
        meta["missing_liquidity"] = int((liq <= 0).sum())
    elif method == "hybrid":
        mw = _normalize_weights(mcap)
        lw = _normalize_weights(liq)
        a = cfg.hybrid_mcap_weight
        b = cfg.hybrid_liquidity_weight
        if a + b <= 0:
            a, b = 0.5, 0.5
        a, b = a / (a + b), b / (a + b)
        # If either side is all-zero (missing data), fall back to the other
        if mw.sum() == 0 and lw.sum() == 0:
            w = pd.Series(1.0 / len(tickers), index=tickers)
            meta["fallback"] = "equal (no mcap/liquidity data)"
        elif mw.sum() == 0:
            w = lw
            meta["fallback"] = "liquidity-only (no mcap data)"
        elif lw.sum() == 0:
            w = mw
            meta["fallback"] = "mcap-only (no liquidity data)"
        else:
            w = a * mw + b * lw
            w = w / w.sum()
        meta["hybrid"] = {"mcap": a, "liquidity": b}
    elif method == "sector_balanced":
        sectors = _load_sectors(db_path)
        # Equal weight per sector, then equal within sector
        groups: dict[str, list[str]] = {}
        for t in tickers:
            groups.setdefault(sectors.get(t) or "Unknown", []).append(t)
        n_sectors = len(groups)
        w_dict: dict[str, float] = {}
        for sector, members in groups.items():
            per = 1.0 / (n_sectors * len(members))
            for t in members:
                w_dict[t] = per
        w = pd.Series(w_dict).reindex(tickers).fillna(0.0)
        meta["sector_count"] = n_sectors
    else:
        logger.warning(
            "global_risk_pulse | unknown weighting %r — falling back to equal",
            method,
        )
        w = pd.Series(1.0 / len(tickers), index=tickers)
        meta["fallback"] = "equal (unknown method)"

    return w, meta


def _normalize_weights(series: pd.Series) -> pd.Series:
    """Force a series to sum to 1.0; replaces nans/negatives with 0."""
    s = series.copy()
    s = s.where(s > 0, 0.0).fillna(0.0)
    tot = s.sum()
    return s / tot if tot > 0 else s


def _load_market_caps(
    tickers: list[str], db_path: str, registry: Any,
) -> pd.Series:
    """Pull latest market cap per ticker from the registry."""
    field_name = "ticker.price.market_cap"
    out: dict[str, float] = {}
    if registry is not None and hasattr(registry, "latest_value"):
        for t in tickers:
            try:
                v = registry.latest_value(t, field_name)
                if v is not None:
                    out[t] = float(v)
            except Exception:  # noqa: BLE001
                continue
    else:
        # Direct SQL fallback — single query, much faster for N=5000
        try:
            placeholders = ",".join("?" for _ in tickers)
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    f"""SELECT entity_id, value_num
                        FROM data_points
                        WHERE field = ?
                          AND entity_type = 'ticker'
                          AND entity_id IN ({placeholders})
                          AND id IN (
                              SELECT MAX(id) FROM data_points
                              WHERE field = ? AND entity_type = 'ticker'
                              GROUP BY entity_id
                          )""",
                    [field_name, *tickers, field_name],
                ).fetchall()
            out = {t.upper(): float(v) for t, v in rows if v is not None}
        except sqlite3.Error:
            pass
    return pd.Series(out).reindex(tickers).fillna(0.0)


def _load_liquidity(tickers: list[str], db_path: str) -> pd.Series:
    """30-day dollar ADV per ticker, computed directly from price_history."""
    if not tickers:
        return pd.Series(dtype=float)
    try:
        placeholders = ",".join("?" for _ in tickers)
        cutoff = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(
                f"""SELECT ticker, AVG(close * volume) AS adv_usd
                    FROM price_history
                    WHERE ticker IN ({placeholders}) AND date >= ?
                    GROUP BY ticker""",
                conn, params=[*tickers, cutoff],
            )
    except sqlite3.Error:
        return pd.Series(0.0, index=tickers)
    if df.empty:
        return pd.Series(0.0, index=tickers)
    df["ticker"] = df["ticker"].str.upper()
    s = df.set_index("ticker")["adv_usd"]
    return s.reindex(tickers).fillna(0.0)


# =============================================================================
# Subcomponents
# =============================================================================
#
# Each subcomponent returns a dict:
#     {"score": float in [-1, 1] | None,
#      "coverage": int,
#      "notes": str}
# Positive score = more risk-off (more market stress).


def _make_sub(score: Optional[float], coverage: int, notes: str = "") -> dict:
    """Standard subcomponent return shape."""
    if score is not None:
        score = float(np.clip(score, -1.0, 1.0))
    return {"score": score, "coverage": int(coverage), "notes": notes}


def _compute_volatility(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    weights: pd.Series,
    cfg: PulseConfig,
    db_path: str,
    registry: Any,
) -> dict:
    """Cross-section of recent realized vol, weighted-averaged and z-scored
    against the long-run baseline (sqrt(252) * stddev of daily returns).
    """
    if returns.shape[0] < cfg.vol_window_days + 2:
        return _make_sub(None, 0, "insufficient history for vol window")

    # 30-day realized vol per ticker (annualized)
    recent = returns.tail(cfg.vol_window_days).std(axis=0) * math.sqrt(252)
    valid = recent.dropna()
    if valid.empty:
        return _make_sub(None, 0, "no tickers with computable vol")

    # Weighted cross-sectional mean
    w = weights.reindex(valid.index).fillna(0.0)
    w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(valid), index=valid.index)
    current_mean_vol = float((valid * w).sum())

    # Long-run baseline: rolling 30-day vol across the whole window,
    # weighted-averaged into a single time series, then mean/std.
    rolling = returns.rolling(cfg.vol_window_days).std() * math.sqrt(252)
    # Reindex weights and align — only tickers we have
    wb = weights.reindex(rolling.columns).fillna(0.0)
    wb = wb / wb.sum() if wb.sum() > 0 else None
    if wb is None:
        return _make_sub(None, 0, "weights empty for baseline")
    baseline_series = rolling.fillna(0).mul(wb, axis=1).sum(axis=1)
    baseline_series = baseline_series[baseline_series > 0].dropna()
    if len(baseline_series) < 60:
        # Not enough history → return raw normalized version
        # Typical equity vol ~15–35% annualized; map (vol-0.2)/0.15
        score = (current_mean_vol - 0.20) / 0.15
        return _make_sub(
            score, len(valid),
            f"baseline too short ({len(baseline_series)}d); "
            f"raw vol={current_mean_vol:.3f}",
        )

    mu = baseline_series.mean()
    sigma = baseline_series.std()
    if sigma <= 0:
        return _make_sub(0.0, len(valid), "degenerate baseline")
    z = (current_mean_vol - mu) / sigma
    # Map z to [-1, +1] via a soft cap at ±2.5σ
    score = float(np.clip(z / 2.5, -1.0, 1.0))
    return _make_sub(
        score, len(valid),
        f"weighted vol={current_mean_vol:.3f}, baseline μ={mu:.3f}, σ={sigma:.3f}, z={z:.2f}",
    )


def _compute_breadth(
    panel: pd.DataFrame, returns: pd.DataFrame, cfg: PulseConfig,
) -> dict:
    """Two-piece breadth: % above SMA(N) and % positive over short window."""
    if panel.shape[0] < cfg.breadth_sma_days + 1:
        return _make_sub(None, 0, "insufficient history for breadth")

    last_row = panel.iloc[-1]
    sma = panel.tail(cfg.breadth_sma_days).mean(axis=0)
    above = (last_row > sma).astype(float)
    above = above.where(last_row.notna() & sma.notna())
    pct_above = float(above.mean()) if above.notna().any() else None

    short_ret = (
        panel.iloc[-1] / panel.iloc[-cfg.breadth_short_lookback - 1] - 1
        if panel.shape[0] > cfg.breadth_short_lookback else pd.Series(dtype=float)
    )
    pct_positive = float((short_ret > 0).mean()) if not short_ret.dropna().empty else None

    parts = []
    if pct_above is not None:
        parts.append(-(2 * pct_above - 1))   # high breadth → negative (risk-on)
    if pct_positive is not None:
        parts.append(-(2 * pct_positive - 1))
    if not parts:
        return _make_sub(None, 0, "neither breadth measure available")

    score = float(np.mean(parts))
    return _make_sub(
        score, int(above.notna().sum()),
        f"pct_above_{cfg.breadth_sma_days}dma={pct_above}, "
        f"pct_positive_{cfg.breadth_short_lookback}d={pct_positive}",
    )


def _compute_correlation(
    returns: pd.DataFrame, cfg: PulseConfig,
) -> dict:
    """Median pairwise correlation over the configured window.

    Sub-samples to ``correlation_subsample_cap`` tickers for performance.
    Sub-sampling is deterministic (hash of ticker symbol) so successive
    runs are comparable.
    """
    window = returns.tail(cfg.correlation_window_days).dropna(axis=1, how="all")
    if window.shape[0] < 10 or window.shape[1] < 5:
        return _make_sub(None, 0, "insufficient data for correlation")

    cols = list(window.columns)
    if len(cols) > cfg.correlation_subsample_cap:
        # Deterministic subsample: take the lowest-hash ``cap`` tickers
        hashed = sorted(
            cols,
            key=lambda t: hashlib.md5(t.encode()).hexdigest(),
        )
        cols = hashed[: cfg.correlation_subsample_cap]
        window = window[cols]

    corr = window.corr()
    if corr.empty:
        return _make_sub(None, 0, "empty correlation matrix")
    # Pull upper triangle (off-diagonal) values
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    vals = corr.values[mask]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return _make_sub(None, 0, "all-NaN correlations")

    median_corr = float(np.median(vals))
    # Map median ρ to [-1, +1]: typical range [0.2, 0.7]. Above ~0.5 is
    # elevated; > 0.7 is crisis territory. Linear map (ρ - 0.4) / 0.3,
    # then clip.
    score = float(np.clip((median_corr - 0.4) / 0.3, -1.0, 1.0))
    return _make_sub(
        score, int(corr.shape[0]),
        f"median pairwise ρ={median_corr:.3f}, n_pairs={vals.size}",
    )


def _compute_concentration(weights: pd.Series) -> dict:
    """Herfindahl-Hirschman concentration of the weighting vector."""
    if weights.empty:
        return _make_sub(None, 0, "no weights")
    w = weights[weights > 0]
    hhi = float((w ** 2).sum())
    # Map HHI to [-1, +1]:
    #   HHI ≈ 1/N (equal weight): low concentration → near -1
    #   HHI = 0.5: very concentrated → +1
    # We use a curve anchored at 1/N → -1 and 0.5 → +1
    n = max(1, len(w))
    floor = 1.0 / n
    if 0.5 <= floor:  # degenerate (N=1 or 2)
        score = 1.0
    else:
        score = float(np.clip(2 * (hhi - floor) / (0.5 - floor) - 1, -1.0, 1.0))
    return _make_sub(score, n, f"HHI={hhi:.4f}, N={n}, floor=1/N={floor:.4f}")


def _compute_dispersion(included: list[str], db_path: str) -> dict:
    """Cross-sectional std-dev of per-ticker macro_stress risk scores
    pulled from ``risk_scores`` (the table populated by
    ``DerivedSignals.recompute_risk_scores``)."""
    if not included:
        return _make_sub(None, 0, "no tickers")
    try:
        placeholders = ",".join("?" for _ in included)
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                f"""SELECT ticker, macro_stress FROM risk_scores
                    WHERE ticker IN ({placeholders})
                      AND ticker NOT IN ('__GLOBAL__', '_GLOBAL_', 'GLOBAL')
                      AND as_of = (
                          SELECT MAX(as_of) FROM risk_scores rs2
                          WHERE rs2.ticker = risk_scores.ticker
                      )""",
                [t.upper() for t in included],
            ).fetchall()
    except sqlite3.Error:
        return _make_sub(None, 0, "risk_scores table not readable")
    if not rows:
        return _make_sub(None, 0, "no per-ticker risk scores available")
    vals = np.array([r[1] for r in rows if r[1] is not None], dtype=float)
    if vals.size < 3:
        return _make_sub(None, len(vals), "too few risk scores for std-dev")
    sigma = float(vals.std(ddof=0))
    # macro_stress scores live in [0, 1]; cross-sectional σ rarely > 0.25.
    # Map σ to [-1, +1]: linear with (σ - 0.05) / 0.10.
    score = float(np.clip((sigma - 0.05) / 0.10, -1.0, 1.0))
    return _make_sub(
        score, int(vals.size),
        f"σ(risk_score)={sigma:.3f}, n={vals.size}",
    )


def _compute_macro_regime(registry: Any, db_path: str) -> dict:
    """Rule-based blend of macro indicators.

    Each indicator is mapped to [-1, +1] (positive = stress), then
    averaged across all available indicators. Missing indicators are
    skipped; coverage records how many entered the average.
    """
    def latest(field: str) -> Optional[float]:
        if registry is not None and hasattr(registry, "latest_value"):
            try:
                v = registry.latest_value("global", field)
                return float(v) if v is not None else None
            except Exception:  # noqa: BLE001
                return None
        # Fallback to direct SQL
        try:
            with sqlite3.connect(db_path) as conn:
                row = conn.execute(
                    """SELECT value_num FROM data_points
                       WHERE entity_id = 'global' AND field = ?
                       ORDER BY as_of DESC, id DESC LIMIT 1""",
                    (field,),
                ).fetchone()
            return float(row[0]) if row and row[0] is not None else None
        except sqlite3.Error:
            return None

    vix = latest("global.vix")
    vix_term = latest("global.vix_term_3m_1m")
    spread = latest("global.yield_curve_2y10y")
    rec = latest("global.recession_prob")
    fsi = latest("global.financial_stress")
    hy = latest("global.hy_oas")

    scores: list[float] = []
    notes: list[str] = []

    if vix is not None:
        # VIX 12 → -1, VIX 32 → +1
        s = float(np.clip((vix - 22) / 10, -1.0, 1.0))
        scores.append(s)
        notes.append(f"vix={vix:.1f}→{s:+.2f}")

    if vix_term is not None:
        # Term ratio < 1 = backwardation = stress
        s = float(np.clip((1.0 - vix_term) * 5, -1.0, 1.0))
        scores.append(s)
        notes.append(f"vix_term={vix_term:.3f}→{s:+.2f}")

    if spread is not None:
        # 2s10s in percentage points: inverted (spread<0) → +1
        s = float(np.clip(-spread / 0.5, -1.0, 1.0))
        scores.append(s)
        notes.append(f"2s10s={spread:.2f}→{s:+.2f}")

    if rec is not None:
        # FRED recession prob is in percent (0..100). 15% → 0, 40% → +1
        rec_pct = rec if rec <= 1.0 else rec / 100.0
        s = float(np.clip((rec_pct - 0.15) / 0.25, -1.0, 1.0))
        scores.append(s)
        notes.append(f"rec_prob={rec_pct:.2f}→{s:+.2f}")

    if fsi is not None:
        # St. Louis FSI: 0 = average, > 0 = stressed; scale by 2
        s = float(np.clip(fsi / 2.0, -1.0, 1.0))
        scores.append(s)
        notes.append(f"fsi={fsi:.2f}→{s:+.2f}")

    if hy is not None:
        # HY OAS in percent: <3 calm, >8 crisis
        s = float(np.clip((hy - 5.0) / 3.0, -1.0, 1.0))
        scores.append(s)
        notes.append(f"hy_oas={hy:.2f}→{s:+.2f}")

    if not scores:
        return _make_sub(None, 0, "no macro indicators available")
    return _make_sub(float(np.mean(scores)), len(scores), "; ".join(notes))


# =============================================================================
# Aggregation, coverage, confidence
# =============================================================================


def _aggregate(
    subcomponents: dict[str, dict],
    weights_in: dict[str, float],
) -> tuple[Optional[float], dict[str, float]]:
    """Weighted sum of subcomponent scores. Missing subcomponents have
    their weight redistributed across present ones, so present
    subcomponents always average to a well-defined number.
    """
    present = {
        k: v["score"] for k, v in subcomponents.items() if v.get("score") is not None
    }
    if not present:
        return None, {k: 0.0 for k in subcomponents}

    raw_weights = {k: float(weights_in.get(k, 0.0)) for k in present}
    tot = sum(raw_weights.values())
    if tot <= 0:
        # All present subcomponents have zero configured weight → equal weight
        norm_weights = {k: 1.0 / len(present) for k in present}
    else:
        norm_weights = {k: w / tot for k, w in raw_weights.items()}

    pulse = sum(norm_weights[k] * present[k] for k in present)

    # Build full weight report including zero entries for missing parts
    applied = {k: round(norm_weights.get(k, 0.0), 4) for k in subcomponents}
    return float(pulse), applied


def _coverage_summary(
    universe_meta: dict,
    included: list[str],
    excluded: dict[str, str],
    db_path: str,
    panel: pd.DataFrame,
) -> dict:
    raw = int(universe_meta.get("raw_count", len(included) + len(excluded)))
    inc = len(included)
    coverage_pct = (inc / raw) if raw > 0 else 0.0

    # Sector coverage
    sectors = _load_sectors(db_path)
    sector_inc: dict[str, dict] = {}
    for t in included:
        s = sectors.get(t) or "Unknown"
        sector_inc.setdefault(s, {"included": 0})["included"] += 1
    for t, _reason in excluded.items():
        s = sectors.get(t) or "Unknown"
        sector_inc.setdefault(s, {"included": 0})

    return {
        "ticker_count": raw,
        "included_tickers": inc,
        "excluded_tickers": len(excluded),
        "coverage_pct": round(coverage_pct, 4),
        "sector_coverage": {
            s: {"included": v.get("included", 0)} for s, v in sector_inc.items()
        },
        "excluded_reasons": dict(excluded),
    }


def _compute_confidence(
    coverage: dict, subcomponents: dict, n_weights: int,
) -> float:
    """Confidence in [0, 1].

    Combines:
      - Universe coverage pct
      - Subcomponent completeness (fraction of subs with non-None score)
      - Sample-size floor (penalty for <30 tickers)
    """
    cov_pct = float(coverage.get("coverage_pct", 0.0))
    sub_present = sum(1 for v in subcomponents.values() if v.get("score") is not None)
    sub_total = len(subcomponents) or 1
    sub_score = sub_present / sub_total

    size_score = min(1.0, n_weights / 30.0)

    return float(np.clip(
        0.5 * cov_pct + 0.3 * sub_score + 0.2 * size_score,
        0.0, 1.0,
    ))


# =============================================================================
# Diagnostics
# =============================================================================


def _build_diagnostics(
    panel: pd.DataFrame,
    returns: pd.DataFrame,
    last_returns: pd.Series,
    weights: pd.Series,
    subcomponents: dict,
    db_path: str,
    cfg: PulseConfig,
    pulse_score: Optional[float],
) -> dict:
    """Build the human-readable diagnostics block."""
    diag: dict = {
        "top_contributors_positive": [],
        "top_contributors_negative": [],
        "regime_label": _regime_label(pulse_score, cfg),
        "delta_vs_previous": None,
        "moved_subcomponents": [],
    }

    # Top contributors = weighted last-day return ranked
    if not last_returns.empty:
        contrib = (last_returns * weights.reindex(last_returns.index).fillna(0)).dropna()
        if not contrib.empty:
            top_pos = contrib.nlargest(10)
            top_neg = contrib.nsmallest(10)
            diag["top_contributors_positive"] = [
                {"ticker": t, "weighted_return": round(float(v), 6)}
                for t, v in top_pos.items()
            ]
            diag["top_contributors_negative"] = [
                {"ticker": t, "weighted_return": round(float(v), 6)}
                for t, v in top_neg.items()
            ]

    # Previous run comparison
    prev = _load_previous_pulse(db_path)
    if prev is not None and pulse_score is not None:
        prev_score = prev.get("pulse_score")
        if isinstance(prev_score, (int, float)):
            diag["delta_vs_previous"] = round(float(pulse_score - prev_score), 4)
            prev_subs = prev.get("subcomponents", {}) or {}
            moved = []
            for k, v in subcomponents.items():
                cur = v.get("score")
                p = prev_subs.get(k) or {}
                old = p.get("score") if isinstance(p, dict) else None
                if cur is not None and old is not None:
                    delta = float(cur - old)
                    if abs(delta) > 0.05:
                        moved.append({"name": k, "delta": round(delta, 4)})
            diag["moved_subcomponents"] = sorted(
                moved, key=lambda x: -abs(x["delta"]),
            )

    return diag


def _regime_label(score: Optional[float], cfg: PulseConfig) -> str:
    if score is None:
        return "unknown"
    if score >= cfg.risk_off_threshold:
        return "RISK_OFF"
    if score <= cfg.risk_on_threshold:
        return "RISK_ON"
    if score >= cfg.risk_off_threshold / 2:
        return "elevated_stress"
    if score <= cfg.risk_on_threshold / 2:
        return "calm"
    return "neutral"


def _serialize_subcomponent(sub: dict) -> dict:
    """Make a subcomponent JSON-serializable (numpy → python)."""
    score = sub.get("score")
    return {
        "score": None if score is None else float(score),
        "coverage": int(sub.get("coverage") or 0),
        "notes": str(sub.get("notes") or ""),
    }


# =============================================================================
# History persistence
# =============================================================================


def _ensure_history_table(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS global_risk_pulse_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                as_of TEXT NOT NULL,
                pulse_score REAL,
                confidence REAL,
                n_tickers INTEGER,
                subcomponents_json TEXT,
                weights_json TEXT,
                computed_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pulse_as_of "
            "ON global_risk_pulse_history(as_of)"
        )
        conn.commit()


def _persist_history(db_path: str, result: dict) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """INSERT INTO global_risk_pulse_history
               (as_of, pulse_score, confidence, n_tickers,
                subcomponents_json, weights_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                result["timestamp_utc"],
                result["pulse_score"],
                result["confidence"],
                result["coverage"]["included_tickers"],
                json.dumps(result["subcomponents"]),
                json.dumps(result["weights"]),
            ),
        )
        conn.commit()


def _load_previous_pulse(db_path: str) -> Optional[dict]:
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """SELECT pulse_score, subcomponents_json
                   FROM global_risk_pulse_history
                   ORDER BY id DESC LIMIT 1"""
            ).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    score, subs_json = row
    try:
        subs = json.loads(subs_json) if subs_json else {}
    except (TypeError, ValueError):
        subs = {}
    return {"pulse_score": score, "subcomponents": subs}


# =============================================================================
# Empty result builder
# =============================================================================


def _empty_result(
    cfg: PulseConfig,
    as_of_dt: datetime,
    reason: str,
    universe_meta: Optional[dict] = None,
    excluded: Optional[dict] = None,
) -> dict:
    return {
        "pulse_score": None,
        "scale": "[-1.0, +1.0]; positive = risk-off, negative = risk-on",
        "subcomponents": {},
        "weights": {"subcomponents": {}, "universe_weighting": {}},
        "coverage": {
            "ticker_count": int((universe_meta or {}).get("raw_count", 0)),
            "included_tickers": 0,
            "excluded_tickers": len(excluded or {}),
            "coverage_pct": 0.0,
            "sector_coverage": {},
            "excluded_reasons": excluded or {},
        },
        "confidence": 0.0,
        "timestamp_utc": as_of_dt.isoformat(),
        "provenance": {
            "computed_by": cfg.version,
            "universe_source": cfg.universe_source,
            "weighting": cfg.weighting,
        },
        "thresholds": {
            "risk_off": cfg.risk_off_threshold,
            "risk_on": cfg.risk_on_threshold,
        },
        "diagnostics": {
            "regime_label": "unknown",
            "reason": reason,
        },
    }
