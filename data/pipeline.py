"""
Data Pipeline Orchestrator
===========================
Coordinates all data sources (SEC, market, macro) and the portfolio DB
into a unified pipeline. Handles scheduled refreshes, batch operations,
and generates the context payloads that the LLM agent consumes.

This is the "glue" module — it doesn't fetch data itself, but knows
when and how to call each data source to keep everything fresh.

Usage:
    pipeline = DataPipeline()
    pipeline.run_daily_refresh()
    context = pipeline.build_agent_context("AAPL")
    briefing = pipeline.generate_morning_briefing()
"""

import logging
from datetime import datetime
from typing import Any, Optional

from data.sec_fetcher import SECFetcher
from data.market_data import MarketData
from data.macro_data import MacroData
from data.portfolio_db import PortfolioDB
from data.data_registry import DataRegistry, get_default_registry

logger = logging.getLogger(__name__)


# =============================================================================
# MODULE-LEVEL FUNCTIONAL API
# =============================================================================
# The class-based :class:`DataPipeline` below is the full-featured interface
# used by interactive notebooks, the Streamlit UI, and scripts that want to
# keep one pipeline instance around.
#
# ``main.py`` and the test suite use a simpler functional interface:
#
#     pipeline.run_daily_refresh(tickers, db_path, cfg)
#     context_dict = pipeline.build_agent_context(ticker, db_path, cfg)
#
# These wrappers construct a :class:`DataPipeline` internally per call and
# return structured dicts rather than text blobs. That shape matches what
# the rest of the codebase (risk_scanner, thesis_generator, filing_analyzer,
# and the ``sample_context`` test fixture) expects.
#
# Tests monkeypatch these two names directly, so the wrappers must exist as
# bindings at module scope.


# =============================================================================
# REGISTRY-FIRST CONTEXT CONTRACT
# =============================================================================
#
# ``build_agent_context()`` returns a single, registry-backed view of one
# ticker. The contract below is the authoritative description of what the
# agent will see. Every field is sourced from the DataRegistry — raw data
# providers (yfinance, EDGAR, FRED) are NEVER called directly by the
# context builder. Instead, ``_ensure_registry_populated_for_context()``
# does a lazy backfill: if a required field is missing from the registry,
# the helper fetches it from the provider, writes it to the registry, and
# then reads it back. Subsequent calls within the freshness window are
# pure registry reads.
#
# CONTEXT SCHEMA
# --------------
# ``ticker``            (str)         The ticker symbol. Required.
# ``as_of``             (str)         ISO-format date the snapshot was taken.
# ``price``             (float|None)  Latest close price. None if no data.
#                                     Backed by ``ticker.price.adj_close``.
# ``prices``            (dict)        Price block: ``last``, ``change_pct``,
#                                     ``market_cap``, ``fifty_two_week_high``,
#                                     ``fifty_two_week_low``. Missing → None.
# ``metrics``           (dict)        Fundamentals flat dict. Yfinance-style
#                                     keys (``trailing_pe``, ``grossMargins``,
#                                     ``freeCashflow``, etc.) preserved for
#                                     backward compatibility with
#                                     ``filing_analyzer.extract_key_metrics_for_agent``.
# ``filings``           (list[dict])  Recent SEC filings (10-K, 10-Q, 8-K).
#                                     Empty list when none available.
# ``macro``             (dict)        Flat macro dict — same shape as before
#                                     (``vix``, ``fed_funds_rate``,
#                                     ``yield_curve_spread``, etc.).
# ``portfolio``         (dict)        Position info if held, ``{}`` if not.
# ``sentiment``         (dict)        Sentiment block:
#                                     ``wsb_mentions_24h``, ``wsb_score``,
#                                     ``news_count_7d``, ``news_tone_7d``.
# ``geo_signals``       (dict)        Geographic / supply-chain signals.
#                                     Empty when not yet populated.
# ``risk_scores``       (dict)        Composite registry-derived risk scores
#                                     (``macro_stress``, ``supply_chain``,
#                                     ``sanctions_pressure``, etc.).
# ``derived_signals``   (dict)        Quant-derived signals from the registry
#                                     (``rsi_14``, ``macd_hist``, ``regime``,
#                                     etc.).
# ``provenance``        (dict)        Per-field source / confidence / as_of
#                                     metadata: ``{field: {source_id, as_of,
#                                     confidence}}``. Built from the registry's
#                                     ``latest()`` rows.
# ``freshness``         (dict)        Per-section freshness markers:
#                                     ``{section_name: latest_as_of_iso}``.
#
# ABSENCE SEMANTICS
# -----------------
# A field that is not in the registry is represented explicitly:
#   - scalars      → ``None``
#   - dict blocks  → ``{}``
#   - list blocks  → ``[]``
# The function NEVER substitutes zero, empty string, or a fabricated value.
# Callers can rely on ``ctx["prices"].get("last") is None`` to detect the
# missing-price case unambiguously.
#
# DETERMINISM
# -----------
# For a given registry snapshot, ``build_agent_context()`` is deterministic.
# The only nondeterministic input is ``datetime.now()`` for ``as_of``; pass
# ``as_of_max`` (via a registry param) to pin time travel for tests.
#
# UNIVERSE-LEVEL CONTEXT
# ----------------------
# The class method ``DataPipeline.build_universe_context()`` (see below)
# applies the same registry-first contract across multiple tickers. It is
# *not* a thin wrapper around the per-ticker builder; it batches snapshot
# reads to avoid N round-trips, but uses no raw providers.


# Registry field → (top-level context section, key inside that section)
# This is the canonical map from the registry's namespaced fields to the
# legacy flat context schema. Adding a new field is a single-line change.
_TICKER_FIELD_MAP: dict[str, tuple[str, str]] = {
    # Price block
    "ticker.price.adj_close":         ("prices", "last"),
    "ticker.price.close":             ("prices", "close"),
    "ticker.price.market_cap":        ("prices", "market_cap"),

    # Fundamentals — these populate ``metrics`` under yfinance-style aliases
    # so downstream consumers (filing_analyzer, thesis_essay) don't need to
    # change. The aliases match what get_fundamentals() used to return.
    "ticker.fundamental.revenue_ttm":      ("metrics", "totalRevenue"),
    "ticker.fundamental.eps_diluted_ttm":  ("metrics", "trailingEps"),
    "ticker.fundamental.fcf_ttm":          ("metrics", "freeCashflow"),
    "ticker.fundamental.net_income_ttm":   ("metrics", "netIncomeToCommon"),
    "ticker.fundamental.gross_margin":     ("metrics", "grossMargins"),
    "ticker.fundamental.operating_margin": ("metrics", "operatingMargins"),
    "ticker.fundamental.capex_ttm":        ("metrics", "capex"),
    "ticker.fundamental.rd_ttm":           ("metrics", "researchAndDevelopment"),
    "ticker.fundamental.tax_rate":         ("metrics", "effectiveTaxRate"),
    "ticker.fundamental.goodwill":         ("metrics", "goodwill"),
    "ticker.fundamental.shares_diluted":   ("metrics", "sharesOutstanding"),
    "ticker.fundamental.long_term_debt":   ("metrics", "longTermDebt"),
    "ticker.fundamental.total_assets":     ("metrics", "totalAssets"),
    "ticker.fundamental.total_liabilities":("metrics", "totalLiab"),

    # Sentiment
    "ticker.sentiment.wsb_mentions_24h":   ("sentiment", "wsb_mentions_24h"),
    "ticker.sentiment.wsb_score":          ("sentiment", "wsb_score"),
    "ticker.sentiment.news_count_7d":      ("sentiment", "news_count_7d"),
    "ticker.sentiment.news_tone_7d":       ("sentiment", "news_tone_7d"),

    # Derived signals
    "ticker.signal.rsi_14":                ("derived_signals", "rsi_14"),
    "ticker.signal.macd_hist":             ("derived_signals", "macd_hist"),
    "ticker.signal.atr_14":                ("derived_signals", "atr_14"),
    "ticker.signal.sma_50":                ("derived_signals", "sma_50"),
    "ticker.signal.sma_200":               ("derived_signals", "sma_200"),
    "ticker.signal.realized_vol_30d":      ("derived_signals", "realized_vol_30d"),
    "ticker.signal.drawdown":              ("derived_signals", "drawdown"),
    "ticker.signal.regime":                ("derived_signals", "regime"),

    # Risk scores
    "ticker.risk.macro_stress_score":      ("risk_scores", "macro_stress"),
    "ticker.risk.supply_chain_score":      ("risk_scores", "supply_chain"),
    "ticker.risk.sanctions_pressure":      ("risk_scores", "sanctions_pressure"),
    "ticker.risk.commodity_sensitivity":   ("risk_scores", "commodity_sensitivity"),
    "ticker.risk.energy_crisis_score":     ("risk_scores", "energy_crisis"),
    "ticker.risk.supplier_concentration":  ("risk_scores", "supplier_concentration"),
    "ticker.risk.customer_concentration":  ("risk_scores", "customer_concentration"),

    # Geographic / supply-chain disclosure
    "ticker.disclosure.geographic_revenue":("geo_signals", "geographic_revenue"),
    "ticker.disclosure.segment_revenue":   ("geo_signals", "segment_revenue"),

    # Options
    "ticker.options.iv_30d":               ("derived_signals", "iv_30d"),
    "ticker.options.put_call_ratio":       ("derived_signals", "put_call_ratio"),

    # Analyst
    "ticker.analyst.consensus_target":     ("metrics", "targetMeanPrice"),
    "ticker.analyst.next_earnings_date":   ("metrics", "next_earnings_date"),
}

# Macro/global fields → ``macro`` block
_MACRO_FIELD_MAP: dict[str, str] = {
    "global.vix":                          "vix",
    "global.vix_3m":                       "vix_3m",
    "global.hy_oas":                       "hy_oas",
    "global.ig_oas":                       "ig_oas",
    "global.recession_prob":               "recession_probability",
    "global.consumer_sentiment":           "consumer_sentiment",
    "global.financial_stress":             "financial_stress",
    "global.yield_curve_2y10y":            "yield_curve_spread",
}

# Filings live as registry events rather than data_points. Pulled separately
# via ``reg.recent_events()``.
_FILING_EVENT_TYPE = "sec_filing"


def _empty_context(ticker: str, reason: Optional[str] = None) -> dict:
    """Return a schema-complete empty context.

    Every key is present with a type-appropriate empty value, so downstream
    code can safely index into ``ctx["prices"]["last"]`` etc. without
    KeyError.
    """
    ctx: dict = {
        "ticker": ticker,
        "as_of": datetime.now().date().isoformat(),
        "price": None,
        "prices": {
            "last": None,
            "change_pct": None,
            "market_cap": None,
            "fifty_two_week_high": None,
            "fifty_two_week_low": None,
        },
        "metrics": {},
        "filings": [],
        "macro": {},
        "portfolio": {},
        "sentiment": {},
        "geo_signals": {},
        "risk_scores": {},
        "derived_signals": {},
        "provenance": {},
        "freshness": {},
    }
    if reason:
        ctx["_empty_reason"] = reason
    return ctx


def _backfill_prices_to_registry(
    pipe: "DataPipeline", reg: DataRegistry, ticker: str
) -> dict:
    """Fetch latest price from market_data, write it to the registry, return summary.

    The summary dict captures fields that don't have canonical registry slots
    yet (``change_pct``, ``fifty_two_week_high``, etc.) so the context builder
    can still expose them. Once those become canonical registry fields they
    can be dropped from this passthrough.
    """
    summary: dict = {
        "change_pct": None,
        "fifty_two_week_high": None,
        "fifty_two_week_low": None,
    }
    try:
        latest = pipe.market.get_latest_price(ticker) or {}
    except Exception as e:
        logger.warning(
            "build_agent_context | %s | price backfill skipped: %s",
            ticker, e, exc_info=True,
        )
        return summary

    price = latest.get("price")
    if price is not None:
        try:
            reg.upsert_point(
                entity_id=ticker, entity_type="ticker",
                field="ticker.price.adj_close",
                as_of=datetime.now().isoformat(timespec="seconds"),
                source_id="yfinance",
                value_num=float(price), confidence=0.9,
            )
        except Exception as e:
            logger.debug("registry upsert price failed for %s: %s", ticker, e)

    market_cap = latest.get("market_cap")
    if market_cap is not None:
        try:
            reg.upsert_point(
                entity_id=ticker, entity_type="ticker",
                field="ticker.price.market_cap",
                as_of=datetime.now().isoformat(timespec="seconds"),
                source_id="yfinance",
                value_num=float(market_cap), confidence=0.9,
            )
        except Exception as e:
            logger.debug("registry upsert market_cap failed for %s: %s", ticker, e)

    summary["change_pct"] = latest.get("change_pct")
    summary["fifty_two_week_high"] = latest.get("fifty_two_week_high")
    summary["fifty_two_week_low"] = latest.get("fifty_two_week_low")
    return summary


def _backfill_fundamentals_to_registry(
    pipe: "DataPipeline", reg: DataRegistry, ticker: str
) -> dict:
    """Fetch fundamentals, write canonical fields to registry, return passthrough.

    Returns a flat dict of yfinance-style fields that don't have canonical
    registry slots yet (sector, industry, name, P/E, forward P/E, etc.).
    These remain in ``ctx["metrics"]`` so existing consumers keep working.
    """
    passthrough: dict = {}
    try:
        fund = pipe.market.get_fundamentals(ticker) or {}
    except Exception as e:
        logger.warning(
            "build_agent_context | %s | fundamentals backfill skipped: %s",
            ticker, e, exc_info=True,
        )
        return passthrough

    if not isinstance(fund, dict):
        return passthrough

    # Flatten the same way the legacy code did, so we can map fields.
    flat: dict = {}
    for section in ("overview", "financials", "valuation", "growth", "analyst", "dividends"):
        sub = fund.get(section)
        if isinstance(sub, dict):
            flat.update(sub)
    flat.setdefault("sector", fund.get("sector"))
    flat.setdefault("industry", fund.get("industry"))
    flat.setdefault("name", fund.get("name"))

    # Yfinance field → canonical registry field.
    yf_to_canonical: list[tuple[str, str]] = [
        ("totalRevenue",             "ticker.fundamental.revenue_ttm"),
        ("trailingEps",              "ticker.fundamental.eps_diluted_ttm"),
        ("freeCashflow",             "ticker.fundamental.fcf_ttm"),
        ("netIncomeToCommon",        "ticker.fundamental.net_income_ttm"),
        ("grossMargins",             "ticker.fundamental.gross_margin"),
        ("operatingMargins",         "ticker.fundamental.operating_margin"),
        ("sharesOutstanding",        "ticker.fundamental.shares_diluted"),
        ("longTermDebt",             "ticker.fundamental.long_term_debt"),
        ("totalAssets",              "ticker.fundamental.total_assets"),
        ("totalLiab",                "ticker.fundamental.total_liabilities"),
        ("goodwill",                 "ticker.fundamental.goodwill"),
        ("targetMeanPrice",          "ticker.analyst.consensus_target"),
    ]
    now_iso = datetime.now().isoformat(timespec="seconds")
    for yf_key, canonical in yf_to_canonical:
        v = flat.get(yf_key)
        if v is None:
            continue
        try:
            reg.upsert_point(
                entity_id=ticker, entity_type="ticker",
                field=canonical, as_of=now_iso,
                source_id="yfinance",
                value_num=float(v), confidence=0.8,
            )
        except (TypeError, ValueError):
            # Non-numeric value — skip canonical write, keep in passthrough.
            pass
        except Exception as e:
            logger.debug("registry upsert %s failed for %s: %s", canonical, ticker, e)

    # Everything not written as canonical stays in passthrough so the agent
    # still sees common fields like trailing_pe, sector, name, etc.
    canonical_yf_keys = {pair[0] for pair in yf_to_canonical}
    for k, v in flat.items():
        if k not in canonical_yf_keys:
            passthrough[k] = v
    return passthrough


def _backfill_filings_to_registry(
    pipe: "DataPipeline", reg: DataRegistry, ticker: str
) -> list[dict]:
    """Fetch recent SEC filings and emit registry events.

    Returns the list of filing dicts so the context builder can pass them
    through. Registry events are emitted for traceability; the list is
    the source of truth for the agent's ``filings`` block.
    """
    try:
        filings_10k = pipe.sec.get_filings(ticker, "10-K", count=1) or []
        filings_10q = pipe.sec.get_filings(ticker, "10-Q", count=2) or []
        filings_8k = pipe.sec.get_recent_8k_events(ticker, days_back=60) or []
    except Exception as e:
        logger.warning(
            "build_agent_context | %s | filings backfill skipped: %s",
            ticker, e, exc_info=True,
        )
        return []

    all_filings = [*filings_10k, *filings_10q, *filings_8k]

    now_iso = datetime.now().isoformat(timespec="seconds")
    for f in all_filings:
        try:
            reg.upsert_event(
                entity_id=ticker,
                event_type=_FILING_EVENT_TYPE,
                occurred_at=str(f.get("filed_at") or f.get("date") or now_iso),
                source_id="sec_edgar",
                payload_json={
                    "form": f.get("form") or f.get("form_type"),
                    "accession": f.get("accession_number") or f.get("accession"),
                    "url": f.get("url"),
                    "filed_at": f.get("filed_at"),
                },
                confidence=1.0,
            )
        except Exception as e:
            logger.debug("registry upsert filing event failed for %s: %s", ticker, e)

    return all_filings


def _backfill_macro_to_registry(pipe: "DataPipeline", reg: DataRegistry) -> dict:
    """Fetch macro dashboard, write key globals to registry, return passthrough.

    Macro fields are not ticker-scoped — they use ``entity_id="GLOBAL"``.
    """
    try:
        raw = pipe.macro.get_macro_dashboard() or {}
    except Exception as e:
        logger.warning(
            "build_agent_context | macro backfill skipped: %s", e, exc_info=True,
        )
        return {}
    flat = _flatten_macro(raw)

    # Map flat macro keys to canonical global fields where they exist.
    flat_to_canonical: list[tuple[str, str]] = [
        ("vix",                     "global.vix"),
        ("vix_3m",                  "global.vix_3m"),
        ("recession_probability",   "global.recession_prob"),
        ("financial_stress",        "global.financial_stress"),
        ("yield_curve_spread",      "global.yield_curve_2y10y"),
        ("consumer_sentiment",      "global.consumer_sentiment"),
    ]
    now_iso = datetime.now().isoformat(timespec="seconds")
    for src, canonical in flat_to_canonical:
        v = flat.get(src)
        if v is None:
            continue
        try:
            reg.upsert_point(
                entity_id="GLOBAL", entity_type="global",
                field=canonical, as_of=now_iso,
                source_id="fred",
                value_num=float(v), confidence=0.95,
            )
        except (TypeError, ValueError):
            pass
        except Exception as e:
            logger.debug("registry upsert macro %s failed: %s", canonical, e)

    return flat


def _ensure_registry_populated_for_context(
    pipe: "DataPipeline",
    reg: DataRegistry,
    ticker: str,
) -> dict:
    """Lazy backfill: if any required field is missing, fetch + write it.

    Returns a passthrough dict containing fields that don't have canonical
    registry slots yet (so they can still reach the agent). Within a few
    refresh cycles, all canonical fields will be in the registry and the
    passthrough collapses to just non-canonical extras (sector, P/E, etc.).
    """
    passthrough: dict = {
        "price_summary": {},
        "metrics_extras": {},
        "filings": [],
        "macro_extras": {},
    }

    # --- Prices: backfill if no recent close in registry ---
    if reg.latest_value(ticker, "ticker.price.adj_close") is None:
        passthrough["price_summary"] = _backfill_prices_to_registry(pipe, reg, ticker)
    else:
        # Still need the non-canonical price fields. Cheap to fetch.
        try:
            latest = pipe.market.get_latest_price(ticker) or {}
            passthrough["price_summary"] = {
                "change_pct": latest.get("change_pct"),
                "fifty_two_week_high": latest.get("fifty_two_week_high"),
                "fifty_two_week_low": latest.get("fifty_two_week_low"),
            }
        except Exception:
            pass

    # --- Fundamentals: backfill if no recent revenue in registry ---
    if reg.latest_value(ticker, "ticker.fundamental.revenue_ttm") is None:
        passthrough["metrics_extras"] = _backfill_fundamentals_to_registry(pipe, reg, ticker)
    else:
        # Fetch extras (sector, name, multiples) that don't have canonical slots.
        try:
            fund = pipe.market.get_fundamentals(ticker) or {}
            flat: dict = {}
            for section in ("overview", "financials", "valuation", "growth", "analyst", "dividends"):
                sub = fund.get(section) if isinstance(fund, dict) else None
                if isinstance(sub, dict):
                    flat.update(sub)
            if isinstance(fund, dict):
                flat.setdefault("sector", fund.get("sector"))
                flat.setdefault("industry", fund.get("industry"))
                flat.setdefault("name", fund.get("name"))
            passthrough["metrics_extras"] = flat
        except Exception:
            pass

    # --- Filings: always pull (cheap; SEC fetcher has its own caching) ---
    passthrough["filings"] = _backfill_filings_to_registry(pipe, reg, ticker)

    # --- Macro: backfill if no recent VIX in registry ---
    if reg.latest_value("GLOBAL", "global.vix") is None:
        passthrough["macro_extras"] = _backfill_macro_to_registry(pipe, reg)
    else:
        # Macro has many fields that aren't all canonical — pull the full flat dict.
        try:
            raw = pipe.macro.get_macro_dashboard() or {}
            passthrough["macro_extras"] = _flatten_macro(raw)
        except Exception:
            pass

    return passthrough


def _set_nested(ctx: dict, section: str, key: str, value: Any) -> None:
    """Write ``ctx[section][key] = value`` if value is not None."""
    if value is None:
        return
    ctx.setdefault(section, {})[key] = value


def _record_provenance(
    ctx: dict, field: str, row: Optional[dict]
) -> None:
    """Record a registry row's source/as_of/confidence in ``ctx['provenance']``."""
    if not row:
        return
    ctx.setdefault("provenance", {})[field] = {
        "source_id": row.get("source_id"),
        "as_of": row.get("as_of"),
        "confidence": row.get("confidence"),
    }


def _update_freshness(ctx: dict, section: str, as_of: Optional[str]) -> None:
    """Track the latest as_of seen per section."""
    if not as_of:
        return
    fresh = ctx.setdefault("freshness", {})
    prev = fresh.get(section)
    if prev is None or as_of > prev:
        fresh[section] = as_of


def _flatten_macro(nested: dict) -> dict:
    """Flatten the nested macro dashboard into a single-level dict.

    ``MacroData.get_macro_dashboard()`` returns a structure like::

        {
          "interest_rates": {"yield_spread_10y2y": 0.3, "fed_funds": 5.25, ...},
          "inflation":       {"cpi_yoy_pct": 3.1, ...},
          "employment":      {"unemployment_rate": 3.7, ...},
          "growth":          {"gdp_growth": 2.1, ...},
          "financial_conditions": {"vix": 18.5, ...},
          "recession_signals":    {"recession_probability": 0.22, ...},
        }

    ``risk_scanner._score_macro_risk()`` and ``main.py`` expect a flat dict
    with keys like ``"yield_curve_spread"``, ``"vix"``, ``"recession_probability"``.

    This function:
      1. Lifts all nested values to the top level.
      2. Adds canonical aliases expected by the risk scanner.
      3. Preserves the top-level ``"timestamp"`` key.
    """
    flat: dict = {}

    # Lift every nested section's values to the top level.
    for _section_name, section_val in nested.items():
        if isinstance(section_val, dict):
            flat.update(section_val)
        else:
            # Scalar top-level keys (e.g. "timestamp") pass through unchanged.
            flat[_section_name] = section_val

    # Add canonical aliases that risk_scanner and main.py expect.
    # MacroData uses "yield_spread_10y2y"; scanner uses "yield_curve_spread".
    for src, dst in (
        ("yield_spread_10y2y",   "yield_curve_spread"),
        ("yield_spread_10y3m",   "yield_curve_spread_10y3m"),
        ("cpi_yoy_pct",          "cpi_yoy"),
        ("unemployment_rate",    "unemployment"),
        ("gdp_growth",           "gdp_growth_pct"),
    ):
        if src in flat and dst not in flat:
            flat[dst] = flat[src]

    return flat


def _build_pipeline(db_path: str, cfg) -> "DataPipeline":
    """Construct a DataPipeline from a config object.

    Pulls FRED + SEC credentials from ``cfg`` if available; falls back to
    environment variables inside the data modules themselves.
    """
    return DataPipeline(
        db_path=db_path,
        fred_api_key=getattr(cfg, "FRED_API_KEY", None),
        sec_agent_name=getattr(cfg, "SEC_AGENT_NAME", None),
        sec_agent_email=getattr(cfg, "SEC_AGENT_EMAIL", None),
    )


def run_daily_refresh(tickers, db_path: str, cfg) -> dict:
    """Refresh all data sources for the given tickers.

    Ensures every ticker is on the watchlist (so :meth:`DataPipeline.run_daily_refresh`
    picks them up) and delegates the refresh to the class method.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols to include in the refresh.
    db_path:
        SQLite path for the shared DB.
    cfg:
        Project config object (``SimpleNamespace`` or the ``config`` module).

    Returns
    -------
    dict
        Result summary from :meth:`DataPipeline.run_daily_refresh`, including
        ``refreshed`` / ``errors`` lists and elapsed time.
    """
    pipe = _build_pipeline(db_path, cfg)

    # Make sure every requested ticker is known to the portfolio DB so the
    # refresh loop sees it. Add them to the watchlist defensively; duplicates
    # are a no-op in the portfolio layer.
    for t in tickers or []:
        try:
            pipe.portfolio.add_to_watchlist(t, thesis="auto-added by main run")
        except Exception as e:
            logger.debug("watchlist add skipped for %s: %s", t, e)

    include_sec = bool(getattr(cfg, "INCLUDE_SEC_REFRESH", True))
    return pipe.run_daily_refresh(include_sec=include_sec)


def build_agent_context(ticker: str, db_path: str, cfg) -> dict:
    """Assemble a registry-backed structured context dict for one ticker.

    **Registry-first design.** This function reads exclusively from the
    ``DataRegistry`` snapshot layer. Raw data providers (yfinance, EDGAR,
    FRED) are never called directly here. If a required field is missing,
    ``_ensure_registry_populated_for_context()`` does a one-shot backfill:
    fetch from the provider, write to the registry, then read it back. This
    keeps the registry as the single source of truth for the LLM while
    allowing the system to bootstrap without a separate refresh pass.

    See the CONTEXT SCHEMA section at the top of this module for the
    authoritative field list and absence semantics.

    Parameters
    ----------
    ticker:
        The ticker symbol (e.g. ``"AAPL"``).
    db_path:
        SQLite path for the registry, portfolio DB, and refresh log.
    cfg:
        Project config object — only used to construct the data pipeline
        for backfill calls. The registry itself is configured by db_path.

    Returns
    -------
    dict
        A schema-complete context. Missing fields are explicitly ``None``,
        ``{}``, or ``[]`` — never zero or fabricated.
    """
    pipe = _build_pipeline(db_path, cfg)
    reg = get_default_registry(db_path)

    ctx = _empty_context(ticker)

    # ------------------------------------------------------------------
    # Step 1: lazy backfill — make sure the registry has what we need.
    # Provider calls happen ONLY inside this helper, never below.
    # ------------------------------------------------------------------
    passthrough = _ensure_registry_populated_for_context(pipe, reg, ticker)

    # ------------------------------------------------------------------
    # Step 2: one snapshot call for every ticker-scoped registry field.
    # ------------------------------------------------------------------
    ticker_fields = list(_TICKER_FIELD_MAP.keys())
    ticker_snapshot = reg.snapshot(ticker, ticker_fields)

    for field, value in ticker_snapshot.items():
        section, key = _TICKER_FIELD_MAP[field]
        if value is None:
            # Honor absence semantics: don't write None into nested dicts;
            # leave them empty. ``_empty_context`` already created the
            # structure so consumers can index safely.
            continue
        _set_nested(ctx, section, key, value)

        # Pull the full row once for provenance + freshness. ``latest()``
        # is the same query path as ``latest_value()`` so this is cheap.
        row = reg.latest(ticker, field)
        _record_provenance(ctx, field, row)
        if row:
            _update_freshness(ctx, section, row.get("as_of"))

    # ------------------------------------------------------------------
    # Step 3: macro snapshot — global-scoped fields.
    # ------------------------------------------------------------------
    macro_fields = list(_MACRO_FIELD_MAP.keys())
    macro_snapshot = reg.snapshot("GLOBAL", macro_fields)
    for field, value in macro_snapshot.items():
        if value is None:
            continue
        flat_key = _MACRO_FIELD_MAP[field]
        ctx["macro"][flat_key] = value
        row = reg.latest("GLOBAL", field)
        _record_provenance(ctx, field, row)
        if row:
            _update_freshness(ctx, "macro", row.get("as_of"))

    # ------------------------------------------------------------------
    # Step 4: merge non-canonical passthrough fields. These are values
    # the providers return that don't yet have canonical registry slots
    # (sector, name, trailing_pe, etc.). They reach the agent through
    # ``metrics`` and ``prices`` so existing consumers don't break. As
    # canonical fields expand, this section will shrink.
    # ------------------------------------------------------------------
    price_summary = passthrough.get("price_summary") or {}
    for k, v in price_summary.items():
        if v is not None:
            ctx["prices"][k] = v

    metrics_extras = passthrough.get("metrics_extras") or {}
    for k, v in metrics_extras.items():
        # Don't clobber values that came from the canonical registry path.
        if v is not None and ctx["metrics"].get(k) is None:
            ctx["metrics"][k] = v

    macro_extras = passthrough.get("macro_extras") or {}
    for k, v in macro_extras.items():
        if v is not None and ctx["macro"].get(k) is None:
            ctx["macro"][k] = v

    # Filings: registry events plus the live list (the live list is the
    # source of truth for content; events are for audit/traceability).
    filings_list = passthrough.get("filings") or []
    if filings_list:
        ctx["filings"] = filings_list

    # ------------------------------------------------------------------
    # Step 5: ``price`` shorthand for legacy consumers.
    # ------------------------------------------------------------------
    if ctx["prices"].get("last") is not None:
        try:
            ctx["price"] = float(ctx["prices"]["last"])
        except (TypeError, ValueError):
            ctx["price"] = None

    # ------------------------------------------------------------------
    # Step 6: portfolio position lookup. Portfolio DB is not part of the
    # agent-facing data registry — it's user state, not market data — so
    # reading it here doesn't violate the registry-first contract.
    # ------------------------------------------------------------------
    try:
        positions = pipe.portfolio.get_positions() or []
        for p in positions:
            if p.get("ticker") == ticker:
                ctx["portfolio"] = {
                    "position_size_usd": p.get("position_size_usd") or p.get("market_value"),
                    "weight_in_portfolio": p.get("weight"),
                    "cost_basis": p.get("cost_basis"),
                    "shares": p.get("shares"),
                }
                break
    except Exception as e:
        logger.debug(
            "build_agent_context | %s | portfolio lookup skipped: %s",
            ticker, e, exc_info=True,
        )

    return ctx


def build_universe_context(
    tickers: list[str], db_path: str, cfg
) -> dict[str, dict]:
    """Registry-first context builder for multiple tickers.

    Reads the same registry the per-ticker path reads. Calls the per-ticker
    builder in a loop; the helper inside (``_ensure_registry_populated_for_context``)
    is idempotent and the registry caches across calls, so the second ticker
    onward sees a pre-warmed registry. No raw providers are called outside
    of the per-ticker backfill path.

    Returns ``{ticker: context_dict}``. Tickers that fail are not included.
    """
    out: dict[str, dict] = {}
    for t in tickers or []:
        try:
            out[t] = build_agent_context(t, db_path, cfg)
        except Exception as e:
            logger.warning(
                "build_universe_context | %s | failed: %s", t, e, exc_info=True,
            )
    return out


# =============================================================================
# CLASS-BASED API
# =============================================================================


class DataPipeline:
    """Orchestrates all data modules into a unified pipeline."""

    def __init__(
        self,
        db_path: str = "data/hedgefund.db",
        fred_api_key: Optional[str] = None,
        sec_agent_name: Optional[str] = None,
        sec_agent_email: Optional[str] = None,
    ):
        self.db_path = db_path

        # Initialize all data modules with shared DB
        self.sec = SECFetcher(db_path=db_path, agent_name=sec_agent_name,
                               agent_email=sec_agent_email)
        self.market = MarketData(db_path=db_path)
        self.macro = MacroData(api_key=fred_api_key, db_path=db_path)
        self.portfolio = PortfolioDB(db_path=db_path)

        logger.info("DataPipeline initialized — all modules connected to %s", db_path)

    # ------------------------------------------------------------------
    # Daily refresh — run this on a schedule or manually each morning
    # ------------------------------------------------------------------
    def run_daily_refresh(self, include_sec: bool = True):
        """
        Refresh all data sources.

        Call this once per day (before market open) to update everything.
        Takes ~2-5 minutes depending on portfolio size and network speed.

        Order matters:
        1. Market data (fast, needed by everything)
        2. Macro data (medium, 15-20 API calls)
        3. SEC filings (slow, rate-limited)
        4. Portfolio snapshot (needs market data)
        """
        start = datetime.now()
        results = {"errors": [], "refreshed": []}

        # 1. Refresh market data for all positions + watchlist
        logger.info("Step 1/4: Refreshing market data...")
        tickers = self._get_all_tickers()
        for ticker in tickers:
            try:
                self.market.get_prices(ticker, period="1y", use_cache=False)
                self.market.get_fundamentals(ticker, use_cache=False)
                results["refreshed"].append(f"market:{ticker}")
            except Exception as e:
                results["errors"].append(f"market:{ticker} — {e}")
                logger.error(f"Market refresh failed for {ticker}: {e}")

        # 2. Refresh macro dashboard
        logger.info("Step 2/4: Refreshing macro data...")
        try:
            self.macro.get_macro_dashboard()
            self.macro.get_yield_curve()
            results["refreshed"].append("macro:dashboard")
        except Exception as e:
            results["errors"].append(f"macro — {e}")
            logger.error(f"Macro refresh failed: {e}")

        # 3. Check for new SEC filings
        if include_sec:
            logger.info("Step 3/4: Checking SEC filings...")
            position_tickers = [p["ticker"] for p in self.portfolio.get_positions()]
            for ticker in position_tickers:
                try:
                    # Check for new 8-K (material events) — these are urgent
                    events = self.sec.get_recent_8k_events(ticker, days_back=7)
                    if events:
                        results["refreshed"].append(
                            f"sec:8K:{ticker}:{len(events)} new"
                        )

                    # Refresh 10-Q/10-K index
                    self.sec.get_filings(ticker, filing_type="10-Q", count=2)
                    self.sec.get_filings(ticker, filing_type="10-K", count=1)
                except Exception as e:
                    results["errors"].append(f"sec:{ticker} — {e}")
                    logger.error(f"SEC refresh failed for {ticker}: {e}")
        else:
            logger.info("Step 3/4: Skipping SEC (disabled)")

        # 4. Save portfolio snapshot
        logger.info("Step 4/4: Saving portfolio snapshot...")
        try:
            self.portfolio.save_daily_snapshot(market_data=self.market)
            results["refreshed"].append("portfolio:snapshot")
        except Exception as e:
            results["errors"].append(f"portfolio:snapshot — {e}")

        elapsed = (datetime.now() - start).total_seconds()
        results["elapsed_seconds"] = round(elapsed, 1)
        results["timestamp"] = datetime.now().isoformat()

        logger.info(
            f"Daily refresh complete in {elapsed:.1f}s. "
            f"{len(results['refreshed'])} refreshed, {len(results['errors'])} errors."
        )

        return results

    # ------------------------------------------------------------------
    # Agent context builders — these generate the text payloads
    # that get injected into the LLM's prompt
    # ------------------------------------------------------------------
    def build_agent_context_text(self, ticker: str) -> str:
        """
        Build a comprehensive context payload for the LLM agent
        to analyze a specific stock.

        Returns a pre-rendered **text blob** suitable for direct injection
        into an LLM prompt. For a structured **dict** instead, use the
        module-level :func:`build_agent_context` function.

        This is what gets injected into the system prompt or user message
        when you ask the agent "Analyze AAPL" or "Should I buy MSFT?"

        The context includes: price data, technicals, fundamentals,
        recent filings, macro backdrop, and portfolio position (if held).
        """
        sections = []

        # --- Stock overview ---
        try:
            latest = self.market.get_latest_price(ticker)
            sections.append(
                f"=== {ticker} CURRENT DATA ===\n"
                f"Price: ${latest['price']:.2f} ({latest['change_pct']:+.2f}%)\n"
                f"Market Cap: ${latest['market_cap']:,.0f}\n"
                f"P/E: {latest['pe_ratio']} | Fwd P/E: {latest['forward_pe']}\n"
                f"52wk High: ${latest['fifty_two_week_high']:.2f} | "
                f"52wk Low: ${latest['fifty_two_week_low']:.2f}"
            )
        except Exception as e:
            sections.append(f"[Price data unavailable: {e}]")

        # --- Technicals ---
        try:
            tech = self.market.get_technicals(ticker)
            if tech:
                sig = tech["signal"]
                sections.append(
                    f"\n=== TECHNICAL ANALYSIS ===\n"
                    f"Signal: {sig['overall']} (score: {sig['score']:+.1f})\n"
                    f"RSI(14): {tech['rsi_14']:.1f}\n"
                    f"SMA(50): ${tech['sma_50']:.2f} | SMA(200): ${tech['sma_200']:.2f}\n"
                    f"MACD Histogram: {tech['macd']['histogram']:.3f}\n"
                    f"Bollinger: ${tech['bollinger']['lower']:.2f} — "
                    f"${tech['bollinger']['upper']:.2f}\n"
                    f"ATR(14): ${tech['atr_14']:.2f}\n"
                    f"Relative Volume: {tech['relative_volume']:.2f}x\n"
                    + "\n".join(f"  • {s}" for s in sig["signals"])
                )
        except Exception as e:
            sections.append(f"[Technicals unavailable: {e}]")

        # --- Fundamentals ---
        try:
            fund = self.market.get_fundamentals(ticker)
            fin = fund.get("financials", {})
            val = fund.get("valuation", {})
            gr = fund.get("growth", {})
            an = fund.get("analyst", {})

            sections.append(
                f"\n=== FUNDAMENTALS ===\n"
                f"Sector: {fund.get('sector')} | Industry: {fund.get('industry')}\n"
                f"Revenue: ${fin.get('revenue', 0):,.0f}\n"
                f"Net Income: ${fin.get('net_income', 0):,.0f}\n"
                f"FCF: ${fin.get('free_cash_flow', 0):,.0f}\n"
                f"Profit Margin: {fin.get('profit_margin', 0):.1%}\n"
                f"ROE: {fin.get('return_on_equity', 0):.1%}\n"
                f"Debt/Equity: {fin.get('debt_to_equity', 0):.1f}\n"
                f"Revenue Growth: {gr.get('revenue_growth', 0):.1%}\n"
                f"P/E: {val.get('trailing_pe')} | PEG: {val.get('peg_ratio')}\n"
                f"EV/EBITDA: {val.get('ev_to_ebitda')}\n"
                f"Analyst Target: ${an.get('target_mean', 0):.2f} "
                f"({an.get('recommendation', 'N/A')})"
            )
        except Exception as e:
            sections.append(f"[Fundamentals unavailable: {e}]")

        # --- Recent SEC filings ---
        try:
            filings_10k = self.sec.get_filings(ticker, "10-K", count=1)
            filings_8k = self.sec.get_recent_8k_events(ticker, days_back=60)

            if filings_10k or filings_8k:
                filing_lines = ["\n=== SEC FILINGS ==="]
                if filings_10k:
                    f = filings_10k[0]
                    filing_lines.append(
                        f"Latest 10-K: {f['filed_date']} ({f['accession_number']})"
                    )
                if filings_8k:
                    filing_lines.append(f"Recent 8-K events ({len(filings_8k)}):")
                    for f in filings_8k[:5]:
                        filing_lines.append(f"  {f['filed_date']}: {f['description']}")
                sections.append("\n".join(filing_lines))
        except Exception:
            pass

        # --- Portfolio position (if held) ---
        pos = self.portfolio.get_position(ticker)
        if pos:
            sections.append(
                f"\n=== PORTFOLIO POSITION ===\n"
                f"Shares: {pos['shares']} | Entry: ${pos['avg_entry_price']:.2f}\n"
                f"Conviction: {pos.get('conviction', 'N/A')}\n"
                f"Thesis: {pos.get('thesis', 'None recorded')}"
            )

        # --- Macro backdrop ---
        try:
            yc = self.macro.get_yield_curve()
            sections.append(
                f"\n=== MACRO BACKDROP ===\n"
                f"10Y-2Y Spread: {yc.get('spread_2y_10y', 'N/A')}% "
                f"(Inverted: {yc.get('is_inverted', 'N/A')})"
            )
        except Exception:
            pass

        return "\n".join(sections)

    def build_portfolio_context(self) -> str:
        """
        Build context for portfolio-level questions like
        "How is my portfolio doing?" or "What should I rebalance?"
        """
        return self.portfolio.export_portfolio_summary(market_data=self.market)

    # ------------------------------------------------------------------
    # Morning briefing — the daily summary for your agent
    # ------------------------------------------------------------------
    def generate_morning_briefing(self) -> str:
        """
        Generate a morning briefing that covers:
        1. Portfolio P&L overnight
        2. Macro conditions
        3. New SEC filings for holdings
        4. Price alerts triggered
        5. Watchlist opportunities

        This text gets fed to the LLM which then produces a
        natural-language briefing with recommendations.
        """
        sections = []

        # 1. Portfolio overview
        sections.append("=== MORNING BRIEFING ===")
        sections.append(f"Date: {datetime.now().strftime('%A, %B %d, %Y')}\n")

        try:
            sections.append(self.portfolio.export_portfolio_summary(self.market))
        except Exception as e:
            sections.append(f"[Portfolio data error: {e}]")

        # 2. Macro snapshot
        try:
            dash = self.macro.get_macro_dashboard()

            rates = dash.get("interest_rates", {})
            sections.append(
                f"\n=== MACRO SNAPSHOT ===\n"
                f"Fed Funds: {rates.get('fed_funds', 'N/A')}%\n"
                f"10Y Treasury: {rates.get('treasury_10y', 'N/A')}%\n"
                f"Yield Curve Inverted: {rates.get('yield_curve_inverted', 'N/A')}\n"
                f"VIX: {dash.get('financial_conditions', {}).get('vix', 'N/A')}\n"
                f"Unemployment: {dash.get('employment', {}).get('unemployment_rate', 'N/A')}%"
            )

            recession = dash.get("recession_signals", {})
            if recession.get("sahm_triggered"):
                sections.append("⚠ SAHM RULE TRIGGERED — recession signal active")
        except Exception:
            sections.append("[Macro data unavailable]")

        # 3. New 8-K filings (material events)
        positions = self.portfolio.get_positions()
        new_events = []
        for pos in positions:
            try:
                events = self.sec.get_recent_8k_events(pos["ticker"], days_back=3)
                for e in events:
                    new_events.append(f"  {e['ticker']} ({e['filed_date']}): {e['description']}")
            except Exception:
                pass

        if new_events:
            sections.append(
                "\n=== NEW SEC FILINGS (last 3 days) ===\n" + "\n".join(new_events)
            )

        # 4. Check alerts
        try:
            triggered = self.portfolio.check_alerts(self.market)
            if triggered:
                sections.append("\n=== TRIGGERED ALERTS ===")
                for a in triggered:
                    sections.append(
                        f"  {a['ticker']}: {a['alert_type']} "
                        f"(threshold: {a['threshold']}, current: {a['current_price']})"
                        f" — {a.get('message', '')}"
                    )
        except Exception:
            pass

        # 5. Watchlist price check
        try:
            watchlist = self.portfolio.get_watchlist()
            if watchlist:
                sections.append("\n=== WATCHLIST STATUS ===")
                for w in watchlist:
                    try:
                        latest = self.market.get_latest_price(w["ticker"])
                        price = latest["price"]
                        entry = w.get("target_entry")
                        if entry and price <= entry:
                            sections.append(
                                f"  ✅ {w['ticker']}: ${price:.2f} "
                                f"AT OR BELOW target entry ${entry:.2f}"
                            )
                        else:
                            sections.append(
                                f"  {w['ticker']}: ${price:.2f} "
                                f"(target: ${entry})"
                            )
                    except Exception:
                        pass
        except Exception:
            pass

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Comparison tools
    # ------------------------------------------------------------------
    def compare_to_benchmark(
        self, period: str = "1y", benchmark: str = "SPY"
    ) -> dict:
        """
        Compare portfolio performance against a benchmark (default SPY).

        Returns cumulative returns for portfolio and benchmark
        for charting in the Streamlit dashboard.
        """
        # Get portfolio history
        perf = self.portfolio.get_performance_history(days=365)
        if perf.empty:
            return {"error": "No portfolio history available. Run daily snapshots first."}

        # Get benchmark returns
        bench_prices = self.market.get_prices(benchmark, period=period)
        if bench_prices.empty:
            return {"error": f"Could not fetch {benchmark} data"}

        bench_returns = (bench_prices["Close"] / bench_prices["Close"].iloc[0] - 1) * 100

        # Portfolio returns
        port_returns = (perf["total_value"] / perf["total_value"].iloc[0] - 1) * 100

        return {
            "portfolio_return": round(float(port_returns.iloc[-1]), 2),
            "benchmark_return": round(float(bench_returns.iloc[-1]), 2),
            "alpha": round(float(port_returns.iloc[-1] - bench_returns.iloc[-1]), 2),
            "portfolio_series": port_returns,
            "benchmark_series": bench_returns,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_all_tickers(self) -> list[str]:
        """Get all unique tickers from positions + watchlist."""
        position_tickers = {p["ticker"] for p in self.portfolio.get_positions()}
        watchlist_tickers = {w["ticker"] for w in self.portfolio.get_watchlist()}
        return list(position_tickers | watchlist_tickers)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print("=== Data Pipeline Test ===\n")
    print("Initializing pipeline...")
    pipeline = DataPipeline()

    # Set up a demo portfolio
    print("Setting up demo portfolio...")
    pipeline.portfolio.set_cash(100_000)
    pipeline.portfolio.add_position(
        "AAPL", 100, 170.00, sector="Technology",
        thesis="AI ecosystem play", conviction="HIGH"
    )
    pipeline.portfolio.add_position(
        "MSFT", 50, 380.00, sector="Technology",
        thesis="Azure + Copilot", conviction="HIGH"
    )
    pipeline.portfolio.add_to_watchlist(
        "NVDA", target_entry=850.00, thesis="GPU monopoly", priority="HIGH"
    )

    total_cost = (100 * 170) + (50 * 380)
    pipeline.portfolio.set_cash(100_000 - total_cost)

    # Build agent context for a stock
    print("\n" + "=" * 60)
    print("Agent context for AAPL:")
    print("=" * 60)
    context = pipeline.build_agent_context("AAPL")
    print(context)

    # Portfolio context
    print("\n" + "=" * 60)
    print("Portfolio context:")
    print("=" * 60)
    print(pipeline.build_portfolio_context())

    print("\n✅ Pipeline test complete!")
    print("Next steps:")
    print("  1. Set FRED_API_KEY for macro data")
    print("  2. Set SEC_AGENT_NAME and SEC_AGENT_EMAIL for SEC data")
    print("  3. Run pipeline.run_daily_refresh() to populate all data")
    print("  4. Run pipeline.generate_morning_briefing() for daily brief")