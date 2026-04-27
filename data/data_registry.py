"""
data_registry.py
================
Canonical normalization layer for Ary Quant.

Every fetcher in the data layer writes into BOTH its own raw tables AND the
unified normalized tables defined here. The agent reads from the unified
tables only, which gives us:

    * one read path for `pipeline.build_agent_context()`
    * source priority + confidence merging in one place
    * automatic dedup by (entity, field, as_of, source) primary key
    * a single audit log of what was fetched, when, and from where

Three universal tables:

    data_sources : registry of every external + derived source we use
    data_points  : every (entity, field, value, as_of) observation
    events       : every discrete event (filing, rating change, sanction add)

Fetchers call:

    >>> from data.data_registry import DataRegistry
    >>> reg = DataRegistry()
    >>> reg.register_source("yfinance", category="price", refresh_cadence="hourly")
    >>> reg.upsert_point(
    ...     entity_id="AAPL", entity_type="ticker",
    ...     field="ticker.price.adj_close",
    ...     value_num=187.43, as_of="2026-04-25",
    ...     source_id="yfinance", confidence=0.9,
    ... )

Readers call:

    >>> reg.latest("AAPL", "ticker.price.adj_close")
    {'value_num': 187.43, 'as_of': '2026-04-25', 'source_id': 'yfinance', ...}
    >>> reg.time_series("AAPL", "ticker.signal.rsi_14", since="2025-01-01")
    pd.Series indexed by as_of

Design rules
------------
* No external network I/O in this file. Pure SQLite + Python.
* Schema migrations are tracked in ``schema_migrations`` and applied
  idempotently on every ``DataRegistry()`` construction.
* Source priority lives in ``SOURCE_PRIORITY`` below; ``latest()`` respects
  it. Higher priority + fresher wins; ties are resolved by confidence.
* Confidence is a 0..1 float; see ``CONFIDENCE_GUIDE`` for the rubric.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# =============================================================================
# DEFAULTS
# =============================================================================

# Default DB path. Each instance can override. Matches the convention used by
# the rest of the data layer (``portfolio_db.py``, ``market_data.py``, etc.).
DEFAULT_DB_PATH = "data/hedgefund.db"

SCHEMA_VERSION = 1


# =============================================================================
# CANONICAL FIELD DICTIONARY
# =============================================================================
# Every field name written into ``data_points`` should appear here. Fetchers
# may write fields not in this dict (forward compatibility), but the registry
# logs a warning when they do.
#
# Format: field_name -> (dtype, description, refresh_cadence, missing_policy)
#
#   dtype           : 'num' | 'text' | 'json'
#   refresh_cadence : 'hourly'|'daily'|'weekly'|'monthly'|'quarterly'|'event'
#   missing_policy  : 'forward_fill'|'null_safe'|'derived_proxy'

CANONICAL_FIELDS: dict[str, tuple[str, str, str, str]] = {
    # ---- Price / market ------------------------------------------------
    "ticker.price.open":              ("num", "Daily open",                            "hourly", "forward_fill"),
    "ticker.price.high":              ("num", "Daily high",                            "hourly", "forward_fill"),
    "ticker.price.low":               ("num", "Daily low",                             "hourly", "forward_fill"),
    "ticker.price.close":             ("num", "Daily close",                           "hourly", "forward_fill"),
    "ticker.price.adj_close":         ("num", "Adjusted close",                        "hourly", "forward_fill"),
    "ticker.price.volume":            ("num", "Volume",                                "hourly", "forward_fill"),
    "ticker.price.market_cap":        ("num", "Market capitalization (USD)",           "daily",  "forward_fill"),

    # ---- Fundamentals (XBRL preferred) --------------------------------
    "ticker.fundamental.revenue_ttm":      ("num", "TTM revenue",                      "quarterly", "forward_fill"),
    "ticker.fundamental.eps_diluted_ttm":  ("num", "TTM diluted EPS",                  "quarterly", "forward_fill"),
    "ticker.fundamental.fcf_ttm":          ("num", "TTM free cash flow",               "quarterly", "forward_fill"),
    "ticker.fundamental.net_income_ttm":   ("num", "TTM net income",                   "quarterly", "forward_fill"),
    "ticker.fundamental.gross_margin":     ("num", "Gross margin %",                   "quarterly", "forward_fill"),
    "ticker.fundamental.operating_margin": ("num", "Operating margin %",               "quarterly", "forward_fill"),
    "ticker.fundamental.capex_ttm":        ("num", "TTM capex (PPE acquisitions)",     "quarterly", "forward_fill"),
    "ticker.fundamental.rd_ttm":           ("num", "TTM R&D expense",                  "quarterly", "forward_fill"),
    "ticker.fundamental.rd_intensity":     ("num", "R&D / revenue",                    "quarterly", "forward_fill"),
    "ticker.fundamental.tax_rate":         ("num", "Effective tax rate",               "quarterly", "forward_fill"),
    "ticker.fundamental.goodwill":         ("num", "Goodwill on balance sheet",        "quarterly", "forward_fill"),
    "ticker.fundamental.goodwill_to_assets": ("num", "Goodwill / total assets",        "quarterly", "forward_fill"),
    "ticker.fundamental.pension_liability":("num", "Pension and OPEB liabilities",     "quarterly", "forward_fill"),
    "ticker.fundamental.shares_diluted":   ("num", "Diluted shares outstanding",       "quarterly", "forward_fill"),
    "ticker.fundamental.sbc_ttm":          ("num", "Stock-based compensation TTM",     "quarterly", "forward_fill"),
    "ticker.fundamental.sbc_to_revenue":   ("num", "SBC / revenue",                    "quarterly", "forward_fill"),
    "ticker.fundamental.dilution_yoy":     ("num", "YoY change in diluted shares",     "quarterly", "forward_fill"),
    "ticker.fundamental.inventory_days":   ("num", "Inventory days",                   "quarterly", "forward_fill"),
    "ticker.fundamental.long_term_debt":   ("num", "Long-term debt",                   "quarterly", "forward_fill"),
    "ticker.fundamental.total_assets":     ("num", "Total assets",                     "quarterly", "forward_fill"),
    "ticker.fundamental.total_liabilities": ("num", "Total liabilities",               "quarterly", "forward_fill"),

    # ---- Disclosure / governance --------------------------------------
    "ticker.disclosure.geographic_revenue":  ("json", "Geographic revenue breakdown",  "quarterly", "null_safe"),
    "ticker.disclosure.segment_revenue":     ("json", "Segment revenue breakdown",     "quarterly", "null_safe"),
    "ticker.disclosure.debt_maturity":       ("json", "Debt maturity schedule",        "quarterly", "null_safe"),
    "ticker.risk.supplier_concentration":    ("num",  "Top supplier % of COGS",        "quarterly", "null_safe"),
    "ticker.risk.customer_concentration":    ("num",  "Top customer % of revenue",     "quarterly", "null_safe"),
    "ticker.disclosure.share_buyback_authorized": ("num", "Authorized buyback (USD)",  "event",     "null_safe"),
    "ticker.disclosure.atm_offering_size":   ("num",  "ATM offering size (USD)",       "event",     "null_safe"),
    "ticker.disclosure.secondary_size":      ("num",  "Secondary offering size (USD)", "event",     "null_safe"),

    # ---- Ownership / insiders -----------------------------------------
    "ticker.ownership.insider_buys_30d":   ("num", "Form 4 buy count last 30 days",    "daily", "forward_fill"),
    "ticker.ownership.insider_sells_30d":  ("num", "Form 4 sell count last 30 days",   "daily", "forward_fill"),
    "ticker.ownership.insider_net_usd_30d":("num", "Form 4 net $ insider activity 30d","daily", "forward_fill"),
    "ticker.ownership.13d_holders":        ("num", "Count of active 13D filers",       "daily", "forward_fill"),
    "ticker.ownership.13g_holders":        ("num", "Count of active 13G filers",       "daily", "forward_fill"),
    "ticker.ownership.13f_holders_count":  ("num", "Number of 13F holders",            "quarterly", "forward_fill"),
    "ticker.ownership.13f_aum_pct_change": ("num", "QoQ change in 13F-reported value", "quarterly", "forward_fill"),
    "ticker.ownership.etf_pct":            ("num", "% of float held by ETFs",          "weekly",    "forward_fill"),
    "ticker.ownership.mutual_fund_pct":    ("num", "% of float held by mutual funds",  "monthly",   "forward_fill"),

    # ---- Short / borrow / options --------------------------------------
    "ticker.short.interest_pct_float":  ("num", "Short interest % of float",           "daily",   "forward_fill"),
    "ticker.short.days_to_cover":       ("num", "Short interest / avg daily volume",   "daily",   "forward_fill"),
    "ticker.short.borrow_fee_bps":      ("num", "Borrow fee in basis points",          "hourly",  "forward_fill"),
    "ticker.short.borrow_available":    ("num", "Shares available to borrow",          "hourly",  "forward_fill"),
    "ticker.options.iv_30d":            ("num", "ATM 30-day implied vol",              "hourly",  "forward_fill"),
    "ticker.options.iv_60d":            ("num", "ATM 60-day implied vol",              "hourly",  "forward_fill"),
    "ticker.options.iv_90d":            ("num", "ATM 90-day implied vol",              "hourly",  "forward_fill"),
    "ticker.options.iv_skew_25d":       ("num", "25-delta put-call IV skew",           "hourly",  "forward_fill"),
    "ticker.options.put_call_ratio":    ("num", "Put/call volume ratio",               "hourly",  "forward_fill"),
    "ticker.options.unusual_activity":  ("num", "Volume/OI z-score, max strike",       "hourly",  "forward_fill"),
    "ticker.options.iv_term_slope":     ("num", "IV(90d) - IV(30d), term-structure",   "hourly",  "forward_fill"),

    # ---- Analyst / earnings -------------------------------------------
    "ticker.analyst.consensus_target":   ("num", "Mean analyst price target",          "daily", "forward_fill"),
    "ticker.analyst.upgrade_count_30d":  ("num", "Upgrades in last 30 days",           "daily", "forward_fill"),
    "ticker.analyst.downgrade_count_30d":("num", "Downgrades in last 30 days",         "daily", "forward_fill"),
    "ticker.analyst.eps_revision_30d":   ("num", "Net EPS estimate revision 30d",      "daily", "forward_fill"),
    "ticker.analyst.next_earnings_date": ("text","Next earnings date (ISO)",           "daily", "forward_fill"),

    # ---- Sentiment / social -------------------------------------------
    "ticker.sentiment.wsb_mentions_24h": ("num", "WSB mention count last 24h",         "hourly", "null_safe"),
    "ticker.sentiment.wsb_score":        ("num", "Aggregated WSB sentiment -1..+1",    "hourly", "null_safe"),
    "ticker.sentiment.news_count_7d":    ("num", "News articles in last 7 days",       "hourly", "null_safe"),
    "ticker.sentiment.news_tone_7d":     ("num", "Mean GDELT tone last 7 days",        "hourly", "null_safe"),

    # ---- Derived signals ----------------------------------------------
    "ticker.signal.rsi_14":            ("num", "14-day RSI",                            "daily", "null_safe"),
    "ticker.signal.macd_hist":         ("num", "MACD histogram",                        "daily", "null_safe"),
    "ticker.signal.atr_14":            ("num", "14-day ATR",                            "daily", "null_safe"),
    "ticker.signal.sma_50":            ("num", "50-day simple MA",                      "daily", "null_safe"),
    "ticker.signal.sma_200":           ("num", "200-day simple MA",                     "daily", "null_safe"),
    "ticker.signal.realized_vol_30d":  ("num", "30-day realized vol (annualized)",      "daily", "null_safe"),
    "ticker.signal.drawdown":          ("num", "Drawdown from 252-day high",            "daily", "null_safe"),
    "ticker.signal.rs_vs_sector_60d":  ("num", "Return spread vs sector ETF, 60d",      "daily", "null_safe"),
    "ticker.signal.regime":            ("text","Trend regime: BULL/BEAR/CHOP",          "daily", "null_safe"),

    # ---- Factor exposures ---------------------------------------------
    "ticker.factor.beta_market":  ("num", "Market beta",                                "weekly", "null_safe"),
    "ticker.factor.beta_smb":     ("num", "Size factor beta",                           "weekly", "null_safe"),
    "ticker.factor.beta_hml":     ("num", "Value factor beta",                          "weekly", "null_safe"),
    "ticker.factor.beta_mom":     ("num", "Momentum factor beta",                       "weekly", "null_safe"),
    "ticker.factor.beta_qmj":     ("num", "Quality minus junk beta",                    "weekly", "null_safe"),

    # ---- Composite risk scores ----------------------------------------
    "ticker.risk.macro_stress_score":   ("num", "Composite macro stress 0..1",          "daily", "derived_proxy"),
    "ticker.risk.supply_chain_score":   ("num", "Supply chain risk 0..1",               "daily", "derived_proxy"),
    "ticker.risk.sanctions_pressure":   ("num", "Sanctions exposure 0..1",              "daily", "derived_proxy"),
    "ticker.risk.commodity_sensitivity":("num", "Sensitivity to input commodities 0..1","weekly","derived_proxy"),
    "ticker.risk.energy_crisis_score":  ("num", "Energy risk exposure 0..1",            "daily", "derived_proxy"),

    # ---- Sector / global ----------------------------------------------
    "sector.return_5d":               ("num", "Sector ETF 5-day return",                "daily", "forward_fill"),
    "sector.return_20d":              ("num", "Sector ETF 20-day return",               "daily", "forward_fill"),
    "sector.return_60d":              ("num", "Sector ETF 60-day return",               "daily", "forward_fill"),
    "sector.breadth_above_50dma":     ("num", "% of sector members above 50-DMA",       "daily", "forward_fill"),

    "global.vix":                     ("num", "VIX index level",                        "daily", "forward_fill"),
    "global.vix_3m":                  ("num", "VIX3M",                                  "daily", "forward_fill"),
    "global.vix_term_3m_1m":          ("num", "VIX3M / VIX1M ratio",                    "daily", "forward_fill"),
    "global.hy_oas":                  ("num", "ICE BofA US HY OAS",                     "daily", "forward_fill"),
    "global.ig_oas":                  ("num", "ICE BofA US IG OAS",                     "daily", "forward_fill"),
    "global.recession_prob":          ("num", "FRED smoothed recession probability",    "monthly","forward_fill"),
    "global.consumer_sentiment":      ("num", "U-Mich consumer sentiment",              "monthly","forward_fill"),
    "global.financial_stress":        ("num", "St. Louis Fed financial stress index",   "weekly", "forward_fill"),
    "global.yield_curve_2y10y":       ("num", "T10Y2Y spread",                          "daily",  "forward_fill"),
    "global.sanctions_added_7d":      ("num", "New sanctioned entities in last 7d",     "daily",  "forward_fill"),
    "global.geopolitical_event_24h":  ("num", "GDELT count, financial themes, 24h",     "hourly", "forward_fill"),

    # ---- Commodity / energy / freight ---------------------------------
    "commodity.spot_usd":             ("num", "Spot price in USD",                      "daily",  "forward_fill"),
    "commodity.storage":              ("num", "Storage level (units source-dependent)", "weekly", "forward_fill"),
    "freight.bdiy":                   ("num", "Baltic Dry Index",                       "daily",  "forward_fill"),
    "freight.scfi":                   ("num", "Shanghai Containerized Freight Index",   "weekly", "forward_fill"),
    "freight.port_congestion_score":  ("num", "Port congestion 0..1",                   "hourly", "forward_fill"),
}


# =============================================================================
# SOURCE PRIORITY
# =============================================================================
# Maps canonical field -> ordered list of source_ids, highest authority first.
# Used by `latest()` and `merge_priority()`. A field not listed here falls
# back to most-recent-wins.

SOURCE_PRIORITY: dict[str, list[str]] = {
    # Fundamentals: SEC XBRL is gold standard, then aggregators.
    "ticker.fundamental.revenue_ttm":     ["sec_xbrl", "finnhub", "yfinance"],
    "ticker.fundamental.eps_diluted_ttm": ["sec_xbrl", "finnhub", "yfinance"],
    "ticker.fundamental.fcf_ttm":         ["sec_xbrl", "yfinance"],
    "ticker.fundamental.net_income_ttm":  ["sec_xbrl", "yfinance"],
    "ticker.fundamental.capex_ttm":       ["sec_xbrl", "yfinance"],
    "ticker.fundamental.rd_ttm":          ["sec_xbrl", "yfinance"],
    "ticker.fundamental.tax_rate":        ["sec_xbrl"],
    "ticker.fundamental.goodwill":        ["sec_xbrl", "yfinance"],
    "ticker.fundamental.pension_liability":["sec_xbrl"],
    "ticker.fundamental.shares_diluted":  ["sec_xbrl", "yfinance"],
    "ticker.fundamental.sbc_ttm":         ["sec_xbrl"],
    "ticker.fundamental.long_term_debt":  ["sec_xbrl", "yfinance"],

    # Insider/ownership: only EDGAR is authoritative.
    "ticker.ownership.insider_buys_30d":  ["sec_form4"],
    "ticker.ownership.insider_sells_30d": ["sec_form4"],
    "ticker.ownership.insider_net_usd_30d":["sec_form4"],
    "ticker.ownership.13d_holders":       ["sec_13d"],
    "ticker.ownership.13g_holders":       ["sec_13g"],
    "ticker.ownership.13f_holders_count": ["sec_13f"],

    # Prices: yfinance primary; could add stooq/tiingo as backup later.
    "ticker.price.adj_close": ["yfinance", "stooq", "tiingo"],
    "ticker.price.close":     ["yfinance", "stooq", "tiingo"],
    "ticker.price.volume":    ["yfinance", "stooq", "tiingo"],

    # Short interest: FINRA primary, yfinance fallback (often stale).
    "ticker.short.interest_pct_float": ["finra", "yfinance"],
    "ticker.short.days_to_cover":      ["finra", "yfinance"],
    "ticker.short.borrow_fee_bps":     ["ibkr_api", "ibkr_scrape"],

    # Macro: FRED is gold.
    "global.vix":               ["fred", "yfinance"],
    "global.vix_3m":            ["fred", "cboe"],
    "global.hy_oas":             ["fred"],
    "global.ig_oas":             ["fred"],
    "global.recession_prob":     ["fred"],
    "global.consumer_sentiment": ["fred"],
    "global.financial_stress":   ["fred"],
    "global.yield_curve_2y10y":  ["fred"],

    # Analyst: Finnhub primary, yfinance fallback.
    "ticker.analyst.consensus_target": ["finnhub", "yfinance"],
    "ticker.analyst.next_earnings_date": ["finnhub", "yfinance"],
}


# =============================================================================
# CONFIDENCE GUIDE
# =============================================================================
# Reference rubric used by fetchers when calling upsert_point. Stored only
# for documentation; not enforced by code.

CONFIDENCE_GUIDE: dict[str, float] = {
    "sec_xbrl": 1.0,
    "fred": 1.0,
    "sec_form4": 1.0,
    "sec_13d": 1.0,
    "sec_13g": 1.0,
    "sec_13f": 1.0,
    "ofac": 1.0,
    "eu_sanctions": 1.0,
    "uk_sanctions": 1.0,
    "un_sanctions": 1.0,
    "finra": 0.9,
    "yfinance": 0.9,
    "finnhub": 0.85,
    "tiingo": 0.85,
    "marketaux": 0.8,
    "gdelt": 0.8,
    "stocktwits": 0.6,
    "reddit_wsb": 0.4,
    "ibkr_scrape": 0.4,
    "marketwatch_scrape": 0.4,
    "derived": 1.0,  # only as good as inputs; weighted at compute time
    "derived_proxy": 0.6,
}


# =============================================================================
# REGISTRY CLASS
# =============================================================================


class DataRegistry:
    """Unified write/read interface over the canonical SQLite schema."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        # WAL mode lets readers and one writer coexist — important once
        # the scheduler runs derived signal recomputes alongside fetches.
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
        except sqlite3.OperationalError:
            pass
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._conn() as conn:
            # Migrations table tracks which versions have been applied.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version     INTEGER PRIMARY KEY,
                    applied_at  TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )

            applied = {
                row[0]
                for row in conn.execute("SELECT version FROM schema_migrations")
            }

            if 1 not in applied:
                self._migration_v1(conn)
                conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES (?)", (1,)
                )
                logger.info("data_registry | applied migration v1")

    @staticmethod
    def _migration_v1(conn: sqlite3.Connection) -> None:
        """Initial unified schema."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_sources (
                source_id        TEXT PRIMARY KEY,
                category         TEXT NOT NULL,
                is_free          INTEGER NOT NULL DEFAULT 1,
                refresh_cadence  TEXT NOT NULL,
                base_priority    INTEGER NOT NULL DEFAULT 5,
                last_run         TEXT,
                last_success     TEXT,
                last_error       TEXT,
                notes            TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_points (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id    TEXT NOT NULL,
                entity_type  TEXT NOT NULL,
                field        TEXT NOT NULL,
                value_num    REAL,
                value_text   TEXT,
                value_json   TEXT,
                as_of        TEXT NOT NULL,
                fetched_at   TEXT NOT NULL DEFAULT (datetime('now')),
                source_id    TEXT NOT NULL,
                confidence   REAL NOT NULL DEFAULT 1.0,
                UNIQUE(entity_id, field, as_of, source_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dp_entity ON data_points(entity_id, field)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dp_field  ON data_points(field, as_of)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dp_source ON data_points(source_id, fetched_at)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id     TEXT,
                entity_type   TEXT,
                event_type    TEXT NOT NULL,
                occurred_at   TEXT NOT NULL,
                fetched_at    TEXT NOT NULL DEFAULT (datetime('now')),
                source_id     TEXT NOT NULL,
                severity      REAL,
                payload_json  TEXT,
                UNIQUE(entity_id, event_type, occurred_at, source_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ev_entity ON events(entity_id, occurred_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ev_type   ON events(event_type, occurred_at)")

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_quality_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at   TEXT NOT NULL DEFAULT (datetime('now')),
                entity_id   TEXT,
                field       TEXT,
                issue       TEXT NOT NULL,
                detail      TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS refresh_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                task          TEXT NOT NULL,
                started_at    TEXT NOT NULL,
                finished_at   TEXT,
                rows_written  INTEGER DEFAULT 0,
                status        TEXT,
                error         TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_refresh_task ON refresh_log(task, started_at)")

    # ------------------------------------------------------------------
    # Source registration
    # ------------------------------------------------------------------
    def register_source(
        self,
        source_id: str,
        category: str,
        refresh_cadence: str,
        is_free: bool = True,
        base_priority: int = 5,
        notes: str = "",
    ) -> None:
        """Idempotent insert of a source into ``data_sources``."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO data_sources
                    (source_id, category, is_free, refresh_cadence, base_priority, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    category = excluded.category,
                    is_free = excluded.is_free,
                    refresh_cadence = excluded.refresh_cadence,
                    base_priority = excluded.base_priority,
                    notes = excluded.notes
                """,
                (source_id, category, int(is_free), refresh_cadence, base_priority, notes),
            )

    def mark_source_run(
        self, source_id: str, success: bool = True, error: Optional[str] = None
    ) -> None:
        """Update last_run / last_success / last_error on a source."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            if success:
                conn.execute(
                    """
                    UPDATE data_sources
                       SET last_run = ?, last_success = ?, last_error = NULL
                     WHERE source_id = ?
                    """,
                    (now, now, source_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE data_sources
                       SET last_run = ?, last_error = ?
                     WHERE source_id = ?
                    """,
                    (now, (error or "")[:1000], source_id),
                )

    # ------------------------------------------------------------------
    # Upserts
    # ------------------------------------------------------------------
    def upsert_point(
        self,
        entity_id: str,
        entity_type: str,
        field: str,
        as_of: Union[str, datetime],
        source_id: str,
        value_num: Optional[float] = None,
        value_text: Optional[str] = None,
        value_json: Optional[Any] = None,
        confidence: float = 1.0,
    ) -> int:
        """Insert or replace a single data point.

        Returns the row id (new or updated). The ``UNIQUE`` constraint on
        ``(entity_id, field, as_of, source_id)`` makes this idempotent.
        """
        if field not in CANONICAL_FIELDS:
            logger.debug("data_registry | non-canonical field used: %s", field)

        as_of_str = as_of.isoformat() if isinstance(as_of, datetime) else str(as_of)
        json_str = (
            json.dumps(value_json, default=str)
            if value_json is not None
            else None
        )

        confidence = max(0.0, min(1.0, float(confidence)))

        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO data_points
                    (entity_id, entity_type, field, value_num, value_text,
                     value_json, as_of, source_id, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id, field, as_of, source_id) DO UPDATE SET
                    value_num   = excluded.value_num,
                    value_text  = excluded.value_text,
                    value_json  = excluded.value_json,
                    fetched_at  = datetime('now'),
                    confidence  = excluded.confidence
                """,
                (
                    entity_id,
                    entity_type,
                    field,
                    value_num,
                    value_text,
                    json_str,
                    as_of_str,
                    source_id,
                    confidence,
                ),
            )
            return cur.lastrowid or 0

    def upsert_points_bulk(self, rows: Iterable[dict]) -> int:
        """Bulk insert. Each row is a dict matching upsert_point kwargs.

        Returns count written. Much faster than calling upsert_point in a loop
        because everything goes in one transaction.
        """
        n = 0
        with self._conn() as conn:
            for r in rows:
                if r.get("field") and r["field"] not in CANONICAL_FIELDS:
                    logger.debug(
                        "data_registry | non-canonical field used: %s", r["field"]
                    )
                as_of = r["as_of"]
                if isinstance(as_of, datetime):
                    as_of = as_of.isoformat()
                json_val = r.get("value_json")
                if json_val is not None and not isinstance(json_val, str):
                    json_val = json.dumps(json_val, default=str)
                conn.execute(
                    """
                    INSERT INTO data_points
                        (entity_id, entity_type, field, value_num, value_text,
                         value_json, as_of, source_id, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entity_id, field, as_of, source_id) DO UPDATE SET
                        value_num   = excluded.value_num,
                        value_text  = excluded.value_text,
                        value_json  = excluded.value_json,
                        fetched_at  = datetime('now'),
                        confidence  = excluded.confidence
                    """,
                    (
                        r["entity_id"],
                        r["entity_type"],
                        r["field"],
                        r.get("value_num"),
                        r.get("value_text"),
                        json_val,
                        str(as_of),
                        r["source_id"],
                        max(0.0, min(1.0, float(r.get("confidence", 1.0)))),
                    ),
                )
                n += 1
        return n

    def upsert_event(
        self,
        event_type: str,
        occurred_at: Union[str, datetime],
        source_id: str,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        severity: Optional[float] = None,
        payload: Optional[Any] = None,
    ) -> int:
        """Insert or replace an event row."""
        occurred_str = (
            occurred_at.isoformat()
            if isinstance(occurred_at, datetime)
            else str(occurred_at)
        )
        payload_str = (
            json.dumps(payload, default=str) if payload is not None else None
        )
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO events
                    (entity_id, entity_type, event_type, occurred_at,
                     source_id, severity, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id, event_type, occurred_at, source_id) DO UPDATE SET
                    severity     = excluded.severity,
                    payload_json = excluded.payload_json,
                    fetched_at   = datetime('now')
                """,
                (
                    entity_id,
                    entity_type,
                    event_type,
                    occurred_str,
                    source_id,
                    severity,
                    payload_str,
                ),
            )
            return cur.lastrowid or 0

    def log_quality_issue(
        self,
        issue: str,
        entity_id: Optional[str] = None,
        field: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO data_quality_log (entity_id, field, issue, detail) VALUES (?, ?, ?, ?)",
                (entity_id, field, issue, detail),
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def latest(
        self,
        entity_id: str,
        field: str,
        as_of_max: Optional[str] = None,
    ) -> Optional[dict]:
        """Return the highest-priority, freshest observation of (entity, field).

        Resolution order:
            1. Pull every row matching (entity, field) [optionally filtered by as_of_max].
            2. Within each source, take the row with the latest as_of.
            3. Across sources, pick by SOURCE_PRIORITY (lower index = higher priority).
            4. Tiebreak by confidence desc, then as_of desc.
        Returns ``None`` if no rows exist.
        """
        sql = (
            "SELECT entity_id, entity_type, field, value_num, value_text, "
            "       value_json, as_of, fetched_at, source_id, confidence "
            "  FROM data_points "
            " WHERE entity_id = ? AND field = ?"
        )
        params: list[Any] = [entity_id, field]
        if as_of_max:
            sql += " AND as_of <= ?"
            params.append(as_of_max)
        sql += " ORDER BY as_of DESC"

        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

        if not rows:
            return None

        # Group by source, keep latest per source
        per_source: dict[str, dict] = {}
        for r in rows:
            sid = r["source_id"]
            if sid not in per_source:
                per_source[sid] = r

        # Priority pick
        priority = SOURCE_PRIORITY.get(field)
        if priority:
            for sid in priority:
                if sid in per_source:
                    return _decode_json(per_source[sid])
        # Fallback: highest confidence, then latest
        best = max(
            per_source.values(),
            key=lambda r: (float(r["confidence"] or 0), r["as_of"]),
        )
        return _decode_json(best)

    def latest_value(
        self,
        entity_id: str,
        field: str,
        as_of_max: Optional[str] = None,
    ) -> Optional[Union[float, str, dict, list]]:
        """Convenience: return just the value (numeric, text, or decoded json)."""
        row = self.latest(entity_id, field, as_of_max=as_of_max)
        if row is None:
            return None
        if row.get("value_num") is not None:
            return row["value_num"]
        if row.get("value_text") is not None:
            return row["value_text"]
        return row.get("value_json")

    def time_series(
        self,
        entity_id: str,
        field: str,
        since: Optional[str] = None,
        until: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> list[dict]:
        """Return a list of rows for (entity, field) ordered by as_of asc."""
        sql = (
            "SELECT as_of, value_num, value_text, value_json, source_id, confidence "
            "  FROM data_points WHERE entity_id = ? AND field = ?"
        )
        params: list[Any] = [entity_id, field]
        if since:
            sql += " AND as_of >= ?"
            params.append(since)
        if until:
            sql += " AND as_of <= ?"
            params.append(until)
        if source_id:
            sql += " AND source_id = ?"
            params.append(source_id)
        sql += " ORDER BY as_of ASC"
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            return [_decode_json(dict(r)) for r in conn.execute(sql, params).fetchall()]

    def recent_events(
        self,
        entity_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        sql = "SELECT * FROM events WHERE 1=1"
        params: list[Any] = []
        if entity_id:
            sql += " AND entity_id = ?"
            params.append(entity_id)
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        if since:
            sql += " AND occurred_at >= ?"
            params.append(since)
        sql += " ORDER BY occurred_at DESC LIMIT ?"
        params.append(int(limit))
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            return [_decode_json(dict(r), payload_key="payload_json") for r in conn.execute(sql, params).fetchall()]

    # ------------------------------------------------------------------
    # Snapshot helpers used by the agent context builder
    # ------------------------------------------------------------------
    def snapshot(
        self,
        entity_id: str,
        fields: Sequence[str],
        as_of_max: Optional[str] = None,
    ) -> dict[str, Any]:
        """Return {field: value} for the given list of fields. Missing fields
        are present in the dict with value None so the LLM context shape is
        stable.
        """
        out: dict[str, Any] = {}
        for f in fields:
            out[f] = self.latest_value(entity_id, f, as_of_max=as_of_max)
        return out

    # ------------------------------------------------------------------
    # Refresh log helpers
    # ------------------------------------------------------------------
    @contextmanager
    def refresh_run(self, task: str):
        """Context manager that wraps a refresh task with logging.

        Usage:
            with reg.refresh_run("hourly_prices") as run:
                run["rows_written"] = fetch_prices(...)
        """
        started = datetime.now().isoformat(timespec="seconds")
        state: dict = {"rows_written": 0, "status": "ok", "error": None}
        try:
            yield state
        except Exception as e:  # noqa: BLE001 — we want to log all failures
            state["status"] = "error"
            state["error"] = repr(e)[:1000]
            raise
        finally:
            finished = datetime.now().isoformat(timespec="seconds")
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO refresh_log
                        (task, started_at, finished_at, rows_written, status, error)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task,
                        started,
                        finished,
                        int(state.get("rows_written", 0)),
                        state.get("status"),
                        state.get("error"),
                    ),
                )

    def last_run(self, task: str) -> Optional[datetime]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT MAX(started_at) FROM refresh_log WHERE task = ? AND status = 'ok'",
                (task,),
            ).fetchone()
        if not row or not row[0]:
            return None
        try:
            return datetime.fromisoformat(row[0])
        except ValueError:
            return None

    def is_due(self, task: str, interval_seconds: int) -> bool:
        last = self.last_run(task)
        if last is None:
            return True
        return (datetime.now() - last).total_seconds() >= interval_seconds


# =============================================================================
# Helpers
# =============================================================================


def _decode_json(row: dict, payload_key: str = "value_json") -> dict:
    """If the row's json column is a non-empty string, decode it in place."""
    val = row.get(payload_key)
    if isinstance(val, str) and val:
        try:
            row[payload_key] = json.loads(val)
        except (TypeError, ValueError):
            pass  # leave as raw string if not valid JSON
    return row


# =============================================================================
# Module-level singleton helper
# =============================================================================
# Most fetchers will share one DataRegistry instance. The lazy accessor below
# keeps us from constructing it at import time (which would create the DB).

_DEFAULT_REGISTRY: Optional[DataRegistry] = None


def get_default_registry(db_path: Optional[str] = None) -> DataRegistry:
    """Return a process-wide DataRegistry. Creates one on first call."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None or (
        db_path is not None and _DEFAULT_REGISTRY.db_path != db_path
    ):
        _DEFAULT_REGISTRY = DataRegistry(db_path or DEFAULT_DB_PATH)
    return _DEFAULT_REGISTRY


# =============================================================================
# CLI smoke test
# =============================================================================
if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name

    reg = DataRegistry(path)
    reg.register_source("yfinance", "price", "hourly", base_priority=2)
    reg.register_source("sec_xbrl", "filing", "quarterly", base_priority=1)
    reg.register_source("derived", "computed", "daily", base_priority=1)

    reg.upsert_point(
        "AAPL", "ticker", "ticker.price.adj_close",
        as_of="2026-04-25", source_id="yfinance",
        value_num=187.43, confidence=0.9,
    )
    reg.upsert_point(
        "AAPL", "ticker", "ticker.fundamental.eps_diluted_ttm",
        as_of="2026-03-31", source_id="sec_xbrl",
        value_num=6.84, confidence=1.0,
    )
    reg.upsert_point(
        "AAPL", "ticker", "ticker.fundamental.eps_diluted_ttm",
        as_of="2026-03-31", source_id="yfinance",
        value_num=6.79, confidence=0.9,
    )

    print("latest(AAPL, eps_diluted_ttm) ->",
          reg.latest_value("AAPL", "ticker.fundamental.eps_diluted_ttm"))
    print("snapshot ->", reg.snapshot(
        "AAPL",
        ["ticker.price.adj_close", "ticker.fundamental.eps_diluted_ttm",
         "ticker.signal.rsi_14"],
    ))

    with reg.refresh_run("smoke") as run:
        run["rows_written"] = 3
    print("last_run(smoke) ->", reg.last_run("smoke"))
    print("is_due(smoke, 60s) ->", reg.is_due("smoke", 60))

    print("OK")
