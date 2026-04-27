"""
refresh_scheduler.py
====================
Cadence-driven orchestrator for the Ary Quant data layer.

Wraps every fetcher in:
    * a refresh_log entry (start, end, rows, status, error)
    * is_due() gating against last_run, so re-running the scheduler is safe
    * graceful skip on missing dependencies (e.g. no FRED key, no yfinance)

Cadences
--------
hourly : fast-moving signals (prices, options, social, news, geopolitical)
daily  : fundamentals, filings, sanctions full snapshots, macro daily
weekly : 13F holdings, factor exposures, ADV registrations
event  : ad hoc handlers (8-K material event, rating change, ownership > 5%)

Public API
----------
    >>> sch = RefreshScheduler(tickers=["AAPL","MSFT","NVDA"])
    >>> sch.run_hourly()
    >>> sch.run_daily()
    >>> sch.run_weekly()
    >>> sch.run_event("share_repurchase_announce", {"ticker": "AAPL"})

The scheduler does not own a process loop — it's designed to be invoked
from cron, an external scheduler (Airflow, APScheduler), or a Streamlit
button. This keeps the file dependency-light.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)


# Default tickers used when none are passed. Caller normally overrides.
DEFAULT_WATCHLIST: list[str] = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]


# Interval seconds for is_due() — guard rails so the same task isn't
# re-run within a sensible cooldown.
INTERVAL_HOURLY = 60 * 50            # ~50 minutes
INTERVAL_DAILY = 60 * 60 * 20        # ~20 hours
INTERVAL_WEEKLY = 60 * 60 * 24 * 6   # ~6 days


@dataclass
class TaskResult:
    name: str
    rows_written: int = 0
    duration_s: float = 0.0
    error: Optional[str] = None


# ----------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------


class RefreshScheduler:
    """Orchestrates cadenced refreshes across the 7 other modules."""

    def __init__(
        self,
        tickers: Optional[list[str]] = None,
        db_path: str = "data/hedgefund.db",
        registry=None,
        market_data=None,
        macro_data=None,
        sec_fetcher=None,
        sentiment_news=None,
        geo_supply=None,
        derived_signals=None,
    ):
        self.tickers = [t.upper() for t in (tickers or DEFAULT_WATCHLIST)]
        self.db_path = db_path
        self._registry = registry
        # Lazy-construct each module so import-time failures don't block
        # other capabilities.
        self._market = market_data
        self._macro = macro_data
        self._sec = sec_fetcher
        self._sentiment = sentiment_news
        self._geo = geo_supply
        self._derived = derived_signals

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------
    @property
    def registry(self):
        if self._registry is None:
            try:
                from data.data_registry import get_default_registry
                self._registry = get_default_registry(self.db_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("scheduler | registry load failed: %s", e)
        return self._registry

    @property
    def market(self):
        if self._market is None:
            try:
                from data.market_data import MarketData
                self._market = MarketData(db_path=self.db_path, registry=self.registry)
            except Exception as e:  # noqa: BLE001
                logger.warning("scheduler | market_data unavailable: %s", e)
        return self._market

    @property
    def macro(self):
        if self._macro is None:
            try:
                from data.macro_data import MacroData
                self._macro = MacroData(db_path=self.db_path, registry=self.registry)
            except Exception as e:  # noqa: BLE001
                logger.warning("scheduler | macro_data unavailable: %s", e)
        return self._macro

    @property
    def sec(self):
        if self._sec is None:
            try:
                from data.sec_fetcher import SECFetcher
                self._sec = SECFetcher(db_path=self.db_path, registry=self.registry)
            except Exception as e:  # noqa: BLE001
                logger.warning("scheduler | sec_fetcher unavailable: %s", e)
        return self._sec

    @property
    def sentiment(self):
        if self._sentiment is None:
            try:
                from data.sentiment_news import SentimentNews
                self._sentiment = SentimentNews(db_path=self.db_path, registry=self.registry)
            except Exception as e:  # noqa: BLE001
                logger.warning("scheduler | sentiment_news unavailable: %s", e)
        return self._sentiment

    @property
    def geo(self):
        if self._geo is None:
            try:
                from data.geo_supply import GeoSupply
                self._geo = GeoSupply(db_path=self.db_path, registry=self.registry)
            except Exception as e:  # noqa: BLE001
                logger.warning("scheduler | geo_supply unavailable: %s", e)
        return self._geo

    @property
    def derived(self):
        if self._derived is None:
            try:
                from data.derived_signals import DerivedSignals
                self._derived = DerivedSignals(db_path=self.db_path, registry=self.registry)
            except Exception as e:  # noqa: BLE001
                logger.warning("scheduler | derived_signals unavailable: %s", e)
        return self._derived

    # ------------------------------------------------------------------
    # Run a single task with logging + is_due gating
    # ------------------------------------------------------------------
    def _run_task(
        self,
        name: str,
        fn: Callable[[], Any],
        interval_seconds: int,
        force: bool = False,
    ) -> TaskResult:
        reg = self.registry
        if reg is not None and not force and not reg.is_due(name, interval_seconds):
            logger.info("scheduler | %s | skipped (not due)", name)
            return TaskResult(name=name, rows_written=0, error="skipped_not_due")

        if reg is None:
            # Run anyway, just no logging
            return self._run_task_unlogged(name, fn)

        result = TaskResult(name=name)
        with reg.refresh_run(name) as state:
            try:
                rv = fn()
                # Try to extract a numeric "rows written" hint
                rows = self._extract_rows(rv)
                state["rows_written"] = rows
                result.rows_written = rows
            except Exception as e:  # noqa: BLE001
                result.error = repr(e)[:500]
                logger.warning("scheduler | %s | failed: %s", name, e)
                # refresh_run will record the error
                raise
        return result

    @staticmethod
    def _run_task_unlogged(name: str, fn: Callable[[], Any]) -> TaskResult:
        try:
            rv = fn()
            return TaskResult(name=name, rows_written=RefreshScheduler._extract_rows(rv))
        except Exception as e:  # noqa: BLE001
            return TaskResult(name=name, error=repr(e)[:500])

    @staticmethod
    def _extract_rows(rv: Any) -> int:
        """Heuristic: handle int returns and dict-of-counts returns."""
        if isinstance(rv, int):
            return rv
        if isinstance(rv, dict):
            total = 0
            for v in rv.values():
                if isinstance(v, int):
                    total += v
            return total
        return 0

    def _try_task(self, *args, **kwargs) -> TaskResult:
        try:
            return self._run_task(*args, **kwargs)
        except Exception:  # noqa: BLE001 — caller-side guard
            return TaskResult(
                name=args[0] if args else "unknown",
                error="handled_exception",
            )

    # ------------------------------------------------------------------
    # Cadenced run blocks
    # ------------------------------------------------------------------
    def run_hourly(self, force: bool = False) -> list[TaskResult]:
        results: list[TaskResult] = []
        # Per-ticker hourly: prices, options, sentiment news/social
        if self.market:
            for tk in self.tickers:
                results.append(self._try_task(
                    f"hourly_prices_{tk}",
                    lambda tk=tk: self.market.sync_prices_to_registry(tk, period="1mo"),
                    INTERVAL_HOURLY, force=force,
                ))
                results.append(self._try_task(
                    f"hourly_options_{tk}",
                    lambda tk=tk: self.market.get_options_chain(tk, max_expiries=2),
                    INTERVAL_HOURLY, force=force,
                ))
        if self.sentiment:
            for tk in self.tickers:
                results.append(self._try_task(
                    f"hourly_sentiment_{tk}",
                    lambda tk=tk: self.sentiment.refresh_ticker(tk),
                    INTERVAL_HOURLY, force=force,
                ))
        if self.geo:
            results.append(self._try_task(
                "hourly_gdelt",
                lambda: self.geo.fetch_gdelt_geopolitical(),
                INTERVAL_HOURLY, force=force,
            ))
        # Derived recompute uses cached data, fast
        if self.derived:
            for tk in self.tickers:
                results.append(self._try_task(
                    f"hourly_derived_{tk}",
                    lambda tk=tk: self.derived.recompute_for(tk),
                    INTERVAL_HOURLY, force=force,
                ))
        return results

    def run_daily(self, force: bool = False) -> list[TaskResult]:
        results: list[TaskResult] = []
        if self.macro:
            results.append(self._try_task(
                "daily_macro",
                lambda: self.macro.refresh_macro(),
                INTERVAL_DAILY, force=force,
            ))
        if self.sec:
            for tk in self.tickers:
                results.append(self._try_task(
                    f"daily_sec_{tk}",
                    lambda tk=tk: self.sec.refresh_ticker_filings(tk),
                    INTERVAL_DAILY, force=force,
                ))
        if self.market:
            for tk in self.tickers:
                results.append(self._try_task(
                    f"daily_market_{tk}",
                    lambda tk=tk: self.market.refresh_ticker_market(tk),
                    INTERVAL_DAILY, force=force,
                ))
        if self.geo:
            results.append(self._try_task(
                "daily_sanctions",
                lambda: self.geo.refresh_global(),
                INTERVAL_DAILY, force=force,
            ))
        if self.derived:
            results.append(self._try_task(
                "daily_sector_heatmap",
                lambda: self.derived.recompute_sector_heatmap(),
                INTERVAL_DAILY, force=force,
            ))
            for tk in self.tickers:
                results.append(self._try_task(
                    f"daily_risk_scores_{tk}",
                    lambda tk=tk: self.derived.recompute_risk_scores(tk),
                    INTERVAL_DAILY, force=force,
                ))
        return results

    def run_weekly(self, force: bool = False) -> list[TaskResult]:
        results: list[TaskResult] = []
        if self.derived:
            for tk in self.tickers:
                results.append(self._try_task(
                    f"weekly_factors_{tk}",
                    lambda tk=tk: self.derived.recompute_factor_exposures(tk),
                    INTERVAL_WEEKLY, force=force,
                ))
        return results

    # ------------------------------------------------------------------
    # Event-driven hooks
    # ------------------------------------------------------------------
    def run_event(self, event_type: str, payload: Optional[dict] = None) -> TaskResult:
        """React to an event by triggering a targeted refresh."""
        payload = payload or {}
        ticker = (payload.get("ticker") or "").upper()
        name = f"event_{event_type}_{ticker or 'global'}"
        try:
            if event_type == "8k_filed" and ticker and self.sec:
                rv = self.sec.ingest_corporate_actions(ticker)
            elif event_type == "rating_change_mention" and ticker and self.sentiment:
                rv = self.sentiment.fetch_analyst_events(ticker)
            elif event_type == "form4" and ticker and self.sec:
                rv = self.sec.ingest_insider_transactions(ticker, count=10)
            elif event_type == "ownership_threshold" and ticker and self.sec:
                rv = self.sec.ingest_ownership_filings(ticker, "SC 13D", count=5)
            elif event_type == "sanctioned_entity_added" and self.geo:
                rv = self.geo.refresh_global()
            else:
                rv = 0
            return TaskResult(name=name, rows_written=self._extract_rows(rv))
        except Exception as e:  # noqa: BLE001
            return TaskResult(name=name, error=repr(e)[:500])

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Return a summary of when each task last ran successfully."""
        if self.registry is None:
            return {}
        out: dict[str, Any] = {}
        with self.registry._conn() as conn:
            rows = conn.execute(
                """SELECT task, MAX(started_at) FROM refresh_log
                    WHERE status = 'ok' GROUP BY task ORDER BY task"""
            ).fetchall()
        for task, last in rows:
            out[task] = last
        return out


# ----------------------------------------------------------------------
# CLI runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cadence", choices=["hourly", "daily", "weekly", "status"])
    p.add_argument("--tickers", nargs="+", default=DEFAULT_WATCHLIST)
    p.add_argument("--db", default="data/hedgefund.db")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    sch = RefreshScheduler(tickers=args.tickers, db_path=args.db)
    if args.cadence == "status":
        for k, v in sch.status().items():
            print(f"{k:50s} last={v}")
    else:
        method = getattr(sch, f"run_{args.cadence}")
        results = method(force=args.force)
        for r in results:
            print(f"{r.name:50s} rows={r.rows_written:5d}  err={r.error or ''}")
