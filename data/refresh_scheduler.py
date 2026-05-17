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
hourly      : fast-moving signals (prices, options, social, news, geopolitical)
daily       : fundamentals, filings, sanctions full snapshots, macro daily,
              Ken French factor returns
weekly      : 13F holdings, factor exposures, ADV registrations
event       : ad hoc handlers (8-K material event, rating change,
              ownership > 5%)
market_open : holdings scan + Slack push for elevated risk flags

Public API
----------
    >>> sch = RefreshScheduler(tickers=["AAPL","MSFT","NVDA"])
    >>> sch.run_hourly()
    >>> sch.run_daily()
    >>> sch.run_weekly()
    >>> sch.run_market_open_scan()
    >>> sch.run_event("share_repurchase_announce", {"ticker": "AAPL"})

The scheduler does not own a process loop — it's designed to be invoked
from cron, an external scheduler (Airflow, APScheduler), or a Streamlit
button. This keeps the file dependency-light.

Slack integration
-----------------
``run_market_open_scan()`` posts elevated risk flags to Slack via
``data.notifiers``. Set ``SLACK_WEBHOOK_URL`` in the environment to
enable; if unset, the scan still runs and persists results — only the
push step is skipped (with an INFO log).

Suggested cron entries
----------------------
::

    # Every hour during US market hours
    0  9-16 * * 1-5  cd /path/to/project && python -m data.refresh_scheduler hourly

    # Daily refresh and factor-returns ingest (run after close)
    30 16   * * 1-5  cd /path/to/project && python -m data.refresh_scheduler daily

    # Holdings scan + Slack push, ~5 minutes before market open
    25  9   * * 1-5  cd /path/to/project && python -m data.refresh_scheduler scan
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
INTERVAL_MARKET_OPEN_SCAN = 60 * 60 * 12  # ~12 hours — cap to once per session


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
    """Orchestrates cadenced refreshes across the data + agent layers."""

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
        slack_webhook_url: Optional[str] = None,
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
        # Slack webhook can be passed explicitly or resolved from env at
        # call time inside data.notifiers. We hold the explicit value so
        # callers (tests, alternative entry points) can override env.
        self.slack_webhook_url = slack_webhook_url

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
        # Ken French factor returns. Pulled daily because the data
        # library updates frequently and the loader is idempotent.
        # Failure here must NOT break the rest of the daily refresh,
        # which is why it sits in its own _try_task guard.
        results.append(self._try_task(
            "daily_factor_returns",
            lambda: self._refresh_factor_returns(),
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
    # Daily holdings scan + Slack push
    # ------------------------------------------------------------------
    def run_market_open_scan(
        self,
        force: bool = False,
        notify: bool = True,
        notify_levels: Iterable[str] = ("HIGH", "MEDIUM"),
    ) -> list[TaskResult]:
        """Pull every current portfolio position and score its risk flags.

        For each position:
          1. Build the agent context (registry-backed snapshot).
          2. Extract the agent-ready metric snapshot.
          3. Run :func:`agent.risk_scanner.compute_risk_flags` with an
             empty ``agent_risks`` list — this is the *deterministic*
             part of the scan and runs without an LLM.
          4. If ``notify`` and the combined level is in
             ``notify_levels``, push a formatted alert to Slack via
             :mod:`data.notifiers`.

        The scan is gated by ``is_due("market_open_scan", 12h)`` so
        running the cron entry twice in one morning doesn't double-fire
        Slack pings. Pass ``force=True`` to override (useful for manual
        re-runs after a config change).

        Returns one :class:`TaskResult` per ticker scanned, with
        ``rows_written`` repurposed as a 0/1 flag for "alert sent".
        """
        # Tickers: union of (a) current portfolio holdings (the spec) and
        # (b) anything explicitly passed to the scheduler — so a watchlist
        # run still covers the canonical tickers if positions table is
        # empty.
        tickers = self._scan_tickers()
        if not tickers:
            logger.info("scheduler | market_open_scan | no holdings to scan")
            return []

        # Hoist imports so the rest of the module stays import-cheap.
        try:
            from data import pipeline as pipeline_mod  # noqa: F401
            from agent import filing_analyzer, risk_scanner
            try:
                import config as cfg_mod
            except Exception:  # noqa: BLE001
                cfg_mod = None
            from data.notifiers import notify_risk_flags, slack_configured
        except Exception as e:  # noqa: BLE001
            logger.error("scheduler | market_open_scan | imports failed: %s", e)
            return [TaskResult(name="market_open_scan", error=repr(e)[:500])]

        slack_on = notify and slack_configured(self.slack_webhook_url)
        if notify and not slack_on:
            logger.info(
                "scheduler | market_open_scan | Slack not configured; "
                "scan will run but no alerts will be pushed."
            )

        results: list[TaskResult] = []
        for tk in tickers:
            name = f"market_open_scan_{tk}"
            # is_due() gate per ticker so a partial failure doesn't
            # block the rest of the universe on retry.
            reg = self.registry
            if reg is not None and not force and not reg.is_due(
                name, INTERVAL_MARKET_OPEN_SCAN
            ):
                logger.info("scheduler | %s | skipped (not due)", name)
                results.append(TaskResult(name=name, error="skipped_not_due"))
                continue

            try:
                if reg is not None:
                    with reg.refresh_run(name) as state:
                        sent = self._scan_one(
                            tk, cfg_mod, pipeline_mod, filing_analyzer,
                            risk_scanner, notify_risk_flags,
                            slack_on, notify_levels,
                        )
                        state["rows_written"] = int(bool(sent))
                else:
                    sent = self._scan_one(
                        tk, cfg_mod, pipeline_mod, filing_analyzer,
                        risk_scanner, notify_risk_flags,
                        slack_on, notify_levels,
                    )
                results.append(TaskResult(name=name, rows_written=int(bool(sent))))
            except Exception as e:  # noqa: BLE001
                logger.warning("scheduler | %s | failed: %s", name, e)
                results.append(TaskResult(name=name, error=repr(e)[:500]))

        # Summary log so cron output is glanceable.
        sent_count = sum(1 for r in results if r.rows_written)
        scanned = sum(1 for r in results if r.error != "skipped_not_due")
        logger.info(
            "scheduler | market_open_scan | scanned=%d alerts_sent=%d",
            scanned, sent_count,
        )
        return results

    def _scan_one(
        self,
        ticker: str,
        cfg_mod: Any,
        pipeline_mod: Any,
        filing_analyzer: Any,
        risk_scanner: Any,
        notify_risk_flags: Callable,
        slack_on: bool,
        notify_levels: Iterable[str],
    ) -> bool:
        """Run risk-scan + optional Slack push for one ticker.

        Returns True iff a Slack message was successfully sent.
        """
        ctx = pipeline_mod.build_agent_context(ticker, self.db_path, cfg_mod)
        price_val = ctx.get("price") or (ctx.get("prices") or {}).get("last") or 0.0
        try:
            price_val = float(price_val) if price_val is not None else 0.0
        except (TypeError, ValueError):
            price_val = 0.0
        key_metrics = filing_analyzer.extract_key_metrics_for_agent(
            ticker=ticker,
            metrics=ctx.get("metrics") or {},
            price=price_val,
        )
        flags = risk_scanner.compute_risk_flags(
            ticker=ticker,
            metrics=key_metrics,
            macro=ctx.get("macro") or {},
            agent_risks=[],  # deterministic scan — no LLM in this path
            config=cfg_mod,
        )
        combined = (flags.get("levels") or {}).get("combined", "LOW")
        logger.info(
            "scheduler | scan | %s | combined=%s", ticker, combined,
        )
        if not slack_on:
            return False
        return bool(notify_risk_flags(
            ticker, flags,
            webhook_url=self.slack_webhook_url,
            include_levels=notify_levels,
        ))

    def _scan_tickers(self) -> list[str]:
        """Tickers covered by the market-open scan.

        Prefer live portfolio positions (the README spec — "auto-scan
        holdings at market open"). If the positions table is empty or
        unreachable, fall back to ``self.tickers`` so the scan still
        produces useful output during paper-trading setup.
        """
        try:
            from data.portfolio_db import PortfolioDB
            pdb = PortfolioDB(db_path=self.db_path)
            held = [p["ticker"] for p in pdb.get_positions() if p.get("ticker")]
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "scheduler | market_open_scan | portfolio load failed: %s", e,
            )
            held = []
        if held:
            return [t.upper() for t in held]
        logger.info(
            "scheduler | market_open_scan | no live positions; "
            "falling back to watchlist (%d tickers)", len(self.tickers),
        )
        return list(self.tickers)

    # ------------------------------------------------------------------
    # Factor returns daily ingest
    # ------------------------------------------------------------------
    def _refresh_factor_returns(self) -> int:
        """Pull the latest Ken French daily factors into ``factor_returns``.

        The loader writes ``F-F_Research_Data_Factors_daily`` (Mkt-RF,
        SMB, HML, RF) and ``F-F_Momentum_Factor_daily`` (MOM). We pull
        the trailing ~30 days because Ken French publishes with a
        multi-week lag, so older data may still be backfilled into
        recent slots.

        Returns the total number of rows upserted across all factors.
        """
        try:
            from data.factor_returns_loader import load_all
        except Exception as e:  # noqa: BLE001
            logger.warning("scheduler | factor_returns import failed: %s", e)
            return 0
        from datetime import timedelta
        since = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
        written = load_all(db_path=self.db_path, since=since)
        return sum(written.values())

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
    p.add_argument(
        "cadence",
        choices=["hourly", "daily", "weekly", "scan", "status"],
        help=(
            "hourly/daily/weekly = standard refresh cadences. "
            "scan = market-open holdings scan with Slack push. "
            "status = print last-run timestamps and exit."
        ),
    )
    p.add_argument("--tickers", nargs="+", default=DEFAULT_WATCHLIST)
    p.add_argument("--db", default="data/hedgefund.db")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--no-notify",
        action="store_true",
        help="(scan only) Run the scan but skip the Slack push step.",
    )
    p.add_argument(
        "--webhook",
        default=None,
        help=(
            "(scan only) Override SLACK_WEBHOOK_URL. Useful for testing "
            "without modifying the environment."
        ),
    )
    args = p.parse_args()

    sch = RefreshScheduler(
        tickers=args.tickers,
        db_path=args.db,
        slack_webhook_url=args.webhook,
    )

    if args.cadence == "status":
        for k, v in sch.status().items():
            print(f"{k:50s} last={v}")
    elif args.cadence == "scan":
        results = sch.run_market_open_scan(
            force=args.force, notify=not args.no_notify,
        )
        for r in results:
            tag = "alert" if r.rows_written else "ok"
            print(f"{r.name:50s} {tag:6s} err={r.error or ''}")
    else:
        method = getattr(sch, f"run_{args.cadence}")
        results = method(force=args.force)
        for r in results:
            print(f"{r.name:50s} rows={r.rows_written:5d}  err={r.error or ''}")