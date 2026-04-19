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
import json
from datetime import datetime, timedelta
from typing import Optional

from data.sec_fetcher import SECFetcher
from data.market_data import MarketData
from data.macro_data import MacroData
from data.portfolio_db import PortfolioDB

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
    """Assemble a structured context dict for one ticker.

    This is the **dict-returning** sibling of
    :meth:`DataPipeline.build_agent_context_text`. Consumers that want a
    pre-rendered LLM prompt should use the text method; everything else
    (``main.py``, ``risk_scanner``, ``thesis_generator``, the Streamlit UI)
    wants the structured dict.

    Returns
    -------
    dict
        Shape::

            {
              "ticker":    str,
              "price":     float,
              "prices":    {"last": float, "history": list, ...},
              "metrics":   dict,       # raw fundamentals from market_data
              "filings":   list[dict], # recent SEC filings
              "macro":     dict,       # macro dashboard snapshot
              "portfolio": dict,       # position info if held, else {}
              "as_of":     str,        # ISO date
            }

        Any individual section that fails to fetch degrades to an empty
        value of the appropriate type rather than raising — one bad
        upstream call must not kill the whole ticker.
    """
    pipe = _build_pipeline(db_path, cfg)

    ctx: dict = {
        "ticker": ticker,
        "as_of": datetime.now().date().isoformat(),
        "price": 0.0,
        "prices": {},
        "metrics": {},
        "filings": [],
        "macro": {},
        "portfolio": {},
    }

    # --- Prices -------------------------------------------------------------
    try:
        latest = pipe.market.get_latest_price(ticker)
        price = float(latest.get("price") or 0.0)
        ctx["price"] = price
        ctx["prices"] = {
            "last": price,
            "change_pct": latest.get("change_pct"),
            "market_cap": latest.get("market_cap"),
            "fifty_two_week_high": latest.get("fifty_two_week_high"),
            "fifty_two_week_low": latest.get("fifty_two_week_low"),
        }
    except Exception as e:
        logger.warning("build_agent_context | %s | price fetch failed: %s", ticker, e)

    # --- Fundamentals (the raw dict the analyzer will normalize) ------------
    try:
        fund = pipe.market.get_fundamentals(ticker)
        # Flatten the nested structure into a single-level dict that
        # filing_analyzer.extract_key_metrics_for_agent can consume.
        flat: dict = {}
        for section in ("financials", "valuation", "growth", "analyst"):
            sub = fund.get(section) if isinstance(fund, dict) else None
            if isinstance(sub, dict):
                flat.update(sub)
        # Keep a couple of top-level descriptors for the prompt builder.
        if isinstance(fund, dict):
            flat.setdefault("sector", fund.get("sector"))
            flat.setdefault("industry", fund.get("industry"))
        ctx["metrics"] = flat
    except Exception as e:
        logger.warning(
            "build_agent_context | %s | fundamentals fetch failed: %s", ticker, e
        )

    # --- Filings ------------------------------------------------------------
    try:
        filings_10k = pipe.sec.get_filings(ticker, "10-K", count=1) or []
        filings_10q = pipe.sec.get_filings(ticker, "10-Q", count=2) or []
        filings_8k = pipe.sec.get_recent_8k_events(ticker, days_back=60) or []
        ctx["filings"] = [*filings_10k, *filings_10q, *filings_8k]
    except Exception as e:
        logger.warning("build_agent_context | %s | filings fetch failed: %s", ticker, e)

    # --- Macro --------------------------------------------------------------
    try:
        ctx["macro"] = pipe.macro.get_macro_dashboard() or {}
    except Exception as e:
        logger.warning("build_agent_context | %s | macro fetch failed: %s", ticker, e)

    # --- Portfolio position (if any) ---------------------------------------
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
        logger.debug("build_agent_context | %s | portfolio lookup skipped: %s", ticker, e)

    return ctx


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