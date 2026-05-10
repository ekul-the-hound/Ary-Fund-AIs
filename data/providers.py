"""
providers.py
============
Production-grade data providers replacing/augmenting the yfinance pipeline.

Three providers, each in its own class:

    TiingoProvider   — Adjusted historical OHLCV + daily fundamentals
                       (replaces yfinance for prices)
    FMPProvider      — Earnings call transcripts, analyst estimates,
                       ratios, DCF (new capabilities)
    FinnhubProvider  — Earnings calendar, real-time WebSocket prices,
                       ownership data (event-driven refresh)

All HTTP failures are caught and logged; methods return empty
DataFrames / lists / None rather than raising. WebSocket reconnects
with exponential backoff.

Environment variables required:

    TIINGO_API_KEY   — https://api.tiingo.com/account/api/token
    FMP_API_KEY      — https://site.financialmodelingprep.com/developer/docs
    FINNHUB_API_KEY  — https://finnhub.io/dashboard

Design choices:

  * Sync HTTP via ``requests`` for simplicity (the heavy stuff is the WS).
  * Async WebSocket via ``websockets`` library, yielding ticks as an
    ``AsyncIterator[dict]``.
  * Top-level convenience functions (``get_prices`` etc.) lazily
    instantiate provider singletons from env vars — drop-in replacement
    for the legacy ``market_data.get_prices`` signature.
  * No SQLite caching here; that lives in the existing data layer
    (market_data.py, macro_data.py). Keeps providers pure I/O.

Why three providers and not one mega-class:

  * Each provider's auth, base URL, error idioms, and rate limits differ.
    A single class would be a pile of conditionals. Three classes share
    only the ``_get_json`` helper at module level.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator, Iterator, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

TIINGO_BASE = "https://api.tiingo.com"
FMP_BASE = "https://financialmodelingprep.com/api"
FINNHUB_BASE = "https://finnhub.io/api/v1"
FINNHUB_WS_URL = "wss://ws.finnhub.io"

DEFAULT_TIMEOUT = 30  # seconds


# ============================================================================
# Shared HTTP helper
# ============================================================================

def _get_json(
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
    provider: str = "",
    ctx: str = "",
) -> Any:
    """GET ``url`` and parse JSON. Returns parsed body, or ``None`` on any
    failure. Logs with provider + context so failures are traceable.

    Never raises. Never returns a partial / malformed response.
    """
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        logger.error("%s | %s | network error: %s", provider, ctx, e)
        return None

    if resp.status_code >= 400:
        logger.error(
            "%s | %s | HTTP %d: %s",
            provider, ctx, resp.status_code, resp.text[:200],
        )
        return None

    try:
        return resp.json()
    except ValueError as e:
        logger.error("%s | %s | JSON decode failed: %s", provider, ctx, e)
        return None


def _empty_price_df() -> pd.DataFrame:
    """Canonical empty DataFrame used when a price fetch fails."""
    return pd.DataFrame(
        columns=["date", "open", "high", "low", "close", "adjusted_close", "volume"]
    )


# ============================================================================
# TiingoProvider
# ============================================================================

class TiingoProvider:
    """Tiingo: clean adjusted prices + daily fundamentals.

    Free tier covers EOD prices for US equities. The fundamentals
    endpoint requires a paid plan; on free tier the call returns an
    empty dict and we log a warning rather than crashing.

    Why Tiingo over yfinance for prices:
      - Officially supported API (yfinance scrapes Yahoo, breaks often)
      - Adjustments for splits/dividends are correct and consistent
      - Stable rate limits documented in writing
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "TIINGO_API_KEY is required. Get a free key at "
                "https://api.tiingo.com/account/api/token"
            )
        self.api_key = api_key
        self._headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }

    # --------------------------------------------------------------
    # Prices
    # --------------------------------------------------------------
    def get_historical_prices(
        self, ticker: str, start: str, end: str
    ) -> pd.DataFrame:
        """Adjusted OHLCV for ``ticker`` between ``start`` and ``end``
        (inclusive, ISO ``YYYY-MM-DD``).

        Returns a DataFrame with columns:
            date, open, high, low, close, adjusted_close, volume

        ``date`` is a ``datetime64[ns]`` column (not the index), sorted
        ascending. Returns an empty DataFrame on any failure.

        The adjusted_close column accounts for splits and dividends.
        Use it for any return / vol / regime calc; raw close is for
        display only.
        """
        ticker = ticker.upper()
        url = f"{TIINGO_BASE}/tiingo/daily/{ticker}/prices"
        params = {"startDate": start, "endDate": end, "format": "json"}

        body = _get_json(
            url, params=params, headers=self._headers,
            provider="tiingo", ctx=f"prices/{ticker}",
        )

        if not body or not isinstance(body, list):
            logger.warning("tiingo | prices/%s | empty response", ticker)
            return _empty_price_df()

        # Tiingo returns: date, open, high, low, close, adjOpen, adjHigh,
        # adjLow, adjClose, volume, adjVolume, divCash, splitFactor
        rows = []
        for r in body:
            try:
                rows.append({
                    "date": r["date"],
                    "open": r.get("open"),
                    "high": r.get("high"),
                    "low": r.get("low"),
                    "close": r.get("close"),
                    "adjusted_close": r.get("adjClose"),
                    "volume": r.get("volume"),
                })
            except (KeyError, TypeError) as e:
                logger.warning("tiingo | prices/%s | malformed row: %s", ticker, e)
                continue

        if not rows:
            return _empty_price_df()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    # --------------------------------------------------------------
    # Fundamentals
    # --------------------------------------------------------------
    def get_fundamentals(self, ticker: str) -> dict:
        """Daily fundamental metrics for ``ticker``.

        Returns a dict with keys (when available):
            marketCap, peRatio, pbRatio, trailingPEG1Y, eps,
            dividendYield, beta

        Free tier note: Tiingo's fundamentals endpoint requires a paid
        addon. On free tier this returns ``{}`` and logs a warning so
        callers can fall back to yfinance / FMP for fundamentals.
        """
        ticker = ticker.upper()
        url = f"{TIINGO_BASE}/tiingo/fundamentals/{ticker}/daily"

        body = _get_json(
            url, headers=self._headers,
            provider="tiingo", ctx=f"fundamentals/{ticker}",
        )

        if not body:
            return {}

        # Endpoint returns a list; take most recent
        latest = body[-1] if isinstance(body, list) and body else body
        if not isinstance(latest, dict):
            return {}

        return {
            "marketCap": latest.get("marketCap"),
            "peRatio": latest.get("peRatio"),
            "pbRatio": latest.get("pbRatio"),
            "trailingPEG1Y": latest.get("trailingPEG1Y"),
            "eps": latest.get("eps") or latest.get("epsActual"),
            "dividendYield": latest.get("divYield"),
            "beta": latest.get("beta"),
            "asOfDate": latest.get("date"),
        }


# ============================================================================
# FMPProvider
# ============================================================================

class FMPProvider:
    """Financial Modeling Prep: transcripts, analyst data, ratios, DCF.

    The earnings-call-transcript endpoint is the killer feature here —
    no other free provider has these in clean text form, and they're
    high-signal input for the thesis_generator (management tone,
    forward guidance, analyst Q&A).
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "FMP_API_KEY is required. Get a free key at "
                "https://site.financialmodelingprep.com/developer/docs"
            )
        self.api_key = api_key

    # --------------------------------------------------------------
    # Transcripts
    # --------------------------------------------------------------
    def get_transcripts(
        self, ticker: str, date: Optional[str] = None
    ) -> list[dict]:
        """Earnings call transcripts for ``ticker``.

        If ``date`` is given (``YYYY-MM-DD``), fetch that specific
        transcript. Otherwise fetch the most recent.

        Returns ``list[dict]`` where each dict has:
            date    — ISO date of the call
            title   — quarter/year label, e.g. "Q3 2026"
            content — full transcript text

        Empty list on failure.
        """
        ticker = ticker.upper()
        url = f"{FMP_BASE}/v3/earning_call_transcript/{ticker}"
        params: dict = {"apikey": self.api_key}
        if date:
            params["date"] = date

        body = _get_json(url, params=params, provider="fmp",
                         ctx=f"transcripts/{ticker}")
        if not body or not isinstance(body, list):
            return []

        out = []
        for r in body:
            try:
                title_parts = []
                q, y = r.get("quarter"), r.get("year")
                if q is not None and y is not None:
                    title_parts.append(f"Q{q} {y}")
                title = " ".join(title_parts) or r.get("symbol", ticker)
                out.append({
                    "date": r.get("date", ""),
                    "title": title,
                    "content": r.get("content", ""),
                })
            except (KeyError, TypeError) as e:
                logger.warning("fmp | transcripts/%s | malformed row: %s",
                               ticker, e)
                continue
        return out

    # --------------------------------------------------------------
    # Analyst estimates
    # --------------------------------------------------------------
    def get_analyst_estimates(self, ticker: str) -> pd.DataFrame:
        """Forward analyst estimates by period.

        Columns:
            period             — fiscal period end date (YYYY-MM-DD)
            consensus          — mean estimated EPS
            high               — high estimated EPS
            low                — low estimated EPS
            mean_estimate      — mean estimated revenue
            revisions_up       — # of upward revisions
            revisions_down     — # of downward revisions

        Empty DataFrame on failure.
        """
        ticker = ticker.upper()
        url = f"{FMP_BASE}/v3/analyst-estimates/{ticker}"
        params = {"apikey": self.api_key}

        body = _get_json(url, params=params, provider="fmp",
                         ctx=f"estimates/{ticker}")

        cols = ["period", "consensus", "high", "low", "mean_estimate",
                "revisions_up", "revisions_down"]
        if not body or not isinstance(body, list):
            return pd.DataFrame(columns=cols)

        rows = []
        for r in body:
            try:
                rows.append({
                    "period": r.get("date"),
                    "consensus": r.get("estimatedEpsAvg"),
                    "high": r.get("estimatedEpsHigh"),
                    "low": r.get("estimatedEpsLow"),
                    "mean_estimate": r.get("estimatedRevenueAvg"),
                    "revisions_up": r.get("numberAnalystEstimatedRevenue"),
                    "revisions_down": r.get("numberAnalystsEstimatedEps"),
                })
            except (KeyError, TypeError):
                continue
        return pd.DataFrame(rows, columns=cols)

    # --------------------------------------------------------------
    # Ratios
    # --------------------------------------------------------------
    def get_ratios(self, ticker: str) -> pd.DataFrame:
        """Per-period financial ratios.

        Columns include: date, peRatio, pbRatio, debtEquityRatio,
        returnOnEquity, returnOnAssets, currentRatio, quickRatio,
        grossProfitMargin, operatingProfitMargin, netProfitMargin.

        Empty DataFrame on failure.
        """
        ticker = ticker.upper()
        url = f"{FMP_BASE}/v3/ratios/{ticker}"
        params = {"apikey": self.api_key}

        body = _get_json(url, params=params, provider="fmp",
                         ctx=f"ratios/{ticker}")
        if not body or not isinstance(body, list):
            return pd.DataFrame()

        df = pd.DataFrame(body)
        # Keep just the canonical columns we care about (if present)
        keep = [
            "date", "period",
            "priceEarningsRatio", "priceToBookRatio",
            "debtEquityRatio", "returnOnEquity", "returnOnAssets",
            "currentRatio", "quickRatio",
            "grossProfitMargin", "operatingProfitMargin", "netProfitMargin",
        ]
        cols = [c for c in keep if c in df.columns]
        if not cols:
            return df  # return raw if FMP changed schema
        df = df[cols].copy()
        # Rename to short canonical names downstream code can rely on
        df = df.rename(columns={
            "priceEarningsRatio": "PE",
            "priceToBookRatio": "PB",
            "debtEquityRatio": "DE",
            "returnOnEquity": "ROE",
            "returnOnAssets": "ROA",
            "grossProfitMargin": "gross_margin",
            "operatingProfitMargin": "operating_margin",
            "netProfitMargin": "net_margin",
        })
        return df

    # --------------------------------------------------------------
    # DCF
    # --------------------------------------------------------------
    def get_dcf(self, ticker: str) -> Optional[float]:
        """Latest DCF value for ``ticker`` from FMP's model.

        Returns the DCF intrinsic value as a float, or ``None`` if
        unavailable. Compare against current price to gauge model's
        implied over/undervaluation.
        """
        ticker = ticker.upper()
        url = f"{FMP_BASE}/v3/discounted-cash-flow/{ticker}"
        params = {"apikey": self.api_key}

        body = _get_json(url, params=params, provider="fmp",
                         ctx=f"dcf/{ticker}")
        if not body:
            return None

        # Endpoint returns either dict or list[dict]
        rec = body[0] if isinstance(body, list) and body else body
        if not isinstance(rec, dict):
            return None

        val = rec.get("dcf")
        try:
            return float(val) if val is not None else None
        except (TypeError, ValueError):
            return None


# ============================================================================
# FinnhubProvider
# ============================================================================

class FinnhubProvider:
    """Finnhub: earnings calendar, real-time WebSocket, ownership.

    Earnings calendar is the trigger for event-driven refreshes — see
    ``earnings_calendar.should_trigger_event_refresh``.

    The real-time WS is a fire-and-forget async generator; a caller
    typically wraps it in an asyncio task and pushes ticks to a queue.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "FINNHUB_API_KEY is required. Get a free key at "
                "https://finnhub.io/dashboard"
            )
        self.api_key = api_key

    # --------------------------------------------------------------
    # Earnings calendar
    # --------------------------------------------------------------
    def get_earnings_calendar(
        self, from_date: str, to_date: str
    ) -> list[dict]:
        """Earnings events between ``from_date`` and ``to_date``
        (inclusive, ISO ``YYYY-MM-DD``).

        Returns a list of dicts:
            ticker        — symbol
            date          — earnings date
            epsEstimate   — analyst consensus EPS estimate
            epsActual     — actual EPS (if reported)
            period        — period end date

        Empty list on failure.
        """
        url = f"{FINNHUB_BASE}/calendar/earnings"
        params = {"from": from_date, "to": to_date, "token": self.api_key}

        body = _get_json(url, params=params, provider="finnhub",
                         ctx=f"earnings_calendar/{from_date}_{to_date}")
        if not body:
            return []

        events = body.get("earningsCalendar") if isinstance(body, dict) else None
        if not events:
            return []

        out = []
        for e in events:
            try:
                out.append({
                    "ticker": e.get("symbol"),
                    "date": e.get("date"),
                    "epsEstimate": e.get("epsEstimate"),
                    "epsActual": e.get("epsActual"),
                    "period": e.get("period"),
                })
            except (KeyError, TypeError):
                continue
        return out

    # --------------------------------------------------------------
    # Ownership
    # --------------------------------------------------------------
    def get_ownership(self, ticker: str) -> pd.DataFrame:
        """Institutional holders of ``ticker``.

        Columns:
            holderName, shares, dateReported, percentHeld, value

        Useful for spotting concentrated ownership / activist positions
        and for cross-referencing 13F filings.

        Empty DataFrame on failure.
        """
        ticker = ticker.upper()
        url = f"{FINNHUB_BASE}/stock/institutional-ownership"
        params = {"symbol": ticker, "token": self.api_key, "limit": 100}

        body = _get_json(url, params=params, provider="finnhub",
                         ctx=f"ownership/{ticker}")

        cols = ["holderName", "shares", "dateReported", "percentHeld", "value"]
        if not body or not isinstance(body, dict):
            return pd.DataFrame(columns=cols)

        records = body.get("data") or []
        if not records:
            return pd.DataFrame(columns=cols)

        # Finnhub returns nested: [{ "reportDate": ..., "investors": [...] }, ...]
        rows = []
        for snapshot in records:
            report_date = snapshot.get("reportDate") or snapshot.get("date")
            investors = snapshot.get("investors") or []
            # Some shapes flatten directly — handle both
            if not investors and isinstance(snapshot, dict) and "name" in snapshot:
                investors = [snapshot]
                report_date = report_date or snapshot.get("dateReported")

            for inv in investors:
                rows.append({
                    "holderName": inv.get("name") or inv.get("holderName"),
                    "shares": inv.get("share") or inv.get("shares"),
                    "dateReported": report_date,
                    "percentHeld": inv.get("percentage") or inv.get("percentHeld"),
                    "value": inv.get("value"),
                })
        return pd.DataFrame(rows, columns=cols)

    # --------------------------------------------------------------
    # Real-time WebSocket price stream
    # --------------------------------------------------------------
    async def create_realtime_price_stream(
        self,
        tickers: list[str],
        max_backoff: float = 60.0,
    ) -> AsyncIterator[dict]:
        """Async generator yielding real-time trades for ``tickers``.

        Each yielded dict:
            {"ticker": str, "price": float, "size": float,
             "exchange": str, "timestamp": int}

        Reconnects on failure with exponential backoff (1s → 2s → 4s …
        capped at ``max_backoff``). Cancellation propagates cleanly via
        ``asyncio.CancelledError``.

        Usage:

            async for tick in finnhub.create_realtime_price_stream(["AAPL"]):
                print(tick)
        """
        # Local import so the module can be imported without `websockets`
        # installed (only required if you actually use the WS feature).
        try:
            import websockets
        except ImportError as e:
            raise ImportError(
                "websockets package required for real-time stream: "
                "pip install websockets"
            ) from e

        ws_url = f"{FINNHUB_WS_URL}?token={self.api_key}"
        backoff = 1.0
        symbols = [t.upper() for t in tickers]

        while True:
            try:
                async with websockets.connect(ws_url) as ws:
                    # Subscribe
                    for sym in symbols:
                        await ws.send(json.dumps({
                            "type": "subscribe", "symbol": sym,
                        }))
                    logger.info("finnhub WS | subscribed to %s", symbols)
                    backoff = 1.0  # reset on successful connect

                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except (TypeError, ValueError):
                            continue

                        if msg.get("type") != "trade":
                            continue

                        for trade in msg.get("data", []) or []:
                            yield {
                                "ticker": trade.get("s"),
                                "price": trade.get("p"),
                                "size": trade.get("v"),
                                "exchange": trade.get("x"),
                                "timestamp": trade.get("t"),
                            }
            except asyncio.CancelledError:
                logger.info("finnhub WS | cancelled")
                raise
            except Exception as e:  # noqa: BLE001 — broad on purpose at WS boundary
                logger.warning(
                    "finnhub WS | disconnected (%s); reconnecting in %.1fs",
                    e, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, max_backoff)


# ============================================================================
# Module-level convenience: lazy singleton instances + thin wrappers
# ============================================================================
#
# These functions are the drop-in replacements for legacy yfinance-based
# helpers. They lazily build the provider on first call, reading API keys
# from env vars. Tests can override providers by injecting attribute
# overrides (see test_data_providers.py).

_tiingo: Optional[TiingoProvider] = None
_fmp: Optional[FMPProvider] = None
_finnhub: Optional[FinnhubProvider] = None


def _get_tiingo() -> TiingoProvider:
    global _tiingo
    if _tiingo is None:
        _tiingo = TiingoProvider(api_key=os.getenv("TIINGO_API_KEY", ""))
    return _tiingo


def _get_fmp() -> FMPProvider:
    global _fmp
    if _fmp is None:
        _fmp = FMPProvider(api_key=os.getenv("FMP_API_KEY", ""))
    return _fmp


def _get_finnhub() -> FinnhubProvider:
    global _finnhub
    if _finnhub is None:
        _finnhub = FinnhubProvider(api_key=os.getenv("FINNHUB_API_KEY", ""))
    return _finnhub


def reset_providers() -> None:
    """Clear cached singletons. Tests use this to inject fresh mocks."""
    global _tiingo, _fmp, _finnhub
    _tiingo = None
    _fmp = None
    _finnhub = None


# --- prices / fundamentals (Tiingo) -----------------------------------------

def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Adjusted historical OHLCV. Drop-in replacement for yfinance fetch.

    Same signature as the legacy ``market_data.get_prices`` so callers
    can be migrated by import-swap alone.
    """
    return _get_tiingo().get_historical_prices(ticker, start, end)


def get_fundamentals(ticker: str) -> dict:
    """Daily fundamental snapshot (marketCap, peRatio, eps, etc.)."""
    return _get_tiingo().get_fundamentals(ticker)


# --- transcripts / analyst (FMP) --------------------------------------------

def get_transcripts(
    ticker: str, date: Optional[str] = None
) -> list[dict]:
    """Earnings call transcripts as ``[{date, title, content}]``."""
    return _get_fmp().get_transcripts(ticker, date)


def get_analyst_data(ticker: str) -> dict:
    """Bundle of FMP analyst products.

    Returns ``{estimates: DataFrame, ratios: DataFrame, dcf: float|None}``.
    Each component fails independently — a missing key returns its empty
    sentinel, the others still populate.
    """
    fmp = _get_fmp()
    return {
        "estimates": fmp.get_analyst_estimates(ticker),
        "ratios": fmp.get_ratios(ticker),
        "dcf": fmp.get_dcf(ticker),
    }


# --- earnings / ownership / WS (Finnhub) ------------------------------------

def get_earnings_events(from_date: str, to_date: str) -> list[dict]:
    """Earnings calendar between two ISO dates."""
    return _get_finnhub().get_earnings_calendar(from_date, to_date)


def get_ownership_data(ticker: str) -> pd.DataFrame:
    """Institutional holders DataFrame."""
    return _get_finnhub().get_ownership(ticker)


async def realtime_price_stream(
    tickers: list[str],
) -> AsyncIterator[dict]:
    """Real-time price ticks via Finnhub WebSocket.

    Wraps the async generator on the singleton. Reconnects automatically.
    """
    async for tick in _get_finnhub().create_realtime_price_stream(tickers):
        yield tick
