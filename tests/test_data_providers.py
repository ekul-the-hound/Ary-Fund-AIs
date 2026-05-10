"""
test_data_providers.py
======================
Comprehensive unit tests for ``providers.py`` and ``earnings_calendar.py``.

All external I/O (HTTP + WebSocket) is mocked via ``unittest.mock``.
WebSocket tests use ``pytest-asyncio``.

Run:

    pytest -v test_data_providers.py
    pytest -v test_data_providers.py -k tiingo
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
import requests

from ui import earnings_calendar
from data import providers
from data.providers import (
    FMPProvider,
    FinnhubProvider,
    TiingoProvider,
    get_analyst_data,
    get_earnings_events,
    get_fundamentals,
    get_ownership_data,
    get_prices,
    get_transcripts,
    realtime_price_stream,
    reset_providers,
)


# ============================================================================
# Helpers
# ============================================================================

def _mock_response(
    status: int = 200,
    json_data: Any = None,
    text: str = "",
) -> MagicMock:
    """Build a MagicMock that quacks like a requests.Response."""
    m = MagicMock()
    m.status_code = status
    m.text = text or json.dumps(json_data) if json_data is not None else ""
    if json_data is not None:
        m.json.return_value = json_data
    else:
        m.json.side_effect = ValueError("No JSON")
    return m


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Each test gets a fresh provider singleton state + refresh log."""
    reset_providers()
    earnings_calendar._last_refresh_ts.clear()
    yield
    reset_providers()
    earnings_calendar._last_refresh_ts.clear()


# ============================================================================
# 1. Provider initialization
# ============================================================================

class TestProviderInit:
    def test_tiingo_requires_key(self):
        with pytest.raises(ValueError, match="TIINGO_API_KEY"):
            TiingoProvider(api_key="")

    def test_fmp_requires_key(self):
        with pytest.raises(ValueError, match="FMP_API_KEY"):
            FMPProvider(api_key="")

    def test_finnhub_requires_key(self):
        with pytest.raises(ValueError, match="FINNHUB_API_KEY"):
            FinnhubProvider(api_key="")

    def test_tiingo_initializes(self):
        p = TiingoProvider(api_key="abc")
        assert p.api_key == "abc"
        assert "Token abc" in p._headers["Authorization"]

    def test_fmp_initializes(self):
        p = FMPProvider(api_key="abc")
        assert p.api_key == "abc"

    def test_finnhub_initializes(self):
        p = FinnhubProvider(api_key="abc")
        assert p.api_key == "abc"


# ============================================================================
# 2. TiingoProvider
# ============================================================================

class TestTiingo:
    PRICES_FIXTURE = [
        {"date": "2026-04-23T00:00:00.000Z", "open": 170.0, "high": 172.0,
         "low": 169.0, "close": 171.5, "adjClose": 171.5, "volume": 50_000_000},
        {"date": "2026-04-22T00:00:00.000Z", "open": 168.0, "high": 170.5,
         "low": 167.0, "close": 170.0, "adjClose": 170.0, "volume": 48_000_000},
        {"date": "2026-04-24T00:00:00.000Z", "open": 171.5, "high": 173.0,
         "low": 170.5, "close": 172.7, "adjClose": 172.7, "volume": 52_000_000},
    ]

    FUNDAMENTALS_FIXTURE = [{
        "date": "2026-04-24",
        "marketCap": 2_700_000_000_000,
        "peRatio": 28.4,
        "pbRatio": 38.1,
        "trailingPEG1Y": 2.1,
        "eps": 6.05,
        "divYield": 0.005,
        "beta": 1.25,
    }]

    def test_get_prices_returns_correct_columns(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, self.PRICES_FIXTURE)
            p = TiingoProvider("k")
            df = p.get_historical_prices("AAPL", "2026-04-22", "2026-04-24")

        assert list(df.columns) == [
            "date", "open", "high", "low", "close", "adjusted_close", "volume",
        ]
        assert len(df) == 3
        assert df["adjusted_close"].iloc[0] == 170.0  # earliest first

    def test_get_prices_sorts_ascending(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, self.PRICES_FIXTURE)
            p = TiingoProvider("k")
            df = p.get_historical_prices("AAPL", "2026-04-22", "2026-04-24")

        dates = df["date"].tolist()
        assert dates == sorted(dates)

    def test_get_prices_dates_are_datetime(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, self.PRICES_FIXTURE)
            p = TiingoProvider("k")
            df = p.get_historical_prices("AAPL", "2026-04-22", "2026-04-24")

        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_get_prices_empty_response(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, [])
            p = TiingoProvider("k")
            df = p.get_historical_prices("AAPL", "2026-04-22", "2026-04-24")

        assert df.empty
        assert list(df.columns) == [
            "date", "open", "high", "low", "close", "adjusted_close", "volume",
        ]

    def test_get_prices_handles_http_error(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(500, text="server error")
            p = TiingoProvider("k")
            df = p.get_historical_prices("AAPL", "2026-04-22", "2026-04-24")

        assert df.empty

    def test_get_prices_handles_network_error(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("dns fail")
            p = TiingoProvider("k")
            df = p.get_historical_prices("AAPL", "2026-04-22", "2026-04-24")

        assert df.empty

    def test_get_fundamentals_returns_dict(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, self.FUNDAMENTALS_FIXTURE)
            p = TiingoProvider("k")
            f = p.get_fundamentals("AAPL")

        assert isinstance(f, dict)
        assert f["marketCap"] == 2_700_000_000_000
        assert f["peRatio"] == 28.4
        assert f["eps"] == 6.05
        assert f["beta"] == 1.25

    def test_get_fundamentals_empty_on_failure(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(403, text="paid only")
            p = TiingoProvider("k")
            f = p.get_fundamentals("AAPL")

        assert f == {}


# ============================================================================
# 3. FMPProvider
# ============================================================================

class TestFMP:
    TRANSCRIPT_FIXTURE = [{
        "symbol": "AAPL", "quarter": 3, "year": 2026,
        "date": "2026-07-25",
        "content": "Operator: Good day. Tim Cook: Hello and welcome...",
    }]

    ESTIMATES_FIXTURE = [
        {
            "date": "2026-12-31", "symbol": "AAPL",
            "estimatedEpsAvg": 7.30, "estimatedEpsHigh": 7.50,
            "estimatedEpsLow": 7.05, "estimatedRevenueAvg": 410_000_000_000,
            "numberAnalystEstimatedRevenue": 28,
            "numberAnalystsEstimatedEps": 2,
        },
    ]

    RATIOS_FIXTURE = [{
        "date": "2026-09-30", "period": "Q3",
        "priceEarningsRatio": 28.4, "priceToBookRatio": 38.1,
        "debtEquityRatio": 1.5, "returnOnEquity": 1.45,
        "returnOnAssets": 0.27, "currentRatio": 1.05, "quickRatio": 0.95,
        "grossProfitMargin": 0.46, "operatingProfitMargin": 0.30,
        "netProfitMargin": 0.25,
    }]

    DCF_FIXTURE = [{"symbol": "AAPL", "date": "2026-04-24",
                    "dcf": 188.5, "Stock Price": 175.0}]

    def test_get_transcripts_no_date(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, self.TRANSCRIPT_FIXTURE)
            p = FMPProvider("k")
            ts = p.get_transcripts("AAPL")

        assert isinstance(ts, list)
        assert len(ts) == 1
        assert set(ts[0].keys()) == {"date", "title", "content"}
        assert ts[0]["title"] == "Q3 2026"
        assert "Tim Cook" in ts[0]["content"]

    def test_get_transcripts_with_date_passes_param(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, self.TRANSCRIPT_FIXTURE)
            p = FMPProvider("k")
            p.get_transcripts("AAPL", date="2026-07-25")

        # Verify the date was sent as a query param
        kwargs = mock_get.call_args.kwargs
        assert kwargs["params"]["date"] == "2026-07-25"

    def test_get_transcripts_empty_on_404(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(404, text="not found")
            p = FMPProvider("k")
            assert p.get_transcripts("ZZZZ") == []

    def test_get_analyst_estimates_columns(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, self.ESTIMATES_FIXTURE)
            p = FMPProvider("k")
            df = p.get_analyst_estimates("AAPL")

        expected_cols = {"period", "consensus", "high", "low",
                         "mean_estimate", "revisions_up", "revisions_down"}
        assert expected_cols.issubset(df.columns)
        assert df.iloc[0]["consensus"] == 7.30
        assert df.iloc[0]["high"] == 7.50

    def test_get_ratios_renames_to_canonical(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, self.RATIOS_FIXTURE)
            p = FMPProvider("k")
            df = p.get_ratios("AAPL")

        # Canonical short names must be present
        for col in ["PE", "PB", "DE", "ROE", "ROA",
                    "currentRatio", "quickRatio",
                    "gross_margin", "operating_margin", "net_margin"]:
            assert col in df.columns
        assert df.iloc[0]["PE"] == 28.4
        assert df.iloc[0]["ROE"] == 1.45

    def test_get_ratios_empty_response(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, [])
            p = FMPProvider("k")
            df = p.get_ratios("AAPL")
        assert df.empty

    def test_get_dcf_returns_float(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, self.DCF_FIXTURE)
            p = FMPProvider("k")
            dcf = p.get_dcf("AAPL")

        assert isinstance(dcf, float)
        assert dcf == 188.5

    def test_get_dcf_returns_none_on_missing(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, [])
            p = FMPProvider("k")
            assert p.get_dcf("AAPL") is None

    def test_get_dcf_returns_none_on_http_error(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(500)
            p = FMPProvider("k")
            assert p.get_dcf("AAPL") is None


# ============================================================================
# 4. FinnhubProvider — HTTP endpoints
# ============================================================================

class TestFinnhubHTTP:
    EARNINGS_FIXTURE = {
        "earningsCalendar": [
            {"symbol": "AAPL", "date": "2026-07-25",
             "epsEstimate": 1.65, "epsActual": 1.72,
             "period": "Q3", "year": 2026, "quarter": 3},
            {"symbol": "MSFT", "date": "2026-07-26",
             "epsEstimate": 3.10, "epsActual": None,
             "period": "Q4", "year": 2026, "quarter": 4},
        ],
    }

    OWNERSHIP_FIXTURE = {
        "data": [{
            "reportDate": "2026-03-31",
            "investors": [
                {"name": "Vanguard", "share": 1_300_000_000,
                 "percentage": 8.4, "value": 220_000_000_000},
                {"name": "BlackRock", "share": 1_050_000_000,
                 "percentage": 6.7, "value": 178_000_000_000},
            ],
        }],
    }

    def test_get_earnings_calendar_shape(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, self.EARNINGS_FIXTURE)
            p = FinnhubProvider("k")
            events = p.get_earnings_calendar("2026-07-20", "2026-07-30")

        assert len(events) == 2
        assert events[0]["ticker"] == "AAPL"
        assert events[0]["epsActual"] == 1.72
        assert events[1]["epsActual"] is None
        for e in events:
            assert {"ticker", "date", "epsEstimate",
                    "epsActual", "period"}.issubset(e.keys())

    def test_get_earnings_calendar_empty(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(
                200, {"earningsCalendar": []})
            p = FinnhubProvider("k")
            assert p.get_earnings_calendar("2026-07-20", "2026-07-21") == []

    def test_get_earnings_calendar_http_error(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(401, text="bad token")
            p = FinnhubProvider("k")
            assert p.get_earnings_calendar("2026-07-20", "2026-07-21") == []

    def test_get_ownership_columns(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, self.OWNERSHIP_FIXTURE)
            p = FinnhubProvider("k")
            df = p.get_ownership("AAPL")

        assert list(df.columns) == [
            "holderName", "shares", "dateReported", "percentHeld", "value",
        ]
        assert len(df) == 2
        assert df.iloc[0]["holderName"] == "Vanguard"
        assert df.iloc[0]["dateReported"] == "2026-03-31"

    def test_get_ownership_empty(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, {"data": []})
            p = FinnhubProvider("k")
            df = p.get_ownership("AAPL")
        assert df.empty
        assert list(df.columns) == [
            "holderName", "shares", "dateReported", "percentHeld", "value",
        ]


# ============================================================================
# 5. FinnhubProvider — WebSocket stream
# ============================================================================

class _MockWS:
    """Async context manager + iterator that simulates websockets.connect."""

    def __init__(self, messages: list[str], raise_after: bool = False):
        self.messages = list(messages)
        self.sent: list[str] = []
        self.raise_after = raise_after

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def send(self, payload: str):
        self.sent.append(payload)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.messages:
            return self.messages.pop(0)
        if self.raise_after:
            # Simulate a disconnect to test reconnect path
            raise ConnectionError("mock disconnect")
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_realtime_stream_yields_trades():
    """The stream should parse Finnhub trade messages and yield dicts."""
    trade_msg = json.dumps({
        "type": "trade",
        "data": [
            {"s": "AAPL", "p": 175.5, "v": 100, "x": "NASDAQ", "t": 1729900000000},
            {"s": "AAPL", "p": 175.6, "v": 50,  "x": "NASDAQ", "t": 1729900001000},
        ],
    })
    # Non-trade message that should be skipped
    ping_msg = json.dumps({"type": "ping"})

    mock_ws = _MockWS([ping_msg, trade_msg], raise_after=True)

    with patch.dict("sys.modules", {"websockets": MagicMock(connect=lambda *a, **kw: mock_ws)}):
        p = FinnhubProvider("k")
        ticks = []
        # Bound the loop: we expect 2 ticks then ConnectionError → reconnect
        # → which would loop forever, so we cancel after collection.
        async def collect():
            async for t in p.create_realtime_price_stream(["AAPL"]):
                ticks.append(t)
                if len(ticks) >= 2:
                    break

        await asyncio.wait_for(collect(), timeout=2.0)

    assert len(ticks) == 2
    assert ticks[0] == {
        "ticker": "AAPL", "price": 175.5, "size": 100,
        "exchange": "NASDAQ", "timestamp": 1729900000000,
    }
    assert ticks[1]["price"] == 175.6
    # Subscribe message was sent
    assert any("subscribe" in s and "AAPL" in s for s in mock_ws.sent)


@pytest.mark.asyncio
async def test_realtime_stream_reconnects_with_backoff():
    """Verify the stream re-enters the connect loop after a disconnect."""
    trade1 = json.dumps({"type": "trade",
                          "data": [{"s": "AAPL", "p": 100.0, "v": 1,
                                    "x": "X", "t": 1}]})
    trade2 = json.dumps({"type": "trade",
                          "data": [{"s": "AAPL", "p": 101.0, "v": 1,
                                    "x": "X", "t": 2}]})

    # First connection yields trade1 then disconnects.
    # Second connection yields trade2.
    ws_session_1 = _MockWS([trade1], raise_after=True)
    ws_session_2 = _MockWS([trade2], raise_after=True)
    sessions = [ws_session_1, ws_session_2]

    def fake_connect(*a, **kw):
        return sessions.pop(0)

    fake_websockets = MagicMock(connect=fake_connect)

    # Patch asyncio.sleep to avoid actually waiting on the backoff.
    with patch.dict("sys.modules", {"websockets": fake_websockets}), \
         patch("data.providers.asyncio.sleep", new=AsyncMock(return_value=None)):
        p = FinnhubProvider("k")
        ticks = []
        async def collect():
            async for t in p.create_realtime_price_stream(["AAPL"]):
                ticks.append(t)
                if len(ticks) >= 2:
                    break
        await asyncio.wait_for(collect(), timeout=2.0)

    assert [t["price"] for t in ticks] == [100.0, 101.0]


# ============================================================================
# 6. Convenience functions
# ============================================================================

class TestConvenience:
    def test_get_prices_uses_tiingo(self, monkeypatch):
        monkeypatch.setenv("TIINGO_API_KEY", "fake")
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, TestTiingo.PRICES_FIXTURE)
            df = get_prices("AAPL", "2026-04-22", "2026-04-24")
        assert isinstance(df, pd.DataFrame)
        assert "adjusted_close" in df.columns
        # Verify the Tiingo URL was called
        assert "tiingo" in mock_get.call_args.args[0]

    def test_get_fundamentals_uses_tiingo(self, monkeypatch):
        monkeypatch.setenv("TIINGO_API_KEY", "fake")
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(
                200, TestTiingo.FUNDAMENTALS_FIXTURE)
            f = get_fundamentals("AAPL")
        assert isinstance(f, dict)
        assert f["peRatio"] == 28.4

    def test_get_transcripts_uses_fmp(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "fake")
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(200, TestFMP.TRANSCRIPT_FIXTURE)
            ts = get_transcripts("AAPL")
        assert len(ts) == 1
        assert "financialmodelingprep" in mock_get.call_args.args[0]

    def test_get_analyst_data_returns_bundle(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "fake")
        # 3 sequential calls: estimates, ratios, dcf
        responses = [
            _mock_response(200, TestFMP.ESTIMATES_FIXTURE),
            _mock_response(200, TestFMP.RATIOS_FIXTURE),
            _mock_response(200, TestFMP.DCF_FIXTURE),
        ]
        with patch("data.providers.requests.get", side_effect=responses):
            data = get_analyst_data("AAPL")

        assert set(data.keys()) == {"estimates", "ratios", "dcf"}
        assert isinstance(data["estimates"], pd.DataFrame)
        assert isinstance(data["ratios"], pd.DataFrame)
        assert data["dcf"] == 188.5

    def test_get_earnings_events_uses_finnhub(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "fake")
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(
                200, TestFinnhubHTTP.EARNINGS_FIXTURE)
            events = get_earnings_events("2026-07-20", "2026-07-30")
        assert len(events) == 2
        assert "finnhub" in mock_get.call_args.args[0]

    def test_get_ownership_data_uses_finnhub(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "fake")
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(
                200, TestFinnhubHTTP.OWNERSHIP_FIXTURE)
            df = get_ownership_data("AAPL")
        assert len(df) == 2
        assert df.iloc[0]["holderName"] == "Vanguard"

    @pytest.mark.asyncio
    async def test_realtime_price_stream_wraps_finnhub(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "fake")
        msg = json.dumps({"type": "trade",
                           "data": [{"s": "AAPL", "p": 1.0, "v": 1,
                                     "x": "X", "t": 1}]})
        mock_ws = _MockWS([msg], raise_after=True)
        with patch.dict("sys.modules",
                         {"websockets": MagicMock(connect=lambda *a, **kw: mock_ws)}):
            ticks = []
            async def collect():
                async for t in realtime_price_stream(["AAPL"]):
                    ticks.append(t)
                    break
            await asyncio.wait_for(collect(), timeout=2.0)
        assert ticks[0]["price"] == 1.0


# ============================================================================
# 7. Error handling — every provider gracefully handles network/HTTP errors
# ============================================================================

class TestErrorHandling:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 500, 502, 503])
    def test_tiingo_prices_handles_all_http_errors(self, status):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(status, text="error")
            p = TiingoProvider("k")
            df = p.get_historical_prices("AAPL", "2026-01-01", "2026-01-02")
        assert df.empty

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 500])
    def test_fmp_handles_http_errors(self, status):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(status, text="error")
            p = FMPProvider("k")
            assert p.get_transcripts("AAPL") == []
            assert p.get_analyst_estimates("AAPL").empty
            assert p.get_dcf("AAPL") is None

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 500])
    def test_finnhub_handles_http_errors(self, status):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.return_value = _mock_response(status, text="error")
            p = FinnhubProvider("k")
            assert p.get_earnings_calendar("2026-07-20", "2026-07-21") == []
            assert p.get_ownership("AAPL").empty

    def test_handles_invalid_json(self):
        # Response succeeds but body is not JSON
        with patch("data.providers.requests.get") as mock_get:
            bad = MagicMock()
            bad.status_code = 200
            bad.text = "not json"
            bad.json.side_effect = ValueError("bad json")
            mock_get.return_value = bad

            p = TiingoProvider("k")
            assert p.get_historical_prices("AAPL", "x", "y").empty

    def test_handles_timeout(self):
        with patch("data.providers.requests.get") as mock_get:
            mock_get.side_effect = requests.Timeout("slow")
            p = FinnhubProvider("k")
            assert p.get_earnings_calendar("a", "b") == []


# ============================================================================
# 8. earnings_calendar module
# ============================================================================

class TestEarningsCalendar:
    def _earnings(self, ticker: str, days_out: int) -> dict:
        d = (datetime.now(timezone.utc).replace(tzinfo=None).date() + timedelta(days=days_out)).isoformat()
        return {"ticker": ticker, "date": d,
                "epsEstimate": 1.5, "period": "Q3"}

    def test_upcoming_filters_by_ticker(self):
        events = [self._earnings("AAPL", 3), self._earnings("MSFT", 4)]
        with patch("ui.earnings_calendar.providers.get_earnings_events",
                    return_value=events):
            res = earnings_calendar.get_upcoming_earnings_week("AAPL")
        assert len(res) == 1
        assert res[0]["ticker"] == "AAPL"

    def test_upcoming_returns_all_when_no_ticker(self):
        events = [self._earnings("AAPL", 3), self._earnings("MSFT", 4)]
        with patch("ui.earnings_calendar.providers.get_earnings_events",
                    return_value=events):
            res = earnings_calendar.get_upcoming_earnings_week()
        assert len(res) == 2

    def test_upcoming_handles_provider_failure(self):
        with patch("ui.earnings_calendar.providers.get_earnings_events",
                    side_effect=RuntimeError("boom")):
            assert earnings_calendar.get_upcoming_earnings_week("AAPL") == []

    def test_should_trigger_for_upcoming_event(self):
        events = [self._earnings("AAPL", 3)]  # 3 days from now
        with patch("ui.earnings_calendar.providers.get_earnings_events",
                    return_value=events):
            assert earnings_calendar.should_trigger_event_refresh("AAPL") is True

    def test_should_trigger_for_recent_event_not_yet_refreshed(self):
        events = [self._earnings("AAPL", -2)]  # 2 days ago
        with patch("ui.earnings_calendar.providers.get_earnings_events",
                    return_value=events):
            # No refresh recorded yet → should trigger
            assert earnings_calendar.should_trigger_event_refresh("AAPL") is True

    def test_should_not_trigger_after_refresh_post_event(self):
        events = [self._earnings("AAPL", -2)]  # 2 days ago
        # Mark refreshed yesterday (after the event)
        earnings_calendar.mark_refreshed(
            "AAPL", when=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1))
        with patch("ui.earnings_calendar.providers.get_earnings_events",
                    return_value=events):
            assert earnings_calendar.should_trigger_event_refresh("AAPL") is False

    def test_should_not_trigger_with_no_events(self):
        with patch("ui.earnings_calendar.providers.get_earnings_events",
                    return_value=[]):
            assert earnings_calendar.should_trigger_event_refresh("AAPL") is False

    def test_should_not_trigger_for_other_ticker(self):
        events = [self._earnings("MSFT", 3)]
        with patch("ui.earnings_calendar.providers.get_earnings_events",
                    return_value=events):
            assert earnings_calendar.should_trigger_event_refresh("AAPL") is False
