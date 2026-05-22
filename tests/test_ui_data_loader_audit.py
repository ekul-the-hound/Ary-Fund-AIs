"""
tests/test_ui_data_loader_audit.py
==================================
Contract audit for the five UI data loaders that live in ``ui/app.py``.

Why these tests
---------------
The Streamlit dashboard's loaders are the *interface* between the
backend (``data.pipeline``, ``data.portfolio_db``, ``data.market_data``,
``data.macro_data``) and the rendering functions. If a backend ever
renames or re-nests one of its output fields, the loaders silently
return ``{}`` or ``None`` and the UI shows blanks instead of crashing —
which means the regression is invisible until someone notices missing
data on the dashboard. These tests pin the contract on **both sides**:

* The mock backend returns the shape the *real* backend documents.
* The loader is invoked, its output is asserted to contain the keys
  + types the *consumer rendering function* reads.

The task brief calls out ``MacroData.get_macro_dashboard()`` as the
primary audit target. That contract gets the deepest coverage
(``TestMacroSnapshotContract`` walks every top-level block).

No UI code is modified by these tests. They lock the existing
behaviour so any later refactor of the backend (or the loader) breaks
loudly, in CI, at the right line.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


# ----------------------------------------------------------------------
# Streamlit shim — the loaders are decorated with @st.cache_data and
# import streamlit at module load. We can't bring in real Streamlit in
# a unit-test environment (it requires a runtime), so we stub the bits
# the loader module touches at import time.
# ----------------------------------------------------------------------
def _install_streamlit_stub() -> None:
    """Install a minimal ``streamlit`` stub in sys.modules.

    The loaders use: ``st.cache_data``, ``st.warning``, ``st.info``,
    ``st.error``, plus a handful of layout functions in the render
    helpers that we don't exercise. Module-level calls in ``app.py``
    also include ``st.set_page_config``. Rather than enumerate every
    Streamlit symbol the file might touch, we use a catch-all
    ``__getattr__`` on the module that returns a no-op callable for
    anything we haven't explicitly defined — keeping the test rig
    robust to future additions in ``app.py``.
    """
    if "streamlit" in sys.modules:
        return  # already stubbed by something earlier in the test run

    # Build the stub as a class so __getattr__ works at module-attribute
    # access level (Python 3.7+ supports module-level __getattr__).
    stub = types.ModuleType("streamlit")

    def cache_data(*_args, **_kwargs):
        """No-op replacement for @st.cache_data when used as a
        parameterised decorator (the form ``app.py`` uses)."""
        def _decorator(fn):
            # Streamlit attaches a `.clear()` method to the wrapped fn.
            # Two of the loaders call it. Provide a no-op so they
            # don't crash.
            fn.clear = lambda: None
            return fn
        return _decorator

    # Explicit attrs we want to ensure exist with sensible defaults.
    stub.cache_data = cache_data
    stub.warning = lambda *_a, **_k: None
    stub.info = lambda *_a, **_k: None
    stub.error = lambda *_a, **_k: None
    stub.markdown = lambda *_a, **_k: None
    stub.subheader = lambda *_a, **_k: None
    stub.metric = lambda *_a, **_k: None
    stub.columns = lambda *_a, **_k: [stub] * 10
    stub.button = lambda *_a, **_k: False
    stub.caption = lambda *_a, **_k: None
    stub.spinner = lambda *_a, **_k: _NullCtx()
    stub.rerun = lambda: None
    stub.set_page_config = lambda *_a, **_k: None
    stub.sidebar = stub  # `with st.sidebar:` and `st.sidebar.something()`
    stub.success = lambda *_a, **_k: None
    stub.write = lambda *_a, **_k: None

    # Catch-all for anything we forgot — returns a no-op callable.
    def _module_getattr(name: str):
        def _noop(*_a, **_k):
            return None
        return _noop
    stub.__getattr__ = _module_getattr  # type: ignore[attr-defined]

    sys.modules["streamlit"] = stub


class _NullCtx:
    """Context-manager stand-in for st.spinner()."""
    def __enter__(self): return self
    def __exit__(self, *exc): return False


_install_streamlit_stub()


def _install_plotly_stub() -> None:
    """Stub out plotly.graph_objects and plotly.subplots.

    ``ui/app.py`` imports both at module load time for the chart
    rendering helpers. The audit tests don't exercise charts, so a
    minimal namespace stub is enough to clear the import and keep the
    loaders themselves testable.
    """
    if "plotly" in sys.modules:
        return
    plotly = types.ModuleType("plotly")
    go = types.ModuleType("plotly.graph_objects")
    subplots = types.ModuleType("plotly.subplots")

    class _Anything:
        """Generic stand-in for plotly figure/trace/layout objects.

        Returns itself for any attribute access or call, so chained
        calls like ``fig.update_layout(...).add_trace(...)`` work
        without us enumerating every plotly API surface.
        """
        def __init__(self, *_a, **_k): pass
        def __getattr__(self, _name): return _Anything()
        def __call__(self, *_a, **_k): return _Anything()

    def _factory(*_a, **_k):
        return _Anything()

    go.Figure = _factory
    go.Scatter = _factory
    go.Candlestick = _factory
    go.Bar = _factory
    subplots.make_subplots = _factory

    sys.modules["plotly"] = plotly
    sys.modules["plotly.graph_objects"] = go
    sys.modules["plotly.subplots"] = subplots


_install_plotly_stub()


# ----------------------------------------------------------------------
# Import the loaders module under test. We import once at module level
# so the @st.cache_data decoration runs against the stub.
# ----------------------------------------------------------------------
# We can't import ``ui.app`` directly because the project root used in
# these tests is wherever we copied the file. The conftest harness
# expects ``ui/app.py`` to be importable from CWD.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Importing app.py runs its module-level Streamlit setup (some
# ``st.set_page_config`` etc). The stub makes that a no-op. We then
# extract the loader functions for direct call.
ui_app = importlib.import_module("ui.app")


# ----------------------------------------------------------------------
# Backend mocks. Each one returns the *documented* contract shape so
# the loader has something realistic to walk.
# ----------------------------------------------------------------------
class FakeMarketData:
    """Mimics data.market_data.MarketData."""
    def __init__(self, db_path: str = "") -> None:
        self.db_path = db_path

    def get_latest_price(self, ticker: str) -> dict:
        return {"price": 150.0}

    def get_prices(self, ticker: str, period: str = "3mo",
                   interval: str = "1d") -> pd.DataFrame:
        idx = pd.date_range("2026-04-01", periods=10, freq="B")
        return pd.DataFrame({
            "open":  [100.0] * 10,
            "high":  [105.0] * 10,
            "low":    [95.0] * 10,
            "close": [102.0] * 10,
            "volume": [1_000_000] * 10,
        }, index=idx)


class FakePortfolioDB:
    """Mimics data.portfolio_db.PortfolioDB enough to back the loader.

    The real class returns more nested fields; we return exactly what
    the loader reads, no more, so we catch missing-key bugs.
    """
    def __init__(self, db_path: str = "") -> None:
        self.db_path = db_path

    def get_portfolio_snapshot(self, market_data=None) -> dict:
        # This is the contract the loader walks (app.py lines 232-243).
        return {
            "positions": [
                {
                    "ticker": "AAPL", "shares": 100,
                    "avg_entry_price": 170.0,
                    "current_price": 195.0,
                    "market_value": 19_500.0,
                    "unrealized_pnl": 2_500.0,
                    "unrealized_pct": 14.71,
                    "portfolio_weight": 50.0,
                    "sector": "Technology",
                    "conviction": "HIGH",
                },
            ],
            "summary": {
                "num_positions": 1,
                "total_value": 19_500.0,
                "unrealized_pnl": 2_500.0,
                "cash": 5_000.0,
            },
        }

    def get_risk_metrics(self, market_data=None) -> dict:
        return {"concentration": "LOW", "num_positions": 1}

    def get_cash(self) -> float:
        return 5_000.0


class FakeMacroData:
    """Mimics data.macro_data.MacroData.get_macro_dashboard() exactly
    as documented in macro_data.py lines 253-350.

    This is the most important fake in the audit — the task explicitly
    flags get_macro_dashboard() as the contract to verify against.
    """
    def __init__(self, db_path: str = "") -> None:
        self.db_path = db_path

    def get_macro_dashboard(self) -> dict:
        # Shape verified against macro_data.py:253-350.
        return {
            "timestamp": "2026-05-21T00:00:00",
            "interest_rates": {
                "fed_funds": 4.25,
                "treasury_2y": 4.50,
                "treasury_10y": 4.10,
                "treasury_30y": 4.30,
                "yield_spread_10y2y": -0.40,
                "yield_curve_inverted": True,
            },
            "inflation": {
                "cpi_yoy_pct": 2.50,
                "breakeven_5y": 2.30,
            },
            "employment": {
                "unemployment_rate": 3.90,
                "initial_claims": 215_000,
            },
            "growth": {
                "gdp_growth_annualized": 2.10,
                "consumer_sentiment": 78.5,
            },
            "financial_conditions": {
                "vix": 17.5,
                "financial_stress_index": -0.50,
                "baa_corporate_spread": 1.80,
            },
            "recession_signals": {
                "recession_probability": 18.0,
                "sahm_rule": 0.40,
                "sahm_triggered": False,
            },
        }


# ----------------------------------------------------------------------
# Pytest fixtures — wire the fakes into the loader module before each
# test so cache_data doesn't carry state across tests.
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_loaders(monkeypatch, tmp_path):
    """Per-test isolation: patch backends to fakes, point the DB at
    a temp file, and clear any st.cache_data state from prior tests.

    autouse=True so every test in this module gets fresh state.
    """
    # 1. Backend module patches. We patch the module-level references
    #    that the loader module already captured at import time.
    monkeypatch.setattr(ui_app, "MarketData", FakeMarketData, raising=True)
    monkeypatch.setattr(ui_app, "MacroData", FakeMacroData, raising=True)

    # portfolio_db is referenced via the module object (the loader does
    # `portfolio_db.PortfolioDB(db_path=...)`), so we replace the module
    # in the loader's namespace.
    fake_pdb_mod = types.SimpleNamespace(PortfolioDB=FakePortfolioDB)
    monkeypatch.setattr(ui_app, "portfolio_db", fake_pdb_mod, raising=True)

    # 2. DB path: the loaders look up _db_path() which reads config.
    #    We bypass that by monkey-patching _db_path to a temp dir.
    monkeypatch.setattr(ui_app, "_db_path", lambda: str(tmp_path / "test.db"))

    # 3. Clear all five cached loaders — st.cache_data hands them a
    #    .clear() method (provided by our stub) — calling it makes
    #    the next invocation actually run.
    for name in (
        "load_portfolio_summary",
        "load_latest_opinion",
        "load_ticker_context",
        "load_price_history",
        "load_macro_snapshot",
    ):
        fn = getattr(ui_app, name, None)
        if fn is not None and hasattr(fn, "clear"):
            fn.clear()


# ======================================================================
# 1. load_portfolio_summary  (app.py lines 186-271)
# ======================================================================
class TestPortfolioSummaryContract:

    def test_returns_dict(self):
        out = ui_app.load_portfolio_summary()
        assert isinstance(out, dict)

    def test_has_all_keys_render_metric_cards_reads(self):
        # render_metric_cards (app.py:558-574) reads:
        #   num_holdings, total_value, total_pnl, avg_risk, cash, holdings
        out = ui_app.load_portfolio_summary()
        required = {"holdings", "num_holdings", "total_value",
                    "total_pnl", "avg_risk", "cash"}
        missing = required - set(out.keys())
        assert not missing, f"loader missing keys: {missing}"

    def test_value_types_match_consumer_expectations(self):
        out = ui_app.load_portfolio_summary()
        assert isinstance(out["holdings"], pd.DataFrame)
        assert isinstance(out["num_holdings"], int)
        # total_value/total_pnl are formatted with f"${x:,.0f}" — must be
        # numeric, not str.
        assert isinstance(out["total_value"], (int, float))
        assert isinstance(out["total_pnl"], (int, float))
        # avg_risk maps to _risk_badge() — must be one of the four
        # documented strings.
        assert out["avg_risk"] in {"low", "moderate", "high", "unknown"}
        # cash is Optional[float]; None must not be silently turned into
        # 0 by the loader.
        assert out["cash"] is None or isinstance(out["cash"], (int, float))

    def test_avg_risk_normalised_to_lowercase(self):
        # The fake returns "LOW"; loader must lowercase before exposing.
        out = ui_app.load_portfolio_summary()
        assert out["avg_risk"] == "low"

    def test_empty_returns_well_shaped_dict(self, monkeypatch):
        # Backend hard-failure path: every key must still exist with a
        # safe default, so render_metric_cards never KeyErrors.
        class _Broken:
            def __init__(self, *_a, **_k):
                raise RuntimeError("simulated DB failure")
        monkeypatch.setattr(
            ui_app, "portfolio_db",
            types.SimpleNamespace(PortfolioDB=_Broken),
            raising=True,
        )
        ui_app.load_portfolio_summary.clear()
        out = ui_app.load_portfolio_summary()
        required = {"holdings", "num_holdings", "total_value",
                    "total_pnl", "avg_risk", "cash"}
        assert set(out.keys()) >= required


# ======================================================================
# 2. load_macro_snapshot  (app.py lines 509-523)
# ======================================================================
class TestMacroSnapshotContract:
    """The headline audit. Walks the full nested dict that
    MacroData.get_macro_dashboard() returns and verifies render_macro_panel
    can locate every key it expects."""

    def test_returns_dict(self):
        out = ui_app.load_macro_snapshot()
        assert isinstance(out, dict)

    def test_top_level_blocks_present(self):
        # macro_data.py:260-268 documents these top-level keys.
        out = ui_app.load_macro_snapshot()
        expected_blocks = {
            "timestamp",
            "interest_rates",
            "inflation",
            "employment",
            "growth",
            "financial_conditions",
            "recession_signals",
        }
        missing = expected_blocks - set(out.keys())
        assert not missing, f"top-level blocks missing: {missing}"

    def test_nested_blocks_are_dicts_not_scalars(self):
        # If any of these became a scalar (because of an upstream bug),
        # the UI's .get("vix") chain would silently return None.
        out = ui_app.load_macro_snapshot()
        for block in ("interest_rates", "inflation", "employment",
                      "growth", "financial_conditions", "recession_signals"):
            assert isinstance(out[block], dict), (
                f"{block} must be a dict, got {type(out[block]).__name__}"
            )

    def test_interest_rates_has_keys_panel_reads(self):
        # render_macro_panel (app.py:802-855) reads:
        #   interest_rates.yield_spread_10y2y
        #   interest_rates.yield_curve_inverted
        out = ui_app.load_macro_snapshot()
        rates = out["interest_rates"]
        for key in ("yield_spread_10y2y", "yield_curve_inverted"):
            assert key in rates, f"interest_rates missing {key}"

    def test_financial_conditions_has_vix(self):
        # render_macro_panel reads financial_conditions.vix
        out = ui_app.load_macro_snapshot()
        assert "vix" in out["financial_conditions"]

    def test_recession_signals_has_recession_probability(self):
        # render_macro_panel reads recession_signals.recession_probability
        out = ui_app.load_macro_snapshot()
        assert "recession_probability" in out["recession_signals"]

    def test_panel_can_resolve_all_four_cards(self):
        """End-to-end: simulate render_macro_panel's exact lookup
        sequence and assert each card has a real value (not None)."""
        snap = ui_app.load_macro_snapshot()
        # The panel merges snapshot with context.get("macro"); we
        # simulate context being absent (the typical case for the
        # macro panel which only has the snapshot).
        macro = {**(snap or {}), **{}}

        rates = macro.get("interest_rates") or {}
        financial = macro.get("financial_conditions") or {}
        recession = macro.get("recession_signals") or {}

        # The lookup chains the panel uses (app.py:816-839)
        recession_prob = (
            recession.get("recession_probability")
            or macro.get("recession_probability")
        )
        term_spread = (
            rates.get("yield_spread_10y2y")
            or macro.get("term_spread")
            or macro.get("yield_curve_spread")
        )
        vix = (
            financial.get("vix")
            or macro.get("vix")
            or macro.get("volatility_index")
        )
        inverted = rates.get("yield_curve_inverted")

        # All four cards must resolve to non-None with the documented
        # fake data. If the contract drifts (e.g. nested → flat), this
        # is where it breaks first.
        assert recession_prob is not None, "recession card has no value"
        assert term_spread is not None, "term-spread card has no value"
        assert vix is not None, "vix card has no value"
        assert isinstance(inverted, bool), (
            "yield_curve_inverted must be bool for Inverted/Normal logic"
        )

    def test_recession_probability_is_numeric(self):
        # Panel does `recession_prob > 1.0` to detect percent vs.
        # fraction. Must be numeric or that comparison crashes.
        snap = ui_app.load_macro_snapshot()
        rp = snap["recession_signals"]["recession_probability"]
        assert isinstance(rp, (int, float))

    def test_empty_on_backend_failure(self, monkeypatch):
        # If MacroData blows up, the loader must return {} not None.
        # render_macro_panel relies on `if not macro:` short-circuit.
        class _Broken:
            def __init__(self, *_a, **_k):
                raise RuntimeError("simulated FRED outage")
        monkeypatch.setattr(ui_app, "MacroData", _Broken, raising=True)
        ui_app.load_macro_snapshot.clear()
        out = ui_app.load_macro_snapshot()
        assert out == {}


# ======================================================================
# 3. load_latest_opinion  (app.py lines 274-304)
# ======================================================================
class TestLatestOpinionContract:

    def test_returns_dict_on_empty_db(self):
        # No agent_opinions table → must return {} not crash
        out = ui_app.load_latest_opinion("AAPL")
        assert out == {}

    def test_returns_payload_when_row_exists(self, tmp_path, monkeypatch):
        import sqlite3, json as _json
        db = tmp_path / "with_opinions.db"
        with sqlite3.connect(db) as conn:
            conn.execute("""
                CREATE TABLE agent_opinions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT, created_at TEXT, payload_json TEXT
                )
            """)
            payload = {
                "outlook": "bullish",
                "confidence": 0.85,
                "price_direction": "moderate_up",
                "time_horizon": "1Y",
                "rationale": "test",
                "key_risks": [],
                "key_opportunities": [],
                "risk_flags": {"levels": {"combined": "LOW"}, "reasons": []},
            }
            conn.execute(
                "INSERT INTO agent_opinions(ticker, created_at, payload_json) "
                "VALUES (?, ?, ?)",
                ("AAPL", "2026-05-21", _json.dumps(payload)),
            )
        monkeypatch.setattr(ui_app, "_db_path", lambda: str(db))
        ui_app.load_latest_opinion.clear()
        out = ui_app.load_latest_opinion("AAPL")
        # Every field the downstream load_ticker_context reconstructor
        # walks must come through.
        for k in ("outlook", "confidence", "price_direction",
                  "time_horizon", "rationale",
                  "key_risks", "key_opportunities", "risk_flags"):
            assert k in out, f"opinion payload missing {k}"


# ======================================================================
# 4. load_ticker_context  (app.py lines 387-445)
# ======================================================================
class TestTickerContextContract:

    def test_returns_dict_when_pipeline_unavailable(self, monkeypatch):
        monkeypatch.setattr(ui_app, "data_pipeline", None, raising=True)
        ui_app.load_ticker_context.clear()
        out = ui_app.load_ticker_context("AAPL", 90)
        assert out == {}

    def test_reconstructs_thesis_and_risk_keys(self, monkeypatch, tmp_path):
        # Wire a fake pipeline that returns empty raw context, plus a
        # populated agent_opinions row so load_ticker_context's
        # reconstruction branch runs.
        import sqlite3, json as _json

        fake_pipeline = types.SimpleNamespace(
            build_agent_context=lambda t, p, c: {"prices": []},
        )
        monkeypatch.setattr(ui_app, "data_pipeline", fake_pipeline,
                            raising=True)
        # Need a real-ish config object — the loader checks `app_config is None`
        monkeypatch.setattr(ui_app, "app_config",
                            types.SimpleNamespace(), raising=True)

        db = tmp_path / "ctx.db"
        with sqlite3.connect(db) as conn:
            conn.execute("""
                CREATE TABLE agent_opinions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT, created_at TEXT, payload_json TEXT
                )
            """)
            payload = {
                "outlook": "neutral",
                "confidence": 0.6,
                "price_direction": "flat",
                "time_horizon": "1Y",
                "rationale": "balanced",
                "key_risks": ["leverage"],
                "key_opportunities": ["growth"],
                "risk_flags": {
                    "levels": {
                        "combined": "MEDIUM",
                        "fundamental": "LOW",
                        "macro": "MEDIUM",
                        "market": "LOW",
                    },
                    "reasons": ["debt"],
                },
            }
            conn.execute(
                "INSERT INTO agent_opinions(ticker, created_at, payload_json) "
                "VALUES (?, ?, ?)",
                ("AAPL", "2026-05-21", _json.dumps(payload)),
            )
        monkeypatch.setattr(ui_app, "_db_path", lambda: str(db))
        ui_app.load_ticker_context.clear()
        ui_app.load_latest_opinion.clear()

        out = ui_app.load_ticker_context("AAPL", 90)
        # The loader builds these two nested dicts (app.py:416-443) —
        # this is the contract render_thesis_panel / render_risk_panel
        # walk top-down.
        assert "thesis" in out, "context missing reconstructed thesis"
        assert "risk" in out, "context missing reconstructed risk"

        thesis = out["thesis"]
        for k in ("outlook", "direction", "confidence", "time_horizon",
                  "summary", "key_risks", "opportunities"):
            assert k in thesis, f"thesis sub-dict missing {k}"

        risk = out["risk"]
        for k in ("combined_level", "fundamental_risk",
                  "macro_risk", "market_risk", "flags"):
            assert k in risk, f"risk sub-dict missing {k}"

        # Type checks the consumer relies on
        assert isinstance(thesis["key_risks"], list)
        assert isinstance(risk["flags"], list)
        # Risk levels are lowercased for badge styling
        assert risk["combined_level"] == "medium"
        assert risk["fundamental_risk"] == "low"


# ======================================================================
# 5. load_price_history  (app.py lines 448-506)
# ======================================================================
class TestPriceHistoryContract:

    def test_returns_dataframe(self):
        df = ui_app.load_price_history("AAPL", 90)
        assert isinstance(df, pd.DataFrame)

    def test_returns_empty_when_market_data_missing(self, monkeypatch):
        monkeypatch.setattr(ui_app, "MarketData", None, raising=True)
        ui_app.load_price_history.clear()
        out = ui_app.load_price_history("AAPL", 90)
        assert isinstance(out, pd.DataFrame)
        assert out.empty

    def test_has_lowercase_ohlcv_columns(self):
        df = ui_app.load_price_history("AAPL", 90)
        if df.empty:
            pytest.skip("fake returned empty; cannot check columns")
        # render_price_chart reads df["close"], df["volume"], etc.
        # Loader normalises columns to lowercase (app.py:487).
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns, f"column {col!r} missing"

    def test_has_datetime_index(self):
        df = ui_app.load_price_history("AAPL", 90)
        if df.empty:
            pytest.skip("fake returned empty; cannot check index")
        assert isinstance(df.index, pd.DatetimeIndex)
