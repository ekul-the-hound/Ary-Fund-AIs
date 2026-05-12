"""
tests/conftest.py
=================

Shared fixtures for the ARY Fund test suite.

What lives here
---------------
- Two import helpers: ``safe_import`` and ``require_module``. They let
  individual tests skip cleanly when a project module or symbol is missing,
  instead of crashing the whole collection phase.
- Domain fixtures: filings, fundamentals, macro snapshots, config object,
  pipeline context, deterministic agent response, and the thesis contract.

Design notes
------------
- All boundaries (LLM, SEC, market data, DB writes) are mocked at the test
  layer; nothing here makes a real network call.
- Fixture values are sized so that:
    * ``sample_metrics`` + ``sample_macro``  -> LOW risk, POSITIVE bias
    * ``risky_metrics``  + ``stressed_macro`` -> HIGH risk, NEGATIVE bias
  This keeps the directional-bias and rule-trigger tests deterministic.
- The project root is prepended to ``sys.path`` so ``import agent.*`` /
  ``import data.*`` work regardless of which directory pytest is launched
  from (the user's Windows shell can be inconsistent here).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Make the project root importable
# ---------------------------------------------------------------------------
# tests/ lives directly under the project root; one parent up is what we want.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------
def _safe_import(module_path: str):
    """Import ``module_path`` or return ``None`` if it isn't available.

    Used by tests that want to patch a module *if it exists* without blowing
    up when the module is missing or partially built.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError:
        return None
    except Exception:
        # Other import-time errors (syntax in dependent module, missing
        # third-party package, etc.) shouldn't tear down the whole suite.
        return None


def _require(module_path: str, *attrs: str):
    """Import ``module_path`` and assert the requested attributes exist.

    Skips the current test (via ``pytest.skip``) if either the module
    cannot be imported or any required attribute is missing. Returns the
    imported module on success so tests can call its public API.
    """
    mod = _safe_import(module_path)
    if mod is None:
        pytest.skip(
            f"module {module_path} not importable",
            allow_module_level=False,
        )
    for a in attrs:
        if not hasattr(mod, a):
            pytest.skip(
                f"{module_path} missing attribute {a}",
                allow_module_level=False,
            )
    return mod


@pytest.fixture
def safe_import():
    """Expose ``_safe_import`` to tests as a fixture."""
    return _safe_import


@pytest.fixture
def require_module():
    """Expose ``_require`` to tests as a fixture."""
    return _require


# ---------------------------------------------------------------------------
# Filings
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_filings() -> List[Dict[str, Any]]:
    """A small but realistic batch of SEC filings.

    Five distinct ``filing_date`` strings across 2023 and 2024 covering 10-K,
    10-Q and 8-K forms, with prose that exercises the tone/red-flag lexicons
    in ``filing_analyzer``.
    """
    return [
        {
            "ticker": "TEST",
            "form_type": "10-K",
            "filing_date": "2024-02-15",
            "text": (
                "Annual report. Revenue grew 12% year over year. Operating "
                "margins expanded. We identified supply-chain concentration "
                "and cybersecurity incidents as material risk factors. "
                "Outlook reaffirmed; momentum continues into the new year."
            ),
            "html_url": "https://example.invalid/test-10k-2024.htm",
        },
        {
            "ticker": "TEST",
            "form_type": "10-Q",
            "filing_date": "2024-05-03",
            "text": (
                "Quarterly report. Revenue up 9% quarter over quarter. "
                "Gross margin 48%. Free cash flow positive. Share repurchases "
                "continued. Management noted strong demand and accelerating "
                "growth in the data-center segment."
            ),
            "html_url": "https://example.invalid/test-10q-2024q1.htm",
        },
        {
            "ticker": "TEST",
            "form_type": "10-Q",
            "filing_date": "2024-08-05",
            "text": (
                "Quarterly report. Revenue up 7% quarter over quarter. "
                "Gross margin 47%. Operating expenses moderated. "
                "Cash conversion remained robust."
            ),
            "html_url": "https://example.invalid/test-10q-2024q2.htm",
        },
        {
            "ticker": "TEST",
            "form_type": "8-K",
            "filing_date": "2024-09-10",
            "text": (
                "Current report. The CFO announced departure effective end "
                "of quarter. The company reaffirmed guidance and outlined "
                "an orderly transition plan."
            ),
            "html_url": "https://example.invalid/test-8k-2024-09.htm",
        },
        {
            "ticker": "TEST",
            "form_type": "10-K",
            "filing_date": "2023-02-20",
            "text": (
                "Prior annual report. Revenue grew 15% on strong product "
                "demand and operating leverage."
            ),
            "html_url": "https://example.invalid/test-10k-2023.htm",
        },
    ]


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_metrics() -> Dict[str, Any]:
    """Benign, healthy fundamentals snapshot.

    Shaped to look like the OUTPUT of
    ``filing_analyzer.extract_key_metrics_for_agent`` so it can be fed
    directly to ``risk_scanner.compute_risk_flags`` and
    ``thesis_generator.generate_thesis`` without further normalisation.

    Values are tuned so:
      * ``risk_scanner._score_fundamental`` returns "LOW"
      * ``thesis_generator._score_fundamentals_bias`` returns a positive bias
      * the test that mutates ``debt_to_ebitda=8.5`` correctly trips HIGH
    """
    return {
        # Identifier and price -------------------------------------------------
        "ticker": "TEST",
        "price": 150.0,

        # Growth ---------------------------------------------------------------
        "revenue_growth_yoy": 0.10,
        "revenue_growth_3y": 0.12,
        "revenue_growth_5y": 0.11,

        # Margins --------------------------------------------------------------
        "gross_margin": 0.55,
        "operating_margin": 0.25,
        "profit_margin": 0.18,
        "margin_trend": "expanding",

        # Leverage -------------------------------------------------------------
        "debt_ebitda": 1.5,
        "debt_to_ebitda": 1.5,           # alias the risk scanner also reads
        "net_debt_ebitda": 1.0,
        "net_debt_to_ebitda": 1.0,
        "interest_coverage": 12.0,

        # Cash flow ------------------------------------------------------------
        "free_cash_flow": 5_000_000_000.0,
        "cash_flow_negative_3_years": False,
        "cash_conversion": 1.05,

        # Valuation ------------------------------------------------------------
        "p_e": 25.0,
        "pe_ratio": 25.0,
        "forward_pe": 22.0,
        "ev_ebitda": 18.0,
        "ev_to_ebitda": 18.0,
        "fcf_yield": 0.06,
        "free_cash_flow_yield": 0.06,
        "market_cap": 100_000_000_000.0,

        # Quality --------------------------------------------------------------
        "roic": 0.18,
    }


@pytest.fixture
def risky_metrics(sample_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """A stressed variant of ``sample_metrics``.

    Starts from the benign snapshot and overwrites the fields the thesis
    generator's fundamentals scorer reads, pushing every sub-signal negative.
    """
    out = dict(sample_metrics)
    out.update({
        "revenue_growth_yoy": -0.05,
        "revenue_growth_3y": -0.08,        # -> -0.8 contribution
        "revenue_growth_5y": -0.05,
        "gross_margin": 0.30,
        "operating_margin": 0.02,
        "profit_margin": -0.01,
        "margin_trend": "compressing",     # -> -0.5

        "debt_ebitda": 6.0,                # -> -1.0 (>= 4.5)
        "debt_to_ebitda": 6.0,
        "net_debt_ebitda": 5.5,
        "net_debt_to_ebitda": 5.5,
        "interest_coverage": 1.2,          # below 2.0 threshold -> HIGH

        "free_cash_flow": -500_000_000.0,
        "cash_flow_negative_3_years": True,
        "cash_conversion": 0.5,            # -> -0.7 (< 0.7)

        "fcf_yield": -0.02,                # -> -0.9 (< 0)
        "free_cash_flow_yield": -0.02,
        "p_e": 80.0,
        "pe_ratio": 80.0,
        "ev_ebitda": 45.0,
        "ev_to_ebitda": 45.0,

        "roic": -0.05,                     # -> -0.8 (< 0)
    })
    return out


# ---------------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_macro() -> Dict[str, Any]:
    """Benign macro snapshot.

    Tuned so ``risk_scanner._score_macro`` returns "LOW" and
    ``thesis_generator._score_macro_bias`` returns a small positive value.
    """
    return {
        "recession_probability": 0.10,     # well below 0.35 medium threshold
        "vix": 15.0,                       # benign volatility regime
        "yield_curve_inverted": False,
        "yield_curve_spread": 0.02,        # +2 bps, gently upward
        "yield_curve_10y_2y": 0.02,
        "ten_year_yield": 0.040,
        "two_year_yield": 0.020,
        "cpi_yoy": 0.025,
        "unemployment_rate": 0.039,
        "fed_funds_rate": 0.0425,
    }


@pytest.fixture
def stressed_macro() -> Dict[str, Any]:
    """A stressed macro backdrop — risk-off across every signal we read."""
    return {
        "recession_probability": 0.70,     # above 0.6 high threshold
        "vix": 32.0,                       # above 28 high threshold
        "yield_curve_inverted": True,
        "yield_curve_spread": -0.005,
        "yield_curve_10y_2y": -0.005,
        "ten_year_yield": 0.038,
        "two_year_yield": 0.043,
        "cpi_yoy": 0.062,
        "unemployment_rate": 0.058,
        "fed_funds_rate": 0.055,
    }


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_config() -> SimpleNamespace:
    """A minimal stand-in for the project's ``config`` module.

    Carries the attributes the modules under test read via ``getattr`` /
    direct attribute access. ``SimpleNamespace`` is used so tests can
    flip flags with ``setattr(sample_config, "USE_MOCK_AGENT", False)``.
    """
    return SimpleNamespace(
        # Risk thresholds mirror config._DEFAULT_THRESHOLDS exactly.
        RISK_THRESHOLDS={
            "debt_ebitda_high": 3.0,
            "debt_ebitda_medium": 2.0,
            "interest_coverage_low": 2.0,
            "fcf_yield_low": 0.02,
            "recession_prob_high": 0.6,
            "recession_prob_medium": 0.35,
            "vix_high": 28.0,
            "vix_medium": 20.0,
            "drawdown_high": 0.25,
            "drawdown_medium": 0.15,
            "realized_vol_high": 0.45,
            "realized_vol_medium": 0.30,
        },
        # Agent layer
        DEFAULT_AGENT_MODEL="mock",
        AGENT_MODELS={"mock": "mock", "dev": "phi3:3.8b", "prod": "qwen3:30b-a3b"},
        USE_MOCK_AGENT=True,
        MOCK_AGENT=True,
        AGENT_TIMEOUT=30.0,
        # Pipeline
        MAX_FILINGS_PER_TICKER=10,
        # DB / universe — tests rarely read these but cheap to provide.
        PORTFOLIO_DB_PATH="test_portfolio.db",
        WATCHLIST=["TEST"],
        # Logging defaults so main._configure_logging can no-op cleanly.
        LOG_LEVEL=30,                      # WARNING
        LOG_FORMAT="%(levelname)s | %(message)s",
        LOG_FILE=None,
    )


# ---------------------------------------------------------------------------
# Pipeline context
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_context(
    sample_filings: List[Dict[str, Any]],
    sample_metrics: Dict[str, Any],
    sample_macro: Dict[str, Any],
) -> Dict[str, Any]:
    """Mimics the dict returned by ``pipeline.build_agent_context``.

    The orchestration tests patch ``build_agent_context`` to return this
    fixture, then let the real ``filing_analyzer``, ``risk_scanner``, and
    ``thesis_generator`` run on top of it. So every section here must be a
    valid input to those modules.
    """
    return {
        "ticker": "TEST",
        "as_of": "2026-05-10",
        "price": sample_metrics["price"],
        "prices": {
            "last": sample_metrics["price"],
            "close": sample_metrics["price"],
            "change_pct": 0.012,
            "market_cap": sample_metrics["market_cap"],
            "fifty_two_week_high": sample_metrics["price"] * 1.15,
            "fifty_two_week_low": sample_metrics["price"] * 0.78,
        },
        "metrics": dict(sample_metrics),
        "filings": list(sample_filings),
        "macro": dict(sample_macro),
        "portfolio": {},
    }


# ---------------------------------------------------------------------------
# DB path
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_db(tmp_path: Path) -> str:
    """A real file path under pytest's per-test tmp directory.

    All DB writes are mocked at the boundary, so this path is only ever
    *passed*, never actually opened by production code during the test.
    Using a real path keeps anything that does ``Path(db_path).parent``
    happy without polluting the working tree.
    """
    return str(tmp_path / "test_portfolio.db")


# ---------------------------------------------------------------------------
# Agent response & thesis contract
# ---------------------------------------------------------------------------
@pytest.fixture
def deterministic_agent_response() -> Dict[str, Any]:
    """A fully-formed agent response payload.

    Matches the JSON shape ``base_agent.ask_agent`` is contracted to emit,
    so it works whether the patched stub returns it directly or wraps it
    inside an ``AgentResponse``.
    """
    return {
        "ticker": "TEST",
        "outlook": "neutral",
        "time_horizon": "1Y",
        "price_direction": "flat",
        "confidence": 0.55,
        "key_risks": [
            "moderate leverage relative to peers",
            "customer concentration in top three accounts",
        ],
        "key_opportunities": [
            "operating leverage as revenue scales",
            "new product cycle in second half",
        ],
        "summary": (
            "TEST screens neutral on a one-year horizon: clean balance sheet "
            "and strong cash conversion offset by valuation that already "
            "discounts continued execution."
        ),
        # Legacy aliases kept for backward-compatible consumers
        "risks": [
            "moderate leverage relative to peers",
            "customer concentration in top three accounts",
        ],
        "thesis": "Balanced setup; conviction modest on a one-year view.",
    }


@pytest.fixture
def thesis_contract() -> Dict[str, Any]:
    """The shape every thesis output must satisfy.

    Used by the contract tests to assert required keys are present and that
    categorical fields take values from the allowed sets.
    """
    return {
        "required_keys": {
            "outlook",
            "time_horizon",
            "price_direction",
            "confidence",
            "key_risks",
            "key_opportunities",
        },
        "valid_outlooks": {"bullish", "neutral", "bearish"},
        "valid_price_directions": {
            "strong_up",
            "moderate_up",
            "flat",
            "moderate_down",
            "strong_down",
        },
    }