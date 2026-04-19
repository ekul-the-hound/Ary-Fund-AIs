"""
Shared fixtures for the hedge-fund agent test suite.

Design notes
------------
- All external boundaries (LLM, SEC, market data, DB writes) are mocked.
- Fixtures produce *realistic* shapes so the logic under test behaves meaningfully.
- `_safe_import` lets individual tests skip gracefully if a module or symbol
  has not been implemented yet, instead of crashing the entire suite at
  collection time.
"""
from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Make the project root importable regardless of where pytest is launched from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------
def _safe_import(module_path: str):
    """Import a module or return None if it does not exist yet."""
    try:
        return importlib.import_module(module_path)
    except ImportError:
        return None


def _require(module_path: str, *attrs: str):
    """Import a module and assert it exposes the given attributes, else skip."""
    mod = _safe_import(module_path)
    if mod is None:
        pytest.skip(f"module {module_path} not importable", allow_module_level=False)
    for a in attrs:
        if not hasattr(mod, a):
            pytest.skip(f"{module_path} missing attribute {a}")
    return mod


@pytest.fixture
def safe_import():
    """Expose the safe-import helper to tests."""
    return _safe_import


@pytest.fixture
def require_module():
    """Expose the require helper to tests."""
    return _require


# ---------------------------------------------------------------------------
# Domain fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_filings() -> list[dict]:
    """A small but realistic set of SEC filings across form types and years."""
    return [
        {
            "ticker": "TEST",
            "form_type": "10-K",
            "filing_date": "2024-02-15",
            "text": (
                "Annual report. Revenue grew 12% year over year. Operating "
                "margins expanded. We identified supply-chain concentration "
                "risk and cybersecurity incidents as material risk factors."
            ),
            "html_url": "https://example.invalid/test-10k-2024.htm",
        },
        {
            "ticker": "TEST",
            "form_type": "10-Q",
            "filing_date": "2024-05-03",
            "text": (
                "Quarterly report. Revenue up 9% Q/Q. Gross margin 48%. "
                "Free cash flow positive. Share repurchases continued."
            ),
            "html_url": "https://example.invalid/test-10q-2024q1.htm",
        },
        {
            "ticker": "TEST",
            "form_type": "10-Q",
            "filing_date": "2024-08-05",
            "text": (
                "Quarterly report. Revenue up 7% Q/Q. Gross margin 47%. "
                "Operating expenses moderated."
            ),
            "html_url": "https://example.invalid/test-10q-2024q2.htm",
        },
        {
            "ticker": "TEST",
            "form_type": "8-K",
            "filing_date": "2024-09-10",
            "text": (
                "Current report. The CFO announced departure effective "
                "end of quarter. The company reaffirmed guidance."
            ),
            "html_url": "https://example.invalid/test-8k-2024-09.htm",
        },
        {
            "ticker": "TEST",
            "form_type": "10-K",
            "filing_date": "2023-02-20",
            "text": "Prior annual report. Revenue grew 15%.",
            "html_url": "https://example.invalid/test-10k-2023.htm",
        },
    ]


@pytest.fixture
def sample_metrics() -> dict:
    """Representative fundamentals dict consumed by analyzer/risk/thesis."""
    return {
        "revenue_growth_yoy": 0.12,
        "gross_margin": 0.48,
        "gross_margin_trend": 0.02,
        "operating_margin": 0.22,
        "operating_margin_trend": 0.01,
        "eps_growth_yoy": 0.14,
        "free_cash_flow": 1_250_000_000.0,
        "free_cash_flow_3y": [900_000_000.0, 1_100_000_000.0, 1_250_000_000.0],
        "cash_conversion": 1.05,
        "debt_to_ebitda": 1.4,
        "pe_ratio": 22.5,
        "ev_to_ebitda": 14.0,
        "price": 184.50,
    }


@pytest.fixture
def risky_metrics(sample_metrics) -> dict:
    """A deliberately stressed metrics dict to trigger high-risk branches."""
    m = dict(sample_metrics)
    m.update(
        {
            "debt_to_ebitda": 6.8,            # very high leverage
            "free_cash_flow": -400_000_000.0,
            "free_cash_flow_3y": [
                -200_000_000.0,
                -350_000_000.0,
                -400_000_000.0,
            ],
            "operating_margin": -0.05,
            "operating_margin_trend": -0.04,
            "eps_growth_yoy": -0.30,
            "revenue_growth_yoy": -0.08,
        }
    )
    return m


@pytest.fixture
def sample_macro() -> dict:
    """Benign macro backdrop."""
    return {
        "recession_probability": 0.18,
        "vix": 14.5,
        "yield_curve_10y_2y": 0.35,     # positive, not inverted
        "inflation_trend": "cooling",
        "policy_rate_proxy": 0.045,
    }


@pytest.fixture
def stressed_macro() -> dict:
    """Recessionary/volatile macro backdrop."""
    return {
        "recession_probability": 0.72,
        "vix": 34.0,
        "yield_curve_10y_2y": -0.55,    # inverted
        "inflation_trend": "rising",
        "policy_rate_proxy": 0.055,
    }


@pytest.fixture
def sample_context(sample_filings, sample_metrics, sample_macro) -> dict:
    """Shape resembling what `pipeline.build_agent_context` returns."""
    return {
        "ticker": "TEST",
        "price": sample_metrics["price"],
        "filings": sample_filings,
        "metrics": sample_metrics,
        "macro": sample_macro,
        "portfolio": {
            "position_size_usd": 500_000.0,
            "weight_in_portfolio": 0.04,
            "cost_basis": 170.00,
        },
        "as_of": "2024-10-01",
    }


@pytest.fixture
def sample_config() -> SimpleNamespace:
    """
    Lightweight config stand-in. SimpleNamespace works because production
    configs are typically attribute-accessed.
    """
    return SimpleNamespace(
        DEFAULT_AGENT_MODEL="mock",
        AGENT_MODEL="mock",
        AGENT_TIMEOUT=10,
        AGENT_MAX_TOKENS=1024,
        OLLAMA_HOST="http://localhost:11434",
        OLLAMA_MODEL="qwen3:30b",
        USE_MOCK_AGENT=True,
        RISK_DEBT_EBITDA_HIGH=4.0,
        RISK_RECESSION_HIGH=0.5,
        DB_PATH=":memory:",
        LOG_LEVEL="ERROR",
    )


@pytest.fixture
def mock_db(tmp_path) -> str:
    """Return a temp SQLite path and pre-create a portfolio_opinions table."""
    db_path = tmp_path / "test_portfolio.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_opinions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def deterministic_agent_response() -> dict:
    """
    Canonical mock-model output used across agent tests.
    Shape mirrors what `ask_agent(...).generated_json` is expected to return.
    """
    return {
        "outlook": "bullish",
        "time_horizon": "1Y",
        "price_direction": "moderate_up",
        "confidence": 0.62,
        "key_risks": [
            "supply-chain concentration",
            "cyber incident exposure",
        ],
        "key_opportunities": [
            "margin expansion",
            "share repurchase authorization",
        ],
        "summary": "Healthy fundamentals with manageable risk factors.",
    }


# ---------------------------------------------------------------------------
# Shape / contract helpers (reused across test files)
# ---------------------------------------------------------------------------
REQUIRED_THESIS_KEYS = {
    "outlook",
    "time_horizon",
    "price_direction",
    "confidence",
    "key_risks",
    "key_opportunities",
}

VALID_OUTLOOKS = {"bullish", "neutral", "bearish"}
VALID_PRICE_DIRECTIONS = {
    "strong_up",
    "moderate_up",
    "flat",
    "moderate_down",
    "strong_down",
}
VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}


@pytest.fixture
def thesis_contract() -> dict[str, Any]:
    return {
        "required_keys": REQUIRED_THESIS_KEYS,
        "valid_outlooks": VALID_OUTLOOKS,
        "valid_price_directions": VALID_PRICE_DIRECTIONS,
        "valid_risk_levels": VALID_RISK_LEVELS,
    }