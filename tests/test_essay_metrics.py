"""
tests/test_essay_metrics.py
===========================
Verifies that ``agent.thesis_essay.generate_thesis_essay`` records a
telemetry row for every real (non-mock) LLM call, labeled
``agent_name="thesis_essay"``, and that a telemetry failure can never
break essay generation.

The essay module does not route through ``base_agent.ask_agent`` — it
owns a private ``_call_ollama_text`` path — so the observability layer
reaches it via the ``metrics.record_metrics`` escape hatch rather than
``instrumented_ask``. These tests exercise that wiring at its real seam:

* ``_call_ollama_text`` is monkeypatched to canned text, so no Ollama
  server is needed and the test is deterministic.
* A fake config forces a non-"mock" model so the instrumented branch
  (not the deterministic fallback) runs.
* ``metrics.record_metrics`` is patched to a spy so we can assert on the
  exact dict the production code hands it — without touching any real
  metrics.db.

One integration test patches ``metrics_db._default_db_path`` to a temp
file and reads the row back through the real DB layer, proving the
record actually lands in the ``agent_metrics`` table with the right
agent label.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from agent import thesis_essay
from agent import metrics as agent_metrics
from data import metrics_db


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
class _FakeConfig:
    """Minimal config that resolves to a real (non-mock) model so the
    instrumented Ollama branch runs. ``AGENT_MODELS`` maps the default
    tag to a non-"mock" string; ``_resolve_model`` returns that string,
    which makes ``generate_thesis_essay`` take the ``_call_ollama_text``
    path instead of the deterministic fallback."""

    DEFAULT_AGENT_MODEL = "prod"
    AGENT_MODELS = {"prod": "qwen3:30b", "mock": "mock"}
    OLLAMA_BASE_URL = "http://localhost:11434"
    AGENT_TIMEOUT = 30
    MAX_TOKENS = 4096
    METRICS_COST_PER_1K_TOKENS = 0.0001


@pytest.fixture
def fake_config():
    return _FakeConfig()


@pytest.fixture
def essay_inputs():
    """The six content inputs ``generate_thesis_essay`` needs. Values are
    only ever interpolated into the prompt string, so light stubs are
    fine — the LLM call itself is mocked."""
    thesis = {
        "ticker": "AAPL",
        "outlook": "bullish",
        "price_direction": "moderate_up",
        "confidence": 0.6,
        "bias_score": 0.3,
        "key_risks": ["valuation"],
        "key_opportunities": ["services growth"],
        "rationale": "Services mix expanding; balance sheet strong.",
        "short_rationale": "AAPL: bullish",
        "valuation": {"current_price": 190.0, "missing_inputs": []},
    }
    metrics_in = {
        "sector": "Technology",
        "revenue_growth_3y": 0.08,
        "debt_ebitda": 1.2,
        "fcf_yield": 0.04,
    }
    macro = {"as_of": "2026-05-30", "regime": "expansion"}
    risk_flags = {"levels": {"combined": "MEDIUM"}}
    filings = {"management_tone": "confident", "red_flags": [], "by_year": {}}
    return thesis, filings, metrics_in, macro, risk_flags


# A long canned essay so the >150-word success guard in
# generate_thesis_essay passes.
_CANNED_ESSAY = " ".join(["AAPL"] + ["word"] * 400)


# ----------------------------------------------------------------------
# Spy helper
# ----------------------------------------------------------------------
class _RecordSpy:
    """Captures every record_metrics call's first positional arg (the
    metadata dict)."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, call_metadata, *args, **kwargs):
        self.calls.append(dict(call_metadata))
        return 1  # mimic the real return type (row id)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_essay_records_metrics_on_success(monkeypatch, fake_config, essay_inputs):
    """A successful essay generation records exactly one telemetry row."""
    thesis, filings, metrics_in, macro, risk_flags = essay_inputs

    monkeypatch.setattr(
        thesis_essay, "_call_ollama_text", lambda **kw: _CANNED_ESSAY
    )
    spy = _RecordSpy()
    # The module imported metrics as a name; patch the attribute it uses.
    monkeypatch.setattr(thesis_essay._metrics, "record_metrics", spy)

    out = thesis_essay.generate_thesis_essay(
        ticker="AAPL",
        thesis=thesis,
        filings_summary=filings,
        metrics=metrics_in,
        macro=macro,
        risk_flags=risk_flags,
        config=fake_config,
    )

    assert out["fallback"] is False
    assert len(spy.calls) == 1


def test_essay_metric_has_agent_name_and_ticker(
    monkeypatch, fake_config, essay_inputs
):
    """The recorded row is labeled agent_name='thesis_essay' and carries
    the ticker, model, token estimates, latency, and success flag."""
    thesis, filings, metrics_in, macro, risk_flags = essay_inputs

    monkeypatch.setattr(
        thesis_essay, "_call_ollama_text", lambda **kw: _CANNED_ESSAY
    )
    spy = _RecordSpy()
    monkeypatch.setattr(thesis_essay._metrics, "record_metrics", spy)

    thesis_essay.generate_thesis_essay(
        ticker="AAPL",
        thesis=thesis,
        filings_summary=filings,
        metrics=metrics_in,
        macro=macro,
        risk_flags=risk_flags,
        config=fake_config,
    )

    rec = spy.calls[0]
    assert rec["agent_name"] == "thesis_essay"
    assert rec["ticker"] == "AAPL"
    assert rec["model"] == "qwen3:30b"
    assert rec["success"] is True
    assert rec["error_message"] is None
    assert rec["prompt_tokens"] > 0
    assert rec["completion_tokens"] > 0
    assert rec["latency_ms"] >= 0


def test_essay_records_failure_metric(monkeypatch, fake_config, essay_inputs):
    """When the LLM call raises, the failure is recorded with success=False
    and a non-empty error_message, and the essay still returns (fallback)."""
    thesis, filings, metrics_in, macro, risk_flags = essay_inputs

    def _boom(**kw):
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr(thesis_essay, "_call_ollama_text", _boom)
    spy = _RecordSpy()
    monkeypatch.setattr(thesis_essay._metrics, "record_metrics", spy)

    out = thesis_essay.generate_thesis_essay(
        ticker="AAPL",
        thesis=thesis,
        filings_summary=filings,
        metrics=metrics_in,
        macro=macro,
        risk_flags=risk_flags,
        config=fake_config,
    )

    # Essay generation survived via deterministic fallback.
    assert out["fallback"] is True
    assert isinstance(out["text"], str) and len(out["text"]) > 0

    # A failure metric was recorded.
    assert len(spy.calls) == 1
    rec = spy.calls[0]
    assert rec["agent_name"] == "thesis_essay"
    assert rec["success"] is False
    assert "ollama unreachable" in (rec["error_message"] or "")


def test_essay_survives_telemetry_failure(monkeypatch, fake_config, essay_inputs):
    """If record_metrics itself raises, essay generation must NOT break —
    telemetry is best-effort and the caller gets a valid essay."""
    thesis, filings, metrics_in, macro, risk_flags = essay_inputs

    monkeypatch.setattr(
        thesis_essay, "_call_ollama_text", lambda **kw: _CANNED_ESSAY
    )

    def _explode(call_metadata, *a, **k):
        raise RuntimeError("metrics db locked")

    monkeypatch.setattr(thesis_essay._metrics, "record_metrics", _explode)

    out = thesis_essay.generate_thesis_essay(
        ticker="AAPL",
        thesis=thesis,
        filings_summary=filings,
        metrics=metrics_in,
        macro=macro,
        risk_flags=risk_flags,
        config=fake_config,
    )

    # The essay is the real (non-fallback) LLM output despite telemetry
    # blowing up — the try/except around record_metrics absorbed it.
    assert out["fallback"] is False
    assert out["text"] == _CANNED_ESSAY


def test_essay_metric_lands_in_db(monkeypatch, fake_config, essay_inputs):
    """Integration: drive the real record_metrics -> metrics_db path against
    a temp DB and read the row back, confirming agent_name='thesis_essay'
    actually persists to the agent_metrics table."""
    thesis, filings, metrics_in, macro, risk_flags = essay_inputs

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    metrics_db.create_metrics_table(db_path)
    # Force every default-path lookup in the DB layer to our temp file.
    monkeypatch.setattr(metrics_db, "_default_db_path", lambda: db_path)

    monkeypatch.setattr(
        thesis_essay, "_call_ollama_text", lambda **kw: _CANNED_ESSAY
    )

    try:
        thesis_essay.generate_thesis_essay(
            ticker="AAPL",
            thesis=thesis,
            filings_summary=filings,
            metrics=metrics_in,
            macro=macro,
            risk_flags=risk_flags,
            config=fake_config,
        )

        rows = metrics_db.get_metrics(
            agent_name="thesis_essay", since_days=1, db_path=db_path
        )
        assert len(rows) == 1
        row = rows[0]
        assert row["agent_name"] == "thesis_essay"
        assert row["ticker"] == "AAPL"
        # Derived fields filled by record_metrics.
        assert row["total_tokens"] == row["prompt_tokens"] + row["completion_tokens"]
        assert row["cost_usd"] is not None
    finally:
        try:
            os.unlink(db_path)
        except (PermissionError, OSError):
            pass