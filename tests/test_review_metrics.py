"""
tests/test_review_metrics.py
============================
Mirror of test_essay_metrics.py for ``agent.thesis_review.review_essay``.

``thesis_review`` shares the essay module's private-Ollama design: it
calls ``_call_ollama_text`` directly rather than ``base_agent.ask_agent``,
so telemetry reaches it through the ``metrics.record_metrics`` escape
hatch. These tests verify a row is recorded for every real review call,
labeled ``agent_name="thesis_review"``, and that telemetry failures never
break review generation.

``review_essay``'s success guard requires >=200 words, so the canned
review text is padded accordingly. ``_call_ollama_text`` in this module
takes positional ``(prompt, model_name, config)`` plus a keyword
``temperature``, so the mock accepts ``*args, **kwargs``.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from agent import thesis_review
from data import metrics_db


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
class _FakeConfig:
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
def review_inputs():
    essay_text = " ".join(["The"] + ["analysis"] * 300)
    metrics_in = {"sector": "Technology", "revenue_growth_3y": 0.08}
    macro = {"as_of": "2026-05-30", "regime": "expansion"}
    risk_flags = {"levels": {"combined": "MEDIUM"}}
    return essay_text, metrics_in, macro, risk_flags


# Padded so review_essay's >=200-word success guard passes. Includes a
# fake score line so _extract_scores has something to parse.
_CANNED_REVIEW = "Thesis Quality: 8/10\n" + " ".join(["critique"] * 300)


class _RecordSpy:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, call_metadata, *args, **kwargs):
        self.calls.append(dict(call_metadata))
        return 1


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_review_records_metrics_on_success(monkeypatch, fake_config, review_inputs):
    essay_text, metrics_in, macro, risk_flags = review_inputs

    monkeypatch.setattr(
        thesis_review, "_call_ollama_text", lambda *a, **k: _CANNED_REVIEW
    )
    spy = _RecordSpy()
    monkeypatch.setattr(thesis_review._metrics, "record_metrics", spy)

    out = thesis_review.review_essay(
        ticker="AAPL",
        essay_text=essay_text,
        metrics=metrics_in,
        macro=macro,
        risk_flags=risk_flags,
        config=fake_config,
    )

    assert out["fallback"] is False
    assert len(spy.calls) >= 1


def test_review_metric_has_agent_name(monkeypatch, fake_config, review_inputs):
    essay_text, metrics_in, macro, risk_flags = review_inputs

    monkeypatch.setattr(
        thesis_review, "_call_ollama_text", lambda *a, **k: _CANNED_REVIEW
    )
    spy = _RecordSpy()
    monkeypatch.setattr(thesis_review._metrics, "record_metrics", spy)

    thesis_review.review_essay(
        ticker="AAPL",
        essay_text=essay_text,
        metrics=metrics_in,
        macro=macro,
        risk_flags=risk_flags,
        config=fake_config,
    )

    # Every recorded row from this module must carry the review label.
    assert spy.calls, "no metrics recorded"
    for rec in spy.calls:
        assert rec["agent_name"] == "thesis_review"
    rec0 = spy.calls[0]
    assert rec0["ticker"] == "AAPL"
    assert rec0["model"] == "qwen3:30b"
    assert rec0["success"] is True
    assert rec0["prompt_tokens"] > 0
    assert rec0["completion_tokens"] > 0


def test_review_records_failure_metric(monkeypatch, fake_config, review_inputs):
    essay_text, metrics_in, macro, risk_flags = review_inputs

    def _boom(*a, **k):
        raise RuntimeError("ollama timeout")

    monkeypatch.setattr(thesis_review, "_call_ollama_text", _boom)
    spy = _RecordSpy()
    monkeypatch.setattr(thesis_review._metrics, "record_metrics", spy)

    out = thesis_review.review_essay(
        ticker="AAPL",
        essay_text=essay_text,
        metrics=metrics_in,
        macro=macro,
        risk_flags=risk_flags,
        config=fake_config,
    )

    assert out["fallback"] is True
    assert len(spy.calls) >= 1
    failure_rows = [r for r in spy.calls if r["success"] is False]
    assert failure_rows, "expected a success=False row"
    assert all(r["agent_name"] == "thesis_review" for r in spy.calls)
    assert "ollama timeout" in (failure_rows[0]["error_message"] or "")


def test_review_survives_telemetry_failure(monkeypatch, fake_config, review_inputs):
    essay_text, metrics_in, macro, risk_flags = review_inputs

    monkeypatch.setattr(
        thesis_review, "_call_ollama_text", lambda *a, **k: _CANNED_REVIEW
    )

    def _explode(call_metadata, *a, **k):
        raise RuntimeError("metrics db locked")

    monkeypatch.setattr(thesis_review._metrics, "record_metrics", _explode)

    out = thesis_review.review_essay(
        ticker="AAPL",
        essay_text=essay_text,
        metrics=metrics_in,
        macro=macro,
        risk_flags=risk_flags,
        config=fake_config,
    )

    # Telemetry blew up but review generation succeeded with real output.
    assert out["fallback"] is False
    assert isinstance(out["text"], str) and len(out["text"].split()) >= 200


def test_review_metric_lands_in_db(monkeypatch, fake_config, review_inputs):
    essay_text, metrics_in, macro, risk_flags = review_inputs

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    metrics_db.create_metrics_table(db_path)
    monkeypatch.setattr(metrics_db, "_default_db_path", lambda: db_path)

    monkeypatch.setattr(
        thesis_review, "_call_ollama_text", lambda *a, **k: _CANNED_REVIEW
    )

    try:
        thesis_review.review_essay(
            ticker="AAPL",
            essay_text=essay_text,
            metrics=metrics_in,
            macro=macro,
            risk_flags=risk_flags,
            config=fake_config,
        )

        rows = metrics_db.get_metrics(
            agent_name="thesis_review", since_days=1, db_path=db_path
        )
        assert len(rows) >= 1
        row = rows[0]
        assert row["agent_name"] == "thesis_review"
        assert row["ticker"] == "AAPL"
        assert row["total_tokens"] == row["prompt_tokens"] + row["completion_tokens"]
    finally:
        try:
            os.unlink(db_path)
        except (PermissionError, OSError):
            pass