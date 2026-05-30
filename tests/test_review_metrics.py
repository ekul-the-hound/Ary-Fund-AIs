"""
tests/test_review_metrics.py
============================
Verifies that ``thesis_review.review_essay`` (the review pass) and
``thesis_review.revise_essay`` (the revision pass) each record a telemetry
row through the observability layer, labeled ``thesis_review`` and
``thesis_revision`` respectively, across mock / success / failure paths —
without changing behaviour or breaking when telemetry fails.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from data import metrics_db
import agent.metrics as agent_metrics
import agent.thesis_review as thesis_review


@pytest.fixture
def tmp_metrics_db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    metrics_db.create_metrics_table(path)
    real_insert = metrics_db.insert_metric
    monkeypatch.setattr(
        agent_metrics.metrics_db, "insert_metric",
        lambda rec, db_path=None: real_insert(rec, db_path=path),
    )
    yield path
    try:
        if os.path.exists(path):
            os.unlink(path)
    except (PermissionError, OSError):
        pass


class _MockCfg:
    DEFAULT_AGENT_MODEL = "mock"
    AGENT_MODELS = {"mock": "mock"}


class _LiveCfg:
    DEFAULT_AGENT_MODEL = "dev"
    AGENT_MODELS = {"dev": "phi3:3.8b"}


_ESSAY = "This is the original essay. " * 60  # ~360 words
_METRICS = {"name": "Apple Inc.", "ticker": "AAPL", "p_e": 37.7}


# ---------------------------------------------------------------------------
# Review pass
# ---------------------------------------------------------------------------

class TestReviewMetricRecording:
    def test_mock_review_records_row(self, tmp_metrics_db):
        out = thesis_review.review_essay(
            "AAPL", _ESSAY, _METRICS, {}, {}, _MockCfg,
        )
        assert out["text"]
        rows = metrics_db.get_metrics(db_path=tmp_metrics_db)
        assert any(r["agent_name"] == "thesis_review" for r in rows)

    def test_live_review_success(self, tmp_metrics_db, monkeypatch):
        monkeypatch.setattr(
            thesis_review, "_call_ollama_text",
            lambda prompt, model, config, temperature=None: (
                "## 1. Executive Summary [SCORE: 7/10]\n" + ("detail " * 200)
            ),
        )
        out = thesis_review.review_essay(
            "AAPL", _ESSAY, _METRICS, {}, {}, _LiveCfg,
        )
        assert not out["fallback"]
        row = [r for r in metrics_db.get_metrics(db_path=tmp_metrics_db)
               if r["agent_name"] == "thesis_review"][0]
        assert row["success"] == 1
        assert row["ticker"] == "AAPL"

    def test_live_review_failure(self, tmp_metrics_db, monkeypatch):
        def _boom(prompt, model, config, temperature=None):
            raise ConnectionError("no ollama")
        monkeypatch.setattr(thesis_review, "_call_ollama_text", _boom)
        out = thesis_review.review_essay(
            "AAPL", _ESSAY, _METRICS, {}, {}, _LiveCfg,
        )
        assert out["fallback"]
        row = [r for r in metrics_db.get_metrics(db_path=tmp_metrics_db)
               if r["agent_name"] == "thesis_review"][0]
        assert row["success"] == 0
        assert row["model"].endswith("(failed)")


# ---------------------------------------------------------------------------
# Revision pass
# ---------------------------------------------------------------------------

class TestRevisionMetricRecording:
    def test_mock_revision_records_row(self, tmp_metrics_db):
        out = thesis_review.revise_essay(
            "AAPL", _ESSAY, "review text", _METRICS, {}, {}, _MockCfg,
        )
        assert out["revision_applied"] is False  # mock keeps original
        rows = metrics_db.get_metrics(db_path=tmp_metrics_db)
        assert any(r["agent_name"] == "thesis_revision" for r in rows)

    def test_live_revision_success(self, tmp_metrics_db, monkeypatch):
        # Return a revision at least 85% of the original length so it's accepted.
        monkeypatch.setattr(
            thesis_review, "_call_ollama_text",
            lambda prompt, model, config, temperature=None: ("word " * 400),
        )
        out = thesis_review.revise_essay(
            "AAPL", _ESSAY, "review text", _METRICS, {}, {}, _LiveCfg,
        )
        assert out["revision_applied"] is True
        row = [r for r in metrics_db.get_metrics(db_path=tmp_metrics_db)
               if r["agent_name"] == "thesis_revision"][0]
        assert row["success"] == 1

    def test_telemetry_failure_does_not_break_revision(self, tmp_metrics_db, monkeypatch):
        monkeypatch.setattr(
            agent_metrics, "record_metrics",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")),
        )
        out = thesis_review.revise_essay(
            "AAPL", _ESSAY, "review text", _METRICS, {}, {}, _MockCfg,
        )
        assert out["text"]
        assert "revision_applied" in out
