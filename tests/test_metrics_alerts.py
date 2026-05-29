"""
tests/test_metrics_alerts.py
============================
Tests for ``agent/metrics_alerts.py``.

Strategy: seed a temp metrics.db with rows whose timestamps are
positioned relative to an injected "now" (so we don't fight the clock),
run ``evaluate_alerts`` and ``check_and_fire_alerts``, assert what
triggers and what gets deduped.

A fake ``notify_fn`` is injected via the public kwarg — the real
``notifiers.notify_slack`` is never called and no webhook is required.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from agent import metrics_alerts as alerter
from data import metrics_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    metrics_db.create_metrics_table(path)
    yield path
    # See note in test_metrics.py: Windows can hold a transient SQLite
    # lock on the temp file, making os.unlink raise during teardown. A
    # cleanup failure must not fail a passing test.
    try:
        if os.path.exists(path):
            os.unlink(path)
    except (PermissionError, OSError):
        pass


class _Cfg:
    """Minimal config with the alerter's knobs. Tests override per-case."""
    ALERT_CONSECUTIVE_FAILURES = 5
    ALERT_LATENCY_P95_MS = 15_000.0
    ALERT_LATENCY_WINDOW_MIN = 60
    ALERT_MIN_SAMPLES_FOR_LATENCY = 10
    ALERT_COST_BUDGET_USD = 1.0
    ALERT_COST_WINDOW_HOURS = 24
    ALERT_COOLDOWN_MINUTES = 60


def _seed(db_path: str, rows: list[dict]) -> None:
    """Bulk-insert rows; lets tests stamp explicit timestamps."""
    for r in rows:
        metrics_db.insert_metric(r, db_path=db_path)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


class _Notifier:
    """Collects sent messages so tests can assert what would have shipped."""

    def __init__(self, succeed: bool = True):
        self.messages: list[str] = []
        self.succeed = succeed

    def __call__(self, text: str, **kw) -> bool:
        self.messages.append(text)
        return self.succeed


# ---------------------------------------------------------------------------
# Rule: consecutive failures
# ---------------------------------------------------------------------------

class TestConsecutiveFailures:
    def test_fires_when_last_N_all_failed(self, tmp_db):
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        # 5 failures in a row, oldest first.
        for i in range(5):
            _seed(tmp_db, [{
                "agent_name": "thesis_generator",
                "success": False, "error_message": "ollama down",
                "timestamp": _iso(now - timedelta(minutes=10 - i)),
                "latency_ms": 100.0,
            }])
        alerts = alerter.evaluate_alerts(config=_Cfg, db_path=tmp_db, now=now)
        rules = [(a.rule, a.scope) for a in alerts]
        assert ("consecutive_failures", "thesis_generator") in rules

    def test_silent_when_one_recent_success(self, tmp_db):
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        # 4 failures then 1 success → last 5 are not all failures
        for i in range(4):
            _seed(tmp_db, [{
                "agent_name": "thesis_generator", "success": False,
                "timestamp": _iso(now - timedelta(minutes=5 - i)),
            }])
        _seed(tmp_db, [{
            "agent_name": "thesis_generator", "success": True,
            "timestamp": _iso(now),
        }])
        alerts = alerter.evaluate_alerts(config=_Cfg, db_path=tmp_db, now=now)
        assert not any(a.rule == "consecutive_failures" for a in alerts)

    def test_silent_below_threshold_count(self, tmp_db):
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        for i in range(4):  # only 4, threshold is 5
            _seed(tmp_db, [{
                "agent_name": "thesis_generator", "success": False,
                "timestamp": _iso(now - timedelta(minutes=4 - i)),
            }])
        alerts = alerter.evaluate_alerts(config=_Cfg, db_path=tmp_db, now=now)
        assert not any(a.rule == "consecutive_failures" for a in alerts)

    def test_per_agent_isolation(self, tmp_db):
        # Agent A failed 5x; agent B is healthy. Only A fires.
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        for i in range(5):
            _seed(tmp_db, [{
                "agent_name": "A", "success": False,
                "timestamp": _iso(now - timedelta(minutes=10 - i)),
            }])
        for i in range(5):
            _seed(tmp_db, [{
                "agent_name": "B", "success": True,
                "timestamp": _iso(now - timedelta(minutes=10 - i)),
            }])
        alerts = alerter.evaluate_alerts(config=_Cfg, db_path=tmp_db, now=now)
        scopes = {a.scope for a in alerts if a.rule == "consecutive_failures"}
        assert scopes == {"A"}

    def test_disabled_when_threshold_zero(self, tmp_db):
        class Cfg(_Cfg):
            ALERT_CONSECUTIVE_FAILURES = 0
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        for i in range(10):
            _seed(tmp_db, [{
                "agent_name": "A", "success": False,
                "timestamp": _iso(now - timedelta(minutes=10 - i)),
            }])
        alerts = alerter.evaluate_alerts(config=Cfg, db_path=tmp_db, now=now)
        assert not any(a.rule == "consecutive_failures" for a in alerts)


# ---------------------------------------------------------------------------
# Rule: latency spike
# ---------------------------------------------------------------------------

class TestLatencySpike:
    def test_fires_when_p95_above_threshold(self, tmp_db):
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        # 15 fast + 5 slow. Nearest-rank P95 of n=20 = position 19 (1-indexed).
        # After sort, positions 16-20 are the 5 slow values → P95 = 40_000.
        for ms in [1000.0] * 15 + [40_000.0] * 5:
            _seed(tmp_db, [{
                "agent_name": "A", "success": True, "latency_ms": ms,
                "timestamp": _iso(now - timedelta(minutes=5)),
            }])
        alerts = alerter.evaluate_alerts(config=_Cfg, db_path=tmp_db, now=now)
        latency = [a for a in alerts if a.rule == "latency_p95"]
        assert latency and latency[0].scope == "A"
        assert latency[0].details["p95_ms"] >= 15_000.0

    def test_silent_when_p95_under_threshold(self, tmp_db):
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        for _ in range(20):
            _seed(tmp_db, [{
                "agent_name": "A", "success": True, "latency_ms": 1500.0,
                "timestamp": _iso(now - timedelta(minutes=5)),
            }])
        alerts = alerter.evaluate_alerts(config=_Cfg, db_path=tmp_db, now=now)
        assert not any(a.rule == "latency_p95" for a in alerts)

    def test_silent_below_min_samples(self, tmp_db):
        # 5 spiky calls but min_samples = 10 → don't alert
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        for _ in range(5):
            _seed(tmp_db, [{
                "agent_name": "A", "success": True, "latency_ms": 30_000.0,
                "timestamp": _iso(now - timedelta(minutes=5)),
            }])
        alerts = alerter.evaluate_alerts(config=_Cfg, db_path=tmp_db, now=now)
        assert not any(a.rule == "latency_p95" for a in alerts)

    def test_window_excludes_old_samples(self, tmp_db):
        # 10 slow calls 2 hours ago + window=60min → outside window
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        for _ in range(20):
            _seed(tmp_db, [{
                "agent_name": "A", "success": True, "latency_ms": 30_000.0,
                "timestamp": _iso(now - timedelta(hours=2)),
            }])
        alerts = alerter.evaluate_alerts(config=_Cfg, db_path=tmp_db, now=now)
        assert not any(a.rule == "latency_p95" for a in alerts)


# ---------------------------------------------------------------------------
# Rule: cost budget
# ---------------------------------------------------------------------------

class TestCostBudget:
    def test_fires_when_over_budget(self, tmp_db):
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        _seed(tmp_db, [{
            "cost_usd": 2.5,
            "timestamp": _iso(now - timedelta(hours=3)),
        }])
        alerts = alerter.evaluate_alerts(config=_Cfg, db_path=tmp_db, now=now)
        cost = [a for a in alerts if a.rule == "cost_budget"]
        assert cost and cost[0].details["spent_usd"] == pytest.approx(2.5)

    def test_silent_under_budget(self, tmp_db):
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        _seed(tmp_db, [{
            "cost_usd": 0.10,
            "timestamp": _iso(now - timedelta(hours=3)),
        }])
        alerts = alerter.evaluate_alerts(config=_Cfg, db_path=tmp_db, now=now)
        assert not any(a.rule == "cost_budget" for a in alerts)


# ---------------------------------------------------------------------------
# Firing + dedup
# ---------------------------------------------------------------------------

class TestFireAndDedup:
    def _seed_failures(self, tmp_db, now, agent="A", n=5):
        for i in range(n):
            _seed(tmp_db, [{
                "agent_name": agent, "success": False,
                "error_message": "boom",
                "timestamp": _iso(now - timedelta(minutes=10 - i)),
                "latency_ms": 100.0,
            }])

    def test_first_fire_calls_notifier(self, tmp_db):
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        self._seed_failures(tmp_db, now)
        notif = _Notifier()
        result = alerter.check_and_fire_alerts(
            config=_Cfg, db_path=tmp_db, notify_fn=notif, now=now,
        )
        assert result["fired"] == 1
        assert result["suppressed"] == 0
        assert len(notif.messages) == 1
        assert "thesis_generator" not in notif.messages[0]  # we used agent "A"
        assert "A" in notif.messages[0]

    def test_second_fire_within_cooldown_is_suppressed(self, tmp_db):
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        self._seed_failures(tmp_db, now)
        notif = _Notifier()
        # First check fires it.
        alerter.check_and_fire_alerts(
            config=_Cfg, db_path=tmp_db, notify_fn=notif, now=now,
        )
        # Second check 5 minutes later — still inside the 60min cooldown.
        result = alerter.check_and_fire_alerts(
            config=_Cfg, db_path=tmp_db, notify_fn=notif,
            now=now + timedelta(minutes=5),
        )
        assert result["fired"] == 0
        assert result["suppressed"] == 1
        assert len(notif.messages) == 1  # unchanged

    def test_fire_after_cooldown_expires(self, tmp_db):
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        self._seed_failures(tmp_db, now)
        notif = _Notifier()
        alerter.check_and_fire_alerts(
            config=_Cfg, db_path=tmp_db, notify_fn=notif, now=now,
        )
        # 61 minutes later → cooldown expired, fires again.
        result = alerter.check_and_fire_alerts(
            config=_Cfg, db_path=tmp_db, notify_fn=notif,
            now=now + timedelta(minutes=61),
        )
        assert result["fired"] == 1
        assert len(notif.messages) == 2

    def test_alert_history_table_grows(self, tmp_db):
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        self._seed_failures(tmp_db, now)
        alerter.check_and_fire_alerts(
            config=_Cfg, db_path=tmp_db, notify_fn=_Notifier(), now=now,
        )
        with sqlite3.connect(tmp_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM alert_history").fetchone()[0]
        assert count == 1

    def test_notifier_failure_does_not_break_check(self, tmp_db):
        # If the notifier raises, the check still completes and records history.
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        self._seed_failures(tmp_db, now)

        def _boom(text, **kw):
            raise ConnectionError("slack 500")

        result = alerter.check_and_fire_alerts(
            config=_Cfg, db_path=tmp_db, notify_fn=_boom, now=now,
        )
        assert result["fired"] == 1
        # History was still recorded, so dedup will work next round.
        with sqlite3.connect(tmp_db) as conn:
            n = conn.execute("SELECT COUNT(*) FROM alert_history").fetchone()[0]
        assert n == 1


# ---------------------------------------------------------------------------
# Empty / corrupt DB
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_db_no_alerts(self, tmp_db):
        now = datetime(2026, 5, 28, 12, 0, tzinfo=timezone.utc)
        assert alerter.evaluate_alerts(config=_Cfg, db_path=tmp_db, now=now) == []

    def test_missing_db_path_does_not_crash(self):
        # Point at a path that doesn't exist; the alerter must return [].
        out = alerter.evaluate_alerts(config=_Cfg, db_path="/tmp/nope.db")
        assert out == []

    def test_dedup_key_format(self):
        a = alerter.Alert(rule="consecutive_failures", scope="X",
                          severity="critical", message="m")
        assert a.dedup_key == "consecutive_failures:X"