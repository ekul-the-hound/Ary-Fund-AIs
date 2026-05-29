"""
agent/metrics_alerts.py
=======================
Threshold-based alerter for the agent telemetry layer.

What this does
--------------
Reads recent rows from ``metrics.db`` and fires alerts when any of these
fail:

1. **Consecutive failures** — the last N calls for an agent are all
   ``success=0``. Catches: backend down, model crash, prompt template
   broken.

2. **Latency spike** — recent P95 latency exceeds a threshold. Catches:
   model swap to a slower variant, GPU thrashing, network degradation.

3. **Cost burn** — cost in the last window exceeds a budget. Catches:
   runaway loops, unexpectedly large prompts.

Where the design lives
----------------------
* **Detection is a pure function** of the rows in ``metrics.db``.
  ``evaluate_alerts(...)`` reads the DB and returns ``list[Alert]``
  without firing anything. Easy to test, easy to dry-run.
* **Firing routes through** ``data.notifiers.notify_slack``, which is
  already the project's "send a thing somewhere" boundary. We don't
  introduce a new notification channel.
* **Dedup is persistent**: an ``alert_history`` table in ``metrics.db``
  records the last time each ``(rule, scope)`` fired. A repeat alert
  within ``cooldown_minutes`` is suppressed. This makes
  ``check_and_fire_alerts`` safe to call repeatedly (e.g. from the
  refresh scheduler) without spamming Slack.

What this is NOT
----------------
* Not a generic monitoring system. No PagerDuty, no SMS, no on-call
  rotation. If you outgrow this, swap ``notify_slack`` for a real
  client.
* Not invoked from inside ``instrumented_ask``. Alerting is a *batch*
  operation — running it per-call would put DB reads on the critical
  path of every LLM request. It belongs in the scheduler or a manual
  `python -m agent.metrics_alerts check` invocation.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from data import metrics_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
# Pulled from ``config`` when present, else these. Centralising them
# here means a user can tune any single knob in config.py without
# editing this module.

_DEFAULTS = {
    "ALERT_CONSECUTIVE_FAILURES": 5,        # N consecutive failures
    "ALERT_LATENCY_P95_MS": 15_000.0,        # ms — qwen3:30b runs ~3.5s, so 15s = clearly degraded
    "ALERT_LATENCY_WINDOW_MIN": 60,          # only consider the last hour for latency
    "ALERT_COST_BUDGET_USD": 1.0,            # USD / day notional
    "ALERT_COST_WINDOW_HOURS": 24,
    "ALERT_COOLDOWN_MINUTES": 60,            # dedup window
    "ALERT_MIN_SAMPLES_FOR_LATENCY": 10,     # don't fire latency alerts on <10 calls
}


def _cfg(config: Any, key: str) -> Any:
    if config is None:
        return _DEFAULTS[key]
    return getattr(config, key, _DEFAULTS[key])


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """One detected threshold breach.

    ``rule`` is the rule name (``consecutive_failures`` /
    ``latency_p95`` / ``cost_budget``). ``scope`` further identifies the
    target (agent name for failure/latency rules, ``"global"`` for cost).
    ``(rule, scope)`` is the dedup key — repeat firings within the
    cooldown for the same pair are suppressed.

    ``message`` is the human-readable Slack body. ``details`` carries
    the raw numbers for logs and tests.
    """

    rule: str
    scope: str
    severity: str            # "warn" | "critical"
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def dedup_key(self) -> str:
        return f"{self.rule}:{self.scope}"


# ---------------------------------------------------------------------------
# Dedup history table
# ---------------------------------------------------------------------------
# Lives in the same metrics.db (it's operational telemetry too). The
# table is created lazily so a fresh install doesn't need a migration
# step.

def _ensure_alert_history(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                dedup_key   TEXT NOT NULL,
                rule        TEXT NOT NULL,
                scope       TEXT NOT NULL,
                severity    TEXT,
                fired_at    TEXT NOT NULL,
                message     TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alert_history_key_ts "
            "ON alert_history(dedup_key, fired_at)"
        )
        conn.commit()


def _last_fired(dedup_key: str, db_path: str) -> Optional[datetime]:
    """Return the most recent fired_at for this dedup key, or None."""
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT fired_at FROM alert_history "
                "WHERE dedup_key = ? ORDER BY fired_at DESC LIMIT 1",
                (dedup_key,),
            ).fetchone()
            if not row or not row[0]:
                return None
            return datetime.fromisoformat(row[0])
    except (sqlite3.OperationalError, ValueError):
        return None


def _record_firing(alert: Alert, db_path: str, now: datetime) -> None:
    _ensure_alert_history(db_path)
    ts = now.isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO alert_history "
            "(dedup_key, rule, scope, severity, fired_at, message) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (alert.dedup_key, alert.rule, alert.scope, alert.severity,
             ts, alert.message),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Rule: consecutive failures (per agent)
# ---------------------------------------------------------------------------

def _check_consecutive_failures(
    db_path: str, config: Any, now: datetime,
) -> list[Alert]:
    """Fire when an agent's last N calls are all failures.

    "Last N" is anchored on call recency, not wall-clock — if an agent
    only runs once an hour, we still want to alert after N straight
    failures regardless of how long that took.
    """
    threshold = int(_cfg(config, "ALERT_CONSECUTIVE_FAILURES"))
    if threshold <= 0:
        return []

    alerts: list[Alert] = []
    # Group by agent: pull the last N rows per agent ordered DESC, check
    # all are failures. SQLite window functions would be cleaner but
    # this is small-N and one query per agent is plenty fast.
    try:
        with sqlite3.connect(db_path) as conn:
            agents = conn.execute(
                "SELECT DISTINCT agent_name FROM agent_metrics "
                "WHERE agent_name IS NOT NULL"
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    for (agent,) in agents:
        try:
            with sqlite3.connect(db_path) as conn:
                rows = conn.execute(
                    "SELECT success, error_message, timestamp "
                    "FROM agent_metrics WHERE agent_name = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (agent, threshold),
                ).fetchall()
        except sqlite3.OperationalError:
            continue

        if len(rows) < threshold:
            continue
        if not all((r[0] or 0) == 0 for r in rows):
            continue

        last_err = next((r[1] for r in rows if r[1]), "unknown error")
        alerts.append(Alert(
            rule="consecutive_failures",
            scope=agent,
            severity="critical",
            message=(
                f":rotating_light: *{agent}*: {threshold} consecutive failures. "
                f"Last error: `{last_err}`. Check Ollama, model availability, "
                f"and recent prompt changes."
            ),
            details={
                "agent": agent, "consecutive": threshold,
                "last_error": last_err,
            },
        ))

    return alerts


# ---------------------------------------------------------------------------
# Rule: latency spike (per agent, last hour)
# ---------------------------------------------------------------------------

def _check_latency_spike(
    db_path: str, config: Any, now: datetime,
) -> list[Alert]:
    """Fire when an agent's P95 latency over the window exceeds the threshold.

    P95 rather than mean: means hide tail spikes (one bad call pulls the
    average up but you'd see it). P95 rather than P99: P99 on small
    sample sizes is too noisy to alert on without flapping.
    """
    threshold_ms = float(_cfg(config, "ALERT_LATENCY_P95_MS"))
    window_min = int(_cfg(config, "ALERT_LATENCY_WINDOW_MIN"))
    min_samples = int(_cfg(config, "ALERT_MIN_SAMPLES_FOR_LATENCY"))

    if threshold_ms <= 0 or window_min <= 0:
        return []

    cutoff_iso = (now - timedelta(minutes=window_min)).isoformat(timespec="seconds")
    alerts: list[Alert] = []

    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT COALESCE(agent_name, '(unlabeled)'), latency_ms "
                "FROM agent_metrics "
                "WHERE timestamp >= ? AND latency_ms IS NOT NULL",
                (cutoff_iso,),
            ).fetchall()
    except sqlite3.OperationalError:
        return []

    by_agent: dict[str, list[float]] = {}
    for name, ms in rows:
        by_agent.setdefault(name, []).append(float(ms))

    for agent, samples in by_agent.items():
        if len(samples) < min_samples:
            continue
        samples.sort()
        # Reuse the same nearest-rank percentile semantics as metrics_db
        import math

        rank = max(1, math.ceil(0.95 * len(samples)))
        p95 = samples[min(rank, len(samples)) - 1]
        if p95 <= threshold_ms:
            continue
        alerts.append(Alert(
            rule="latency_p95",
            scope=agent,
            severity="warn",
            message=(
                f":turtle: *{agent}*: P95 latency = {p95:,.0f} ms "
                f"over the last {window_min} min "
                f"(threshold {threshold_ms:,.0f} ms, n={len(samples)})."
            ),
            details={
                "agent": agent, "p95_ms": p95, "threshold_ms": threshold_ms,
                "samples": len(samples), "window_min": window_min,
            },
        ))

    return alerts


# ---------------------------------------------------------------------------
# Rule: cost budget (global, last 24h)
# ---------------------------------------------------------------------------

def _check_cost_budget(
    db_path: str, config: Any, now: datetime,
) -> list[Alert]:
    """Fire when total cost over the window exceeds the daily budget."""
    budget = float(_cfg(config, "ALERT_COST_BUDGET_USD"))
    window_h = int(_cfg(config, "ALERT_COST_WINDOW_HOURS"))
    if budget <= 0 or window_h <= 0:
        return []

    cutoff_iso = (now - timedelta(hours=window_h)).isoformat(timespec="seconds")
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) "
                "FROM agent_metrics WHERE timestamp >= ?",
                (cutoff_iso,),
            ).fetchone()
            spent = float(row[0] or 0.0)
    except sqlite3.OperationalError:
        return []

    if spent <= budget:
        return []

    return [Alert(
        rule="cost_budget",
        scope="global",
        severity="warn",
        message=(
            f":money_with_wings: Notional cost = ${spent:,.4f} over the last "
            f"{window_h}h (budget ${budget:,.4f}). Check for runaway loops "
            f"or oversized prompts."
        ),
        details={"spent_usd": spent, "budget_usd": budget,
                 "window_hours": window_h},
    )]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_alerts(
    config: Any = None,
    db_path: Optional[str] = None,
    now: Optional[datetime] = None,
) -> list[Alert]:
    """Run every rule and return the alerts that would fire.

    Pure function on the DB contents: does not write, does not send.
    The ``now`` kwarg is injectable for testability (so tests can
    simulate "now" relative to seeded timestamps).
    """
    db_path = db_path or _resolve_db_path(config)
    now = now or datetime.now(timezone.utc)

    alerts: list[Alert] = []
    alerts.extend(_check_consecutive_failures(db_path, config, now))
    alerts.extend(_check_latency_spike(db_path, config, now))
    alerts.extend(_check_cost_budget(db_path, config, now))
    return alerts


def check_and_fire_alerts(
    config: Any = None,
    db_path: Optional[str] = None,
    notify_fn: Any = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Evaluate rules, dedup against history, fire surviving alerts.

    Returns a summary dict: ``{"evaluated": N, "suppressed": M, "fired": K,
    "alerts": [...]}``.

    ``notify_fn`` is injectable for tests; defaults to
    ``notifiers.notify_slack`` (which itself returns False quietly when
    no webhook is configured, so this is safe to call on a laptop).
    """
    db_path = db_path or _resolve_db_path(config)
    now = now or datetime.now(timezone.utc)
    cooldown = timedelta(minutes=int(_cfg(config, "ALERT_COOLDOWN_MINUTES")))

    if notify_fn is None:
        try:
            from data.notifiers import notify_slack as _ns
            notify_fn = _ns
        except Exception:  # noqa: BLE001 — keep the alerter usable offline
            notify_fn = lambda text, **kw: False  # noqa: E731

    candidates = evaluate_alerts(config=config, db_path=db_path, now=now)

    fired: list[Alert] = []
    suppressed = 0
    for alert in candidates:
        last = _last_fired(alert.dedup_key, db_path)
        if last is not None and (now - last) < cooldown:
            suppressed += 1
            logger.info(
                "alerts | suppress %s (last fired %s ago < cooldown %s)",
                alert.dedup_key, now - last, cooldown,
            )
            continue
        try:
            notify_fn(alert.message)
        except Exception as e:  # noqa: BLE001
            logger.warning("alerts | notify_fn raised (non-fatal): %s", e)
        _record_firing(alert, db_path, now)
        fired.append(alert)

    return {
        "evaluated": len(candidates),
        "suppressed": suppressed,
        "fired": len(fired),
        "alerts": [a.dedup_key for a in fired],
    }


# ---------------------------------------------------------------------------
# Internal: resolve db path with the same precedence as metrics_db
# ---------------------------------------------------------------------------

def _resolve_db_path(config: Any) -> str:
    if config is not None:
        path = getattr(config, "METRICS_DB_PATH", None)
        if path:
            return str(path)
    # Fall back to metrics_db's own default
    return metrics_db._default_db_path()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# ``python -m agent.metrics_alerts check`` — convenient for a cron job
# or for ad-hoc validation. Outputs JSON so it composes cleanly with
# downstream shell tooling.

if __name__ == "__main__":  # pragma: no cover - exercised by humans
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(description="Agent metrics alerter")
    parser.add_argument(
        "command", choices=["check", "dry-run"],
        help="check = evaluate and fire; dry-run = evaluate only.",
    )
    parser.add_argument("--db", default=None, help="Path to metrics.db")
    args = parser.parse_args()

    try:
        import config as _config
    except Exception:  # noqa: BLE001
        _config = None

    if args.command == "dry-run":
        out = [a.__dict__ for a in evaluate_alerts(
            config=_config, db_path=args.db,
        )]
    else:
        out = check_and_fire_alerts(config=_config, db_path=args.db)
    print(_json.dumps(out, indent=2, default=str))
