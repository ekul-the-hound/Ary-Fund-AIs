"""
Metrics Database
================
Operational telemetry store for the agent layer, kept deliberately
**separate** from ``portfolio.db``.

Why a separate database?
------------------------
``portfolio.db`` holds *state the system reasons about* — positions,
trades, agent opinions, risk scores. ``metrics.db`` holds *telemetry
about the system itself* — how many tokens a call burned, how long it
took, whether it failed. Mixing the two would:

* couple a high-write-rate operational table (one row per LLM call)
  to the portfolio schema, inflating its size and backup footprint;
* risk a telemetry migration accidentally touching portfolio tables;
* make it harder to wipe/rotate telemetry independently.

The two never join, so there is no cost to keeping them apart. This
module is purely additive: nothing else in the system reads from it
except the Streamlit metrics tab and the optional alerting hook.

Design
------
* Pure functional API (no class) — matches the module-level functional
  style ``main.py`` already uses for ``save_agent_opinion``, and keeps
  monkeypatching trivial in tests.
* ``insert_metric`` is forgiving: unknown keys are ignored, missing
  keys default to NULL. Telemetry must never raise into the call path
  it is measuring.
* Every public reader accepts ``db_path`` so tests can point at a temp
  file; it defaults to ``config.METRICS_DB_PATH`` when omitted.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default DB path resolution
# ---------------------------------------------------------------------------
# Resolved lazily (not at import) so tests can run without a fully-formed
# config, and so a missing config attribute degrades to a local file
# rather than an ImportError.

def _default_db_path() -> str:
    try:
        import config  # type: ignore

        path = getattr(config, "METRICS_DB_PATH", None)
        if path:
            return str(path)
    except Exception:  # noqa: BLE001 — config may be absent in some contexts
        pass
    return "metrics.db"


# Columns we persist. Keeping this list in one place lets ``insert_metric``
# filter unknown keys and lets the reader build SELECTs without drift.
_COLUMNS = (
    "timestamp",
    "agent_name",
    "model",
    "ticker",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_ms",
    "success",
    "error_message",
    "cost_usd",
    "run_id",
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def create_metrics_table(db_path: Optional[str] = None) -> None:
    """Create the ``agent_metrics`` table if absent. Idempotent.

    Also creates two indexes — by timestamp and by agent_name — because
    every dashboard query filters on a time window and most also group
    by agent. Without them the table degrades to a full scan once it
    grows past a few thousand rows (which, at one row per LLM call,
    happens fast).
    """
    db_path = db_path or _default_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_metrics (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT NOT NULL,
                agent_name       TEXT,
                model            TEXT,
                ticker           TEXT,
                prompt_tokens    INTEGER,
                completion_tokens INTEGER,
                total_tokens     INTEGER,
                latency_ms       REAL,
                success          INTEGER,
                error_message    TEXT,
                cost_usd         REAL,
                run_id           TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_ts "
            "ON agent_metrics(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_agent "
            "ON agent_metrics(agent_name)"
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def insert_metric(record: dict, db_path: Optional[str] = None) -> int:
    """Insert one telemetry row. Returns the new row id (or -1 on failure).

    Forgiving by design: keys not in ``_COLUMNS`` are dropped, absent
    columns become NULL. ``timestamp`` defaults to now (UTC, ISO) if the
    caller didn't supply one. This function NEVER raises — telemetry must
    not break the call it is measuring; on error it logs and returns -1.
    """
    db_path = db_path or _default_db_path()
    row = {k: record.get(k) for k in _COLUMNS}
    if not row.get("timestamp"):
        row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Normalise the bool→int for SQLite (which has no native bool).
    if row.get("success") is not None:
        row["success"] = int(bool(row["success"]))

    cols = ", ".join(_COLUMNS)
    placeholders = ", ".join("?" for _ in _COLUMNS)
    values = [row[c] for c in _COLUMNS]

    try:
        create_metrics_table(db_path)
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                f"INSERT INTO agent_metrics ({cols}) VALUES ({placeholders})",
                values,
            )
            conn.commit()
            return int(cur.lastrowid or -1)
    except Exception as e:  # noqa: BLE001 — telemetry must be non-fatal
        logger.warning("metrics insert failed (non-fatal): %s", e)
        return -1


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def _cutoff_iso(since_days: int) -> str:
    """ISO timestamp ``since_days`` ago, UTC. Used as a ``timestamp >= ?``
    lower bound."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    return cutoff.isoformat(timespec="seconds")


def get_metrics(
    agent_name: Optional[str] = None,
    since_days: int = 7,
    limit: int = 1000,
    db_path: Optional[str] = None,
) -> list[dict]:
    """Return recent metric rows as dicts, newest first.

    Filters to the last ``since_days`` and optionally to one agent.
    Returns ``[]`` (never raises) if the table doesn't exist yet.
    """
    db_path = db_path or _default_db_path()
    where = ["timestamp >= ?"]
    params: list[Any] = [_cutoff_iso(since_days)]
    if agent_name:
        where.append("agent_name = ?")
        params.append(agent_name)
    params.append(int(limit))

    sql = (
        "SELECT id, " + ", ".join(_COLUMNS) + " FROM agent_metrics "
        "WHERE " + " AND ".join(where) +
        " ORDER BY timestamp DESC LIMIT ?"
    )
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        # Table not created yet — treat as empty.
        return []


def get_token_spent_since_days(
    days: int = 7,
    db_path: Optional[str] = None,
) -> float:
    """Total tokens (prompt + completion) consumed in the window.

    Returns a float (SQLite SUM yields a numeric / None). 0.0 when no
    rows or no table.
    """
    db_path = db_path or _default_db_path()
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) "
                "FROM agent_metrics WHERE timestamp >= ?",
                (_cutoff_iso(days),),
            ).fetchone()
            return float(row[0] or 0.0)
    except sqlite3.OperationalError:
        return 0.0


def get_cost_spent_since_days(
    days: int = 7,
    db_path: Optional[str] = None,
) -> float:
    """Total cost_usd in the window. Companion to token spend."""
    db_path = db_path or _default_db_path()
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) "
                "FROM agent_metrics WHERE timestamp >= ?",
                (_cutoff_iso(days),),
            ).fetchone()
            return float(row[0] or 0.0)
    except sqlite3.OperationalError:
        return 0.0


def _percentile(sorted_vals: list[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile on a pre-sorted list.

    Nearest-rank (rather than linear interpolation) is chosen because
    latency distributions are spiky and small-N; interpolation invents
    values between real samples, which is misleading when you have, say,
    8 calls. With nearest-rank, every reported P99 is an actual observed
    latency.
    """
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    # rank in [1, n]; ceil(pct * n)
    import math

    rank = max(1, math.ceil((pct / 100.0) * len(sorted_vals)))
    return float(sorted_vals[min(rank, len(sorted_vals)) - 1])


def get_latency_stats(
    agent_name: Optional[str] = None,
    since_days: int = 7,
    db_path: Optional[str] = None,
) -> dict:
    """Return ``{count, p50, p90, p99, mean}`` for latency_ms in the window.

    Percentiles computed in Python (nearest-rank) rather than in SQL
    because SQLite has no native percentile function and the row counts
    here (hundreds to low thousands) make an in-memory sort trivial. All
    percentile keys are ``None`` when there are no rows.
    """
    db_path = db_path or _default_db_path()
    where = ["timestamp >= ?", "latency_ms IS NOT NULL"]
    params: list[Any] = [_cutoff_iso(since_days)]
    if agent_name:
        where.append("agent_name = ?")
        params.append(agent_name)

    sql = (
        "SELECT latency_ms FROM agent_metrics WHERE "
        + " AND ".join(where)
        + " ORDER BY latency_ms ASC"
    )
    try:
        with sqlite3.connect(db_path) as conn:
            vals = [float(r[0]) for r in conn.execute(sql, params).fetchall()]
    except sqlite3.OperationalError:
        vals = []

    if not vals:
        return {"count": 0, "p50": None, "p90": None, "p99": None, "mean": None}

    return {
        "count": len(vals),
        "p50": _percentile(vals, 50),
        "p90": _percentile(vals, 90),
        "p99": _percentile(vals, 99),
        "mean": sum(vals) / len(vals),
    }


def get_success_rate_by_agent(
    since_days: int = 7,
    db_path: Optional[str] = None,
) -> dict[str, dict]:
    """Return ``{agent_name: {total, ok, rate}}`` over the window.

    ``rate`` is in [0, 1]. Rows with NULL agent_name are bucketed under
    the literal string ``"(unlabeled)"`` so they're visible rather than
    silently dropped — an unlabeled call is itself a finding (someone
    forgot to pass ``agent_name``).
    """
    db_path = db_path or _default_db_path()
    sql = (
        "SELECT COALESCE(agent_name, '(unlabeled)') AS a, "
        "COUNT(*) AS total, "
        "COALESCE(SUM(success), 0) AS ok "
        "FROM agent_metrics WHERE timestamp >= ? GROUP BY a"
    )
    out: dict[str, dict] = {}
    try:
        with sqlite3.connect(db_path) as conn:
            for a, total, ok in conn.execute(sql, (_cutoff_iso(since_days),)):
                total = int(total or 0)
                ok = int(ok or 0)
                out[a] = {
                    "total": total,
                    "ok": ok,
                    "rate": (ok / total) if total else 0.0,
                }
    except sqlite3.OperationalError:
        return {}
    return out
