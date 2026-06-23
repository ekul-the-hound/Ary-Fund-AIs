"""
ui/state.py
===========

The spine of the ARY QUANT v2 dashboard: session-state contracts, stage
tracking, and a non-blocking job queue for long LLM runs.

Three responsibilities
-----------------------
1. ACTIVE-TICKER CONTRACT
   A thin, well-documented wrapper around the single source of truth that
   the *existing* dashboard already uses: ``st.session_state["active_ticker"]``.
   The screener writes it on row-click; the sidebar writes it via on_change;
   every view reads it. v2 keeps that exact contract so the proven handoff
   never breaks — these accessors just make call sites readable and give us
   one place to add validation/normalization.

2. STAGE TRACKING (Research Pipeline funnel)
   Per-ticker completion state for the Screen -> Analyze -> Memo -> Review
   -> Decision funnel. Crucially this is DERIVED FROM REAL PERSISTED STATE
   (portfolio.db), not a UI-local guess: a stage is "done" if its artifact
   actually exists (an opinion row, a memo, a review score, a recorded
   thesis). So the rail reflects work that genuinely happened, even across
   restarts.

3. JOB QUEUE (non-blocking LLM runs)
   The current UI blocks for 30s-2min on a synchronous spinner every time
   you generate an opinion or memo. This queue runs that work on a thread
   pool and lets the UI poll for completion, so an analyst can fire
   ``gen NVDA`` and ``gen AMD`` and keep reading.

   CRITICAL STREAMLIT CONSTRAINT
   -----------------------------
   Background threads in Streamlit must NEVER call ``st.*`` or mutate
   ``st.session_state`` — once the script run that spawned them ends, those
   threads have no ScriptRunContext and any st call warns/misbehaves. So:

       * Worker functions here are pure backend calls (main._process_ticker,
         essay generation, pdf rendering). They take plain args and return
         plain dicts. They do NOT import or touch streamlit.
       * Results land in a module-level, thread-safe ``_JOB_REGISTRY``
         keyed by job id, guarded by a Lock.
       * The UI polls ``poll_jobs()`` / ``get_job()`` on each normal rerun
         and reflects status. A lightweight auto-refresh (st.rerun on a
         timer, owned by the caller) drives the polling cadence.

   This is the only pattern that is safe under Streamlit's execution model;
   a naive ``threading.Thread`` that calls ``st.rerun()`` from inside the
   worker will throw NoSessionContext and can corrupt session state.

This module imports streamlit (for the session-state accessors used on the
MAIN thread only) but the WORKER side is import-light and st-free. The job
registry lives at module scope so it is shared across reruns within a single
server process (Streamlit reuses the process; session_state is per-session,
the registry is per-process — that's intentional, jobs are global).
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import streamlit as st

logger = logging.getLogger("ary_quant.ui.state")


# ======================================================================
# 1. ACTIVE-TICKER CONTRACT
# ======================================================================
# The key name MUST match what ui/screener.py and the sidebar already
# write. Do not rename it — the whole cross-view handoff depends on it.
ACTIVE_TICKER_KEY = "active_ticker"
_DEFAULT_TICKER = "NVDA"


def init_active_ticker(default: str = _DEFAULT_TICKER) -> None:
    """Ensure the active-ticker key exists before any widget reads it.

    Call once at the very top of the app, BEFORE building any widget that
    references the active ticker, to avoid the 'value not yet in state on
    first render' race that the original app.py documents.
    """
    if ACTIVE_TICKER_KEY not in st.session_state:
        st.session_state[ACTIVE_TICKER_KEY] = (default or _DEFAULT_TICKER).upper()


def get_active_ticker(default: str = _DEFAULT_TICKER) -> str:
    """Read the current active ticker, normalized to upper-case."""
    val = st.session_state.get(ACTIVE_TICKER_KEY) or default
    return str(val).strip().upper()


def set_active_ticker(ticker: str) -> bool:
    """Set the active ticker (used by screener row-clicks, palette, nav).

    Returns True if the value actually changed (so callers can decide
    whether a rerun is warranted). Normalizes to upper-case and ignores
    empty input. This mirrors exactly what the screener does today —
    write the key, then the caller triggers a rerun — so existing
    behavior is preserved.
    """
    new = (ticker or "").strip().upper()
    if not new:
        return False
    old = st.session_state.get(ACTIVE_TICKER_KEY)
    st.session_state[ACTIVE_TICKER_KEY] = new
    return new != old


# ======================================================================
# 2. STAGE TRACKING
# ======================================================================
class Stage(str, Enum):
    """The Research Pipeline funnel stages, in order."""
    SCREEN = "screen"
    ANALYZE = "analyze"
    MEMO = "memo"
    REVIEW = "review"
    DECISION = "decision"


STAGE_ORDER: list[Stage] = [
    Stage.SCREEN, Stage.ANALYZE, Stage.MEMO, Stage.REVIEW, Stage.DECISION,
]

STAGE_LABELS: dict[Stage, str] = {
    Stage.SCREEN: "Screen",
    Stage.ANALYZE: "Analyze",
    Stage.MEMO: "Memo",
    Stage.REVIEW: "Review",
    Stage.DECISION: "Decision",
}


@dataclass
class StageStatus:
    """Per-ticker funnel state, derived from persisted artifacts.

    ``done`` flags which stages have a real artifact backing them.
    ``review_score`` (if present) is surfaced on the rail so a weak memo is
    visible without opening it. ``held`` reflects whether the ticker is an
    open position (Decision stage essentially complete).
    """
    ticker: str
    done: dict[Stage, bool] = field(default_factory=dict)
    review_score: Optional[float] = None
    held: bool = False

    def is_done(self, stage: Stage) -> bool:
        return bool(self.done.get(stage, False))

    def furthest_done(self) -> Optional[Stage]:
        """The last stage (in order) that is complete, or None."""
        last: Optional[Stage] = None
        for s in STAGE_ORDER:
            if self.done.get(s):
                last = s
        return last


def compute_stage_status(
    ticker: str,
    *,
    opinion: Optional[dict[str, Any]] = None,
    held: bool = False,
    has_open_thesis: bool = False,
) -> StageStatus:
    """Derive funnel completion from a persisted opinion dict + portfolio flags.

    This is intentionally PURE (no DB calls): the caller fetches the latest
    opinion (via the app's cached ``load_latest_opinion``) and the position
    info (via portfolio_db) and passes them in. Keeping I/O out of here makes
    it unit-testable and keeps the funnel logic in one obvious place.

    Stage semantics
    ---------------
    * SCREEN   : always considered reachable/done once we're looking at the
                 name — screening is how you got here. Marked done.
    * ANALYZE  : an agent opinion exists (outlook/confidence/risk computed).
    * MEMO     : the opinion carries a non-empty essay.
    * REVIEW   : the opinion carries a review with a numeric overall score.
    * DECISION : the ticker is held OR has an open recorded thesis.

    Absent inputs degrade gracefully to "not done" for that stage.
    """
    opinion = opinion or {}
    done: dict[Stage, bool] = {s: False for s in STAGE_ORDER}

    # Screen: reaching this name means it cleared/entered the funnel.
    done[Stage.SCREEN] = True

    # Analyze: a real opinion with at least an outlook or confidence.
    has_opinion = bool(
        opinion.get("outlook")
        or opinion.get("confidence") is not None
        or opinion.get("risk_flags")
    )
    done[Stage.ANALYZE] = has_opinion

    # Memo: a non-empty essay string.
    essay = opinion.get("essay")
    done[Stage.MEMO] = bool(isinstance(essay, str) and essay.strip())

    # Review: a numeric overall score.
    review = opinion.get("review") or {}
    review_scores = review.get("scores") or {}
    overall = review_scores.get("overall")
    review_score = float(overall) if isinstance(overall, (int, float)) else None
    done[Stage.REVIEW] = review_score is not None

    # Decision: held position or an open thesis on the books.
    done[Stage.DECISION] = bool(held or has_open_thesis)

    return StageStatus(
        ticker=ticker.upper(),
        done=done,
        review_score=review_score,
        held=held,
    )


# ======================================================================
# 3. JOB QUEUE
# ======================================================================
class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    """A single background unit of work.

    ``kind`` is a short label ('opinion', 'memo', 'report') used for the
    tray UI. ``ticker`` is the subject. ``result`` holds the worker's
    return value on success; ``error`` holds the exception text on failure.
    Timestamps support a simple elapsed display.
    """
    id: str
    kind: str
    ticker: str
    state: JobState = JobState.QUEUED
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    label: str = ""

    def elapsed(self) -> float:
        """Seconds elapsed (running -> now, finished -> total, queued -> 0)."""
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)


# Module-level (per-process) job registry. Shared across reruns and
# sessions within one Streamlit server process. Guarded by a Lock because
# worker threads write to it concurrently with the main thread reading.
_JOB_REGISTRY: dict[str, Job] = {}
_JOB_LOCK = threading.Lock()

# A single shared thread pool. Small by design: local Ollama serves one
# generation at a time anyway (GPU-bound), so 2 workers is plenty and
# avoids hammering the model server. Created lazily so importing this
# module is cheap and side-effect-free.
_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()
_MAX_WORKERS = 2


def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _EXECUTOR is None:
                _EXECUTOR = ThreadPoolExecutor(
                    max_workers=_MAX_WORKERS,
                    thread_name_prefix="ary-job",
                )
    return _EXECUTOR


def _run_job(job_id: str, fn: Callable[..., Any],
             args: tuple, kwargs: dict) -> None:
    """Thread-pool target. Pure backend work — NO streamlit calls here.

    Updates the registry under lock at start, on success, and on failure.
    Any exception is captured as text rather than propagated, so one failed
    job never takes down the pool or the UI. The worker ``fn`` must itself
    be st-free (it runs main._process_ticker, essay generation, etc.).
    """
    # Mark running.
    with _JOB_LOCK:
        job = _JOB_REGISTRY.get(job_id)
        if job is None:
            return
        job.state = JobState.RUNNING
        job.started_at = time.time()

    try:
        result = fn(*args, **kwargs)
        with _JOB_LOCK:
            job = _JOB_REGISTRY.get(job_id)
            if job is not None:
                job.result = result
                job.state = JobState.DONE
                job.finished_at = time.time()
        logger.info("job %s (%s) done in %.1fs", job_id, job.kind, job.elapsed())
    except Exception as exc:  # noqa: BLE001 — must never crash the pool
        tb = traceback.format_exc()
        with _JOB_LOCK:
            job = _JOB_REGISTRY.get(job_id)
            if job is not None:
                job.error = f"{exc}"
                job.state = JobState.ERROR
                job.finished_at = time.time()
        logger.error("job %s (%s) failed: %s\n%s", job_id, job, exc, tb)


def submit_job(kind: str, ticker: str, fn: Callable[..., Any],
               *args: Any, label: str = "", **kwargs: Any) -> str:
    """Submit a background job and return its id.

    ``fn`` is the pure backend callable (e.g. a lambda that calls
    ``main._process_ticker(ticker, db_path, cfg)``). It MUST NOT touch
    streamlit. ``label`` is an optional human string for the tray. The
    returned id is used to poll status via ``get_job`` / ``poll_jobs``.

    De-duplication: if an identical (kind, ticker) job is already QUEUED or
    RUNNING, its id is returned instead of starting a second one — firing
    ``gen NVDA`` twice shouldn't run the chain twice.
    """
    ticker = (ticker or "").strip().upper()

    # Dedup against in-flight jobs.
    with _JOB_LOCK:
        for jid, j in _JOB_REGISTRY.items():
            if (j.kind == kind and j.ticker == ticker
                    and j.state in (JobState.QUEUED, JobState.RUNNING)):
                return jid

    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, kind=kind, ticker=ticker,
              label=label or f"{kind} · {ticker}")
    with _JOB_LOCK:
        _JOB_REGISTRY[job_id] = job

    _get_executor().submit(_run_job, job_id, fn, args, kwargs)
    logger.info("job %s submitted (%s · %s)", job_id, kind, ticker)
    return job_id


def get_job(job_id: str) -> Optional[Job]:
    """Snapshot of a single job, or None if unknown."""
    with _JOB_LOCK:
        return _JOB_REGISTRY.get(job_id)


def poll_jobs(*, kinds: Optional[set[str]] = None,
              ticker: Optional[str] = None,
              include_finished: bool = True) -> list[Job]:
    """Return a snapshot list of jobs, newest first, optionally filtered.

    Safe to call on every rerun — it only reads the registry under lock and
    returns shallow copies of the dataclass references (callers should treat
    them as read-only snapshots). Use ``kinds`` / ``ticker`` to scope to a
    view (e.g. the Desk shows only the active ticker's jobs).
    """
    tkr = (ticker or "").strip().upper() or None
    with _JOB_LOCK:
        jobs = list(_JOB_REGISTRY.values())
    if kinds:
        jobs = [j for j in jobs if j.kind in kinds]
    if tkr:
        jobs = [j for j in jobs if j.ticker == tkr]
    if not include_finished:
        jobs = [j for j in jobs if j.state in (JobState.QUEUED, JobState.RUNNING)]
    jobs.sort(key=lambda j: j.submitted_at, reverse=True)
    return jobs


def has_active_jobs() -> bool:
    """True if any job is QUEUED or RUNNING (drives the auto-refresh timer).

    The app should only burn reruns polling while something is actually in
    flight. When this is False, the tray is static and no auto-refresh is
    needed.
    """
    with _JOB_LOCK:
        return any(
            j.state in (JobState.QUEUED, JobState.RUNNING)
            for j in _JOB_REGISTRY.values()
        )


def clear_finished_jobs() -> int:
    """Remove DONE/ERROR jobs from the registry. Returns count removed.

    Wired to a 'clear' control in the tray so old jobs don't accumulate
    across a long session. In-flight jobs are left untouched.
    """
    removed = 0
    with _JOB_LOCK:
        for jid in list(_JOB_REGISTRY.keys()):
            if _JOB_REGISTRY[jid].state in (JobState.DONE, JobState.ERROR):
                del _JOB_REGISTRY[jid]
                removed += 1
    return removed


def consume_job_result(job_id: str) -> Any:
    """Pop a finished job's result and delete the job. Returns result or None.

    Used when a view wants to apply a completed job's output exactly once
    (e.g. clear caches after an opinion finishes) and then forget it. If the
    job isn't finished, returns None and leaves it in place.
    """
    with _JOB_LOCK:
        job = _JOB_REGISTRY.get(job_id)
        if job is None or job.state not in (JobState.DONE, JobState.ERROR):
            return None
        result = job.result
        del _JOB_REGISTRY[job_id]
    return result


# ======================================================================
# Auto-refresh helper
# ======================================================================
_LAST_AUTOREFRESH_KEY = "_ary_last_autorefresh_ts"


def maybe_autorefresh(*, interval_s: float = 2.0) -> None:
    """Trigger a periodic rerun ONLY while jobs are in flight.

    Streamlit has no built-in polling, so to reflect job progress we rerun
    the script on a timer — but only when something is actually running, to
    avoid needless churn. The cadence is throttled via a session timestamp.

    Call this once near the end of the main render. It is a no-op when no
    jobs are active or when the interval hasn't elapsed.
    """
    if not has_active_jobs():
        return
    now = time.time()
    last = st.session_state.get(_LAST_AUTOREFRESH_KEY, 0.0)
    if now - last >= interval_s:
        st.session_state[_LAST_AUTOREFRESH_KEY] = now
        # Sleep a hair so we don't busy-loop the CPU between reruns; this
        # runs on the main thread and is bounded by interval_s.
        time.sleep(min(0.2, interval_s))
        st.rerun()


__all__ = [
    # active ticker
    "ACTIVE_TICKER_KEY", "init_active_ticker",
    "get_active_ticker", "set_active_ticker",
    # stages
    "Stage", "STAGE_ORDER", "STAGE_LABELS", "StageStatus",
    "compute_stage_status",
    # jobs
    "JobState", "Job", "submit_job", "get_job", "poll_jobs",
    "has_active_jobs", "clear_finished_jobs", "consume_job_result",
    "maybe_autorefresh",
]

# D:\Ary Fund\ui\state.py
