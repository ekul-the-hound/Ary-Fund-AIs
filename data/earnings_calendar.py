"""
earnings_calendar.py
====================
Earnings calendar queries + event-driven refresh trigger logic.

Used by ``thesis_generator`` and ``refresh_scheduler`` to decide when
to re-pull data for a ticker — e.g. an upcoming earnings event in the
next 7 days warrants a fresh transcript / analyst estimate fetch.

Named ``earnings_calendar.py`` (not ``calendar.py``) deliberately to
avoid shadowing Python's stdlib ``calendar`` module on import.

Public API:

    get_upcoming_earnings_week(ticker=None) -> list[dict]
    should_trigger_event_refresh(ticker, lookback_days=7) -> bool
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from data import providers

logger = logging.getLogger(__name__)


# ============================================================================
# Module-level state for last-refresh tracking
# ============================================================================
# ``should_trigger_event_refresh`` needs to know whether we've refreshed
# *since* the most recent earnings event. We keep that as a simple
# in-memory dict; production code should persist this in the
# refresh_log table (refresh_scheduler.py is the right home for that).
_last_refresh_ts: dict[str, datetime] = {}


def _utcnow() -> datetime:
    """timezone-naive UTC now (matches what's stored in registry / logs)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def mark_refreshed(ticker: str, when: Optional[datetime] = None) -> None:
    """Record that ``ticker`` was just refreshed. Call this at the end
    of any pipeline run so the next ``should_trigger_event_refresh``
    call has the right baseline."""
    _last_refresh_ts[ticker.upper()] = when or _utcnow()


def _last_refresh(ticker: str) -> Optional[datetime]:
    return _last_refresh_ts.get(ticker.upper())


# ============================================================================
# Calendar lookups
# ============================================================================

def get_upcoming_earnings_week(
    ticker: Optional[str] = None,
    days_ahead: int = 7,
) -> list[dict]:
    """Earnings events in the next ``days_ahead`` days.

    If ``ticker`` is provided, filter the calendar to that symbol only;
    otherwise return all events in the window (useful for the morning
    briefing's "this week's earnings" panel).

    Each dict: ``{ticker, date, epsEstimate, period}``. Returns ``[]``
    on any failure — never raises.
    """
    today = _utcnow().date()
    end = today + timedelta(days=days_ahead)
    from_str = today.isoformat()
    to_str = end.isoformat()

    try:
        events = providers.get_earnings_events(from_str, to_str)
    except Exception as e:  # noqa: BLE001
        logger.error("earnings_calendar | upcoming fetch failed: %s", e)
        return []

    if not events:
        return []

    if ticker is not None:
        tk = ticker.upper()
        events = [e for e in events if (e.get("ticker") or "").upper() == tk]

    # Trim to the public contract: {ticker, date, epsEstimate, period}
    out = []
    for e in events:
        out.append({
            "ticker": e.get("ticker"),
            "date": e.get("date"),
            "epsEstimate": e.get("epsEstimate"),
            "period": e.get("period"),
        })
    return out


# ============================================================================
# Event-driven refresh trigger
# ============================================================================

def should_trigger_event_refresh(
    ticker: str,
    lookback_days: int = 7,
) -> bool:
    """Decide whether ``ticker`` warrants a fresh data pull right now.

    Returns ``True`` if EITHER:

      1. There is an earnings event for ``ticker`` within the next
         ``lookback_days`` days. (Anticipate the print: refresh
         transcripts, estimates, ratios so the thesis generator has
         current data when results drop.)

      2. There was an earnings event in the past ``lookback_days`` days
         AND we have not refreshed since that event. (We need
         post-print data: actual EPS, the new transcript, revised
         estimates.)

    Otherwise returns ``False``.

    Why both windows: missing the lookahead means stale prep; missing
    the lookbehind means the thesis is built on yesterday's numbers
    while the market is reacting to today's print.
    """
    ticker = ticker.upper()
    today = _utcnow().date()
    window_start = today - timedelta(days=lookback_days)
    window_end = today + timedelta(days=lookback_days)

    try:
        events = providers.get_earnings_events(
            window_start.isoformat(), window_end.isoformat(),
        )
    except Exception as e:  # noqa: BLE001
        logger.error(
            "earnings_calendar | trigger check failed for %s: %s", ticker, e,
        )
        return False

    if not events:
        return False

    last_refresh = _last_refresh(ticker)

    for e in events:
        if (e.get("ticker") or "").upper() != ticker:
            continue

        date_str = e.get("date")
        if not date_str:
            continue
        try:
            ev_date = datetime.fromisoformat(date_str).date()
        except ValueError:
            logger.warning(
                "earnings_calendar | bad date for %s: %r", ticker, date_str,
            )
            continue

        # Case 1: future event within window — always trigger
        if ev_date >= today:
            return True

        # Case 2: past event within window — trigger if we haven't
        # refreshed since
        if last_refresh is None or last_refresh.date() < ev_date:
            return True

    return False
