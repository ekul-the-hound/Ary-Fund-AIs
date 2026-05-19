"""
rag/document_loaders/transcripts.py
===================================
Loads earnings call transcripts from your existing providers.

The chunker handles the structural side — splitting on speaker turns,
tagging management vs analyst. This loader's only job is to fetch
recent transcripts and wrap them in ``RawDocument``.

Provider contract
-----------------
The expected provider interface is whatever your ``providers.py``
already exposes for transcripts. Specifically, we look for one of:

* ``provider.get_earnings_transcripts(ticker, limit=N)``
  → list of dicts with keys ``year``, ``quarter``, ``date``,
    ``transcript`` or ``content`` (the full text)

* ``provider.get_earnings_transcript(ticker, year, quarter)``
  → single dict with ``content`` and metadata

If your provider uses different method names, only ``_fetch_for_ticker``
needs to change. Everything downstream is provider-agnostic.

How many quarters to index
--------------------------
Four quarters covers a full fiscal year — enough for sequential
guidance analysis. Going deeper (8 quarters / 2 years) is the right
move for longer-horizon strategy work but bloats the corpus. The
``n_quarters`` knob makes this a per-instance decision.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from rag.document_loaders import RawDocument

logger = logging.getLogger(__name__)


class TranscriptsLoader:
    """Yields RawDocuments for earnings call transcripts.

    Parameters
    ----------
    provider:
        Any object exposing ``get_earnings_transcripts(ticker, limit=N)``
        OR ``get_earnings_transcript(ticker, year, quarter)``. The
        loader probes for both at call time and uses whichever exists.
    n_quarters:
        How many most-recent transcripts to fetch per ticker. Default
        4 = one fiscal year of calls.
    """

    def __init__(self, provider: Any, n_quarters: int = 4):
        self.provider = provider
        self.n_quarters = n_quarters

    def load_for_ticker(self, ticker: str) -> Iterable[RawDocument]:
        """Yield RawDocuments for one ticker's recent transcripts."""
        ticker = ticker.upper()
        rows = self._fetch_for_ticker(ticker)
        if not rows:
            return
        # Sort newest first, take top N
        rows.sort(
            key=lambda r: (r.get("year") or 0, r.get("quarter") or 0),
            reverse=True,
        )
        for row in rows[: self.n_quarters]:
            doc = self._build_document(ticker, row)
            if doc:
                yield doc

    def _fetch_for_ticker(self, ticker: str) -> list[dict]:
        """Probe the provider for a transcript-fetching method.

        Returns a list of transcript dicts. Empty list on failure.
        """
        # Path 1: bulk fetcher
        if hasattr(self.provider, "get_earnings_transcripts"):
            try:
                rows = self.provider.get_earnings_transcripts(
                    ticker, limit=self.n_quarters,
                )
                return list(rows or [])
            except Exception as e:  # noqa: BLE001
                logger.warning("transcripts | %s | bulk fetch failed: %s",
                               ticker, e)
                return []

        # Path 2: per-quarter fetcher — try the last N quarters
        if hasattr(self.provider, "get_earnings_transcript"):
            from datetime import datetime
            now = datetime.now()
            current_q = (now.month - 1) // 3 + 1
            rows = []
            # Walk backwards from current quarter
            year, q = now.year, current_q
            for _ in range(self.n_quarters + 2):  # +2 buffer for misses
                try:
                    t = self.provider.get_earnings_transcript(ticker, year, q)
                    if t and (t.get("content") or t.get("transcript")):
                        t.setdefault("year", year)
                        t.setdefault("quarter", q)
                        rows.append(t)
                except Exception:  # noqa: BLE001
                    pass
                q -= 1
                if q < 1:
                    q = 4
                    year -= 1
                if len(rows) >= self.n_quarters:
                    break
            return rows

        logger.warning("transcripts | provider has no recognized transcript method")
        return []

    def _build_document(self, ticker: str, row: dict) -> Optional[RawDocument]:
        """Wrap one transcript row in a RawDocument."""
        text = (row.get("transcript") or row.get("content") or "").strip()
        if not text:
            return None

        year = row.get("year")
        quarter = row.get("quarter")
        date = row.get("date") or row.get("event_date") or ""
        # Date may come as datetime or ISO string — normalize to ISO date
        if hasattr(date, "strftime"):
            date = date.strftime("%Y-%m-%d")
        else:
            date = str(date)[:10] if date else ""

        # Stable doc_id. We prefer the year/quarter combo over the
        # raw date because earnings calls slip a few days year to
        # year, and "Q1 2025" is the durable identity.
        if year and quarter:
            doc_id = f"{ticker}_Q{quarter}_{year}_transcript"
        elif date:
            doc_id = f"{ticker}_{date}_transcript"
        else:
            # Last resort — hash-based ID so we still get uniqueness
            import hashlib
            h = hashlib.sha256(text[:200].encode()).hexdigest()[:10]
            doc_id = f"{ticker}_transcript_{h}"

        title = f"{ticker} earnings call"
        if year and quarter:
            title += f" Q{quarter} {year}"
        elif date:
            title += f" {date}"

        return RawDocument(
            doc_id=doc_id,
            doc_type="transcript",
            text=text,
            title=title,
            ticker=ticker,
            as_of=date or (f"{year}-{(quarter-1)*3 + 1:02d}-01"
                           if year and quarter else None),
            metadata={
                "year": year,
                "quarter": quarter,
                "event_date": date,
            },
        )
