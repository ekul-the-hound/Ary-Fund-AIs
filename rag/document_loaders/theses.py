"""
rag/document_loaders/theses.py
==============================
Loads previously-generated agent opinions, theses, and thesis essays
from the portfolio_db.

Why index your own theses?
--------------------------
At inference time, the agent has no memory of past conversations.
Indexing old theses gives it back its own track record:

* When generating a new AAPL thesis, retrieve the prior 3 AAPL theses
  so the agent can write things like "previous thesis assumed
  Services growth would continue; that assumption played out — now
  the question is whether it can sustain another year."

* When ``thesis_review.py`` scores a thesis well, that thesis becomes
  a positive example that future generation can build on.

Phase 4 of the RAG plan is the long-term payoff of indexing theses:
the system learns from its own best reasoning without retraining a
single LLM weight.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from rag.document_loaders import RawDocument

logger = logging.getLogger(__name__)


class ThesesLoader:
    """Yields RawDocuments for stored theses.

    Parameters
    ----------
    portfolio_db:
        Instance of your project's portfolio DB. Must expose
        ``get_thesis_history(ticker) -> list[dict]`` where each row
        has ``thesis_text``, ``essay_text``, ``created_at``, and
        ``score`` (from thesis_review).
    min_score:
        Skip theses with review score below this threshold. Indexing
        bad reasoning teaches the agent bad habits. Default 0.6.
    n_per_ticker:
        Cap on how many recent theses to index per ticker. Default 10
        — enough for evolution analysis, not enough to drown out
        primary sources.
    """

    def __init__(self, portfolio_db, min_score: float = 0.6, n_per_ticker: int = 10):
        self.db = portfolio_db
        self.min_score = min_score
        self.n_per_ticker = n_per_ticker

    def load_for_ticker(self, ticker: str) -> Iterable[RawDocument]:
        ticker = ticker.upper()
        try:
            rows = self.db.get_thesis_history(ticker) or []
        except Exception as e:  # noqa: BLE001
            logger.warning("theses_loader | %s | history fetch failed: %s", ticker, e)
            return

        # Sort by created_at desc, take top N
        rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        rows = rows[: self.n_per_ticker]

        for row in rows:
            doc = self._build_document(ticker, row)
            if doc:
                yield doc

    def _build_document(self, ticker: str, row: dict) -> Optional[RawDocument]:
        score = row.get("score")
        if score is not None and score < self.min_score:
            return None  # quality filter

        # Prefer the essay over the short thesis: essays have more
        # reasoning structure and chunk better. Fall back to thesis
        # text if no essay was generated.
        text = (row.get("essay_text") or "").strip()
        if not text:
            text = (row.get("thesis_text") or "").strip()
        if not text:
            return None

        created_at = row.get("created_at") or ""
        # Strip time component for the doc_id so it's stable across
        # tz/precision wobble
        date = created_at[:10] if created_at else "unknown"

        doc_id = f"{ticker}_thesis_{date}"
        if row.get("id"):
            # Disambiguate if multiple theses on the same date
            doc_id = f"{doc_id}_{row['id']}"

        return RawDocument(
            doc_id=doc_id,
            doc_type="thesis",
            text=text,
            title=f"{ticker} thesis essay ({date})",
            ticker=ticker,
            as_of=date,
            metadata={
                "thesis_id": row.get("id"),
                "score": score,
                "model": row.get("model"),
                "stance": row.get("stance"),  # bull/bear/neutral if your schema has it
            },
        )
