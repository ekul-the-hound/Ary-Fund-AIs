"""
rag/document_loaders/filings.py
===============================
Loads SEC filings from your existing ``sec_fetcher`` module.

Pulls 10-K, 10-Q, 8-K, and DEF 14A filings, fetches full text for
each, yields one ``RawDocument`` per filing. The chunker handles
section parsing downstream.

Filing type prioritization
--------------------------
* **10-K** — annual. Highest signal density. Always indexed.
* **10-Q** — quarterly. Most content is restatement of the 10-K plus
  three months of MD&A. We index the most recent 2 per ticker —
  enough for quarter-over-quarter comparison, not enough to bloat
  the corpus.
* **8-K** — material event. Selective: only Item 5.02 (officer
  changes), Item 2.02 (results), Item 1.01 (material agreements),
  Item 7.01 (Reg FD). Most 8-Ks are routine and not worth indexing.
* **DEF 14A** — proxy. Compensation, governance, related-party
  transactions. Underused gold for governance research.

The "what to index" choices are driven by ARGS to the loader rather
than hardcoded, so different fund strategies can tune them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Iterable, Optional

from rag.document_loaders import RawDocument

logger = logging.getLogger(__name__)


# 8-K items worth indexing. The Form 8-K has ~30 possible items;
# most are routine compliance fluff. These are the material ones.
INDEXABLE_8K_ITEMS = {
    "1.01",  # Entry into material definitive agreement
    "1.02",  # Termination of material definitive agreement
    "2.02",  # Results of operations and financial condition
    "2.05",  # Costs associated with exit/disposal activities
    "3.02",  # Unregistered sales of equity securities
    "5.02",  # Departure/election of directors or principal officers
    "5.07",  # Submission of matters to a vote of security holders
    "7.01",  # Reg FD disclosure
    "8.01",  # Other events (material)
}


class FilingsLoader:
    """Yields RawDocuments for SEC filings of the given tickers.

    Parameters
    ----------
    sec_fetcher:
        Instance of your project's ``SECFetcher`` class. Passed in
        rather than imported so this file remains testable without
        the full data layer present.
    n_10k:
        Number of most-recent 10-Ks per ticker. Default 3 covers
        the last ~3 years of annual filings.
    n_10q:
        Most-recent 10-Qs per ticker. Default 2 — enough for
        sequential comparison, sparse enough to keep corpus lean.
    n_8k:
        Most-recent 8-Ks per ticker matching INDEXABLE_8K_ITEMS.
    n_proxy:
        Most-recent proxy statements per ticker. Default 2.
    """

    def __init__(
        self,
        sec_fetcher,
        n_10k: int = 3,
        n_10q: int = 2,
        n_8k: int = 10,
        n_proxy: int = 2,
    ):
        self.sec = sec_fetcher
        self.n_10k = n_10k
        self.n_10q = n_10q
        self.n_8k = n_8k
        self.n_proxy = n_proxy

    def load_for_ticker(self, ticker: str) -> Iterable[RawDocument]:
        """Yield RawDocuments for one ticker. Yields lazily so the
        indexer can stream rather than buffering everything.
        """
        ticker = ticker.upper()

        # Pull filings list — your sec_fetcher.get_filings returns
        # rows with form_type, filing_date, accession_number, etc.
        try:
            filings = self.sec.get_filings(ticker)
        except Exception as e:  # noqa: BLE001
            logger.warning("filings_loader | %s | get_filings failed: %s", ticker, e)
            return

        # Group by form type and take the most recent N of each.
        by_type: dict[str, list] = {"10-K": [], "10-Q": [], "8-K": [], "DEF 14A": []}
        for f in filings or []:
            ft = (f.get("form_type") or "").upper().replace(" ", "")
            if ft in ("10-K", "10K"):
                by_type["10-K"].append(f)
            elif ft in ("10-Q", "10Q"):
                by_type["10-Q"].append(f)
            elif ft in ("8-K", "8K"):
                by_type["8-K"].append(f)
            elif ft.startswith("DEF14A") or ft == "DEF14A":
                by_type["DEF 14A"].append(f)

        # Sort each by date desc (sec_fetcher usually returns this way,
        # but be defensive — re-sort here).
        for ft in by_type:
            by_type[ft].sort(key=lambda x: x.get("filing_date") or "", reverse=True)

        # Yield in priority order so the indexer hits highest-signal
        # filings first.
        targets = (
            [(f, "10-K") for f in by_type["10-K"][: self.n_10k]]
            + [(f, "DEF 14A") for f in by_type["DEF 14A"][: self.n_proxy]]
            + [(f, "10-Q") for f in by_type["10-Q"][: self.n_10q]]
            + [(f, "8-K") for f in by_type["8-K"][: self.n_8k]]
        )

        for filing, ft in targets:
            doc = self._build_document(ticker, filing, ft)
            if doc:
                yield doc

    def _build_document(self, ticker: str, filing: dict, form_type: str) -> Optional[RawDocument]:
        """Fetch full text and assemble a RawDocument. Returns None
        on failure (the indexer just skips it)."""
        accession = filing.get("accession_number") or filing.get("accession")
        filing_date = filing.get("filing_date") or filing.get("date")
        if not accession:
            return None

        try:
            text = self.sec.get_filing_text(ticker, accession)
        except Exception as e:  # noqa: BLE001
            logger.warning("filings_loader | %s | text fetch failed for %s: %s",
                           ticker, accession, e)
            return None
        if not text or not text.strip():
            return None

        # For 8-Ks, filter by item: only index if it touches one of
        # the items we care about. We check the text since item
        # numbers are inside the document body.
        if form_type == "8-K":
            if not any(f"Item {it}" in text for it in INDEXABLE_8K_ITEMS):
                return None

        # Build a stable doc_id. Format: TICKER_FORM_DATE so the
        # ID survives re-fetches and is human-readable in logs.
        doc_id = f"{ticker}_{form_type.replace(' ', '')}_{filing_date}"

        title = f"{ticker} {form_type} filed {filing_date}"

        return RawDocument(
            doc_id=doc_id,
            doc_type="filing",
            text=text,
            title=title,
            ticker=ticker,
            as_of=filing_date,
            source_url=filing.get("url"),
            metadata={
                "form_type": form_type,
                "accession_number": accession,
                "filing_date": filing_date,
            },
        )
