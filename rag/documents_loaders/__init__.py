"""
rag/document_loaders/__init__.py
================================
Adapters from your existing data layer to the RAG ingestion pipeline.

Each loader knows one source (SEC filings, theses, fund notes). Each
yields ``RawDocument`` records — a uniform shape the indexer can
consume regardless of where the text came from.

Why loaders are a sub-package
-----------------------------
Adding a new source type is the most common future extension. Putting
each in its own file means:

* ``filings.py`` doesn't need to import ``theses.py``, so a missing
  dependency in one file doesn't break the others.
* Tests can mock one loader without touching the rest.
* The indexer's contract is "iterate over loaders that yield
  RawDocument" — agnostic to source.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RawDocument:
    """A document about to be chunked.

    All loaders yield this shape. The chunker reads ``text`` and
    ``doc_type``; the indexer reads ``doc_id`` for deduplication and
    ``metadata`` for chunk metadata propagation.

    Attributes
    ----------
    doc_id:
        Stable, human-readable identifier (e.g.
        ``"AAPL_10-K_2024-09-28"``). Used as a primary key in
        ``rag_documents`` and as a chunk-ID prefix.
    doc_type:
        ``"filing" | "transcript" | "thesis" | "note"``. Drives
        chunker selection and shows up in metadata for filtering.
    text:
        The actual content to be chunked. Empty/whitespace-only docs
        are silently skipped by the indexer.
    title:
        Human-friendly label for display (e.g.
        ``"Apple Inc. 10-K, fiscal year 2024"``).
    ticker:
        Primary ticker association. None for non-ticker docs (e.g.
        macro essays). Filters on this field at retrieval time.
    as_of:
        ISO date string for the document's effective date.
    source_url:
        Link back to the original source where available.
    metadata:
        Extra fields specific to the source (e.g. accession number
        for filings, speaker for transcripts).
    """
    doc_id: str
    doc_type: str
    text: str
    title: Optional[str] = None
    ticker: Optional[str] = None
    as_of: Optional[str] = None
    source_url: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def text_sha256(self) -> str:
        """Hash of the raw text. Indexer uses this to skip unchanged docs."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()
