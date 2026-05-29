"""
rag/document_loaders/__init__.py
================================
Document-loader package. Each loader here translates a source-of-truth
in the rest of the codebase (SEC fetcher, portfolio DB, notes folder,
earnings transcript store) into a stream of :class:`RawDocument`
objects that the indexer can consume.

What this file owns
-------------------
The :class:`RawDocument` dataclass. Every loader emits these; the
indexer, chunker, and learning loop all read them. It's the contract
between the "where data lives" side of the system and the "how data
gets embedded" side.

Why a dataclass and not a dict
------------------------------
* Static fields catch typos at construction time (a dict with a
  ``ticker`` key vs ``tikker`` key both "work" — until retrieval
  silently filters everything out).
* ``text_sha256`` is computed lazily and cached, so the indexer can
  call it cheaply on every document during the unchanged-check fast
  path.
* IDE autocomplete and type checkers actually help.

The shape
---------
Required:
    ``doc_id``    — stable identifier. The indexer uses this to
                    detect re-runs and delete stale chunks.
    ``doc_type``  — 'filing' | 'transcript' | 'thesis' | 'note' (or
                    anything else; the chunker falls through to
                    unstructured for unknown types).
    ``text``      — the actual prose.

Optional:
    ``title``      — human-readable display label.
    ``ticker``     — equity ticker (None for non-ticker docs like
                     fund-wide notes).
    ``as_of``      — ISO date string for the document content.
                     Used by the retriever's ``as_of_after`` filter.
    ``source_url`` — link back to the original source. Stored in the
                     tracking DB; not put on chunks.
    ``metadata``   — extra loader-specific keys (form_type,
                     accession_number, speaker, etc.). These are
                     copied onto every chunk's metadata by the
                     indexer.

doc_id stability
----------------
Loaders must produce deterministic ``doc_id`` values. Same source
content → same ID across runs. This is what makes incremental
indexing work: the indexer compares the current SHA-256 of the text
against the previously-recorded one, keyed on ``doc_id``.

Examples that satisfy this:
    ``AAPL_10-K_2024-09-28``
    ``MSFT_transcript_2024-04-25``
    ``thesis_42``
    ``note_supply_chain_china_2024-03-01``

Examples that DON'T (because they change every run):
    ``filing_{uuid4()}``
    ``thesis_{datetime.now().isoformat()}``
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RawDocument:
    """A single document on its way into the RAG index.

    See module docstring for the field semantics. Construct one of
    these from your loader, hand it to ``Indexer.index_document``,
    and the indexer handles the rest.
    """

    # Required
    doc_id: str
    doc_type: str
    text: str

    # Optional
    title: Optional[str] = None
    ticker: Optional[str] = None
    as_of: Optional[str] = None
    source_url: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    # Internal cache for the text hash. We compute it lazily on first
    # access (so a loader that emits a 5MB filing isn't paying the
    # SHA-256 cost until the indexer actually needs it) and memoize
    # so the indexer's "skip if unchanged" check is O(1) on re-call.
    _text_sha256_cache: Optional[str] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Be defensive about None text — the indexer's empty-text
        # guard expects str, not NoneType.
        if self.text is None:
            self.text = ""
        # metadata=None is a common mistake from loaders that build the
        # dict conditionally; coerce to empty dict so downstream
        # ``doc.metadata.items()`` doesn't crash.
        if self.metadata is None:
            self.metadata = {}

    @property
    def text_sha256(self) -> str:
        """SHA-256 of the document text, hex-encoded.

        Used by the indexer to decide whether a document has changed
        since the last run. Cached on first access — the hash of a
        10-K is non-trivial and the indexer reads this property
        repeatedly during a batch run.
        """
        if self._text_sha256_cache is None:
            h = hashlib.sha256()
            h.update(self.text.encode("utf-8", errors="replace"))
            # Bypass the dataclass field assignment for the cache to
            # avoid triggering anything fancy; direct attr set is fine.
            object.__setattr__(self, "_text_sha256_cache", h.hexdigest())
        return self._text_sha256_cache


__all__ = ["RawDocument"]