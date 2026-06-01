"""
rag/bm25_index.py
=================
Lexical retrieval via BM25.

Why this exists
---------------
Pure vector search has a known failure mode: rare proper nouns and
exact-string queries. A query like "CHIPS Act" or "Form 8-K Item 5.02"
or a specific ticker embeds toward generic financial concepts — the
embedding doesn't capture that the user wants those exact strings.

BM25 is the opposite. It scores by literal word overlap, weighted by
how rare the words are across the corpus. Combining BM25 + vector
search via reciprocal rank fusion gets you both:

    * vector search       → handles paraphrase
    * BM25                → handles exact-string match
    * RRF                 → merges without needing score normalization

The merge is rank-based, not score-based. BM25 scores are unbounded
(any positive number) and cosine similarity is in [-1, 1]; mixing
raw scores requires careful calibration. RRF sidesteps this by using
only the position of each document in each ranking.

In-memory index, with persistence
---------------------------------
The BM25 index lives in RAM. For ~250K chunks the memory cost is
~50MB and queries take ~5ms. We pickle the index to disk so process
restart doesn't pay the full rebuild cost (~30 seconds for a large
corpus). Rebuild triggers:

    * No saved index file
    * Number of chunks in tracking DB differs from saved chunk count
    * Caller passes ``force_rebuild=True``

Tokenization
------------
BM25 needs tokens, not characters. We use a deliberately simple
tokenizer (lowercase, alphanumeric runs, drop stopwords) rather than
a model-specific BPE/SentencePiece tokenizer. Why? Because BM25's
strength is exact-string matching, and "matching" means the same
tokens come out for both query and document. A simple universal
tokenizer maximizes match rate; a model-specific one would split
"Foxconn" into subword pieces that wouldn't align between query and
chunk reliably.

Tokens are also case-folded, so "Apple" and "apple" match. For
financial text this is the right default — ticker case does carry
information but most users write tickers in mixed case.
"""

from __future__ import annotations

import logging
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from rag.vector_store import COLLECTION_RESEARCH, VectorStore

logger = logging.getLogger(__name__)


# A small English stopword list. We don't import NLTK or scikit-learn
# just for this — the list is tiny and stable.
STOPWORDS = frozenset((
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "and", "or", "but", "if", "then", "so", "as", "from", "up", "down",
    "i", "you", "he", "she", "we", "what", "which", "who", "whom",
    "how", "where", "when", "why",
    "not", "no", "yes",
))

# Anything that's a run of alphanumerics. Strips punctuation, keeps
# tickers and numeric values intact (so "8-K" tokenizes as ["8", "k"],
# which is unfortunate but rare in practice).
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric runs, drop stopwords.

    See module docstring for why this is deliberately simple.
    """
    if not text:
        return []
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


# ----------------------------------------------------------------------
# BM25 index class
# ----------------------------------------------------------------------

class BM25Index:
    """In-memory BM25 index with disk persistence.

    Parameters
    ----------
    persist_path:
        Where to pickle the index. Default colocates with the
        vector store. Pass ``None`` to disable persistence entirely
        (rebuild every time).
    """

    def __init__(
        self,
        persist_path: Optional[str] = "data/chroma/bm25_index.pkl",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.persist_path = persist_path
        self.k1 = k1
        self.b = b
        # State that gets serialized
        self._bm25 = None          # rank_bm25.BM25Okapi instance
        self._chunk_ids: list[str] = []
        self._chunk_count = 0      # for staleness detection

    @property
    def ready(self) -> bool:
        """True iff the index has been built and can be queried."""
        return self._bm25 is not None and len(self._chunk_ids) > 0

    # ------------------------------------------------------------------
    # Build / load / save
    # ------------------------------------------------------------------
    def build_from_store(
        self,
        store: VectorStore,
        collection: str = COLLECTION_RESEARCH,
    ) -> int:
        """Pull every chunk from the vector store, tokenize, build BM25.

        Returns the number of chunks indexed. Most expensive operation
        in this file — ~30 seconds for ~250K chunks. Cache the result.
        """
        from rank_bm25 import BM25Okapi

        col = store._collection(collection)
        # Chroma's get() with no filter returns everything. include=
        # ["documents"] fetches the text; the IDs always come along.
        all_data = col.get(include=["documents"])
        ids = all_data.get("ids", [])
        docs = all_data.get("documents", [])

        if not ids:
            logger.info("bm25 | no chunks in %s, skipping build", collection)
            self._bm25 = None
            self._chunk_ids = []
            self._chunk_count = 0
            return 0

        # Tokenize. This is the slow step for large corpora — pure
        # Python regex on every doc. For 250K * 500-token chunks
        # it takes ~20 seconds on commodity hardware.
        tokenized = [tokenize(d) for d in docs]

        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self._chunk_ids = list(ids)
        self._chunk_count = len(ids)

        logger.info("bm25 | built index over %d chunks", len(ids))

        # Persist
        if self.persist_path:
            self._save()

        return len(ids)

    def _save(self) -> None:
        if not self.persist_path:
            return
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "chunk_ids": self._chunk_ids,
                "chunk_count": self._chunk_count,
                "k1": self.k1,
                "b": self.b,
            }, f)

    def load(self) -> bool:
        """Try to load a previously-saved index. Returns True on success."""
        if not self.persist_path or not Path(self.persist_path).exists():
            return False
        try:
            with open(self.persist_path, "rb") as f:
                state = pickle.load(f)
            self._bm25 = state["bm25"]
            self._chunk_ids = state["chunk_ids"]
            self._chunk_count = state["chunk_count"]
            logger.info("bm25 | loaded index over %d chunks", self._chunk_count)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("bm25 | load failed: %s; will rebuild", e)
            return False

    def ensure_built(
        self,
        store: VectorStore,
        tracking_db_path: Optional[str] = None,
        collection: str = COLLECTION_RESEARCH,
        force_rebuild: bool = False,
    ) -> None:
        """Load from disk if possible; rebuild if stale or missing.

        Staleness check: compare the saved chunk_count against the
        current chunk count in the vector store. Both are chunk counts
        from the same source, so they match exactly when the corpus is
        unchanged and differ the moment chunks are added/removed/re-indexed
        — including re-indexes that change a document's chunk count, which
        the old document-count proxy silently missed.

        ``tracking_db_path`` is accepted for backwards compatibility but is
        no longer used for the staleness comparison; the store's own count
        is authoritative.
        """
        if force_rebuild:
            self.build_from_store(store, collection=collection)
            return

        # Try loading
        if not self.ready:
            self.load()

        # If we still don't have an index, build fresh
        if not self.ready:
            self.build_from_store(store, collection=collection)
            return

        # Staleness check: chunks-in-store vs chunks-in-index (like-for-like).
        try:
            current = store.count(collection)
        except Exception:  # noqa: BLE001
            current = self._chunk_count  # can't tell — assume fresh
        if current != self._chunk_count:
            logger.info(
                "bm25 | chunk count changed (%d → %d), rebuilding",
                self._chunk_count, current,
            )
            self.build_from_store(store, collection=collection)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, query_text: str, k: int = 30) -> list[tuple[str, float]]:
        """Return top-K (chunk_id, score) pairs by BM25 relevance.

        Returns empty list if the index isn't built. Caller is
        responsible for combining with vector results via RRF.
        """
        if not self.ready:
            return []
        tokens = tokenize(query_text)
        if not tokens:
            return []
        # rank_bm25's get_scores returns a parallel array of scores,
        # one per document. We argsort to get the top K.
        scores = self._bm25.get_scores(tokens)
        # Get top-K indices. argpartition is faster than full sort
        # for k << n, but for small K the difference is negligible.
        import numpy as np
        top_idx = np.argsort(scores)[-k:][::-1]
        return [(self._chunk_ids[i], float(scores[i]))
                for i in top_idx if scores[i] > 0]


# ----------------------------------------------------------------------
# Reciprocal Rank Fusion
# ----------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists by reciprocal rank fusion.

    Parameters
    ----------
    ranked_lists:
        Each inner list is an ordered list of document IDs, best-first.
        Different lists can have different lengths and don't need to
        contain the same IDs.
    k:
        RRF constant. The published default is 60. Higher k means
        rank differences matter less; lower k means they matter more.
        Almost no one tunes this — 60 works.

    Returns
    -------
    Sorted list of ``(doc_id, fused_score)`` pairs, best-first.

    Notes
    -----
    RRF is rank-based; it ignores the underlying scores entirely.
    That's the whole point — BM25 scores and cosine similarity are
    on different scales, and combining them via score weighting
    requires calibration that breaks every time you swap a model.
    Rank-based fusion is robust to all of that.

    Reference: Cormack, Clarke & Buettcher (2009),
    "Reciprocal Rank Fusion outperforms Condorcet and individual Rank
    Learning Methods."
    """
    fused: Counter[str] = Counter()
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    # Counter.most_common gives us sorted output for free
    return fused.most_common()