"""
rag/vector_store.py
===================
Persistent vector storage. Wraps ChromaDB to provide a stable, narrow
interface to the rest of the RAG system.

Why a wrapper around ChromaDB?
------------------------------
* **Vendor decoupling.** If we switch from ChromaDB to Qdrant, FAISS,
  or Weaviate, only this file changes. Every caller in ``indexer.py``,
  ``retriever.py``, and tests speaks our wrapper API, not Chroma's.
* **Domain operations.** "Delete all chunks for doc X," "stats per
  ticker," "ingest a batch of Chunk dataclass instances" — Chroma's
  raw API is dict-of-lists, which is fast but awkward to work with.

Two collections
---------------
* ``research_docs``      — long-form prose chunks (filings, transcripts,
                           theses, notes)
* ``structured_signals`` — short one-sentence facts derived from the
                           data_registry (Phase 2)

Both are configured with cosine distance. They're queried separately
and results merged at the retriever layer when both are needed.

Why two collections instead of one?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Long chunks (~500 tokens) and short facts (~30 tokens) have different
distributional properties in embedding space. Long chunks tend to
dominate top-K rankings when mixed (more "surface area" to partially
match any query). Separate indexes prevent this, at the cost of
querying twice.

Distance vs. similarity
-----------------------
ChromaDB returns distances in cosine space: ``distance = 1 - cosine_sim``.
Lower is more similar. We convert to similarity (``1 - distance``)
before returning to callers, because "higher = better" matches every
other ranking system the user will encounter.

Metadata filtering
------------------
ChromaDB's ``where=`` clause accepts dict-style filters with operators
``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``, ``$lte``, ``$in``, ``$nin``.
Filters apply BEFORE ANN search, which is the whole reason filters
matter: filtering 100K chunks to 200 first, then running ANN over those
200, is enormously more efficient and accurate than running ANN over
100K and discarding 99.8% of results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from rag.chunker import Chunk

logger = logging.getLogger(__name__)


# Collection names. Centralized as constants so a typo can't silently
# create a third collection.
COLLECTION_RESEARCH = "research_docs"
COLLECTION_SIGNALS = "structured_signals"


# ----------------------------------------------------------------------
# Retrieval result dataclass
# ----------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single chunk returned from a vector search.

    Attributes
    ----------
    chunk_id:
        Primary key of the chunk in the vector store.
    text:
        The chunk content (what gets shown to the LLM).
    score:
        Cosine similarity in ``[-1, 1]``, where 1 is "identical
        direction." For chunks pulled from text-embedding models,
        practical values usually fall in ``[0.2, 0.9]`` and anything
        above 0.7 is a strong match.
    metadata:
        Ticker, doc_id, doc_type, section, etc. Same shape as the
        Chunk that was indexed.
    """
    chunk_id: str
    text: str
    score: float
    metadata: dict


# ----------------------------------------------------------------------
# VectorStore class
# ----------------------------------------------------------------------

class VectorStore:
    """Persistent ChromaDB-backed vector store with two collections.

    Construction is lazy: the ChromaDB client and collections are
    created on first access, so tests can construct a store pointed at
    a temp directory without paying full init cost.

    Parameters
    ----------
    persist_path:
        Directory where ChromaDB writes its on-disk index. Created if
        it doesn't exist.
    embedding_dim:
        Vector dimension. Must match the Embedder. Validated on
        every upsert to catch dim-mismatch bugs early.
    """

    def __init__(
        self,
        persist_path: str = "data/chroma",
        embedding_dim: int = 768,
    ):
        self.persist_path = persist_path
        self.embedding_dim = embedding_dim
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collections: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lazy ChromaDB setup
    # ------------------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            import chromadb
            # PersistentClient writes to disk. The alternative,
            # EphemeralClient, is in-memory only and forgets on exit —
            # useful in tests but never in production.
            self._client = chromadb.PersistentClient(path=self.persist_path)
            logger.info("VectorStore | persist_path=%s | dim=%d",
                        self.persist_path, self.embedding_dim)
        return self._client

    def _collection(self, name: str):
        """Get or create a collection. Cached after first access."""
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    @property
    def research(self):
        return self._collection(COLLECTION_RESEARCH)

    @property
    def signals(self):
        return self._collection(COLLECTION_SIGNALS)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        collection: str = COLLECTION_RESEARCH,
        batch_size: int = 256,
    ) -> int:
        """Insert or update a batch of chunks with their embeddings.

        ChromaDB's upsert is idempotent: same ID → updates the row;
        new ID → inserts. This is exactly what we want, since chunk
        IDs are deterministic hashes — re-running the indexer on
        unchanged documents is a no-op.

        Parameters
        ----------
        chunks:
            List of Chunk dataclass instances.
        embeddings:
            Shape (len(chunks), self.embedding_dim). Must align with
            ``chunks`` row by row.
        collection:
            Which collection to write to. Default is ``research_docs``.
        batch_size:
            ChromaDB documents large-batch behavior is fine but RAM
            is finite. 256 is a safe default that keeps each upsert
            under ~1 MB for 768-dim embeddings.

        Returns
        -------
        Number of chunks written.
        """
        if not chunks:
            return 0

        # Shape sanity check — catches dimension-mismatch bugs early.
        # Without this, ChromaDB will raise a cryptic error 200ms into
        # the upsert. With it, we get a clear message at the boundary.
        if embeddings.shape != (len(chunks), self.embedding_dim):
            raise ValueError(
                f"Embedding shape mismatch: got {embeddings.shape}, "
                f"expected ({len(chunks)}, {self.embedding_dim})"
            )

        col = self._collection(collection)
        n_written = 0
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            batch = chunks[start:end]
            batch_emb = embeddings[start:end]
            # Inject doc_id into metadata so where-filters on it work.
            # The Chunk dataclass carries doc_id as a top-level field,
            # but ChromaDB only indexes the metadata dict, not the
            # Python attributes. We copy it in defensively here.
            metadatas = []
            for c in batch:
                md = dict(c.metadata)
                md.setdefault("doc_id", c.doc_id)
                metadatas.append(_clean_metadata(md))
            col.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=batch_emb.tolist(),
                documents=[c.text for c in batch],
                metadatas=metadatas,
            )
            n_written += len(batch)

        logger.info("upsert_chunks | wrote %d to %s", n_written, collection)
        return n_written

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------
    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        where: Optional[dict] = None,
        collection: str = COLLECTION_RESEARCH,
    ) -> list[RetrievalResult]:
        """ANN search over a collection.

        Parameters
        ----------
        query_embedding:
            Shape ``(self.embedding_dim,)`` — a single query vector.
        k:
            Number of neighbors to return.
        where:
            Optional metadata filter. ChromaDB syntax:
                ``{"ticker": "AAPL"}``           — exact match
                ``{"doc_type": {"$in": [...]}}`` — set membership
                ``{"as_of": {"$gte": "2024-01-01"}}`` — comparison

        Returns
        -------
        List of RetrievalResult, ordered by similarity descending.
        """
        if query_embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"Query embedding shape {query_embedding.shape}, "
                f"expected ({self.embedding_dim},)"
            )

        col = self._collection(collection)
        # Chroma expects a list of query vectors (it can batch
        # multiple queries in one call). We only ever pass one.
        raw = col.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where,
        )

        # Chroma returns dict-of-lists keyed by query index. Since we
        # only query once, every list-of-lists has one entry: [0].
        ids = raw["ids"][0]
        distances = raw["distances"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]

        results: list[RetrievalResult] = []
        for chunk_id, dist, doc, md in zip(ids, distances, documents, metadatas):
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=doc,
                # Convert distance → similarity. For cosine space:
                # similarity = 1 - distance. Bounded in [-1, 1].
                score=float(1.0 - dist),
                metadata=dict(md or {}),
            ))
        return results

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------
    def delete_document(self, doc_id: str, collection: str = COLLECTION_RESEARCH) -> int:
        """Delete every chunk belonging to a document.

        Used when a document has been re-ingested (stale chunks
        cleanup) or removed from the corpus entirely.

        Returns the count of chunks deleted, as best as we can tell —
        ChromaDB doesn't return this directly, so we query first.
        """
        col = self._collection(collection)
        # Count before deleting so we can report
        try:
            existing = col.get(where={"doc_id": doc_id}, include=[])
            count = len(existing.get("ids", []))
        except Exception:  # noqa: BLE001
            count = 0
        col.delete(where={"doc_id": doc_id})
        if count:
            logger.info("delete_document | %s | removed %d chunks", doc_id, count)
        return count

    def delete_by_ids(self, chunk_ids: list[str], collection: str = COLLECTION_RESEARCH) -> int:
        if not chunk_ids:
            return 0
        col = self._collection(collection)
        col.delete(ids=chunk_ids)
        return len(chunk_ids)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    def count(self, collection: str = COLLECTION_RESEARCH) -> int:
        """Total chunks in a collection."""
        return self._collection(collection).count()

    def chunk_ids_for_doc(
        self, doc_id: str, collection: str = COLLECTION_RESEARCH
    ) -> set[str]:
        """All chunk IDs currently in the store for a given document.

        Used by the indexer's stale-chunk detection: compare
        existing IDs vs the IDs produced by re-chunking, and delete
        any in the first set but not the second.
        """
        col = self._collection(collection)
        res = col.get(where={"doc_id": doc_id}, include=[])
        return set(res.get("ids", []))

    def stats(self) -> dict:
        """Quick descriptive stats — useful for the CLI status command."""
        return {
            "research_docs_count": self.count(COLLECTION_RESEARCH),
            "structured_signals_count": self.count(COLLECTION_SIGNALS),
            "persist_path": self.persist_path,
            "embedding_dim": self.embedding_dim,
        }


# ----------------------------------------------------------------------
# Metadata cleaning
# ----------------------------------------------------------------------

def _clean_metadata(md: dict) -> dict:
    """ChromaDB metadata values must be str/int/float/bool/None.
    Drop anything that doesn't qualify (e.g., nested dicts, lists) so
    upsert doesn't error out.

    Why this is needed: callers might pass arbitrarily-shaped
    metadata, especially during early development. Rather than failing
    deep inside Chroma's internals, we coerce or drop at the boundary
    and emit a warning.
    """
    cleaned = {}
    for k, v in md.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, (list, tuple)):
            # Chroma doesn't support array values. Join to string.
            cleaned[k] = ", ".join(str(x) for x in v)
        else:
            cleaned[k] = str(v)
    return cleaned


# ----------------------------------------------------------------------
# Module-level convenience
# ----------------------------------------------------------------------

_default_store: Optional[VectorStore] = None


def get_default_store(**kwargs) -> VectorStore:
    """Process-wide singleton (parallel to get_default_embedder)."""
    global _default_store
    if _default_store is None:
        _default_store = VectorStore(**kwargs)
    return _default_store
