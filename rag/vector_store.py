"""
rag/vector_store.py — pure-Python vector store (no native deps)
================================================================
Drop-in replacement for the ChromaDB-backed store. Same public API
(VectorStore, RetrievalResult, COLLECTION_* constants, get_default_store,
_clean_metadata) so nothing downstream changes.

Why this exists
---------------
ChromaDB's native layer (chroma-hnswlib) segfaults on write on this
Windows/Python 3.12 box, and downgrading to a non-crashing version needs
a C++ compiler to build the hnswlib extension. For a corpus of a few
thousand chunks, an exact brute-force cosine search in NumPy is more than
fast enough (milliseconds) and has ZERO native dependencies — no compiler,
no segfaults, no version pinning.

Storage model
-------------
Each collection persists as two files under ``persist_path``:
  * ``<collection>.npz``  — the embedding matrix (float32, n x dim)
  * ``<collection>.json`` — parallel lists: ids, documents, metadatas

On load we read both into memory. Writes are append/replace-by-id, then
flush both files. Last-write-wins, single-process — matching how the
indexer uses it.

Search
------
Brute-force cosine similarity: normalize stored vectors and the query,
dot-product, top-k. With a few thousand 768-dim vectors this is a single
small matmul — sub-millisecond. The ``where`` filter is applied BEFORE
ranking, mirroring Chroma's filter-then-search semantics, and supports
the operators the retriever uses: plain equality plus ``$eq, $ne, $gt,
$gte, $lt, $lte, $in, $nin``.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from rag.chunker import Chunk

logger = logging.getLogger(__name__)


COLLECTION_RESEARCH = "research_docs"
COLLECTION_SIGNALS = "structured_signals"


# ----------------------------------------------------------------------
# Retrieval result dataclass  (identical to the Chroma version)
# ----------------------------------------------------------------------

@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    metadata: dict


# ----------------------------------------------------------------------
# Where-filter evaluation (Chroma-compatible subset)
# ----------------------------------------------------------------------

def _matches_where(md: dict, where: Optional[dict]) -> bool:
    """Return True if metadata dict ``md`` satisfies a Chroma-style
    ``where`` filter. Supports implicit equality and the operators
    $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, plus top-level $and/$or.
    """
    if not where:
        return True

    for key, cond in where.items():
        if key == "$and":
            if not all(_matches_where(md, c) for c in cond):
                return False
            continue
        if key == "$or":
            if not any(_matches_where(md, c) for c in cond):
                return False
            continue

        actual = md.get(key)
        if isinstance(cond, dict):
            for op, want in cond.items():
                if not _apply_op(actual, op, want):
                    return False
        else:
            # implicit equality
            if actual != cond:
                return False
    return True


def _apply_op(actual: Any, op: str, want: Any) -> bool:
    try:
        if op == "$eq":
            return actual == want
        if op == "$ne":
            return actual != want
        if op == "$gt":
            return actual is not None and actual > want
        if op == "$gte":
            return actual is not None and actual >= want
        if op == "$lt":
            return actual is not None and actual < want
        if op == "$lte":
            return actual is not None and actual <= want
        if op == "$in":
            return actual in want
        if op == "$nin":
            return actual not in want
    except TypeError:
        # Mixed-type comparison (e.g. None > "2024"); treat as no-match.
        return False
    # Unknown operator → ignore (don't exclude).
    return True


# ----------------------------------------------------------------------
# In-memory + on-disk collection
# ----------------------------------------------------------------------

class _Collection:
    """One named collection: parallel arrays of ids/vectors/docs/metas,
    persisted to <persist_path>/<name>.npz + .json."""

    def __init__(self, name: str, persist_dir: Path, dim: int):
        self.name = name
        self.dim = dim
        self._npz = persist_dir / f"{name}.npz"
        self._json = persist_dir / f"{name}.json"

        self.ids: list[str] = []
        self.docs: list[str] = []
        self.metas: list[dict] = []
        self.vecs: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._id_pos: dict[str, int] = {}
        self._load()

    # -- persistence ----------------------------------------------------
    def _load(self) -> None:
        if self._json.exists():
            try:
                payload = json.loads(self._json.read_text(encoding="utf-8"))
                self.ids = payload.get("ids", [])
                self.docs = payload.get("docs", [])
                self.metas = payload.get("metas", [])
                if self._npz.exists():
                    arr = np.load(self._npz)["vecs"].astype(np.float32)
                else:
                    arr = np.zeros((len(self.ids), self.dim), dtype=np.float32)
                # Defensive: keep arrays consistent in length.
                if arr.shape[0] != len(self.ids):
                    n = min(arr.shape[0], len(self.ids))
                    arr = arr[:n]
                    self.ids = self.ids[:n]
                    self.docs = self.docs[:n]
                    self.metas = self.metas[:n]
                self.vecs = arr
                self._reindex()
            except Exception as e:  # noqa: BLE001
                logger.warning("vector_store | %s | load failed, starting empty: %s",
                               self.name, e)
                self._reset_empty()

    def _reset_empty(self) -> None:
        self.ids, self.docs, self.metas = [], [], []
        self.vecs = np.zeros((0, self.dim), dtype=np.float32)
        self._id_pos = {}

    def _reindex(self) -> None:
        self._id_pos = {cid: i for i, cid in enumerate(self.ids)}

    def flush(self) -> None:
        self._json.write_text(
            json.dumps({"ids": self.ids, "docs": self.docs, "metas": self.metas}),
            encoding="utf-8",
        )
        np.savez_compressed(self._npz, vecs=self.vecs.astype(np.float32))

    # -- writes ---------------------------------------------------------
    def upsert(self, ids, embeddings, documents, metadatas) -> None:
        emb = np.asarray(embeddings, dtype=np.float32)
        new_rows = []
        for i, cid in enumerate(ids):
            if cid in self._id_pos:
                pos = self._id_pos[cid]
                self.vecs[pos] = emb[i]
                self.docs[pos] = documents[i]
                self.metas[pos] = metadatas[i]
            else:
                self._id_pos[cid] = len(self.ids)
                self.ids.append(cid)
                self.docs.append(documents[i])
                self.metas.append(metadatas[i])
                new_rows.append(emb[i])
        if new_rows:
            self.vecs = np.vstack([self.vecs, np.asarray(new_rows, dtype=np.float32)]) \
                if self.vecs.shape[0] else np.asarray(new_rows, dtype=np.float32)
        self.flush()

    def delete_ids(self, chunk_ids) -> None:
        drop = set(chunk_ids)
        keep = [i for i, cid in enumerate(self.ids) if cid not in drop]
        self._gather(keep)
        self.flush()

    def delete_where(self, where: dict) -> int:
        keep, removed = [], 0
        for i, md in enumerate(self.metas):
            if _matches_where(md, where):
                removed += 1
            else:
                keep.append(i)
        self._gather(keep)
        self.flush()
        return removed

    def _gather(self, keep_idx: list[int]) -> None:
        self.ids = [self.ids[i] for i in keep_idx]
        self.docs = [self.docs[i] for i in keep_idx]
        self.metas = [self.metas[i] for i in keep_idx]
        self.vecs = self.vecs[keep_idx] if keep_idx else np.zeros((0, self.dim), np.float32)
        self._reindex()

    # -- reads ----------------------------------------------------------
    def ids_where(self, where: dict) -> list[str]:
        return [cid for cid, md in zip(self.ids, self.metas) if _matches_where(md, where)]

    def get(self, ids=None, where=None, include=None):
        """ChromaDB-compatible get(). Returns a dict with 'ids' always,
        plus 'documents'/'metadatas' when requested via ``include``.

        Selection: by explicit ``ids`` if given, else by ``where`` filter,
        else everything. Used by the BM25 builder (all docs) and the
        hybrid retriever (fetch specific chunk_ids).
        """
        include = include or []
        if ids is not None:
            idset = set(ids)
            sel = [i for i, cid in enumerate(self.ids) if cid in idset]
        elif where:
            sel = [i for i, md in enumerate(self.metas) if _matches_where(md, where)]
        else:
            sel = list(range(len(self.ids)))

        out = {"ids": [self.ids[i] for i in sel]}
        if "documents" in include:
            out["documents"] = [self.docs[i] for i in sel]
        if "metadatas" in include:
            out["metadatas"] = [dict(self.metas[i]) for i in sel]
        return out

    def count(self) -> int:
        return len(self.ids)

    def search(self, query_vec: np.ndarray, k: int, where: Optional[dict]):
        if not self.ids:
            return []
        # Candidate filter first (filter-then-rank).
        if where:
            cand = [i for i, md in enumerate(self.metas) if _matches_where(md, where)]
            if not cand:
                return []
        else:
            cand = range(len(self.ids))
        cand = list(cand)

        mat = self.vecs[cand]
        # Cosine similarity = normalized dot product.
        q = query_vec.astype(np.float32)
        qn = q / (np.linalg.norm(q) + 1e-12)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        matn = mat / norms
        sims = matn @ qn  # (len(cand),)

        top = np.argsort(-sims)[:k]
        out = []
        for j in top:
            idx = cand[int(j)]
            out.append((self.ids[idx], self.docs[idx], dict(self.metas[idx]),
                        float(sims[int(j)])))
        return out


# ----------------------------------------------------------------------
# VectorStore — same public surface as the Chroma version
# ----------------------------------------------------------------------

class VectorStore:
    def __init__(
        self,
        persist_path: str = "data/chroma",
        embedding_dim: int = 768,
    ):
        self.persist_path = persist_path
        self.embedding_dim = embedding_dim
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, _Collection] = {}
        self._lock = threading.Lock()
        logger.info("VectorStore (pure-python) | persist_path=%s | dim=%d",
                    persist_path, embedding_dim)

    # lazy collection access
    def _collection(self, name: str) -> _Collection:
        if name not in self._collections:
            with self._lock:
                if name not in self._collections:
                    self._collections[name] = _Collection(
                        name, Path(self.persist_path), self.embedding_dim
                    )
        return self._collections[name]

    @property
    def research(self):
        return self._collection(COLLECTION_RESEARCH)

    @property
    def signals(self):
        return self._collection(COLLECTION_SIGNALS)

    # -- writes ---------------------------------------------------------
    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        collection: str = COLLECTION_RESEARCH,
        batch_size: int = 256,
    ) -> int:
        if not chunks:
            return 0
        if embeddings.shape != (len(chunks), self.embedding_dim):
            raise ValueError(
                f"Embedding shape mismatch: got {embeddings.shape}, "
                f"expected ({len(chunks)}, {self.embedding_dim})"
            )
        col = self._collection(collection)
        ids, docs, metas = [], [], []
        for c in chunks:
            md = dict(c.metadata)
            md.setdefault("doc_id", c.doc_id)
            metas.append(_clean_metadata(md))
            ids.append(c.chunk_id)
            docs.append(c.text)
        col.upsert(ids, embeddings, docs, metas)
        logger.info("upsert_chunks | wrote %d to %s", len(chunks), collection)
        return len(chunks)

    # -- query ----------------------------------------------------------
    def query(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        where: Optional[dict] = None,
        collection: str = COLLECTION_RESEARCH,
    ) -> list[RetrievalResult]:
        if query_embedding.shape != (self.embedding_dim,):
            raise ValueError(
                f"Query embedding shape {query_embedding.shape}, "
                f"expected ({self.embedding_dim},)"
            )
        col = self._collection(collection)
        hits = col.search(query_embedding, k, where)
        return [
            RetrievalResult(chunk_id=cid, text=doc, score=score, metadata=md)
            for (cid, doc, md, score) in hits
        ]

    # -- deletes --------------------------------------------------------
    def delete_document(self, doc_id: str, collection: str = COLLECTION_RESEARCH) -> int:
        col = self._collection(collection)
        removed = col.delete_where({"doc_id": doc_id})
        if removed:
            logger.info("delete_document | %s | removed %d chunks", doc_id, removed)
        return removed

    def delete_by_ids(self, chunk_ids: list[str], collection: str = COLLECTION_RESEARCH) -> int:
        if not chunk_ids:
            return 0
        self._collection(collection).delete_ids(chunk_ids)
        return len(chunk_ids)

    # -- inspection -----------------------------------------------------
    def count(self, collection: str = COLLECTION_RESEARCH) -> int:
        return self._collection(collection).count()

    def chunk_ids_for_doc(
        self, doc_id: str, collection: str = COLLECTION_RESEARCH
    ) -> set[str]:
        return set(self._collection(collection).ids_where({"doc_id": doc_id}))

    def stats(self) -> dict:
        return {
            "research_docs_count": self.count(COLLECTION_RESEARCH),
            "structured_signals_count": self.count(COLLECTION_SIGNALS),
            "persist_path": self.persist_path,
            "embedding_dim": self.embedding_dim,
        }


# ----------------------------------------------------------------------
# Metadata cleaning (identical contract to the Chroma version)
# ----------------------------------------------------------------------

def _clean_metadata(md: dict) -> dict:
    cleaned = {}
    for k, v in md.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, (list, tuple)):
            cleaned[k] = ", ".join(str(x) for x in v)
        else:
            cleaned[k] = str(v)
    return cleaned


# ----------------------------------------------------------------------
# Module-level convenience
# ----------------------------------------------------------------------

_default_store: Optional[VectorStore] = None


def get_default_store(**kwargs) -> VectorStore:
    global _default_store
    if _default_store is None:
        _default_store = VectorStore(**kwargs)
    return _default_store