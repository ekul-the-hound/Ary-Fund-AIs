"""
rag/embedder.py
===============
Converts text into fixed-length vectors ("embeddings") that capture meaning.
Similar meanings → nearby vectors in the embedding space.

This is the lowest-level building block of the RAG pipeline. Every other
RAG module — chunker, vector store, retriever — depends on this one,
either directly (to embed something) or transitively (to know the
embedding dimension, which downstream sizing depends on).

Backend selection
-----------------
Tries backends in priority order and uses the first that succeeds:

    1. Ollama running ``nomic-embed-text`` (768-dim, best quality)
    2. sentence-transformers ``all-MiniLM-L6-v2`` (384-dim, CPU-friendly)

Caller code is backend-agnostic. The active backend is detected at
construction time and reported via :attr:`Embedder.backend_name` and
:attr:`Embedder.dimension`.

Why a fallback?
^^^^^^^^^^^^^^^
The Ollama daemon might be down. The user might not have pulled the
model. The network might glitch. A graceful fallback to a pure-Python
embedder means RAG keeps working even on a machine that hasn't been
configured. The fallback is lower-quality (384 dims, ~6 MTEB points
behind nomic) but it works without any setup.

Caching
-------
Every embedding is cached in SQLite, keyed by
``sha256(text + model_name)``. Two reasons:

* **Within a run**: a chunk's text repeated across documents (boilerplate
  disclaimers in 10-Ks, for example) gets embedded once.
* **Across runs**: when you re-index an unchanged document, the chunks
  are skipped.

Including the model name in the cache key means switching models
invalidates the cache automatically — vectors from different models live
in different spaces and can't be compared.

Nomic prefixes
--------------
``nomic-embed-text`` was trained with task prefixes to distinguish how
text will be used:

* ``search_document: <text>`` — text being added to the index
* ``search_query: <text>``    — text being searched WITH

Skipping the prefix works but costs a few MTEB points. We add them
transparently. Callers tell us the role via the ``role`` parameter.

For MiniLM, prefixes are no-ops (the model wasn't trained with them),
so we pass the raw text through. The branch is hidden inside
:meth:`Embedder.embed`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------

# Default endpoint when Ollama is running locally with default settings.
OLLAMA_DEFAULT_URL = "http://localhost:11434"

# Nomic prefixes — see module docstring for why these exist.
NOMIC_DOC_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "

# Role enum for type-safe API. Callers say `role="document"` or
# `role="query"`; we use this to pick the right prefix.
Role = Literal["document", "query"]


# ----------------------------------------------------------------------
# Backend probes
# ----------------------------------------------------------------------

def _probe_ollama(url: str, model: str, timeout: float = 2.0) -> bool:
    """Check whether Ollama is reachable AND the model is pulled.

    A reachable Ollama with the wrong model loaded is just as broken
    as no Ollama at all — we'd get a 404 on every call. So we probe
    both at once by asking the embeddings endpoint to embed a tiny
    test string.

    Returns True iff we successfully got an embedding back.
    """
    try:
        resp = requests.post(
            f"{url}/api/embeddings",
            json={"model": model, "prompt": "x"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return False
        data = resp.json()
        return isinstance(data.get("embedding"), list) and len(data["embedding"]) > 0
    except Exception:  # noqa: BLE001 — probe must never raise
        return False


# ----------------------------------------------------------------------
# Embedder class
# ----------------------------------------------------------------------

class Embedder:
    """Text → vector. Backend-agnostic, cached, batch-friendly.

    Parameters
    ----------
    cache_db_path:
        SQLite file for the embedding cache. Created if it doesn't
        exist. Pass ``None`` to disable caching (useful in tests).
    ollama_url:
        Override for non-default Ollama installs.
    ollama_model:
        Override to try a different Ollama embedding model.
    force_backend:
        If set, skip probing and use this backend regardless. Mainly
        for tests; in production code, let auto-detection win.
    n_workers:
        Thread pool size for parallel HTTP requests to Ollama. Ollama
        serializes per-model internally, but parallel network round
        trips still hide latency.
    """

    def __init__(
        self,
        cache_db_path: Optional[str] = "data/rag_embeddings.db",
        ollama_url: str = OLLAMA_DEFAULT_URL,
        ollama_model: str = "nomic-embed-text",
        force_backend: Optional[Literal["ollama", "minilm"]] = None,
        n_workers: int = 4,
    ):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.n_workers = n_workers

        # Auto-detect backend. Subclasses or tests can override via
        # force_backend, but normal callers let this run.
        if force_backend == "ollama" or (
            force_backend is None and _probe_ollama(ollama_url, ollama_model)
        ):
            self.backend_name = "ollama"
            self.model_name = ollama_model
            self.dimension = 768
            self._minilm_model = None  # not loaded
        else:
            self.backend_name = "minilm"
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.dimension = 384
            # Lazy-import sentence_transformers so users without it
            # installed can still construct the class (and fail at
            # embed-time with a clearer error).
            try:
                from sentence_transformers import SentenceTransformer
                self._minilm_model = SentenceTransformer(self.model_name)
            except ImportError as e:
                raise RuntimeError(
                    "Neither Ollama nor sentence-transformers is available. "
                    "Install one of:\n"
                    "  * Ollama with `ollama pull nomic-embed-text`, OR\n"
                    "  * `pip install sentence-transformers`"
                ) from e

        logger.info(
            "Embedder ready | backend=%s | model=%s | dim=%d",
            self.backend_name, self.model_name, self.dimension,
        )

        # Cache setup
        self._cache_db_path = cache_db_path
        self._cache_lock = threading.Lock()
        if cache_db_path:
            Path(cache_db_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_cache_db()

    # ------------------------------------------------------------------
    # Cache machinery
    # ------------------------------------------------------------------
    def _init_cache_db(self) -> None:
        with sqlite3.connect(self._cache_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    vec BLOB NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

    def _cache_key(self, text: str, role: Role) -> str:
        # Include both model and role in the key. Role matters because
        # nomic embeds the same text differently with doc vs query
        # prefixes; we don't want a cache hit to return the wrong one.
        h = hashlib.sha256()
        h.update(self.model_name.encode("utf-8"))
        h.update(b"\x00")
        h.update(role.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[np.ndarray]:
        if not self._cache_db_path:
            return None
        with self._cache_lock, sqlite3.connect(self._cache_db_path) as conn:
            row = conn.execute(
                "SELECT vec FROM embedding_cache WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32)

    def _cache_put(self, key: str, vec: np.ndarray) -> None:
        if not self._cache_db_path:
            return
        # Store as float32 to halve disk footprint (default is float64).
        # Embedding precision is not affected at any reasonable level.
        v32 = vec.astype(np.float32, copy=False)
        with self._cache_lock, sqlite3.connect(self._cache_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embedding_cache (key, model, vec) "
                "VALUES (?, ?, ?)",
                (key, self.model_name, v32.tobytes()),
            )

    def clear_cache(self) -> int:
        """Wipe the cache. Returns number of rows deleted."""
        if not self._cache_db_path:
            return 0
        with self._cache_lock, sqlite3.connect(self._cache_db_path) as conn:
            cur = conn.execute("DELETE FROM embedding_cache")
            return cur.rowcount

    # ------------------------------------------------------------------
    # Backend-specific implementations
    # ------------------------------------------------------------------
    def _ollama_embed_one(self, text: str) -> np.ndarray:
        """Single round-trip to Ollama. Raises on failure (caller handles)."""
        resp = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": self.ollama_model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embedding"], dtype=np.float32)

    def _minilm_embed_batch(self, texts: list[str]) -> np.ndarray:
        """sentence-transformers handles batching internally and runs
        the whole batch through one forward pass — much faster than
        looping calls."""
        return self._minilm_model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed(
        self,
        texts: Iterable[str],
        role: Role = "document",
    ) -> np.ndarray:
        """Embed a batch of strings.

        Parameters
        ----------
        texts:
            The strings to embed. Empty strings are mapped to zero
            vectors (so callers don't have to filter).
        role:
            "document" for content being indexed, "query" for content
            being searched WITH. This affects the Nomic prefix; for
            MiniLM the parameter is ignored.

        Returns
        -------
        np.ndarray of shape (len(texts), self.dimension), dtype float32.
        """
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, self.dimension), dtype=np.float32)

        # Resolve cache first. We collect (index, cached_vector) for
        # hits, and (index, prepared_text) for misses; we'll only
        # compute embeddings for misses.
        prefix = (
            NOMIC_DOC_PREFIX if (self.backend_name == "ollama" and role == "document")
            else NOMIC_QUERY_PREFIX if (self.backend_name == "ollama" and role == "query")
            else ""
        )
        cache_keys: list[str] = []
        prepared: list[str] = []
        for t in text_list:
            if not t:
                prepared.append("")
                cache_keys.append("")  # sentinel — empty strings never hit cache
            else:
                prepared.append(prefix + t)
                cache_keys.append(self._cache_key(t, role))

        # Try cache for each non-empty input
        out: list[Optional[np.ndarray]] = [None] * len(text_list)
        miss_indices: list[int] = []
        for i, key in enumerate(cache_keys):
            if not key:  # empty string sentinel
                out[i] = np.zeros(self.dimension, dtype=np.float32)
                continue
            cached = self._cache_get(key)
            if cached is not None and len(cached) == self.dimension:
                out[i] = cached
            else:
                miss_indices.append(i)

        # Compute the misses
        if miss_indices:
            miss_texts = [prepared[i] for i in miss_indices]
            if self.backend_name == "ollama":
                vecs = self._ollama_embed_batch_parallel(miss_texts)
            else:
                vecs = self._minilm_embed_batch(miss_texts)
            for i, vec in zip(miss_indices, vecs):
                out[i] = vec
                self._cache_put(cache_keys[i], vec)

        # Stack into (n, dim). At this point every slot is filled.
        return np.stack(out, axis=0).astype(np.float32, copy=False)

    def embed_one(self, text: str, role: Role = "document") -> np.ndarray:
        """Convenience for a single string. Returns shape (dim,)."""
        return self.embed([text], role=role)[0]

    def _ollama_embed_batch_parallel(self, texts: list[str]) -> list[np.ndarray]:
        """Call Ollama in parallel with a thread pool. Order is preserved."""
        results: list[Optional[np.ndarray]] = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            futures = {ex.submit(self._ollama_embed_one, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.warning("ollama embed | row %d failed: %s — using zero vector", i, e)
                    results[i] = np.zeros(self.dimension, dtype=np.float32)
        # All slots are populated (either real or zero)
        return results  # type: ignore[return-value]


# ----------------------------------------------------------------------
# Module-level convenience
# ----------------------------------------------------------------------

_default_embedder: Optional[Embedder] = None
_default_lock = threading.Lock()


def get_default_embedder(**kwargs) -> Embedder:
    """Lazily construct and return a process-wide singleton.

    Most call sites just want "the" embedder, not a configured one.
    The first call decides the configuration; subsequent calls return
    the same instance. Tests that need a clean instance should
    construct ``Embedder`` directly.
    """
    global _default_embedder
    with _default_lock:
        if _default_embedder is None:
            _default_embedder = Embedder(**kwargs)
        return _default_embedder
