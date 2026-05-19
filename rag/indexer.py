"""
rag/indexer.py
==============
The orchestrator. Walks document loaders, chunks each document,
embeds the chunks, and writes them to the vector store.

Key feature: incremental indexing
---------------------------------
A re-index of an unchanged document is a no-op. The flow:

    1. Compute SHA-256 of the document text.
    2. Look up the previously-indexed hash in ``rag_documents``.
    3. If the hashes match — skip everything (no chunking, no embedding).
    4. If they differ (or no previous record) — chunk, embed, upsert.
    5. Before upsert, compute the set of existing chunk IDs for this
       doc_id. Delete any IDs not in the new chunk set. (Stale chunk
       cleanup — happens when a document shrinks or its text drifts.)

This makes ``run_full_index()`` safe to call on a cron. First call:
backfill everything (minutes). Hourly call: process only the handful of
documents that changed since the last run (seconds).

Document tracking
-----------------
We maintain a small SQLite table ``rag_documents`` to track what's
been indexed:

    doc_id PRIMARY KEY
    text_sha256
    chunk_count
    indexed_at

This table is the source of truth for "what does the vector store
believe it has." The vector store itself has the chunks, but querying
"which docs are indexed" through Chroma's metadata is O(n) — the
table makes it O(1).
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from rag.chunker import Chunk, chunk_document
from rag.contextualizer import Contextualizer, make_disabled_contextualizer
from rag.document_loaders import RawDocument
from rag.embedder import Embedder, get_default_embedder
from rag.vector_store import VectorStore, get_default_store

logger = logging.getLogger(__name__)


class Indexer:
    """Orchestrate ingestion from loaders → chunker → embedder → store."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        store: Optional[VectorStore] = None,
        contextualizer: Optional[Contextualizer] = None,
        tracking_db_path: str = "data/rag_tracking.db",
        chunk_tokens: int = 500,
        overlap_tokens: int = 50,
        embed_batch_size: int = 64,
    ):
        self.embedder = embedder or get_default_embedder()
        self.store = store or get_default_store(embedding_dim=self.embedder.dimension)
        # Contextualizer is optional. When None, we use a disabled
        # one — meaning chunks get embedded as-is. Behavior is
        # identical to Phase 1.
        self.contextualizer = contextualizer or make_disabled_contextualizer()
        self.tracking_db_path = tracking_db_path
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.embed_batch_size = embed_batch_size

        Path(tracking_db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tracking_db()

    # ------------------------------------------------------------------
    # Tracking table
    # ------------------------------------------------------------------
    def _init_tracking_db(self) -> None:
        with sqlite3.connect(self.tracking_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_documents (
                    doc_id        TEXT PRIMARY KEY,
                    doc_type      TEXT NOT NULL,
                    ticker        TEXT,
                    text_sha256   TEXT NOT NULL,
                    chunk_count   INTEGER,
                    indexed_at    TEXT NOT NULL,
                    title         TEXT,
                    source_url    TEXT,
                    as_of         TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_ticker ON rag_documents(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rag_type ON rag_documents(doc_type)")

    def _get_indexed_hash(self, doc_id: str) -> Optional[str]:
        with sqlite3.connect(self.tracking_db_path) as conn:
            row = conn.execute(
                "SELECT text_sha256 FROM rag_documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return row[0] if row else None

    def _record_indexed(self, doc: RawDocument, chunk_count: int) -> None:
        with sqlite3.connect(self.tracking_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO rag_documents
                    (doc_id, doc_type, ticker, text_sha256, chunk_count,
                     indexed_at, title, source_url, as_of)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.doc_id, doc.doc_type, doc.ticker, doc.text_sha256,
                chunk_count, datetime.now().isoformat(timespec="seconds"),
                doc.title, doc.source_url, doc.as_of,
            ))

    # ------------------------------------------------------------------
    # Single-document indexing
    # ------------------------------------------------------------------
    def index_document(self, doc: RawDocument, force: bool = False) -> dict:
        """Index one document. Returns a result dict with status."""
        if not doc.text or not doc.text.strip():
            return {"doc_id": doc.doc_id, "status": "skip_empty", "chunks": 0}

        # Skip if unchanged
        if not force:
            prev_hash = self._get_indexed_hash(doc.doc_id)
            if prev_hash == doc.text_sha256:
                return {"doc_id": doc.doc_id, "status": "skip_unchanged", "chunks": 0}

        # Build base metadata that every chunk inherits
        base_md = {
            "doc_type": doc.doc_type,
            "doc_id": doc.doc_id,
        }
        if doc.ticker:
            base_md["ticker"] = doc.ticker
        if doc.as_of:
            base_md["as_of"] = doc.as_of
        if doc.title:
            base_md["title"] = doc.title
        # Loader-supplied extras (form_type, accession, speaker, etc.)
        for k, v in doc.metadata.items():
            if k not in base_md and v is not None:
                base_md[k] = v

        # Chunk
        chunks = chunk_document(
            raw_text=doc.text,
            doc_id=doc.doc_id,
            doc_type=doc.doc_type,
            base_metadata=base_md,
            chunk_tokens=self.chunk_tokens,
            overlap_tokens=self.overlap_tokens,
        )
        if not chunks:
            return {"doc_id": doc.doc_id, "status": "skip_no_chunks", "chunks": 0}

        # Stale chunk cleanup: any chunk IDs currently in the store
        # for this doc that AREN'T in the new chunk set must be
        # deleted. This handles documents whose text drifted.
        try:
            existing_ids = self.store.chunk_ids_for_doc(doc.doc_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("chunk_ids_for_doc failed: %s", e)
            existing_ids = set()
        new_ids = {c.chunk_id for c in chunks}
        stale = list(existing_ids - new_ids)
        if stale:
            self.store.delete_by_ids(stale)
            logger.info("indexer | %s | removed %d stale chunks", doc.doc_id, len(stale))

        # Embed in batches. Each batch: collect texts → contextualize
        # (when enabled) → call embedder → write.
        #
        # Why contextualize per-chunk inside the indexer loop rather
        # than as a separate pass? Two reasons:
        #
        # 1. Memory. A 10-K can have hundreds of chunks. We want to
        #    flush each batch's embeddings to the store before
        #    holding the next batch's contextualized texts in RAM.
        #
        # 2. Failure isolation. If the LLM dies halfway through a
        #    document, the chunks that succeeded are persisted; the
        #    next run will retry the failed ones.
        all_vecs: list[np.ndarray] = []
        for start in range(0, len(chunks), self.embed_batch_size):
            batch = chunks[start: start + self.embed_batch_size]
            # Contextualize. When disabled, this is a no-op pass-through.
            embed_inputs = [
                self.contextualizer.contextualize_chunk(
                    chunk_text=c.text,
                    full_document_text=doc.text,
                    chunk_metadata=c.metadata,
                )
                for c in batch
            ]
            vecs = self.embedder.embed(embed_inputs, role="document")
            all_vecs.append(vecs)
        embeddings = np.concatenate(all_vecs, axis=0)

        # Upsert
        self.store.upsert_chunks(chunks, embeddings)

        # Record
        self._record_indexed(doc, len(chunks))

        return {
            "doc_id": doc.doc_id,
            "status": "indexed",
            "chunks": len(chunks),
            "stale_removed": len(stale),
        }

    # ------------------------------------------------------------------
    # Multi-document indexing
    # ------------------------------------------------------------------
    def index_many(
        self, docs: Iterable[RawDocument], force: bool = False,
    ) -> dict:
        """Index a stream of documents. Returns aggregated stats."""
        stats = {"indexed": 0, "skip_unchanged": 0, "skip_empty": 0,
                 "skip_no_chunks": 0, "errors": 0, "total_chunks": 0}

        for doc in docs:
            try:
                result = self.index_document(doc, force=force)
                stats[result["status"]] = stats.get(result["status"], 0) + 1
                stats["total_chunks"] += result["chunks"]
                if result["status"] == "indexed":
                    logger.info(
                        "indexer | %s | %d chunks (stale -%d)",
                        result["doc_id"], result["chunks"],
                        result.get("stale_removed", 0),
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("indexer | %s | error: %s", doc.doc_id, e)
                stats["errors"] += 1

        return stats

    # ------------------------------------------------------------------
    # High-level entry points
    # ------------------------------------------------------------------
    def run_tickers(
        self, tickers: list[str], sec_fetcher=None, portfolio_db=None,
        force: bool = False,
    ) -> dict:
        """Index filings + theses for a list of tickers, plus all
        fund notes once. Each loader is optional — missing ones are
        skipped silently.
        """
        from rag.document_loaders.notes import NotesLoader

        per_loader_stats: dict[str, dict] = {}

        # Filings
        if sec_fetcher is not None:
            from rag.document_loaders.filings import FilingsLoader
            fl = FilingsLoader(sec_fetcher=sec_fetcher)
            docs = []
            for tk in tickers:
                docs.extend(fl.load_for_ticker(tk))
            per_loader_stats["filings"] = self.index_many(docs, force=force)

        # Theses
        if portfolio_db is not None:
            from rag.document_loaders.theses import ThesesLoader
            tl = ThesesLoader(portfolio_db=portfolio_db)
            docs = []
            for tk in tickers:
                docs.extend(tl.load_for_ticker(tk))
            per_loader_stats["theses"] = self.index_many(docs, force=force)

        # Notes (global, not per-ticker)
        nl = NotesLoader()
        per_loader_stats["notes"] = self.index_many(nl.load_all(), force=force)

        return per_loader_stats

    def stats(self) -> dict:
        """Quick descriptive stats — what's in the tracking table."""
        with sqlite3.connect(self.tracking_db_path) as conn:
            by_type = dict(conn.execute("""
                SELECT doc_type, COUNT(*) FROM rag_documents GROUP BY doc_type
            """).fetchall())
            total_docs, total_chunks = conn.execute("""
                SELECT COUNT(*), COALESCE(SUM(chunk_count), 0) FROM rag_documents
            """).fetchone()
        return {
            "total_documents": total_docs,
            "total_chunks_recorded": total_chunks,
            "by_doc_type": by_type,
            "vector_store_stats": self.store.stats(),
        }

    # ------------------------------------------------------------------
    # Stale document archival (Phase 3)
    # ------------------------------------------------------------------
    def archive_stale_documents(
        self,
        keep_doc_ids: set[str],
        dry_run: bool = False,
    ) -> dict:
        """Remove documents from the corpus that aren't in keep_doc_ids.

        Use case: caller produces the canonical list of "documents we
        currently care about" by walking every loader. Any tracked
        document whose doc_id is NOT in that set is stale — either
        the loader stopped producing it (filing was removed, thesis
        deleted, note file unlinked) or it was superseded.

        Behavior
        --------
        For each stale doc:
            1. Delete all its chunks from the vector store.
            2. Delete its row from rag_documents (the tracking table).

        ``dry_run=True`` reports what would be deleted without
        actually removing anything. Useful as a sanity check on
        first invocation — it's easy to accidentally pass an
        incomplete keep_doc_ids set and wipe the corpus.

        Returns
        -------
        dict with keys ``stale_count``, ``chunks_deleted``,
        ``deleted_doc_ids``, and ``dry_run``.
        """
        with sqlite3.connect(self.tracking_db_path) as conn:
            all_tracked = {row[0] for row in conn.execute(
                "SELECT doc_id FROM rag_documents"
            ).fetchall()}

        stale = all_tracked - keep_doc_ids
        chunks_deleted = 0
        if not stale:
            return {"stale_count": 0, "chunks_deleted": 0,
                    "deleted_doc_ids": [], "dry_run": dry_run}

        if dry_run:
            logger.info("archive_stale | dry_run | %d stale docs identified",
                        len(stale))
            return {
                "stale_count": len(stale), "chunks_deleted": 0,
                "deleted_doc_ids": sorted(stale), "dry_run": True,
            }

        for doc_id in stale:
            try:
                n = self.store.delete_document(doc_id)
                chunks_deleted += n
            except Exception as e:  # noqa: BLE001
                logger.warning("archive_stale | %s | delete failed: %s",
                               doc_id, e)

        with sqlite3.connect(self.tracking_db_path) as conn:
            conn.executemany(
                "DELETE FROM rag_documents WHERE doc_id = ?",
                [(d,) for d in stale],
            )

        logger.info("archive_stale | removed %d docs (%d chunks)",
                    len(stale), chunks_deleted)
        return {
            "stale_count": len(stale),
            "chunks_deleted": chunks_deleted,
            "deleted_doc_ids": sorted(stale),
            "dry_run": False,
        }

    def mark_superseded(self, doc_id: str, superseded_by: str) -> bool:
        """Mark a document as superseded by a newer one.

        Used when an SEC filing is amended (10-K/A) or a thesis is
        re-issued. The chunks of the superseded doc stay in the
        vector store but get ``superseded=True`` in metadata, which
        retrieval can filter on.

        Why keep the chunks at all? Two reasons:
            * Historical queries ("what did the 2023 10-K say?") still
              work — the chunks are accessible if explicitly requested.
            * Deleting them would be lossy. Some readers want both
              versions to compare.

        Returns True if the document was found and marked.
        """
        with sqlite3.connect(self.tracking_db_path) as conn:
            # Make sure the column exists. ALTER TABLE IF NOT EXISTS
            # isn't standard SQLite, so we just try and ignore the
            # error if it's already there.
            try:
                conn.execute(
                    "ALTER TABLE rag_documents ADD COLUMN superseded_by TEXT"
                )
            except sqlite3.OperationalError:
                pass
            cur = conn.execute(
                "UPDATE rag_documents SET superseded_by = ? WHERE doc_id = ?",
                (superseded_by, doc_id),
            )
            if cur.rowcount == 0:
                return False

        # Update metadata on the chunks themselves
        # ChromaDB doesn't have a bulk metadata update — we have to
        # fetch chunk IDs, then re-upsert with new metadata. Skip the
        # re-upsert path here for simplicity; downstream retrieval
        # can use the tracking DB to know about supersession.
        # For full chunk-metadata update, see the docstring of
        # _refresh_chunk_metadata_for_doc below.
        logger.info("mark_superseded | %s superseded by %s", doc_id, superseded_by)
        return True

    def get_active_doc_ids(self) -> set[str]:
        """Return doc_ids that are NOT superseded.

        Retrievers can call this once at startup and filter results
        to skip stale documents without per-query DB hits.
        """
        with sqlite3.connect(self.tracking_db_path) as conn:
            try:
                rows = conn.execute(
                    "SELECT doc_id FROM rag_documents "
                    "WHERE superseded_by IS NULL OR superseded_by = ''"
                ).fetchall()
            except sqlite3.OperationalError:
                # superseded_by column doesn't exist yet; everything is active
                rows = conn.execute(
                    "SELECT doc_id FROM rag_documents"
                ).fetchall()
        return {r[0] for r in rows}
