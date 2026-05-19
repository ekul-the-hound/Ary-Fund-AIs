"""
rag/contextualizer.py
=====================
Anthropic-style Contextual Retrieval.

Before embedding a chunk, prepend a 1-3 sentence LLM-generated summary
that situates the chunk inside its source document. Then embed the
combined string. Store the original chunk text for display; only the
embedding changes.

Anthropic's published result on this technique: a ~49% reduction in
retrieval failures, measured on technical documentation. The
mechanism is straightforward — a chunk that originally read
"The Company's exposure to this risk increased materially..."
becomes embedded as if it read
"Apple Inc., 2024 10-K, Risk Factors section, regulatory risk in
mainland China. The Company's exposure to this risk increased
materially..."
which embeds in roughly the right direction for queries that mention
Apple and China.

Reference
---------
https://www.anthropic.com/news/contextual-retrieval

When to use
-----------
* On for indexing of long-form prose (filings, transcripts, theses).
* Off for short structured signals (one-sentence facts) where there's
  no surrounding document to situate them in.
* Off in tests where determinism matters more than quality.

Caching
-------
LLM calls are expensive. The contextualizer caches by
``sha256(chunk_text + doc_id)`` so re-indexing an unchanged chunk
skips the LLM call. The cache lives in the same SQLite file as
embedding cache (different table) to keep all RAG cache in one place.

Failure handling
----------------
If the LLM call fails (Ollama down, timeout, bad response), we fall
back to no contextualization for that chunk. The chunk still gets
indexed — just with the original text as the embedding input. A
partially-contextualized corpus is better than no corpus.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# The contextualization prompt
# ----------------------------------------------------------------------
#
# This prompt is the one part of contextualization where wording
# matters. Three properties to optimize for:
#
# 1. Brevity. Each contextualization adds tokens to the embedding.
#    Long preambles bloat without proportional retrieval lift.
#
# 2. Anchoring. The summary should include the document's identity
#    (ticker, doc type, section) so retrieval can match on it.
#
# 3. Faithfulness. The summary should describe what the chunk
#    actually says, not what the document says generally. Otherwise
#    every chunk in a 10-K gets the same generic summary and
#    contextualization adds noise.

CONTEXTUALIZE_PROMPT = """You are summarizing a chunk of a financial document so it can be \
retrieved by similarity search.

<document>
{document_summary}
</document>

<chunk>
{chunk_text}
</chunk>

Write 1-2 short sentences that:
1. Identify the source (company, document type, section if applicable).
2. State what THIS SPECIFIC CHUNK discusses — not the document generally.

Output ONLY the summary. No preamble, no quotes, no XML tags. \
Keep it under 60 words."""


# A separate, simpler prompt for documents that are short enough to
# pass entirely as the "document" context (theses, notes). The model
# can read the whole thing, so the prompt asks for chunk-specific
# summary that may reference document context implicitly.
CONTEXTUALIZE_PROMPT_SHORT_DOC = """You are summarizing a chunk so it can be retrieved by similarity search.

<document>
{document_text}
</document>

<chunk>
{chunk_text}
</chunk>

Write 1-2 short sentences identifying what this chunk discusses, \
referencing relevant context from the document. \
Output ONLY the summary. Keep it under 60 words."""


# ----------------------------------------------------------------------
# Document summary builder
# ----------------------------------------------------------------------

def build_document_summary(
    doc_id: str,
    doc_type: str,
    title: Optional[str] = None,
    ticker: Optional[str] = None,
    as_of: Optional[str] = None,
    section: Optional[str] = None,
    extra: Optional[dict] = None,
) -> str:
    """Compress a document's identity into a short header string.

    Used as the "<document>" block in the contextualize prompt when
    the full document is too long to pass to the LLM (a 200-page
    10-K won't fit in 8K of context). The header captures everything
    a 1-2 sentence summary needs to know.
    """
    parts = []
    if title:
        parts.append(title)
    elif ticker:
        parts.append(f"{ticker} {doc_type}")
    else:
        parts.append(doc_type)
    if as_of:
        parts.append(f"dated {as_of}")
    if section:
        parts.append(f"in the section: {section}")
    if extra:
        for k, v in extra.items():
            if v and k not in ("doc_id", "doc_type", "ticker", "as_of",
                               "title", "section", "n_tokens",
                               "chunk_index", "section_sub_index",
                               "section_total_chunks", "turn_sub_index"):
                parts.append(f"{k}={v}")
    return ", ".join(str(p) for p in parts if p)


# ----------------------------------------------------------------------
# Contextualizer class
# ----------------------------------------------------------------------

class Contextualizer:
    """Adds LLM-generated context prefixes to chunks before embedding.

    Parameters
    ----------
    llm_fn:
        Callable ``(prompt: str) -> str``. Wraps whatever LLM you use.
        For Ary Quant the natural choice is your ``base_agent.ask_agent``
        with a system message that produces short summaries. Pass
        None to disable contextualization (the class becomes a no-op
        passthrough).
    cache_db_path:
        SQLite cache for summaries. Default colocates with the
        embedding cache.
    max_doc_chars:
        If a document is shorter than this, pass its full text to the
        prompt. Otherwise pass only a metadata header (built from
        doc_id, ticker, section). 8000 chars ≈ 2000 tokens — fits in
        most local models' context with room to spare.
    timeout:
        Seconds before giving up on the LLM call. On failure we fall
        back to no contextualization for that chunk.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        cache_db_path: str = "data/rag_embeddings.db",
        max_doc_chars: int = 8000,
        timeout: float = 30.0,
    ):
        self.llm_fn = llm_fn
        self.cache_db_path = cache_db_path
        self.max_doc_chars = max_doc_chars
        self.timeout = timeout
        self._lock = threading.Lock()
        if cache_db_path:
            Path(cache_db_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_cache_db()

    @property
    def enabled(self) -> bool:
        return self.llm_fn is not None

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _init_cache_db(self) -> None:
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS contextualize_cache (
                    key TEXT PRIMARY KEY,
                    context_prefix TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

    @staticmethod
    def _cache_key(chunk_text: str, doc_id: str) -> str:
        h = hashlib.sha256()
        h.update(doc_id.encode())
        h.update(b"\x00")
        h.update(chunk_text.encode())
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        with self._lock, sqlite3.connect(self.cache_db_path) as conn:
            row = conn.execute(
                "SELECT context_prefix FROM contextualize_cache WHERE key = ?",
                (key,),
            ).fetchone()
        return row[0] if row else None

    def _cache_put(self, key: str, prefix: str) -> None:
        with self._lock, sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO contextualize_cache (key, context_prefix) "
                "VALUES (?, ?)",
                (key, prefix),
            )

    # ------------------------------------------------------------------
    # The actual contextualization
    # ------------------------------------------------------------------
    def contextualize_chunk(
        self,
        chunk_text: str,
        full_document_text: str,
        chunk_metadata: dict,
    ) -> str:
        """Return ``context_prefix + chunk_text`` ready for embedding.

        If contextualization is disabled or fails, returns the raw
        chunk text unchanged.
        """
        if not self.enabled or not chunk_text.strip():
            return chunk_text

        doc_id = chunk_metadata.get("doc_id", "")
        key = self._cache_key(chunk_text, doc_id)

        # Cache hit
        cached = self._cache_get(key)
        if cached is not None:
            return f"{cached}\n\n{chunk_text}" if cached else chunk_text

        # Build prompt — short doc gets full text, long doc gets header
        if len(full_document_text) <= self.max_doc_chars:
            prompt = CONTEXTUALIZE_PROMPT_SHORT_DOC.format(
                document_text=full_document_text,
                chunk_text=chunk_text,
            )
        else:
            summary = build_document_summary(
                doc_id=doc_id,
                doc_type=chunk_metadata.get("doc_type", ""),
                title=chunk_metadata.get("title"),
                ticker=chunk_metadata.get("ticker"),
                as_of=chunk_metadata.get("as_of"),
                section=chunk_metadata.get("section"),
                extra=chunk_metadata,
            )
            prompt = CONTEXTUALIZE_PROMPT.format(
                document_summary=summary,
                chunk_text=chunk_text,
            )

        # Call the LLM
        t0 = time.perf_counter()
        try:
            result = self.llm_fn(prompt)
            elapsed = time.perf_counter() - t0
            if not isinstance(result, str):
                logger.warning("contextualizer | non-string result for %s", doc_id)
                self._cache_put(key, "")
                return chunk_text
            prefix = result.strip()
            # Defensive trim — if the model rambled past 200 words we'd
            # bloat every embedding. Take first 3 sentences.
            if len(prefix) > 600:
                # Split on sentence-ending punctuation
                import re
                sentences = re.split(r"(?<=[.!?])\s+", prefix)
                prefix = " ".join(sentences[:3])
            self._cache_put(key, prefix)
            logger.debug(
                "contextualizer | %s | %d → %d chars in %.0fms",
                doc_id, len(chunk_text), len(prefix), elapsed * 1000,
            )
            return f"{prefix}\n\n{chunk_text}"
        except Exception as e:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            logger.warning(
                "contextualizer | %s | LLM failed after %.0fms: %s",
                doc_id, elapsed * 1000, e,
            )
            # Cache the failure as empty string so we don't keep retrying
            self._cache_put(key, "")
            return chunk_text


# ----------------------------------------------------------------------
# Factory helpers
# ----------------------------------------------------------------------

def make_ollama_contextualizer(
    model: str = "phi3:3.8b",
    base_url: str = "http://localhost:11434",
    **kwargs,
) -> Contextualizer:
    """Build a Contextualizer backed by Ollama directly.

    The default model is phi3:3.8b — fast, runs on your hardware, and
    is good enough for summary generation. For higher quality at
    indexing time (which only runs once per chunk lifetime), pass
    ``model="qwen3:30b-a3b"`` or similar.
    """
    import requests

    def llm_fn(prompt: str) -> str:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 200},
            },
            timeout=kwargs.get("timeout", 30.0),
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    return Contextualizer(llm_fn=llm_fn, **kwargs)


def make_disabled_contextualizer() -> Contextualizer:
    """A Contextualizer with llm_fn=None — every method becomes a
    pass-through. Useful in tests and as the default when the user
    hasn't enabled contextualization."""
    return Contextualizer(llm_fn=None)
