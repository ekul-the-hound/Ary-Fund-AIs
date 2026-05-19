"""
rag/query_expander.py
=====================
LLM-driven query expansion.

The problem
-----------
A single query rarely covers all angles of what a user is asking. The
question "is Apple's dividend safe?" has at least four sub-aspects:

    * dividend coverage / payout ratio
    * free cash flow trend
    * balance sheet strength
    * past behavior in downturns

A single embedding of "is Apple's dividend safe?" retrieves chunks
about dividends generally but probably misses the chunks that have
the actual numbers on coverage or FCF.

Multi-query retrieval fixes this. We ask an LLM to expand the user's
question into 3-5 sub-queries covering different angles. Each is
retrieved independently; we dedupe and merge.

Compared to Phase 2's hardcoded 3-query template, this:
    * handles arbitrary questions (not just "tell me about X")
    * adapts to the actual user intent
    * costs one extra LLM call per retrieval (~500ms)

Caching
-------
The same question expands to the same sub-queries. Cache keys are
sha256(question + model_name). Cache hit ratio in practice is low
(most questions are unique) but the storage is negligible.

Failure modes
-------------
* LLM unavailable → fall back to using the original question only.
* LLM returns malformed output → also fall back to the original.

Either way, retrieval continues to work; quality just degrades to
single-query.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# The expansion prompt. Tuned for:
#   1. Brevity — sub-queries are SHORT, not paragraphs.
#   2. Coverage — different aspects, not paraphrases.
#   3. Format — JSON list so parsing is trivial.
EXPAND_PROMPT = """You are helping retrieve relevant documents for a financial research question.

The user is asking: "{question}"

Generate exactly {n} short search queries that cover DIFFERENT aspects of this \
question. Each query should be 3-8 words. Avoid duplicates or paraphrases — each \
query should target a distinct angle.

Output ONLY a JSON array of strings. No preamble, no explanation, no code fence.

Example output: ["dividend coverage ratio", "free cash flow trend", "balance sheet debt"]
"""


class QueryExpander:
    """Expands a user question into multiple retrieval queries via LLM.

    Parameters
    ----------
    llm_fn:
        Callable ``(prompt: str) -> str``. Pass None to disable
        expansion (every call returns ``[question]``).
    cache_db_path:
        SQLite cache for expansions.
    n_expansions:
        How many sub-queries to generate. 3-5 is the sweet spot.
        More queries = more retrieval coverage but also more total
        chunks to merge (and more LLM tokens).
    include_original:
        If True (default), the user's original question is included
        in the returned list alongside the LLM-generated sub-queries.
        Useful when the original phrasing is itself a good retrieval
        query.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        cache_db_path: str = "data/rag_embeddings.db",
        n_expansions: int = 4,
        include_original: bool = True,
    ):
        self.llm_fn = llm_fn
        self.cache_db_path = cache_db_path
        self.n_expansions = n_expansions
        self.include_original = include_original
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
                CREATE TABLE IF NOT EXISTS query_expand_cache (
                    key TEXT PRIMARY KEY,
                    expansions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

    def _cache_key(self, question: str) -> str:
        h = hashlib.sha256()
        h.update(f"n={self.n_expansions}|inc_orig={self.include_original}".encode())
        h.update(b"\x00")
        h.update(question.encode())
        return h.hexdigest()

    def _cache_get(self, key: str) -> Optional[list[str]]:
        with self._lock, sqlite3.connect(self.cache_db_path) as conn:
            row = conn.execute(
                "SELECT expansions_json FROM query_expand_cache WHERE key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def _cache_put(self, key: str, expansions: list[str]) -> None:
        with self._lock, sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO query_expand_cache (key, expansions_json) "
                "VALUES (?, ?)",
                (key, json.dumps(expansions)),
            )

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------
    def expand(self, question: str) -> list[str]:
        """Return a list of retrieval queries.

        Always includes the original question (if ``include_original``).
        Falls back to ``[question]`` if expansion fails.
        """
        question = (question or "").strip()
        if not question:
            return []

        # Disabled → just return the original
        if not self.enabled:
            return [question]

        # Cache
        key = self._cache_key(question)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        # LLM call
        try:
            raw = self.llm_fn(
                EXPAND_PROMPT.format(question=question, n=self.n_expansions)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("query_expander | LLM call failed: %s", e)
            return [question] if self.include_original else []

        expansions = self._parse_expansions(raw)
        if not expansions:
            logger.warning("query_expander | got no usable expansions from LLM output: %r",
                           raw[:200])
            self._cache_put(key, [question] if self.include_original else [])
            return [question] if self.include_original else []

        # Optionally prepend the original
        result = ([question] + expansions) if self.include_original else expansions
        # Dedupe while preserving order
        seen = set()
        deduped = []
        for q in result:
            qn = q.strip().lower()
            if qn and qn not in seen:
                seen.add(qn)
                deduped.append(q.strip())

        self._cache_put(key, deduped)
        return deduped

    @staticmethod
    def _parse_expansions(raw: str) -> list[str]:
        """Pull the JSON array out of the LLM's raw output.

        Strategies, in order:
            1. Direct json.loads of the whole output.
            2. Find the first [...] block and parse it.
            3. Fall back to splitting on newlines / bullet markers.
        """
        if not raw:
            return []
        raw = raw.strip()

        # 1. Direct parse
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
        except json.JSONDecodeError:
            pass

        # 2. Find first JSON array
        m = re.search(r"\[\s*(?:\".*?\"\s*,?\s*)+\]", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
            except json.JSONDecodeError:
                pass

        # 3. Bullet / line fallback. Look for lines starting with -, *,
        # a digit, or a quote. Strip those markers.
        lines = raw.splitlines()
        result = []
        for line in lines:
            line = line.strip().lstrip("-*").lstrip("0123456789.").strip()
            line = line.strip("\"'`")
            if line and len(line) < 200:  # sanity limit
                result.append(line)
        return result[:10]  # cap


# ----------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------

def make_ollama_expander(
    model: str = "phi3:3.8b",
    base_url: str = "http://localhost:11434",
    **kwargs,
) -> QueryExpander:
    """Build a QueryExpander backed by Ollama."""
    import requests

    def llm_fn(prompt: str) -> str:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200},
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    return QueryExpander(llm_fn=llm_fn, **kwargs)


def make_disabled_expander() -> QueryExpander:
    """A no-op expander. Always returns ``[question]`` unchanged."""
    return QueryExpander(llm_fn=None)
