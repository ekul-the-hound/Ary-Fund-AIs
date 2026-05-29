"""
rag/document_loaders/notes.py
=============================
Loads hand-written analyst notes from a flat directory of markdown
files.

Convention
----------
Place ``.md`` files in ``data/fund_notes/``. The loader walks the
directory recursively. Each file becomes one RawDocument.

Front matter (optional)
-----------------------
If a file starts with a YAML front-matter block, fields from it are
copied into the document's metadata. Common fields:

    ---
    ticker: AAPL
    as_of: 2024-11-12
    author: jane
    tags: [supply-chain, china]
    ---

If no front matter is present, the loader infers ``ticker`` from the
filename if it matches the pattern ``TICKER_anything.md`` (e.g.
``AAPL_supply_chain_notes.md``).

Why support fund notes?
-----------------------
These are the one source of information competitors don't have. Your
fund's internal reasoning, post-mortems on closed positions, sector
deep-dives — none of this is in 10-Ks. Indexing them gives the agent
access to institutional knowledge.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from rag.document_loaders import RawDocument

logger = logging.getLogger(__name__)


# Match ticker prefix in filenames: AAPL_anything.md, MSFT-research.md
FILENAME_TICKER_RE = re.compile(r"^([A-Z]{1,5})[-_]", re.UNICODE)


class NotesLoader:
    """Walk a directory of .md files and yield RawDocuments."""

    def __init__(self, notes_dir: str = "data/fund_notes"):
        self.notes_dir = Path(notes_dir)

    def load_all(self) -> Iterable[RawDocument]:
        if not self.notes_dir.exists():
            logger.info("notes_loader | %s does not exist, skipping", self.notes_dir)
            return
        for path in sorted(self.notes_dir.rglob("*.md")):
            doc = self._load_file(path)
            if doc:
                yield doc

    def _load_file(self, path: Path) -> Optional[RawDocument]:
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning("notes_loader | %s | read failed: %s", path, e)
            return None
        if not text.strip():
            return None

        # Parse optional YAML front matter
        metadata = {}
        ticker = None
        as_of = None
        body = text
        if text.startswith("---\n"):
            end = text.find("\n---\n", 4)
            if end > 0:
                fm = text[4:end]
                body = text[end + 5:]
                metadata = self._parse_front_matter(fm)
                ticker = (metadata.get("ticker") or "").upper() or None
                as_of = metadata.get("as_of")

        # Infer ticker from filename if not in front matter
        if not ticker:
            m = FILENAME_TICKER_RE.match(path.name)
            if m:
                ticker = m.group(1).upper()

        # Default as_of to file mtime if not specified
        if not as_of:
            as_of = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")

        # Doc ID: relative path with extension stripped, slashes → underscores
        rel = path.relative_to(self.notes_dir).with_suffix("")
        doc_id = "note_" + str(rel).replace("/", "_").replace("\\", "_")

        title = metadata.get("title") or path.stem.replace("_", " ").title()

        return RawDocument(
            doc_id=doc_id,
            doc_type="note",
            text=body.strip(),
            title=title,
            ticker=ticker,
            as_of=as_of,
            metadata=metadata,
        )

    @staticmethod
    def _parse_front_matter(fm: str) -> dict:
        """Tiny YAML subset parser. Handles ``key: value`` and
        ``key: [a, b, c]`` only. We don't pull in PyYAML for two
        fields — that would be over-engineering."""
        out = {}
        for line in fm.splitlines():
            line = line.rstrip()
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                items = [v.strip().strip("'\"") for v in val[1:-1].split(",")]
                out[key] = [v for v in items if v]
            else:
                out[key] = val.strip("'\"")
        return out
