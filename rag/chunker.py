"""
rag/chunker.py
==============
Splits documents into retrieval units ("chunks") of approximately
500 tokens with 50 token overlap.

Why three strategies?
---------------------
Different document types have different structural markers, and the
chunker is most useful when it respects them.

* ``chunk_sec_filing``    — 10-K, 10-Q, 8-K, DEF 14A, etc. Parses
  ``Item N.`` headers and chunks within each section. Section title
  goes into chunk metadata so retrieval can filter or boost on it.

* ``chunk_transcript``    — Earnings calls and similar. Splits on
  speaker turns. Speaker name and role go into metadata.

* ``chunk_unstructured``  — Theses, notes, anything without a known
  structure. Recursive splitter: try paragraph boundaries first, fall
  back to sentence boundaries, fall back to hard token cuts.

The chunk dataclass
-------------------
Every chunk carries its own metadata. The retriever uses this metadata
both for filtering (``where ticker = 'AAPL'``) and for display (showing
"Item 1A — Risk Factors" alongside the chunk text).

Tokens are counted approximately. We don't import the embedder's
tokenizer — that would couple this module to a specific model and
mostly buy us nothing. ~4 characters per token for English prose is
close enough for sizing purposes.

The ID is deterministic
-----------------------
A chunk's ``chunk_id`` is a SHA-256 of ``(doc_id, chunk_index, text)``.
Two consequences:

1. Re-chunking the same document produces the same IDs. The vector
   store sees the same primary key and updates in place rather than
   creating duplicates.

2. If the document text changes, the IDs change too. Stale chunks can
   be detected by comparing "IDs currently in the store for this doc"
   vs. "IDs the new chunker run produces" — anything in the first set
   but not the second is stale and should be deleted.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Sizing constants
# ----------------------------------------------------------------------

DEFAULT_CHUNK_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50

# Rough proxy: tokens ≈ chars / 4 for English prose. The "real"
# tokenizer (BPE, SentencePiece, etc.) varies by model, but the
# approximation is good enough for chunk sizing.
CHARS_PER_TOKEN = 4


def approx_token_count(text: str) -> int:
    """Approximate token count from char count. Off by ±15% vs real
    tokenizers, which is fine — we only need this to size chunks."""
    return max(1, len(text) // CHARS_PER_TOKEN)


# ----------------------------------------------------------------------
# Chunk dataclass
# ----------------------------------------------------------------------

@dataclass
class Chunk:
    """A single retrieval unit.

    Attributes
    ----------
    chunk_id:
        Deterministic SHA-256 hash. Stable across re-chunking runs as
        long as the input text is unchanged. Used as the primary key
        in the vector store.
    doc_id:
        Identifier of the source document (e.g. "AAPL_10-K_2025-09-28").
        Lets us delete or update all chunks of a document at once.
    text:
        The actual content. This is what gets embedded.
    metadata:
        Filter and display info. Typical keys:
            * doc_type:    'filing' | 'transcript' | 'thesis' | 'note'
            * ticker:      'AAPL' (or None for non-ticker docs)
            * as_of:       ISO date of the document
            * section:     'Item 1A — Risk Factors' (filings)
            * speaker:     'Tim Cook' (transcripts)
            * chunk_index: position within the source document
            * n_tokens:    approximate token count
    """
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, chunk_index: int, text: str) -> str:
        h = hashlib.sha256()
        h.update(doc_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(chunk_index).encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()[:16]  # 16 hex chars = 64 bits, plenty of unique IDs


# ----------------------------------------------------------------------
# Recursive splitting (the bottom-layer primitive)
# ----------------------------------------------------------------------

# These splitters are tried in order. The first one that produces
# pieces small enough to fit the chunk size wins. ``\n\n`` for
# paragraph breaks, ``\n`` for line breaks, ``. `` for sentence
# boundaries, then hard cuts.
SPLITTERS = ["\n\n", "\n", ". ", " "]


def _recursive_split(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
    splitters: list[str] = SPLITTERS,
) -> list[str]:
    """Split text into roughly equal-sized pieces, preferring natural
    boundaries.

    Algorithm
    ---------
    1. If the whole text fits in one chunk, return it as-is.
    2. Otherwise, try each splitter in order:
        a. Split on that delimiter.
        b. Greedily pack pieces back together until adding the next
           one would exceed the chunk size.
        c. If packing succeeds (every accumulated piece fits), we're
           done; return the packed chunks.
        d. Otherwise the first piece alone is too big — recurse into
           that piece with the next splitter.
    3. If all splitters fail (we hit the hard fallback), do a brute
       character-based split.

    Overlap is added at the end by re-walking the chunk list and
    prepending the tail of the previous chunk to each subsequent one.
    """
    chunk_size_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    if approx_token_count(text) <= chunk_tokens:
        return [text.strip()] if text.strip() else []

    # Try splitters in order
    chunks: list[str] = []
    for sep in splitters:
        pieces = text.split(sep)
        if len(pieces) == 1:
            continue  # this separator doesn't appear, move on

        current = ""
        for piece in pieces:
            # Adding `piece + sep` to current — does it fit?
            candidate = current + (sep if current else "") + piece
            if len(candidate) <= chunk_size_chars:
                current = candidate
            else:
                # Current chunk is full. Save it, start a new one
                # with this piece.
                if current.strip():
                    chunks.append(current.strip())
                # Edge case: a single piece may itself exceed the
                # chunk size. Recurse with the next splitter level.
                if len(piece) > chunk_size_chars:
                    next_splitters = splitters[splitters.index(sep) + 1:]
                    if next_splitters:
                        chunks.extend(_recursive_split(
                            piece, chunk_tokens, overlap_tokens,
                            splitters=next_splitters,
                        ))
                        current = ""
                    else:
                        # Last-resort hard chop
                        chunks.extend(_hard_chop(piece, chunk_size_chars))
                        current = ""
                else:
                    current = piece
        if current.strip():
            chunks.append(current.strip())

        if chunks:
            break  # This splitter level worked; don't try shorter delimiters

    if not chunks:
        # No splitter worked. Hard chop.
        chunks = _hard_chop(text, chunk_size_chars)

    # Add overlap: prepend the tail of chunk[i-1] to chunk[i]
    if overlap_chars > 0 and len(chunks) > 1:
        with_overlap = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap_chars:]
            # Use a space to glue if the previous tail doesn't already
            # end at a word boundary; this prevents word-fusion artifacts.
            glue = "" if prev_tail.endswith(" ") or chunks[i].startswith(" ") else " "
            with_overlap.append(prev_tail + glue + chunks[i])
        chunks = with_overlap

    return chunks


def _hard_chop(text: str, chunk_size_chars: int) -> list[str]:
    """Fallback: split at character boundaries with no respect for
    semantics. Last resort when no separator exists in the text."""
    return [text[i:i + chunk_size_chars]
            for i in range(0, len(text), chunk_size_chars)]


# ----------------------------------------------------------------------
# SEC filing chunker (the main one)
# ----------------------------------------------------------------------

# Matches lines like "Item 1.", "Item 1A.", "ITEM 7A." at the start of
# a line. Captures the number, the optional letter suffix, and any
# trailing title text up to end of line.
#
# The (?im) flags: case-insensitive, multiline (^ matches start-of-line).
ITEM_HEADER_RE = re.compile(
    r"(?im)^\s*Item\s+(\d+)([A-Z]?)\.?\s*([^\n]{0,200}?)\s*$",
)


def chunk_sec_filing(
    raw_text: str,
    doc_id: str,
    base_metadata: dict,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Section-aware chunker for SEC filings.

    Walks ``raw_text`` looking for ``Item N.`` headers. Each section
    (header → next header) is chunked independently using the
    recursive splitter. The section name flows into metadata so
    retrieval can filter on it ("only Item 1A chunks") or display it
    ("from Apple's Risk Factors section").

    If no Item headers are found, falls back to unstructured chunking.
    This handles edge cases like filings that use a non-standard
    template (some 8-Ks) and short proxy statements.
    """
    matches = list(ITEM_HEADER_RE.finditer(raw_text))
    if not matches:
        logger.info("chunker | %s | no Item headers — falling back to unstructured", doc_id)
        return chunk_unstructured(
            raw_text, doc_id, base_metadata,
            chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
        )

    # Build (section_name, section_text) tuples. The text of section i
    # is from match[i].end() to match[i+1].start() (or end-of-doc for
    # the last section).
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        item_num, item_suffix, title = m.group(1), m.group(2), m.group(3)
        section_name = f"Item {item_num}{item_suffix}"
        if title:
            section_name += f" — {title.strip()}"
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[body_start:body_end].strip()
        if body:
            sections.append((section_name, body))

    # Chunk each section. Maintain a single global chunk_index so
    # ordering across sections is preserved.
    chunks: list[Chunk] = []
    chunk_index = 0
    for section_name, body in sections:
        pieces = _recursive_split(body, chunk_tokens, overlap_tokens)
        n_pieces = len(pieces)
        for sub_idx, piece in enumerate(pieces):
            md = dict(base_metadata)
            md["section"] = section_name
            md["chunk_index"] = chunk_index
            md["section_sub_index"] = sub_idx
            md["section_total_chunks"] = n_pieces
            md["n_tokens"] = approx_token_count(piece)
            chunk_id = Chunk.make_id(doc_id, chunk_index, piece)
            chunks.append(Chunk(
                chunk_id=chunk_id, doc_id=doc_id, text=piece, metadata=md,
            ))
            chunk_index += 1

    return chunks


# ----------------------------------------------------------------------
# Transcript chunker
# ----------------------------------------------------------------------

# Earnings call transcripts typically look like:
#   "Tim Cook - Chief Executive Officer
#
#    Thanks, operator. Good afternoon, everyone...
#
#    Toni Sacconaghi - Analyst, Bernstein
#
#    Hi, thanks. Tim, I wanted to ask about..."
#
# The defining property of a speaker header is that it's a line BY
# ITSELF — flanked by blank lines on both sides. Without that
# requirement, we'd match phrases like "Thanks, Suhasini." inside the
# body as a fake speaker. The lookbehind/lookahead enforce isolation.
#
# Pattern:
#   ``(?<=\n\n)`` — preceded by a blank line (or start-of-doc, handled
#                   below by a separate pass)
#   ``[A-Z]...``  — 1-4 capitalized words (the name)
#   `` [-,] ``    — separator
#   ``[^\n]+``    — title text on the same line
#   ``\n\n``      — followed by a blank line

TRANSCRIPT_SPEAKER_RE = re.compile(
    r"(?:^|\n\n)"
    r"([A-Z][a-zA-Z\.\-']+(?:\s+[A-Z][a-zA-Z\.\-']+){0,3})"
    r"\s+[-,]\s+"
    r"([^\n]{2,120})"
    r"\n\n",
)


def chunk_transcript(
    raw_text: str,
    doc_id: str,
    base_metadata: dict,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk an earnings call transcript by speaker turn.

    A speaker turn is the text between two speaker headers. Long turns
    are split further with the recursive splitter; short turns become
    a single chunk. Speaker name and title go into metadata so a query
    can be filtered to "only CFO turns" or "only analyst Q&A."
    """
    matches = list(TRANSCRIPT_SPEAKER_RE.finditer(raw_text))
    if not matches:
        logger.info("chunker | %s | no speaker turns — falling back to unstructured", doc_id)
        return chunk_unstructured(
            raw_text, doc_id, base_metadata,
            chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
        )

    chunks: list[Chunk] = []
    chunk_index = 0
    for i, m in enumerate(matches):
        speaker = m.group(1).strip()
        title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[body_start:body_end].strip()
        if not body:
            continue

        # Heuristic: "Analyst" or "Q&A" in title => this is a question
        # turn; otherwise a management/prepared remarks turn. Helps
        # retrieval target Q&A specifically.
        is_qa = any(w in title.lower() for w in ("analyst", "q&a", "questions"))
        role = "analyst" if is_qa else "management"

        pieces = _recursive_split(body, chunk_tokens, overlap_tokens)
        for sub_idx, piece in enumerate(pieces):
            md = dict(base_metadata)
            md["speaker"] = speaker
            md["speaker_title"] = title
            md["speaker_role"] = role
            md["chunk_index"] = chunk_index
            md["turn_sub_index"] = sub_idx
            md["n_tokens"] = approx_token_count(piece)
            chunks.append(Chunk(
                chunk_id=Chunk.make_id(doc_id, chunk_index, piece),
                doc_id=doc_id, text=piece, metadata=md,
            ))
            chunk_index += 1

    return chunks


# ----------------------------------------------------------------------
# Unstructured chunker (theses, notes, anything else)
# ----------------------------------------------------------------------

def chunk_unstructured(
    raw_text: str,
    doc_id: str,
    base_metadata: dict,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Recursive splitter for prose without known structure.

    Used for theses, fund notes, agent opinions, and as a fallback
    when section parsing of structured docs fails.
    """
    pieces = _recursive_split(raw_text, chunk_tokens, overlap_tokens)
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        md = dict(base_metadata)
        md["chunk_index"] = i
        md["n_tokens"] = approx_token_count(piece)
        chunks.append(Chunk(
            chunk_id=Chunk.make_id(doc_id, i, piece),
            doc_id=doc_id, text=piece, metadata=md,
        ))
    return chunks


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------

def chunk_document(
    raw_text: str,
    doc_id: str,
    doc_type: str,
    base_metadata: Optional[dict] = None,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Pick the right chunker for the document type.

    Parameters
    ----------
    raw_text:
        The document body.
    doc_id:
        Identifier for this document (used to make chunk IDs).
    doc_type:
        One of 'filing', 'transcript', 'thesis', 'note', or anything
        else (which falls through to unstructured).
    base_metadata:
        Additional metadata to copy onto every chunk produced. Things
        like ``{"ticker": "AAPL", "as_of": "2025-09-28"}``.

    Returns
    -------
    List of Chunk objects, empty if the document had no content.
    """
    base_metadata = dict(base_metadata or {})
    base_metadata["doc_type"] = doc_type
    base_metadata["doc_id"] = doc_id

    if not raw_text or not raw_text.strip():
        return []

    if doc_type == "filing":
        return chunk_sec_filing(raw_text, doc_id, base_metadata,
                                chunk_tokens, overlap_tokens)
    elif doc_type == "transcript":
        return chunk_transcript(raw_text, doc_id, base_metadata,
                                chunk_tokens, overlap_tokens)
    else:
        return chunk_unstructured(raw_text, doc_id, base_metadata,
                                  chunk_tokens, overlap_tokens)
