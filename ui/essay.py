"""Essay loader — bridges ``agent.thesis_essay`` to the Streamlit UI.

The essay (the long-form 'briefing') is expensive (30s–2min on a local
30B model) and not currently persisted by the pipeline, so we cache it
per-ticker in ``st.session_state`` for the life of the browser session.

A single public function, ``get_or_generate_essay(ticker, context)``,
returns the full essay dict or an error marker. The caller decides
whether to block on generation or show a 'generate' CTA — this module
doesn't render UI itself.
"""
from __future__ import annotations

import logging
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# Keyed by ticker. Stored under st.session_state so it survives reruns
# but not full page reloads (which is the right lifecycle: a new session
# means the user may want fresh data).
_SESSION_KEY = "ary_quant_essay_cache"


def _cache() -> dict[str, dict[str, Any]]:
    if _SESSION_KEY not in st.session_state:
        st.session_state[_SESSION_KEY] = {}
    return st.session_state[_SESSION_KEY]


def get_cached_essay(ticker: str) -> dict[str, Any] | None:
    """Return the cached essay dict for this ticker, or None."""
    return _cache().get(ticker)


def clear_cached_essay(ticker: str) -> None:
    _cache().pop(ticker, None)


def _import_thesis_essay() -> Any:
    """Try to import thesis_essay from every plausible location.

    Returns the module on success, or None if nothing works. Covers:
      - Flat layout: `thesis_essay.py` at repo root
      - Packaged layout: `agent/thesis_essay.py`
      - Alternate packages: `ary_quant/thesis_essay.py`, `src/thesis_essay.py`
      - Explicit filesystem search from the current file up to 4 parents
        (handles weird cwd situations when Streamlit runs from odd places)
    """
    import importlib
    import sys
    from pathlib import Path

    candidates = [
        "thesis_essay",
        "quant.thesis_essay",
        "agent.thesis_essay",
        "ary_quant.thesis_essay",
        "src.thesis_essay",
    ]
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
        except Exception as e:
            logger.warning("import %s raised non-ImportError: %s", name, e)
            continue

    # Last resort: walk up from this file looking for thesis_essay.py on disk
    # and add its parent directory to sys.path. This catches the case where
    # the file exists but isn't on the Python path for any reason.
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "thesis_essay.py"
        if candidate.is_file():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            try:
                return importlib.import_module("thesis_essay")
            except Exception as e:
                logger.warning("fs-located thesis_essay at %s failed: %s", candidate, e)
                break

    return None


def generate_essay(
    ticker: str,
    context: dict[str, Any],
    config_module: Any,
) -> dict[str, Any]:
    """Generate a fresh essay and cache it. Returns the essay dict.

    The essay dict shape (from ``thesis_essay.generate_thesis_essay``):
        {
            "text": str,
            "model_used": str,
            "elapsed_ms": float,
            "fallback": bool,
            "word_count": int,
        }

    On any failure we return an error marker dict with a ``text`` field
    describing what happened, so the UI can still render something.
    """
    thesis_essay = _import_thesis_essay()
    if thesis_essay is None:
        return _error_marker(
            "thesis_essay module not found. Checked: `thesis_essay`, "
            "`quant.thesis_essay`, `agent.thesis_essay`, `ary_quant.thesis_essay`, "
            "`src.thesis_essay`. Move thesis_essay.py to a package that's on "
            "sys.path, or adjust the candidate list in ui/essay.py."
        )

    thesis = context.get("thesis") or {}
    # context has 'metrics' from pipeline.build_agent_context; thesis_essay
    # expects the same shape thesis_generator.generate_thesis was built on
    # (it's happy with either the raw metrics dict or the 'key_metrics'
    # shape from filing_analyzer, since it only reads named keys).
    metrics = context.get("metrics") or context.get("key_metrics") or {}
    macro = context.get("macro") or {}
    risk_flags = context.get("risk") or {}
    filings_summary = (
        context.get("filings_summary")
        or context.get("filings")
        or {}
    )
    # If filings is a list (pipeline's shape), wrap it minimally so the
    # essay formatter doesn't choke. thesis_essay._format_filings_block
    # tolerates a plain list under 'recent' well enough.
    if isinstance(filings_summary, list):
        filings_summary = {"recent": filings_summary}

    try:
        essay = thesis_essay.generate_thesis_essay(
            ticker=ticker,
            thesis=thesis,
            filings_summary=filings_summary,
            metrics=metrics,
            macro=macro,
            risk_flags=risk_flags,
            config=config_module,
        )
    except Exception as e:
        logger.exception("Essay generation failed for %s", ticker)
        return _error_marker(f"Essay generation raised: {e}")

    if not isinstance(essay, dict) or not essay.get("text"):
        return _error_marker("Essay generator returned empty output.")

    _cache()[ticker] = essay
    return essay


def _error_marker(message: str) -> dict[str, Any]:
    return {
        "text": f"⚠️ Could not generate a briefing:\n\n{message}",
        "model_used": "error",
        "elapsed_ms": 0.0,
        "fallback": True,
        "word_count": 0,
        "error": True,
    }