"""
fix_scheduler_offline.py
=======================

Propagate the HuggingFace offline flags to the refresh_scheduler entry point.

CONTEXT
-------
``main.py`` sets these at the very top so cached HF models (the reranker
cross-encoder and the sentence-transformers embedder) load without
re-validating against huggingface.co on every startup:

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

But ``refresh_scheduler.py`` is its OWN entry point (invoked from cron /
APScheduler / a Streamlit button), and it never set these. So when the
scheduler's daily cadence runs the RAG index / reranker, HuggingFace tries to
phone home and re-validate — adding latency and failing outright if the box is
offline.

THE FIX
-------
Insert the same ``os.environ.setdefault`` block into refresh_scheduler.py,
immediately AFTER ``from __future__ import annotations`` (which must remain the
first statement) and BEFORE any other import — because the HF libraries read
these env vars at import time, so the flags must be set before anything pulls
them in.

SAFETY
------
* Targets refresh_scheduler.py.
* Backs up to refresh_scheduler.py.bak before writing.
* Idempotent: detects HF_HUB_OFFLINE and does nothing on re-run.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_scheduler_offline.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("data") / "refresh_scheduler.py"

ANCHOR = "from __future__ import annotations"

INSERT = '''
# Load cached HuggingFace models (reranker cross-encoder, embedder) without
# re-validating against huggingface.co on every run. Must be set before any
# import pulls in a huggingface library, so it lives here at the top of this
# entry point (the scheduler is invoked independently of main.py, which sets
# the same flags). See fix_scheduler_offline.py.
import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")'''


def _fail(msg: str) -> None:
    print(f"[fix_scheduler_offline] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if "HF_HUB_OFFLINE" in src:
        print("[fix_scheduler_offline] Already applied — HF_HUB_OFFLINE present. "
              "Nothing to do.")
        return

    if ANCHOR not in src:
        _fail("could not find 'from __future__ import annotations' to anchor "
              "the insert. Not editing blindly.")

    # Insert right after the __future__ line (keeping __future__ first).
    src = src.replace(ANCHOR, ANCHOR + INSERT, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_scheduler_offline] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Inserted HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE at the top of")
    print("    refresh_scheduler.py (after __future__, before other imports)")
    print()
    print("The scheduler's RAG/reranker steps will no longer re-validate against")
    print("huggingface.co on each run.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_scheduler_offline.py