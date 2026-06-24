"""
fix_review_timeout.py
=====================

Fix for the thesis-review step always falling back to the deterministic
writer with:

    agent.thesis_review | NVDA | LLM failed: timed out -> fallback

ROOT CAUSE
----------
The essay and review steps have mismatched time budgets for similar output:

    * thesis_essay._call_ollama_text   -> timeout floor 360s   (SUCCEEDS)
    * thesis_review._call_ollama_text  -> timeout floor 180s   (TIMES OUT)

…yet the review is instructed to write AT LEAST 1200 words
(``_REVIEW_MIN_WORDS = 1200``) with ``num_predict`` = MAX_TOKENS (4096). So the
review is asked to generate as much or more text than the essay, in HALF the
time, on an 8 GB RTX 2080. It reliably exceeds 180s and falls back — which is
why every memo shows the amber "deterministic fallback" banner even though
Ollama and llama3.1:8b are perfectly healthy.

THE FIX
-------
Raise the review's timeout floor from 180.0 to 360.0, matching the essay step
(the proven-working reference for ~1200-word generations on this hardware).
This is the minimal change: it does not touch token budgets, prompts, models,
or the word-count target — it just gives the review the same wall-clock budget
the essay already gets.

This single edit covers BOTH the review and the revision call, because both go
through the same ``_call_ollama_text`` helper.

SAFETY
------
* Targets agent/thesis_review.py.
* Backs up to agent/thesis_review.py.bak before writing.
* Idempotent: detects the 360.0 floor and does nothing on re-run.
* Verifies ast.parse before saving; aborts on any surprise.

Usage (from project root, venv active):
    python fix_review_timeout.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("agent") / "thesis_review.py"

OLD = 'timeout = max(float(getattr(config, "AGENT_TIMEOUT", 30)), 180.0)'
NEW = ('timeout = max(float(getattr(config, "AGENT_TIMEOUT", 30)), 360.0)  '
       '# raised 180->360 to match the essay step; review writes 1200+ words')


def _fail(msg: str) -> None:
    print(f"[fix_review_timeout] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    # Idempotency.
    if "360.0" in src and "180.0" not in src.split("_call_ollama_text", 1)[-1][:400]:
        # Crude but safe: if the review helper region already mentions 360.0
        # and no longer has the 180.0 floor, treat as applied.
        if NEW.split("#")[0].strip() in src:
            print("[fix_review_timeout] Already applied — review timeout floor "
                  "is 360.0. Nothing to do.")
            return

    if OLD not in src:
        # Maybe already patched, or formatting differs.
        if 'getattr(config, "AGENT_TIMEOUT", 30)), 360.0' in src:
            print("[fix_review_timeout] Already applied (360.0 floor present).")
            return
        _fail("could not find the review timeout line "
              "('...AGENT_TIMEOUT...180.0'). The file may have changed; not "
              "editing blindly.")

    src = src.replace(OLD, NEW, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_review_timeout] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Raised review/revision timeout floor 180s -> 360s")
    print()
    print("Re-run `gen NVDA` in the app. The review step should now complete")
    print("with the real LLM, and the amber 'deterministic fallback' banner")
    print("should disappear (model line: llama3.1:8b, no '(failed)').")
    print()
    print("If it STILL times out at 360s, the review generation is genuinely")
    print("too slow on this GPU — next options are lowering num_predict for the")
    print("review call or trimming _REVIEW_MIN_WORDS (1200). Tell me and I'll")
    print("patch that instead.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_review_timeout.py
