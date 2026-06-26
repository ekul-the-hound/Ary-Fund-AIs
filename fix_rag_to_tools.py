"""
fix_rag_to_tools.py
==================

Move the RAG learning panel from the Lab's view radio to the sidebar "Tools"
section (next to Metrics / Debug).

WHAT IT DOES
------------
app_v2.py:
  * Adds "RAG" to _UTILITY_DESTINATIONS (so a button appears in Tools).
  * Adds a render branch: when destination == "RAG", call lab's
    _render_rag_learning_panel().

lab.py:
  * Removes "RAG learning" from the view radio and its dispatch branch
    (the panel function _render_rag_learning_panel + _rag_job_worker stay
    defined in lab.py and are imported by app_v2 — no need to move the code).

SAFETY
------
* Targets ui/app_v2.py and ui/lab.py.
* Backs up each to .py.bak before writing.
* Idempotent: detects prior application on each file.
* Verifies ast.parse on each before saving.
* If lab.py doesn't have the RAG radio entry (e.g. fix_lab_rag_button wasn't
  applied), it warns but still wires the Tools button — as long as the panel
  function exists.

Usage (from project root, venv active):
    python fix_rag_to_tools.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

APP = Path("ui") / "app_v2.py"
LAB = Path("ui") / "lab.py"

# --- app_v2.py edits -----------------------------------------------------
UTIL_ANCHOR = '_UTILITY_DESTINATIONS = ["Metrics", "Debug"]'
UTIL_INSERT = '_UTILITY_DESTINATIONS = ["Metrics", "Debug", "RAG"]'

# Add a dispatch branch. Anchor on the Metrics branch in the main router.
DISPATCH_ANCHOR = '''    elif dest == "Metrics":
        _render_metrics_destination(backend)'''
DISPATCH_INSERT = '''    elif dest == "Metrics":
        _render_metrics_destination(backend)
    elif dest == "RAG":
        _render_rag_destination()'''

# Add the render function. Anchor before _render_metrics_destination.
RENDERFN_ANCHOR = "def _render_metrics_destination(backend: dict[str, Any]) -> None:"
RENDERFN_INSERT = '''def _render_rag_destination() -> None:
    """Render the RAG learning panel (defined in lab.py) as a Tools page."""
    st.markdown("### RAG learning")
    try:
        try:
            from ui.lab import _render_rag_learning_panel
        except Exception:
            from lab import _render_rag_learning_panel  # type: ignore
        _render_rag_learning_panel()
    except Exception as e:  # noqa: BLE001
        st.error(f"RAG learning panel unavailable: {e}")


def _render_metrics_destination(backend: dict[str, Any]) -> None:'''

# --- lab.py edits (remove the RAG radio option + dispatch) ---------------
LAB_RADIO_WITH_RAG = '''        view = st.radio(
            "View",
            ["Per-ticker bench", "Extended models", "Portfolio structure",
             "RAG learning"],
            horizontal=True,
            key="lab_view")'''
LAB_RADIO_WITHOUT = '''        view = st.radio(
            "View",
            ["Per-ticker bench", "Extended models", "Portfolio structure"],
            horizontal=True,
            key="lab_view")'''

LAB_DISPATCH_WITH_RAG = '''        elif view == "RAG learning":
            _render_rag_learning_panel()
        else:
            _render_structure_panel(held_tickers, price_loader)'''
LAB_DISPATCH_WITHOUT = '''        else:
            _render_structure_panel(held_tickers, price_loader)'''


def _fail(msg: str) -> None:
    print(f"[fix_rag_to_tools] ABORT: {msg}")
    sys.exit(1)


def _patch_app() -> str:
    src = APP.read_text(encoding="utf-8")
    if '"RAG"' in src and "_render_rag_destination" in src:
        return "app: already applied"

    if UTIL_ANCHOR not in src:
        _fail("app_v2: could not find _UTILITY_DESTINATIONS anchor.")
    if DISPATCH_ANCHOR not in src:
        _fail("app_v2: could not find the Metrics dispatch branch anchor.")
    if RENDERFN_ANCHOR not in src:
        _fail("app_v2: could not find _render_metrics_destination anchor.")

    src = src.replace(UTIL_ANCHOR, UTIL_INSERT, 1)
    src = src.replace(DISPATCH_ANCHOR, DISPATCH_INSERT, 1)
    src = src.replace(RENDERFN_ANCHOR, RENDERFN_INSERT, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"app_v2 patched file does not parse ({e}); not saving.")

    shutil.copy2(APP, APP.with_suffix(".py.bak"))
    APP.write_text(src, encoding="utf-8")
    return "app: patched"


def _patch_lab() -> str:
    if not LAB.exists():
        return "lab: not found (skipped)"
    src = LAB.read_text(encoding="utf-8")

    if "RAG learning" not in src:
        return ("lab: no 'RAG learning' radio entry found (nothing to remove; "
                "panel fn assumed present)")

    changed = False
    if LAB_RADIO_WITH_RAG in src:
        src = src.replace(LAB_RADIO_WITH_RAG, LAB_RADIO_WITHOUT, 1)
        changed = True
    if LAB_DISPATCH_WITH_RAG in src:
        src = src.replace(LAB_DISPATCH_WITH_RAG, LAB_DISPATCH_WITHOUT, 1)
        changed = True

    if not changed:
        return "lab: RAG radio present but anchors didn't match (left as-is)"

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"lab patched file does not parse ({e}); not saving.")

    shutil.copy2(LAB, LAB.with_suffix(".py.bak"))
    LAB.write_text(src, encoding="utf-8")
    return "lab: RAG removed from Lab radio"


def main() -> None:
    if not APP.exists():
        _fail(f"{APP} not found. Run from the project root (D:\\\\Ary Fund).")

    r_app = _patch_app()
    r_lab = _patch_lab()

    print("[fix_rag_to_tools] SUCCESS")
    print(f"  • {r_app}")
    print(f"  • {r_lab}")
    print()
    print("Restart Streamlit. 'RAG' now appears in the sidebar Tools section")
    print("(next to Metrics / Debug); the Lab radio no longer lists it.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_rag_to_tools.py
