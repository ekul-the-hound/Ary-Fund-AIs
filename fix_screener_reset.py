"""
fix_screener_reset.py
=====================

Fix for the Screener "Clear filters" button crashing with:

    StreamlitAPIException: `st.session_state._flt_mcap` cannot be modified
    after the widget with key `_flt_mcap` is instantiated.

THE BUG
-------
``_reset_filters()`` does ``st.session_state["_flt_mcap"] = []`` (and similar
assignments). But ``_flt_mcap`` is a multiselect WIDGET key, and
``render_screener_tab`` calls ``_render_filter_bar()`` — which instantiates
that widget — BEFORE the Clear-filters button runs. Streamlit forbids
assigning to a widget-backed key after the widget has been created in the same
script run. So clicking Clear -> rerun -> the widget is created first -> the
button's ``_reset_filters()`` assignment throws.

This only manifested once the screener became a standalone page (render order
shifted); it was latent before.

THE FIX (the Streamlit-blessed pattern)
---------------------------------------
You cannot reset a widget by assignment after it exists — but you CAN delete
its key before it is instantiated, which makes the widget re-initialize from
its default next run. So:

  1. The button no longer calls ``_reset_filters()`` directly. It sets a flag
     ``st.session_state["_screener_do_reset"] = True`` and reruns.
  2. At the VERY TOP of ``render_screener_tab`` — before any widget is created
     — we check the flag; if set, we DELETE all filter widget keys (and the
     paired ``_widget`` / min/max keys) and clear the flag. Deletion before
     instantiation is allowed, so the widgets come back at their defaults with
     no exception.
  3. ``_reset_filters()`` is rewritten to DELETE keys rather than assign, and
     is what the top-of-render flag handler calls.

This fixes the crash everywhere (the original app too), changes no filtering
logic, and preserves the defaults.

SAFETY
------
* Targets the real path ``ui/screener.py``.
* Backs up to ``ui/screener.py.bak`` before writing.
* Idempotent: detects the flag pattern and does nothing on re-run.
* Verifies ``ast.parse`` before saving; aborts on any structural surprise.

Usage (from project root, venv active):
    python fix_screener_reset.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "screener.py"


def _fail(msg: str) -> None:
    print(f"[fix_screener_reset] ABORT: {msg}")
    sys.exit(1)


# The new _reset_filters body: deletion instead of assignment.
_NEW_RESET = '''def _reset_filters() -> None:
    """Clear all filter state by DELETING widget keys.

    Assigning to a widget-backed key after the widget is instantiated raises
    StreamlitAPIException. Deleting the key (so the widget re-initializes from
    its default on the next run) is the supported reset pattern. This function
    is invoked from the top of render_screener_tab BEFORE any widget is built,
    via the `_screener_do_reset` flag set by the Clear-filters button.
    """
    keys = list(_FILTER_DEFAULTS.keys()) + [
        f"{k}_widget" for k in _FILTER_DEFAULTS
    ] + [
        "_flt_mcap", "_flt_sector", "_flt_rating",
        "_flt_price_min", "_flt_price_max",
        "_flt_change_widget", "_flt_pe_widget", "_flt_epsg_widget",
        "_flt_dy_widget", "_flt_revg_widget", "_flt_peg_widget",
        "_flt_roe_widget", "_flt_beta_widget",
    ]
    for k in keys:
        st.session_state.pop(k, None)
'''


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active. (Expected ui/screener.py per the "
              "traceback path.)")

    src = TARGET.read_text(encoding="utf-8")
    original = src

    # --- Idempotency check ------------------------------------------------
    if "_screener_do_reset" in src:
        print("[fix_screener_reset] Already applied — '_screener_do_reset' "
              "flag present. Nothing to do.")
        return

    # --- Change 1: replace the _reset_filters function body ---------------
    # Find 'def _reset_filters() -> None:' and replace through to the next
    # top-level 'def ' (or blank-line boundary). We bound it by locating the
    # function start and the following 'def ' at column 0.
    start_marker = "def _reset_filters() -> None:"
    if start_marker not in src:
        _fail("could not find 'def _reset_filters() -> None:'. File may have "
              "changed; not editing blindly.")
    start = src.index(start_marker)
    # Find the next top-level def after this one.
    next_def = src.find("\ndef ", start + len(start_marker))
    if next_def == -1:
        _fail("could not bound the _reset_filters function. Not editing.")
    # Preserve trailing blank lines between functions: capture up to next_def.
    src = src[:start] + _NEW_RESET + src[next_def:]

    # --- Change 2: button sets a flag instead of calling _reset_filters ---
    old_button = (
        "        ):\n"
        "            _reset_filters()\n"
        "            st.rerun()\n"
    )
    new_button = (
        "        ):\n"
        "            # Set a flag and rerun; the actual key deletion happens at\n"
        "            # the top of render_screener_tab BEFORE widgets are built,\n"
        "            # which is the only safe place to reset widget-backed keys.\n"
        "            st.session_state[\"_screener_do_reset\"] = True\n"
        "            st.rerun()\n"
    )
    if old_button not in src:
        _fail("could not find the Clear-filters button call site "
              "('_reset_filters()' + 'st.rerun()'). Not editing. (Change 1 "
              "not saved.)")
    src = src.replace(old_button, new_button, 1)

    # --- Change 3: flag handler at the very top of render_screener_tab ----
    render_marker = (
        "def render_screener_tab() -> None:\n"
        '    """Render the entire Screener tab. Safe to call inside a `with tab_x:`."""\n'
    )
    if render_marker not in src:
        _fail("could not find render_screener_tab signature + docstring to "
              "insert the reset handler. Not editing. (Earlier changes not "
              "saved.)")
    handler = (
        render_marker
        + "    # Handle a pending Clear-filters request BEFORE any widget is\n"
          "    # instantiated. Deleting widget keys here (rather than assigning\n"
          "    # to them after the widgets exist) is the only crash-free reset.\n"
          "    if st.session_state.pop(\"_screener_do_reset\", False):\n"
          "        _reset_filters()\n"
    )
    src = src.replace(render_marker, handler, 1)

    # --- Verify parse -----------------------------------------------------
    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving. Original "
              "untouched.")

    if src == original:  # pragma: no cover
        _fail("no net change produced — please report this.")

    # --- Back up + write --------------------------------------------------
    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_screener_reset] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Rewrote _reset_filters to DELETE keys (not assign)")
    print("  • Clear-filters button now sets a flag + reruns")
    print("  • Reset handler runs at the top of render_screener_tab, before")
    print("    any widget is instantiated")
    print()
    print("Reload the Screener tab and click 'Clear filters' — it should reset")
    print("every filter without the StreamlitAPIException.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_screener_reset.py
