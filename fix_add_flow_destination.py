"""
fix_add_flow_destination.py
===========================
Register the **Money Flow Graph** as a top-nav destination in
``ui/app_v2.py``.

Follows the project's fix-script conventions:
  * backs the target up to ``ui/app_v2_py.bak`` before writing
  * idempotent — re-running is a no-op
  * anchor-based ``str.replace`` (no line numbers)
  * verifies the result with ``ast.parse`` before saving; restores on failure

Three edits:
  1. add ``"Flow"`` to ``_DESTINATIONS``
  2. add a ``_render_flow_destination(backend)`` renderer
  3. route ``dest == "Flow"`` to it in ``main()``

Run from the project root:
    python fix_add_flow_destination.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

# --- anchors (must match ui/app_v2.py verbatim) --------------------------
ANCHOR_DESTINATIONS = (
    '_DESTINATIONS = ["Desk", "Board", "Screener", "Lab", "Analyzer", "Jobs"]'
)
REPLACE_DESTINATIONS = (
    '_DESTINATIONS = ["Desk", "Board", "Screener", "Lab", "Analyzer", '
    '"Flow", "Jobs"]'
)

ANCHOR_ROUTE = (
    '    elif dest == "Analyzer":\n'
    '        _render_analyzer_destination(backend, side["lookback"])\n'
)
REPLACE_ROUTE = (
    '    elif dest == "Analyzer":\n'
    '        _render_analyzer_destination(backend, side["lookback"])\n'
    '    elif dest == "Flow":\n'
    '        _render_flow_destination(backend)\n'
)

ANCHOR_FUNC = (
    'def _render_metrics_destination(backend: dict[str, Any]) -> None:'
)
FLOW_FUNC = '''def _render_flow_destination(backend: dict[str, Any]) -> None:
    """SEC money-flow network (ui/money_flow).

    Company-to-company capital & supply relationships assembled by
    data.money_graph and rendered by ui/money_flow_template.html.
    """
    try:
        from ui.money_flow import render_money_flow_destination
    except Exception:
        try:
            from money_flow import render_money_flow_destination  # type: ignore
        except Exception as e:  # noqa: BLE001
            st.error(
                f"Money Flow Graph unavailable: {e}. Ensure "
                "`ui/money_flow.py`, `ui/money_flow_template.html`, and "
                "`data/money_graph.py` are present.")
            return
    render_money_flow_destination(backend)


'''
REPLACE_FUNC = FLOW_FUNC + ANCHOR_FUNC


def _find_target() -> Path:
    for p in (Path("ui/app_v2.py"), Path("app_v2.py")):
        if p.exists():
            return p
    print("ERROR: could not find ui/app_v2.py (run from the project root).")
    sys.exit(1)


def main() -> None:
    target = _find_target()
    src = target.read_text(encoding="utf-8")

    already = (
        '"Flow"' in src
        and "def _render_flow_destination" in src
        and 'dest == "Flow"' in src
    )
    if already:
        print(f"✓ {target}: Flow destination already registered — no changes.")
        return

    # --- verify anchors exist before touching anything -------------------
    missing = []
    if ANCHOR_DESTINATIONS not in src and REPLACE_DESTINATIONS not in src:
        missing.append("_DESTINATIONS list")
    if ANCHOR_ROUTE not in src and 'dest == "Flow"' not in src:
        missing.append("Analyzer route block in main()")
    if ANCHOR_FUNC not in src:
        missing.append("_render_metrics_destination anchor")
    if missing:
        print("ERROR: expected anchors not found in ui/app_v2.py: "
              + ", ".join(missing))
        print("The file may have drifted. Aborting without changes.")
        sys.exit(1)

    patched = src
    if REPLACE_DESTINATIONS not in patched:
        patched = patched.replace(ANCHOR_DESTINATIONS, REPLACE_DESTINATIONS, 1)
    if 'dest == "Flow"' not in patched:
        patched = patched.replace(ANCHOR_ROUTE, REPLACE_ROUTE, 1)
    if "def _render_flow_destination" not in patched:
        patched = patched.replace(ANCHOR_FUNC, REPLACE_FUNC, 1)

    # --- validate before writing ----------------------------------------
    try:
        ast.parse(patched)
    except SyntaxError as e:
        print(f"ERROR: patched source failed to parse ({e}). No changes made.")
        sys.exit(1)

    backup = target.with_name(target.name.replace(".py", "_py.bak"))
    shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")

    print(f"✓ Backed up  {target} -> {backup}")
    print(f"✓ Patched    {target}")
    print("  • added \"Flow\" to _DESTINATIONS")
    print("  • added _render_flow_destination(backend)")
    print("  • routed dest == \"Flow\" in main()")
    print("\nRestart Streamlit:  streamlit run ui/app_v2.py")


if __name__ == "__main__":
    main()
