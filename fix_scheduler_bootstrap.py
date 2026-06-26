"""
fix_scheduler_bootstrap.py
=========================

Fix the scheduler's "registry load failed: No module named 'data'" warning.

CAUSE
-----
refresh_scheduler.py does ``from data.data_registry import ...`` (and
data.market_data, data.macro_data, ...). Those resolve only if the project
ROOT is on sys.path. When you run it as a script —
``python data/refresh_scheduler.py`` — the script's own directory (data/) is
on sys.path, not the root, so ``import data.*`` fails and the registry silently
falls back to a degraded mode. (The module form ``python -m data.refresh_
scheduler`` would work, but the script form is what you ran.)

FIX
---
Insert the same project-root bootstrap screener_data.py uses: compute the root
as this file's parent's parent, chdir there, and put it on sys.path. Then both
the script form and the module form work, and the registry loads.

SAFETY
------
* Targets data/refresh_scheduler.py.
* Backs up to data/refresh_scheduler.py.bak before writing.
* Idempotent: detects the bootstrap marker and does nothing on re-run.
* Verifies ast.parse before saving.
* Inserts right after the HF offline-flags block (added earlier), before the
  other imports — so flags stay first and imports resolve.

Usage (from project root, venv active):
    python fix_scheduler_bootstrap.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("data") / "refresh_scheduler.py"

ANCHOR = '''import os as _os
_os.environ.setdefault("HF_HUB_OFFLINE", "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")'''

INSERT = ANCHOR + '''

# --- Project-root bootstrap (so `from data.*` works when run as a script) ---
# Running this file as `python data/refresh_scheduler.py` puts data/ on
# sys.path, not the project root, so `import data.data_registry` fails and the
# registry falls back to a degraded mode. Compute the root (this file's
# parent's parent), chdir there, and add it to sys.path so both the script
# form and the module form (`python -m data.refresh_scheduler`) resolve.
import sys as _sys
from pathlib import Path as _Path
_SCHED_ROOT = _Path(__file__).resolve().parent.parent
try:
    _os.chdir(_SCHED_ROOT)
except Exception:
    pass
if str(_SCHED_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SCHED_ROOT))'''


def _fail(msg: str) -> None:
    print(f"[fix_scheduler_bootstrap] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if "_SCHED_ROOT = _Path(__file__)" in src:
        print("[fix_scheduler_bootstrap] Already applied — bootstrap present. "
              "Nothing to do.")
        return

    if ANCHOR not in src:
        _fail("could not find the HF offline-flags block to anchor the "
              "bootstrap. Did fix_scheduler_offline.py run? Not editing "
              "blindly.")

    src = src.replace(ANCHOR, INSERT, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_scheduler_bootstrap] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Added project-root chdir + sys.path bootstrap")
    print()
    print("Re-run; the 'No module named data' warning should be gone:")
    print("    python data/refresh_scheduler.py status")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_scheduler_bootstrap.py
