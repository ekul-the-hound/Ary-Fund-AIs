"""
fix_theme_openbb.py
===================

Restyle ARY QUANT to an OpenBB Workspace-style dark theme.

WHAT IT DOES
------------
PATCH 1 (ui/components.py):
    Replaces the hardcoded color-constant block with a single THEME token
    dict (OpenBB-inspired dark palette), then rewires every existing
    constant (RISK_COLORS, OUTLOOK_COLORS, ZONE_COLORS, FRESH_*,
    _NEUTRAL_TEXT, _HAIRLINE, _CARD_BG) to point at it. No downstream
    call site changes — every existing name keeps working.

PATCH 2 (ui/palette.py):
    Rewires the two stray hardcoded hex literals (job-running blue
    #3b82f6, minimized-tray gray #9ca3af) to THEME tokens via the
    already-imported ``C`` alias.

WHAT IT DOES NOT DO
-------------------
Does not touch charts.py / pdf_renderer.py (report styling is a separate
decision) and does not create .streamlit/config.toml — that file ships
alongside this script; copy it to D:\\Ary Fund\\.streamlit\\config.toml.

CONVENTIONS
-----------
* Timestamped .bak backup before writing.
* ast.parse verification of the patched source before it hits disk.
* Idempotent: re-running detects the THEME marker / absent literals and
  skips cleanly.

USAGE (from project root, venv active)
--------------------------------------
    python fix_theme_openbb.py

Then fully restart Streamlit (Ctrl+C, relaunch) — a stale process will
show the old colors and look like a failed fix.

TWEAKING THE PALETTE
--------------------
All colors live in the THEME dict this script installs at the top of
ui/components.py. Values are close approximations of OpenBB Workspace's
dark look (exact tokens aren't published); to pixel-match, eyedropper an
OpenBB screenshot and edit the dict — one line per role.
"""
from __future__ import annotations

import ast
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPONENTS = ROOT / "ui" / "components.py"
PALETTE = ROOT / "ui" / "palette.py"

# Marker proving PATCH 1 already ran.
THEME_MARKER = "THEME: dict[str, str] = {"

# ----------------------------------------------------------------------
# PATCH 1 — components.py color block -> THEME tokens
# ----------------------------------------------------------------------
# The span we replace starts at the RISK_COLORS definition and ends at
# the _CARD_BG line (inclusive). Anchors must match the file exactly.
BLOCK_START = 'RISK_COLORS: dict[str, str] = {'
BLOCK_END = '_CARD_BG = "rgba(148,163,184,0.06)"'

NEW_BLOCK = '''\
# ----------------------------------------------------------------------
# THEME — single source of truth for every color in the dashboard.
# OpenBB Workspace-style dark palette (close approximation; OpenBB does
# not publish exact tokens). To retune the whole UI, edit values here —
# nothing below or downstream hardcodes hex.
# ----------------------------------------------------------------------
THEME: dict[str, str] = {
    # structural
    "bg":       "#0B0E14",   # app canvas — near-black navy
    "surface":  "#131722",   # cards / panels
    "raised":   "#1A1F2B",   # hover / elevated surfaces
    "border":   "#232936",   # hairlines, card borders
    "text":     "#E6E8EB",   # primary text
    "text_dim": "#8A919E",   # secondary text, captions
    "accent":   "#2962FF",   # the one interactive blue
    # semantic (brightened one step vs the old Tailwind-600 values so
    # they read cleanly on the darker canvas)
    "good":     "#22C55E",   # low risk / bullish / safe / fresh
    "warn":     "#EAB308",   # medium risk / grey zone / recent
    "bad":      "#EF4444",   # high risk / bearish / distress
    "severe":   "#B91C1C",   # severe risk
    "muted":    "#7C8591",   # unknown / neutral / stale
    # translucent fills
    "card_bg":  "rgba(148,163,184,0.05)",
}

RISK_COLORS: dict[str, str] = {
    "low": THEME["good"],
    "medium": THEME["warn"],
    "moderate": THEME["warn"],   # alias used by portfolio concentration bucket
    "high": THEME["bad"],
    "severe": THEME["severe"],
    "unknown": THEME["muted"],
}

OUTLOOK_COLORS: dict[str, str] = {
    "bullish": THEME["good"],
    "neutral": THEME["muted"],
    "bearish": THEME["bad"],
    "unknown": THEME["muted"],
}

# Distress zones from risk_scanner.altman_z (zone in {distress, grey, safe}).
ZONE_COLORS: dict[str, str] = {
    "safe": THEME["good"],
    "grey": THEME["warn"],
    "distress": THEME["bad"],
    "unknown": THEME["muted"],
}

# Freshness staleness ramp, applied to a section's latest as_of.
FRESH_FRESH = THEME["good"]      # < 1 day
FRESH_RECENT = THEME["warn"]     # < 7 days
FRESH_STALE = THEME["muted"]     # older / unknown

_NEUTRAL_TEXT = THEME["text_dim"]
_HAIRLINE = THEME["border"]
_CARD_BG = THEME["card_bg"]'''

# ----------------------------------------------------------------------
# PATCH 2 — palette.py stray literals -> THEME tokens
# ----------------------------------------------------------------------
# (old_literal, new_text). Both lines already sit inside contexts where
# the replacement is valid Python: the first is a plain string in a
# tuple assignment; the second is inside an f-string, so an {expr}
# interpolation slots straight in.
PALETTE_SUBS = [
    ('"#3b82f6"', 'C.THEME["accent"]'),      # running-job glyph color
    ("color:#9ca3af;", "color:{C._NEUTRAL_TEXT};"),  # minimized tray text
]


def _backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(path.suffix + f".{stamp}.bak")
    shutil.copy2(path, bak)
    return bak


def _verify(source: str, label: str) -> None:
    try:
        ast.parse(source)
    except SyntaxError as e:
        print(f"[ABORT] patched {label} does not parse: {e}")
        sys.exit(1)


def patch_components() -> None:
    if not COMPONENTS.exists():
        print(f"[ABORT] {COMPONENTS} not found — run from the project root.")
        sys.exit(1)

    src = COMPONENTS.read_text(encoding="utf-8")

    if THEME_MARKER in src:
        print("[SKIP] components.py already has the THEME block — nothing to do.")
        return

    start = src.find(BLOCK_START)
    end = src.find(BLOCK_END)
    if start == -1 or end == -1:
        print("[ABORT] color-block anchors not found in components.py — "
              "file differs from the expected baseline; not guessing.")
        sys.exit(1)
    end += len(BLOCK_END)

    patched = src[:start] + NEW_BLOCK + src[end:]
    _verify(patched, "components.py")

    bak = _backup(COMPONENTS)
    COMPONENTS.write_text(patched, encoding="utf-8")
    print(f"[OK] components.py patched (backup: {bak.name})")


def patch_palette() -> None:
    if not PALETTE.exists():
        print(f"[WARN] {PALETTE} not found — skipping PATCH 2.")
        return

    src = PALETTE.read_text(encoding="utf-8")
    patched = src
    applied = 0
    for old, new in PALETTE_SUBS:
        if old in patched:
            patched = patched.replace(old, new)
            applied += 1
        elif new in patched:
            pass  # already rewired on a previous run
        else:
            print(f"[WARN] literal {old!r} not found in palette.py — skipped.")

    if applied == 0:
        print("[SKIP] palette.py already rewired — nothing to do.")
        return

    _verify(patched, "palette.py")
    bak = _backup(PALETTE)
    PALETTE.write_text(patched, encoding="utf-8")
    print(f"[OK] palette.py patched, {applied} literal(s) rewired "
          f"(backup: {bak.name})")


def main() -> None:
    print("=== fix_theme_openbb ===")
    patch_components()
    patch_palette()
    print()
    print("Next steps:")
    print("  1. Copy config.toml to .streamlit\\config.toml (create the "
          "folder if needed).")
    print("  2. Fully restart Streamlit (Ctrl+C, then "
          "`streamlit run ui/app_v2.py`).")
    print("  3. To tweak any color, edit the THEME dict at the top of "
          "ui/components.py.")


if __name__ == "__main__":
    main()
