"""
fix_registry_split_tests.py
===========================

Fixes the six PRE-EXISTING test failures in tests/test_all.py that the
OpenBB work surfaced (they fail on a clean checkout too — verified via
git stash). NONE were caused by the OpenBB install.

Root causes (two independent bugs, both in the tests / data layer — not
in today's changes):

FIX 1 — XBRL canonical-field gap (1 failure)
    test_xbrl_concept_map_canonical_fields_present
    ``sec_fetcher.XBRL_CONCEPT_MAP`` maps "ticker.fundamental.
    operating_income_ttm" but data_registry.CANONICAL_FIELDS never
    registered it. Real inconsistency in your own data layer. Fix: add
    the missing canonical field entry, mirroring its neighbours.

FIX 2 — registry-split isolation in build_agent_context (5 failures)
    test_no_raw_provider_access_when_registry_populated
    test_full_snapshot_returns_complete_schema
    test_partial_snapshot_returns_safe_fallbacks
    test_universe_context_uses_registry_only
    test_empty_registry_does_not_raise
    (also hardens test_provenance_and_freshness_recorded, which only
    passed by luck reading live data.)

    These tests seed a registry on a temp DB and pass ``cfg =
    SimpleNamespace()``. But build_agent_context — correctly, for
    production — ignores db_path for the registry and resolves it to
    the real data/hedgefund.db (the registry-split fix), UNLESS cfg
    carries MARKET_DB_PATH / DATA_DB_PATH / REGISTRY_DB_PATH. With a
    bare cfg it read your live hedgefund.db, so the seeded 215.43 / VIX
    17.8 were shadowed by today's real AAPL (~289) and live VIX. The
    tests pre-date that production fix.

    Fix: give the tests' cfg a MARKET_DB_PATH pointing at the same
    temp DB the fixture seeded, so the registry resolves to it. We do
    NOT change 215.43 -> live values — that would bake a live-data
    dependency into the suite and hide the isolation the tests exist
    to guard.

CONVENTIONS: timestamped .bak, ast.parse verify before write,
idempotent (re-run detects both fixes already applied and skips).

USAGE (project root, venv active):
    python fix_registry_split_tests.py
    python -m pytest tests/test_all.py -k "snapshot or registry or xbrl or provenance" -q
"""
from __future__ import annotations

import ast
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "data" / "data_registry.py"
TESTS = ROOT / "tests" / "test_all.py"

# ---- FIX 1: canonical field addition ---------------------------------
REG_ANCHOR = (
    '    "ticker.fundamental.net_income_ttm":   ("num", "TTM net income",'
    '                   "quarterly", "forward_fill"),\n'
)
REG_NEWLINE = (
    '    "ticker.fundamental.operating_income_ttm": ("num", "TTM operating '
    'income (EBIT)",     "quarterly", "forward_fill"),\n'
)
REG_MARKER = '"ticker.fundamental.operating_income_ttm":'

# ---- FIX 2: test cfg isolation ---------------------------------------
TEST_OLD = "    cfg = SimpleNamespace()\n"
TEST_NEW = "    cfg = SimpleNamespace(MARKET_DB_PATH=tmp_db_path)\n"


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


def fix_registry() -> None:
    if not REGISTRY.exists():
        print(f"[ABORT] {REGISTRY} not found — run from project root.")
        sys.exit(1)
    src = REGISTRY.read_text(encoding="utf-8")
    if REG_MARKER in src:
        print("[SKIP] operating_income_ttm already in CANONICAL_FIELDS.")
        return
    if REG_ANCHOR not in src:
        print("[ABORT] net_income_ttm anchor not found in data_registry.py — "
              "file differs from expected baseline; not guessing.")
        sys.exit(1)
    patched = src.replace(REG_ANCHOR, REG_ANCHOR + REG_NEWLINE, 1)
    _verify(patched, "data_registry.py")
    bak = _backup(REGISTRY)
    REGISTRY.write_text(patched, encoding="utf-8")
    print(f"[OK] data_registry.py: added operating_income_ttm "
          f"(backup: {bak.name})")


def fix_tests() -> None:
    if not TESTS.exists():
        print(f"[ABORT] {TESTS} not found.")
        sys.exit(1)
    src = TESTS.read_text(encoding="utf-8")
    n = src.count(TEST_OLD)
    if n == 0:
        if TEST_NEW in src:
            print("[SKIP] test cfg already isolated (MARKET_DB_PATH set).")
            return
        print("[ABORT] no bare 'cfg = SimpleNamespace()' lines found — "
              "test file differs from expected baseline.")
        sys.exit(1)
    patched = src.replace(TEST_OLD, TEST_NEW)
    _verify(patched, "test_all.py")
    bak = _backup(TESTS)
    TESTS.write_text(patched, encoding="utf-8")
    print(f"[OK] test_all.py: isolated {n} test(s) to temp registry DB "
          f"(backup: {bak.name})")


def main() -> None:
    print("=== fix_registry_split_tests ===")
    fix_registry()
    fix_tests()
    print()
    print("Verify:")
    print("  python -m pytest tests/test_all.py -k "
          "\"snapshot or registry or xbrl or provenance\" -q")


if __name__ == "__main__":
    main()
