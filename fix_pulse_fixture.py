"""
fix_pulse_fixture.py — fix the stale-seed bug in the global-risk-pulse tests.

Run from D:\\Ary Fund:   python fix_pulse_fixture.py

ROOT CAUSE: the base_universe fixture seeds prices with _seed_universe's
default end_date=datetime(2026, 5, 13). The pulse freshness filter excludes
any ticker whose latest price is older than max_staleness_days (7) relative to
as_of. The 15 tests using this fixture omit as_of (so it defaults to now), and
once wall-clock time passed ~7 days beyond 2026-05-13, every seeded ticker
became stale -> included_tickers=0 -> empty result -> 11 tests fail (and ~4
more pass only vacuously on the empty result). This is a test-fixture
staleness bug; the production pulse code is correct.

FIX: seed the fixture's universe ending at datetime.now(), so the data is
always fresh relative to the default as_of=now. The 9 as_of-pinned tests are
untouched (they call _seed_universe directly with their own args and pin
as_of=2026-05-13, so they stay internally consistent).

Surgical: replaces only the one-line fixture body. Refuses to write unless it
matches exactly once or is already patched. Verifies `datetime` is imported.
Re-parses before and after; writes UTF-8 without BOM.
"""
import ast
import io
import sys

PATH = r"D:\Ary Fund\tests\test_all.py"

OLD = (
    "@pytest.fixture\n"
    "def base_universe(db: str) -> list[str]:\n"
    "    return _seed_universe(db)\n"
)

NEW = (
    "@pytest.fixture\n"
    "def base_universe(db: str) -> list[str]:\n"
    "    # Seed ending today so the data is fresh relative to the default\n"
    "    # as_of=now used by the fixture-based tests (the pulse freshness\n"
    "    # filter excludes prices older than max_staleness_days). The\n"
    "    # as_of-pinned tests call _seed_universe directly and are unaffected.\n"
    "    return _seed_universe(db, end_date=datetime.now())\n"
)


def main() -> int:
    with io.open(PATH, "r", encoding="utf-8-sig") as f:
        src = f.read()

    # Guard: datetime must be importable in this module for the new call.
    if "datetime" not in src:
        print("ERROR: 'datetime' not referenced in test_all.py — cannot rely "
              "on it being imported. Aborting; tell me and I'll add the import.")
        return 1

    if "_seed_universe(db, end_date=datetime.now())" in src:
        print("Already patched — base_universe seeds end_date=datetime.now().")
        return 0

    count = src.count(OLD)
    if count == 0:
        print("ERROR: base_universe fixture block not found as expected. "
              "No changes made. Paste the fixture and I'll match it.")
        return 1
    if count > 1:
        print(f"ERROR: fixture block found {count} times (expected 1). Aborting.")
        return 1

    patched = src.replace(OLD, NEW)

    try:
        ast.parse(patched)
    except SyntaxError as e:
        print(f"ERROR: patched source does not parse ({e}). No changes written.")
        return 1

    with io.open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(patched)

    with io.open(PATH, "r", encoding="utf-8") as f:
        check = f.read()
    try:
        ast.parse(check)
    except SyntaxError as e:
        print(f"ERROR: file on disk does not parse after write ({e}).")
        return 1

    print("PATCHED OK — base_universe now seeds end_date=datetime.now().")
    return 0


if __name__ == "__main__":
    sys.exit(main())
