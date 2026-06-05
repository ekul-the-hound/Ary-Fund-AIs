"""
fix_sector_offline.py — make _sector_for strictly offline.

Run from D:\\Ary Fund:   python fix_sector_offline.py

BUG: _sector_for calls _md.get_fundamentals(tk, use_cache=True), which is
cache-FIRST, not cache-ONLY. On a stale/missing cache entry it falls through
to a live yfinance fetch, which 404s for delisted/renamed tickers. That makes
the "offline" peer recompute non-deterministic: it hits the network whenever
the fundamentals cache has aged past its 24h TTL.

FIX: read sector directly from the fundamentals_cache table (data_json blob),
ignoring the TTL. Sector is effectively immutable (a company's GICS sector
doesn't change), so any cached row is authoritative and there is never a
reason to re-fetch it from the network. This makes the recompute fully
offline and deterministic regardless of cache age.

Surgical: replaces exactly the _sector_for function body and nothing else.
Refuses to write unless the expected block is found exactly once, re-parses
with ast before and after writing, and writes UTF-8 without BOM.
"""
import ast
import io
import sys

PATH = r"D:\Ary Fund\data\pipeline.py"

OLD = (
    '    def _sector_for(tk: str):\n'
    '        """Cache-first sector tag (yfinance .info), offline. None if absent."""\n'
    '        if _md is None:\n'
    '            return None\n'
    '        try:\n'
    '            f = _md.get_fundamentals(tk, use_cache=True)\n'
    '        except Exception:  # noqa: BLE001\n'
    '            return None\n'
    '        if not isinstance(f, dict):\n'
    '            return None\n'
    '        s = f.get("sector")\n'
    '        return s.strip() if isinstance(s, str) and s.strip() else None\n'
)

NEW = (
    '    def _sector_for(tk: str):\n'
    '        """Sector tag read DIRECTLY from the fundamentals_cache table.\n'
    '\n'
    '        Strictly offline: no get_fundamentals (which is cache-FIRST and\n'
    '        falls through to a live yfinance fetch on a stale entry), and no\n'
    '        TTL check. Sector is immutable, so any cached row is authoritative.\n'
    '        Returns None if the ticker has no cached fundamentals row.\n'
    '        """\n'
    '        if _md is None:\n'
    '            return None\n'
    '        try:\n'
    '            import json as _json\n'
    '            import sqlite3 as _sqlite3\n'
    '            with _sqlite3.connect(_md.db_path) as _conn:\n'
    '                _row = _conn.execute(\n'
    '                    "SELECT data_json FROM fundamentals_cache WHERE ticker = ?",\n'
    '                    (tk,),\n'
    '                ).fetchone()\n'
    '            if not _row or not _row[0]:\n'
    '                return None\n'
    '            _data = _json.loads(_row[0])\n'
    '        except Exception:  # noqa: BLE001\n'
    '            return None\n'
    '        if not isinstance(_data, dict):\n'
    '            return None\n'
    '        s = _data.get("sector")\n'
    '        return s.strip() if isinstance(s, str) and s.strip() else None\n'
)


def main() -> int:
    with io.open(PATH, "r", encoding="utf-8-sig") as f:
        src = f.read()

    count = src.count(OLD)
    if count == 0:
        print("ERROR: expected _sector_for block not found. No changes made.")
        if "Sector tag read DIRECTLY from the fundamentals_cache" in src:
            print("NOTE: looks like it's already patched.")
        return 1
    if count > 1:
        print(f"ERROR: block found {count} times (expected 1). Aborting.")
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

    print("PATCHED OK — _sector_for now reads fundamentals_cache directly "
          "(offline, no TTL, no network fallthrough).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
