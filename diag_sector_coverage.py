"""
diag_sector_coverage.py — how many hydrated tickers we have per sector.

Stage 2 (sector-relative risk-count penalty) computes a per-sector mean/std
of risk counts. A sector needs >= _MIN_PEERS hydrated names to form a
usable distribution. This reports, per sector: how many universe tickers
now have filing text AND a resolvable sector, so we know which sectors
will get real peer stats vs fall back to no-signal.

    python diag_sector_coverage.py
"""
import sqlite3
from collections import Counter

import config

try:
    from data.universe import US_UNIVERSE
except Exception:
    from universe import US_UNIVERSE

# Pull sector for each ticker the same way the metrics path would. The
# universe module may carry a ticker->sector map; fall back to the DB if not.
sector_of = {}
try:
    from data.universe import SECTOR_MAP  # type: ignore
    sector_of = dict(SECTOR_MAP)
except Exception:
    pass

conn = sqlite3.connect(config.PORTFOLIO_DB_PATH)

# Which tickers have filing text?
try:
    have = {r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM sec_filings "
        "WHERE full_text IS NOT NULL AND full_text != ''"
    ).fetchall()}
except sqlite3.OperationalError:
    have = set()

# Try to get sector from a metrics/fundamentals table if no SECTOR_MAP.
if not sector_of:
    for tbl, tcol, scol in [
        ("metrics", "ticker", "sector"),
        ("fundamentals", "ticker", "sector"),
        ("company_info", "ticker", "sector"),
    ]:
        try:
            for t, s in conn.execute(f"SELECT {tcol}, {scol} FROM {tbl}").fetchall():
                if s:
                    sector_of[t] = s
            if sector_of:
                print(f"(sector source: {tbl}.{scol})")
                break
        except sqlite3.OperationalError:
            continue

uni = set(US_UNIVERSE)
counts = Counter()
no_sector = 0
for t in uni:
    if t not in have:
        continue
    s = sector_of.get(t)
    if not s:
        no_sector += 1
        continue
    counts[s] += 1

print(f"\nhydrated universe tickers : {len(have & uni)}")
print(f"  with a resolvable sector: {sum(counts.values())}")
print(f"  WITHOUT a sector tag    : {no_sector}")
print(f"\nper-sector hydrated counts (>=2 needed for peer stats):")
for sector, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    flag = "OK" if n >= 2 else "THIN"
    print(f"  {sector:28} {n:3}  {flag}")
