"""
diag_sector_coverage2.py — per-sector hydrated coverage (CORRECTED).

The first version looked for sector in the wrong places. Sector actually
lives in fundamentals_cache.data_json (a JSON blob, yfinance .info), keyed
by ticker. This reads it the real way and reports, per sector, how many
universe tickers have BOTH filing text AND a resolvable sector — i.e. how
many usable peers each sector has for Stage 2 (>= 2 needed).

    python diag_sector_coverage2.py
"""
import json
import sqlite3
from collections import Counter

import config

try:
    from data.universe import US_UNIVERSE
except Exception:
    from universe import US_UNIVERSE

conn = sqlite3.connect(config.PORTFOLIO_DB_PATH)

# Tickers with filing text.
try:
    have_text = {r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM sec_filings "
        "WHERE full_text IS NOT NULL AND full_text != ''"
    ).fetchall()}
except sqlite3.OperationalError:
    have_text = set()

# Sector per ticker from fundamentals_cache JSON blobs.
sector_of = {}
try:
    for tk, blob in conn.execute(
        "SELECT ticker, data_json FROM fundamentals_cache"
    ).fetchall():
        try:
            d = json.loads(blob)
            s = (d.get("sector") or "").strip()
            if s:
                sector_of[tk] = s
        except Exception:
            continue
except sqlite3.OperationalError:
    print("(no fundamentals_cache table found)")

uni = set(US_UNIVERSE)
counts = Counter()       # sector -> tickers with text AND sector
text_no_sector = 0
for t in uni:
    if t not in have_text:
        continue
    s = sector_of.get(t)
    if not s:
        text_no_sector += 1
        continue
    counts[s] += 1

print(f"universe size                  : {len(uni)}")
print(f"hydrated (filing text)         : {len(have_text & uni)}")
print(f"  ... with a resolvable sector : {sum(counts.values())}")
print(f"  ... text but NO sector       : {text_no_sector}")
print(f"\nper-sector usable peers (>=2 needed for peer stats):")
usable = 0
for sector, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    flag = "OK" if n >= 2 else "THIN"
    if n >= 2:
        usable += 1
    print(f"  {sector:26} {n:3}  {flag}")
print(f"\nsectors with usable peer stats : {usable}")
