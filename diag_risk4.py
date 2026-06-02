"""
diag_risk4.py — find MSFT's real Item 1A header and the right boundary.

Findings so far:
  * AAPL/COST: real Item 1A is the 2nd match; the section is truncated
    because a stray "Item 7" cross-reference sits ~800 chars in. The
    reliable end boundary is the next Item 1B / 1C / 2 header, not 7.
  * MSFT: only ONE "Item 1A. Risk Factors" match (the TOC). The real
    section header must be formatted differently. Let's find it.

This prints:
  1. Every occurrence of the looser token "risk factors" in MSFT (to locate
     the real section heading however it's spelled).
  2. For AAPL/COST/MSFT, the nearest Item 1B / 1C / 2 boundary AFTER the
     last 'Item 1A. Risk Factors' match, so we can size the real section.

    python diag_risk4.py
"""
import re
import config
from data.sec_fetcher import SECFetcher

s = SECFetcher(db_path=config.PORTFOLIO_DB_PATH)

risk_factors_loose = re.compile(r"risk\s*factors", re.IGNORECASE)
item1a = re.compile(r"item\s*1a[\.\s]*risk\s*factors", re.IGNORECASE)
boundary = re.compile(r"item\s*(?:1b|1c|2)\b", re.IGNORECASE)

for tkr in ["MSFT", "AAPL", "COST"]:
    rows = s._get_cached_filings(tkr, "10-K", 1, None, None)
    if not rows:
        print(f"\n{tkr}: no cached 10-K"); continue
    t = s.get_filing_text(rows[0]["accession_number"])
    print(f"\n===== {tkr} (len {len(t):,}) =====")

    # All loose 'risk factors' mentions with a little context
    rf = list(risk_factors_loose.finditer(t))
    print(f"  'risk factors' (loose): {len(rf)} matches")
    for m in rf[:8]:
        ctx = re.sub(r"\s+", " ", t[max(0, m.start()-30):m.start()+40]).strip()
        print(f"      @{m.start():>7} ...{ctx!r}")

    # Last strict Item 1A, then nearest boundary after it
    m1a = list(item1a.finditer(t))
    if m1a:
        start = m1a[-1].end()
        nb = boundary.search(t, start)
        end = nb.start() if nb else len(t)
        print(f"  last Item1A end @ {start}, next 1B/1C/2 boundary @ "
              f"{end}  -> region {end - start:,} chars")
