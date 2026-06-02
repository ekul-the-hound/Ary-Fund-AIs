"""
diag_risk3.py — map where Item headers actually appear in real filings.

The largest-region heuristic overshoots for MSFT/NVDA/GOOGL (swallows the
whole doc) and undershoots for AAPL/COST (stops too early). Before changing
the boundary logic again, this shows the ACTUAL byte positions of every
"Item 1A", "Item 1B", "Item 2", "Item 7", "Item 8" header in each filing,
so we can see what real boundaries exist to anchor on.

    python diag_risk3.py
"""
import re
import config
from data.sec_fetcher import SECFetcher

s = SECFetcher(db_path=config.PORTFOLIO_DB_PATH)

# Find each header type and report its positions.
PATTERNS = {
    "Item 1A": re.compile(r"item\s*1a[\.\s]*risk\s*factors", re.IGNORECASE),
    "Item 1B": re.compile(r"item\s*1b", re.IGNORECASE),
    "Item 2":  re.compile(r"item\s*2\b", re.IGNORECASE),
    "Item 7":  re.compile(r"item\s*7\b", re.IGNORECASE),
    "Item 7A": re.compile(r"item\s*7a\b", re.IGNORECASE),
    "Item 8":  re.compile(r"item\s*8\b", re.IGNORECASE),
}

for tkr in ["AAPL", "COST", "MSFT"]:
    rows = s._get_cached_filings(tkr, "10-K", 1, None, None)
    if not rows:
        print(f"\n{tkr}: no cached 10-K")
        continue
    t = s.get_filing_text(rows[0]["accession_number"])
    print(f"\n===== {tkr}  (len {len(t):,}) =====")
    for name, pat in PATTERNS.items():
        positions = [m.start() for m in pat.finditer(t)]
        # show first few positions
        shown = positions[:6]
        print(f"  {name:8} : {len(positions)} match(es) at {shown}")
    # Show the text right after the LAST 'Item 1A. Risk Factors' header,
    # which should be the real section start.
    m1a = list(PATTERNS["Item 1A"].finditer(t))
    if m1a:
        last = m1a[-1]
        head = re.sub(r"\s+", " ", t[last.end():last.end()+400]).strip()
        print(f"  --- text after LAST Item 1A header ---\n  {head[:380]!r}")
