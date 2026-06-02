"""
diag_risk5.py — verify the HYBRID risk extraction against real filings.

Confirms, for each watchlist 10-K, whether a real Item 1A section was
isolated or the whole-doc fallback was used, and how many risk sentences
came out. Expect every ticker to yield a non-trivial, sane count of
actual risk-factor sentences (not cover-page boilerplate).

    python diag_risk5.py
"""
import config
from data.sec_fetcher import SECFetcher
from agent import filing_analyzer as fa

s = SECFetcher(db_path=config.PORTFOLIO_DB_PATH)

for tkr in ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "COST", "PEP", "AXP"]:
    rows = s._get_cached_filings(tkr, "10-K", 1, None, None)
    if not rows:
        print(f"\n{tkr}: no cached 10-K")
        continue
    t = s.get_filing_text(rows[0]["accession_number"])
    region = fa._select_risk_region(t)
    risks = fa._extract_risk_sentences(t)
    used = "Item1A section" if region is not None else "WHOLE-DOC fallback"
    rlen = len(region) if region is not None else 0
    print(f"\n===== {tkr} =====")
    print(f"  path           : {used}"
          + (f" ({rlen:,} chars)" if region is not None else ""))
    print(f"  risk sentences : {len(risks)}")
    for r in risks[:4]:
        print(f"      - {r[:100]}")
