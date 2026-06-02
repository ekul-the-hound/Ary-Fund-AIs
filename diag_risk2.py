"""
diag_risk2.py — confirm the FIXED risk extraction against real filings.

Runs the new _select_risk_region + _extract_risk_sentences against the
cached 10-Ks for the tickers that previously reported risks=0 (AAPL, COST)
plus MSFT for contrast. Expect each to now select a large Item 1A region
and extract a healthy list of risk sentences.

    python diag_risk2.py
"""
import config
from data.sec_fetcher import SECFetcher
from agent import filing_analyzer as fa

s = SECFetcher(db_path=config.PORTFOLIO_DB_PATH)

for tkr in ["AAPL", "COST", "MSFT", "NVDA", "GOOGL"]:
    rows = s._get_cached_filings(tkr, "10-K", 1, None, None)
    if not rows:
        print(f"\n{tkr}: no cached 10-K")
        continue
    t = s.get_filing_text(rows[0]["accession_number"])
    region = fa._select_risk_region(t)
    risks = fa._extract_risk_sentences(t)
    print(f"\n===== {tkr} =====")
    print(f"  full_text length      : {len(t):,}")
    print(f"  selected region length: {len(region):,}  "
          f"({'real section' if len(region) >= fa._MIN_RISK_REGION_CHARS else 'WHOLE-DOC fallback'})")
    print(f"  risk sentences         : {len(risks)}")
    for r in risks[:5]:
        print(f"      - {r[:100]}")
