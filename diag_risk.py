"""
diag_risk.py — isolate why AAPL / COST report risks=0.

Every 10-K has an Item 1A, so zero extracted risk factors means the
extraction is failing at one of three stages: the Item 1A regex match,
the sentence split, or the length/cue filters. This prints the survivor
count at each stage for two failing tickers (AAPL, COST) alongside a
known-good one (MSFT) for contrast.

    python diag_risk.py
"""
import re
import config
from data.sec_fetcher import SECFetcher
from agent import filing_analyzer as fa

s = SECFetcher(db_path=config.PORTFOLIO_DB_PATH)

for tkr in ["AAPL", "COST", "MSFT"]:   # MSFT works -> baseline for contrast
    rows = s._get_cached_filings(tkr, "10-K", 1, None, None)
    if not rows:
        print(f"\n{tkr}: no cached 10-K")
        continue
    t = s.get_filing_text(rows[0]["accession_number"])
    print(f"\n===== {tkr} =====")
    print(f"  full_text length      : {len(t):,}")

    m = fa._RISK_SECTION_RE.search(t)
    print(f"  Item 1A regex matched : {bool(m)}")
    region = m.group(1) if m else t
    tag = "Item1A" if m else "WHOLE DOC fallback"
    print(f"  captured region length: {len(region):,}  (using {tag})")

    sents = fa._SENT_SPLIT_RE.split(region)
    print(f"  sentences after split : {len(sents):,}")

    lenok = [x for x in (re.sub(r"\s+", " ", q).strip() for q in sents)
             if 40 <= len(x) <= 400]
    print(f"  survive 40-400 len    : {len(lenok):,}")

    cues = ("risk", "could adversely", "may adversely", "uncertain",
            "depend on", "subject to")
    cued = [x for x in lenok if any(c in x.lower() for c in cues)]
    print(f"  contain a cue word    : {len(cued):,}   <-- this is risks=")

    if lenok and not cued:
        print(f"  sample length-ok sentence (no cue): {lenok[0][:160]!r}")
    if len(sents) <= 3:
        print(f"  RAW region head: {region[:300]!r}")
