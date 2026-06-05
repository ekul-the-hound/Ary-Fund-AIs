"""
diag_probe_metrics.py — find exactly where universe tickers get dropped.

We've confirmed: fundamentals_cache is full (560) and fresh (0 stale), the
universe list is full, build_agent_context is no longer in the path. Yet the
peer recompute yields 3 tickers. This probes the ACTUAL cheap reads
(_sector_for / _risk_count_for equivalents) for a sample of universe tickers
and prints what each returns, so we see the drop point directly instead of
guessing.

    python diag_probe_metrics.py
"""
import config
from data.market_data import MarketData
from data.sec_fetcher import SECFetcher
from agent.filing_analyzer import summarize_filings_by_year

try:
    from data.universe import US_UNIVERSE
except Exception:
    from universe import US_UNIVERSE

db_path = config.PORTFOLIO_DB_PATH
md = MarketData(db_path=db_path)
sec = SECFetcher(db_path=db_path)

# 1) What does get_universe_tickers actually return?
from data import peer_stats as ps
uni = ps.get_universe_tickers(config)
print(f"get_universe_tickers -> {len(uni)} tickers; first 10: {uni[:10]}\n")

# 2) Probe the cheap reads for the first 15 universe names.
print(f"{'ticker':6} {'sector (get_fundamentals)':28} {'risk_count':>10}")
sample = uni[:15]
sector_ok = 0
for tk in sample:
    # sector via the SAME call _sector_for uses
    try:
        f = md.get_fundamentals(tk, use_cache=True)
        sec_val = (f.get("sector") if isinstance(f, dict) else None) or "(none)"
    except Exception as e:
        sec_val = f"ERR: {e}"

    # risk count via cached filings
    rc = 0
    try:
        filings = []
        for kind in ("10-K", "10-Q"):
            cached = sec._get_cached_filings(tk, kind, 3, None, None)
            for fl in cached:
                acc = fl.get("accession_number")
                if not acc:
                    continue
                txt = sec.get_filing_text(acc)
                if txt:
                    filings.append({**fl, "text": txt})
        if filings:
            summ = summarize_filings_by_year(tk, filings)
            rc = int(summ.get("risk_factor_count") or 0)
    except Exception as e:
        rc = f"ERR: {e}"

    if sec_val not in ("(none)",) and not str(sec_val).startswith("ERR"):
        sector_ok += 1
    print(f"{tk:6} {str(sec_val):28} {str(rc):>10}")

print(f"\n{sector_ok}/{len(sample)} sample tickers returned a sector.")

# 3) Now run the FULL compute via the real path and report sector count.
print("\nRunning full compute_all_sector_peer_stats over the universe...")

def _metrics_for(tk):
    try:
        f = md.get_fundamentals(tk, use_cache=True)
    except Exception:
        return None
    if not isinstance(f, dict):
        return None
    s = f.get("sector")
    s = s.strip() if isinstance(s, str) and s.strip() else None
    if s is None:
        return None
    # risk count
    rc = 0
    try:
        filings = []
        for kind in ("10-K", "10-Q"):
            for fl in sec._get_cached_filings(tk, kind, 3, None, None):
                acc = fl.get("accession_number")
                if not acc:
                    continue
                txt = sec.get_filing_text(acc)
                if txt:
                    filings.append({**fl, "text": txt})
        if filings:
            rc = int(summarize_filings_by_year(tk, filings).get("risk_factor_count") or 0)
    except Exception:
        rc = 0
    return {"sector": s, "risk_factor_count": rc}

stats = ps.compute_all_sector_peer_stats(_metrics_for, uni)
print(f"\nRESULT: {len(stats)} sector(s)")
for s in sorted(stats):
    rc = stats[s].get("risk_factor_count")
    if rc:
        print(f"  {s:26} mean={rc['mean']:6.1f} std={rc['std']:6.1f} n={rc['n']}")
