"""
diag_stage2_peer_risk.py — validate sector-relative risk-count scoring.

Forces a full-universe peer-stats recompute (bypassing the cache), then
prints, per sector, the risk_factor_count distribution (mean/std/n) and —
for each watchlist ticker — its risk count, sector Z-score, and the
resulting filings-bias penalty. This is the direct Stage 2 test: it shows
whether the penalty discriminates sensibly, without a full pipeline run and
without depending on (or being poisoned by) the cache.

    python diag_stage2_peer_risk.py

Writes a correct peer_stats_cache.json as a side effect, so a subsequent
`python main.py ...` will serve real stats (until something rewrites it).
"""
import logging

import config
from data import pipeline as pl
from data import peer_stats as ps
from agent.thesis_generator import _risk_count_penalty

logging.basicConfig(level=logging.WARNING)  # quiet the per-ticker chatter

WATCHLIST = ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "COST", "PEP", "AXP"]

# True risk counts observed in the run logs, for reference in the printout.
KNOWN = {"MSFT": 119, "AAPL": 67, "NVDA": 52, "GOOGL": 25,
         "AMZN": 51, "COST": 67, "PEP": 20, "AXP": 103}


def main():
    db_path = config.PORTFOLIO_DB_PATH
    print("Forcing full-universe peer-stats recompute (this is the slow part)...\n")

    # Reuse the pipeline's own computation path so we exercise the real
    # _metrics_for + _risk_count_for closures, then force=True to bypass cache.
    stats = pl.get_sector_peer_stats(db_path, config, force=True)

    n_sectors = len(stats)
    print(f"computed stats for {n_sectors} sector(s)\n")

    # Show the risk_factor_count distribution per sector.
    print("=== risk_factor_count distribution by sector ===")
    for sector in sorted(stats):
        rc = stats[sector].get("risk_factor_count")
        if not rc:
            continue
        print(f"  {sector:26} mean={rc['mean']:6.1f}  std={rc['std']:6.1f}  n={rc['n']}")

    # For each watchlist ticker, resolve its sector, Z-score, and penalty.
    print("\n=== watchlist: risk count vs sector peers ===")
    print(f"  {'ticker':6} {'sector':24} {'count':>5} {'mean':>6} {'std':>6} {'z':>5} {'penalty':>8}")
    for tk in WATCHLIST:
        snap = None
        try:
            # Reuse the same per-ticker metric builder the recompute used.
            ctx = pl.build_agent_context(tk, db_path, config)
            metrics_raw = ctx.get("metrics") or {}
            sector = (metrics_raw.get("sector") or "").strip().lower()
        except Exception as e:
            print(f"  {tk:6} (context failed: {e})")
            continue

        peer = ps.peer_stats_for_sector(stats, sector) if sector else None
        rc = (peer or {}).get("risk_factor_count") if peer else None
        count = KNOWN.get(tk, 0)
        if rc and rc.get("std", 0) > 0:
            z = (count - rc["mean"]) / rc["std"]
            pen = _risk_count_penalty(count, peer)
            print(f"  {tk:6} {sector:24} {count:5} {rc['mean']:6.1f} {rc['std']:6.1f} "
                  f"{z:5.2f} {-pen:8.2f}")
        else:
            print(f"  {tk:6} {sector:24} {count:5}    (no usable sector distribution)")

    print("\nDone. A correct cache is now written; the next main.py run will use it.")


if __name__ == "__main__":
    main()
