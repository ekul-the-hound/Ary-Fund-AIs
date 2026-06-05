"""
diag_providers.py — live smoke test of Tiingo / FMP / Finnhub providers.

Run from D:\\Ary Fund:   python diag_providers.py

Read-only. Calls one representative function per provider and reports
whether it authenticated and returned usable data. This validates the
keys + endpoints BEFORE any pipeline integration, so we catch a bad key
or changed schema at the surface instead of buried in the data flow.

Does NOT test the realtime WebSocket stream (async, long-running) — that
is validated separately only if you use it.
"""
import sys
from datetime import date, timedelta

try:
    from data import providers
except Exception as e:  # noqa: BLE001
    print(f"FATAL: cannot import data.providers: {e}")
    sys.exit(1)

TICKER = "MSFT"
today = date.today()
start = (today - timedelta(days=30)).isoformat()
end = today.isoformat()
cal_from = today.isoformat()
cal_to = (today + timedelta(days=14)).isoformat()


def _peek(obj, n=3):
    """Compact description of a result for eyeballing shape."""
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return f"DataFrame rows={len(obj)} cols={list(obj.columns)[:8]}"
    except Exception:
        pass
    if isinstance(obj, dict):
        return f"dict keys={list(obj.keys())[:8]}"
    if isinstance(obj, list):
        head = obj[0] if obj else None
        if isinstance(head, dict):
            return f"list len={len(obj)} first_keys={list(head.keys())[:8]}"
        return f"list len={len(obj)}"
    if obj is None:
        return "None"
    return f"{type(obj).__name__}: {str(obj)[:80]}"


def check(name, fn):
    print(f"\n--- {name} ---")
    try:
        rv = fn()
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL ({type(e).__name__}): {str(e)[:200]}")
        return False
    desc = _peek(rv)
    # Heuristic: empty results often mean auth/endpoint issue OR free-tier gate.
    empty = (
        rv is None
        or (hasattr(rv, "__len__") and len(rv) == 0)
    )
    flag = "EMPTY (check key/tier/endpoint)" if empty else "OK"
    print(f"  {flag}: {desc}")
    return not empty


results = {}

# Tiingo — prices (core), fundamentals (paid-tier on free plan, may be empty)
results["tiingo.get_prices"] = check(
    "Tiingo get_prices (adjusted OHLCV)",
    lambda: providers.get_prices(TICKER, start, end),
)
results["tiingo.get_fundamentals"] = check(
    "Tiingo get_fundamentals (note: paid tier on free plan)",
    lambda: providers.get_fundamentals(TICKER),
)

# FMP — analyst bundle (estimates + ratios + dcf), transcripts
results["fmp.get_analyst_data"] = check(
    "FMP get_analyst_data (estimates/ratios/dcf)",
    lambda: providers.get_analyst_data(TICKER),
)
results["fmp.get_transcripts"] = check(
    "FMP get_transcripts (earnings calls)",
    lambda: providers.get_transcripts(TICKER),
)

# Finnhub — earnings calendar, ownership
results["finnhub.get_earnings_events"] = check(
    "Finnhub get_earnings_events (calendar)",
    lambda: providers.get_earnings_events(cal_from, cal_to),
)
results["finnhub.get_ownership_data"] = check(
    "Finnhub get_ownership_data (institutional holders)",
    lambda: providers.get_ownership_data(TICKER),
)

print("\n" + "=" * 60)
ok = sum(1 for v in results.values() if v)
print(f"SUMMARY: {ok}/{len(results)} returned non-empty data")
for k, v in results.items():
    print(f"  {'OK ' if v else '–– '} {k}")
print("\nNote: an EMPTY result isn't necessarily broken — Tiingo "
      "fundamentals and some FMP endpoints are gated on paid tiers. "
      "What matters is which ones return data you actually want to use.")
