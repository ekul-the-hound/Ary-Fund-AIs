import sys; sys.path.insert(0, ".")
import config
from agent import risk_scanner as RS

_orig = RS.compute_risk_flags
def _probe(*args, **kwargs):
    macro = kwargs.get("macro")
    if macro is None and len(args) >= 3:
        macro = args[2]
    print("=== compute_risk_flags CALLED (real gen path) ===")
    print("  macro empty?:", not bool(macro), "| type:", type(macro).__name__)
    if isinstance(macro, dict):
        print("  macro keys:", list(macro.keys())[:8])
        print("  vix:", macro.get("vix"), "| recession_prob:", macro.get("recession_probability"))
    r = _orig(*args, **kwargs)
    print("  -> macro reasons:", r.get("reasons", {}).get("macro"))
    return r
RS.compute_risk_flags = _probe

import main
print("Running _process_ticker for NVDA (real gen path)...")
result = main._process_ticker("NVDA", config.PORTFOLIO_DB_PATH, config)
print("done. opinion produced:", bool(result))
