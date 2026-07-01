# Monkeypatch compute_risk_flags to log what macro it actually receives,
# then trigger the same analyze path gen uses.
import sys; sys.path.insert(0, ".")
import config
from agent import risk_scanner as RS

_orig = RS.compute_risk_flags
def _probe(*args, **kwargs):
    macro = kwargs.get("macro")
    if macro is None and len(args) >= 3:
        macro = args[2]
    print("=== compute_risk_flags CALLED ===")
    print("  macro type:", type(macro).__name__)
    print("  macro is empty:", not bool(macro))
    if isinstance(macro, dict):
        print("  macro keys:", list(macro.keys())[:8])
        print("  vix:", macro.get("vix"), "recession_prob:", macro.get("recession_probability"))
    result = _orig(*args, **kwargs)
    print("  -> macro reasons:", result.get("reasons", {}).get("macro"))
    return result
RS.compute_risk_flags = _probe

# Now run the actual analyze pipeline for NVDA
from agent import main as agent_main
# find the entry function
fns = [n for n in dir(agent_main) if "analyz" in n.lower() or n in ("run","main","generate_opinion","analyze_ticker")]
print("candidate entry fns:", fns)
