import sys; sys.path.insert(0, ".")
from data.pipeline import build_agent_context
import config
ctx = build_agent_context("NVDA", config.PORTFOLIO_DB_PATH, config)
print("context macro dict:", ctx.get("macro"))
print()
print("Specifically:")
for k in ("vix","yield_curve_spread","recession_probability","yield_curve_inverted"):
    print(f"  {k} =", ctx.get("macro", {}).get(k))
