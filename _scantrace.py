import sys; sys.path.insert(0, ".")
import config
from data.pipeline import build_agent_context
from agent import risk_scanner

ctx = build_agent_context("NVDA", config.PORTFOLIO_DB_PATH, config)
macro = ctx.get("macro") or {}
print("macro dict gen would pass -- keys:", list(macro.keys())[:12])
print("  vix =", macro.get("vix"), "| recession_probability =", macro.get("recession_probability"))

# Now call the scanner exactly like main.py does
rf = risk_scanner.compute_risk_flags(
    ticker="NVDA",
    metrics=ctx.get("metrics") or {},
    macro=macro,
    agent_risks=[],
    config=config,
)
print("macro reasons from scanner:", rf.get("reasons", {}).get("macro"))
print("macro level:", rf.get("levels", {}).get("macro"))
