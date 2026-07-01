import sys; sys.path.insert(0, ".")
import config
from data import pipeline

# Replicate EXACTLY what gen does: what db_path does gen pass to build_agent_context?
# Check main.py analyze - it uses the db_path passed to it. Find the default.
db = config.PORTFOLIO_DB_PATH
print("db_path gen uses:", db)

ctx = pipeline.build_agent_context("NVDA", db, config)
mac = ctx.get("macro") or {}
print("macro present in context:", bool(mac), "| vix:", mac.get("vix"))

# Is there a SECOND build path? check if main.analyze rebuilds context differently
import inspect
from data import pipeline as P
# does build_agent_context depend on cfg having FRED for macro to populate?
print("macro key count:", len(mac))
