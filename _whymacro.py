import sqlite3, json
c = sqlite3.connect("data/portfolio.db")
row = c.execute("SELECT payload_json FROM agent_opinions WHERE ticker='NVDA' ORDER BY id DESC LIMIT 1").fetchone()
d = json.loads(row[0])
rf = d.get("risk_flags", {})
print("=== levels ===", rf.get("levels"))
print("=== reasons.macro  ===", rf.get("reasons", {}).get("macro"))
print("=== reasons.market ===", rf.get("reasons", {}).get("market"))
print()
km = d.get("key_metrics", {})
for k in ("realized_vol","volatility","drawdown","max_drawdown","rsi"):
    print(f"key_metrics[{k}] =", km.get(k))
print()
print("as_of:", d.get("as_of"))
