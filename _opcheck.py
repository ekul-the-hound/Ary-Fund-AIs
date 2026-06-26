import sqlite3
c = sqlite3.connect("data/portfolio.db")
rows = c.execute("SELECT ticker, length(payload_json) FROM agent_opinions ORDER BY id DESC LIMIT 10").fetchall()
print("opinions in data/portfolio.db:", rows)
try:
    import config
    print("PORTFOLIO_DB_PATH =", getattr(config, "PORTFOLIO_DB_PATH", "NOT SET"))
except Exception as e:
    print("config import err:", e)
