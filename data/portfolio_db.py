"""
Portfolio Database
==================
SQLite-backed portfolio management system. Tracks positions, trades,
watchlists, P&L, and risk metrics. This is the central state store
that the LLM agent and quant models read/write to.

All money values are stored in USD. Positions are updated in real-time
by pulling latest prices from MarketData.

Usage:
    db = PortfolioDB()
    db.add_position("AAPL", shares=100, entry_price=170.00)
    db.record_trade("AAPL", "BUY", shares=50, price=175.00)
    snapshot = db.get_portfolio_snapshot()
    risk = db.get_risk_metrics()
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# MODULE-LEVEL FUNCTIONAL API
# =============================================================================
# The class-based :class:`PortfolioDB` below is the full-featured interface
# used by the UI, the quant modules, and interactive scripts.
#
# ``main.py`` uses a simpler functional save: one call per ticker per run,
# storing the final agent opinion as a JSON blob. Keeping it as a
# module-level function makes it trivial for tests to monkeypatch via
# ``monkeypatch.setattr(portfolio_db, "save_agent_opinion", ...)``.


def _ensure_agent_opinions_table(db_path: str) -> None:
    """Create the ``agent_opinions`` table if it doesn't exist.

    Schema matches the ``mock_db`` test fixture: ``(id, ticker, created_at,
    payload_json)``. Idempotent.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_opinions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_agent_opinion(
    ticker: str,
    opinion: dict,
    db_path: Optional[str] = None,
) -> int:
    """Persist one agent opinion to the ``agent_opinions`` table.

    Parameters
    ----------
    ticker:
        Ticker this opinion is about.
    opinion:
        The final merged opinion dict (``main._assemble_final_opinion``
        output). Serialized to JSON for storage.
    db_path:
        SQLite DB path. If omitted, falls back to the default path used by
        :class:`PortfolioDB`.

    Returns
    -------
    int
        The newly-inserted row id.
    """
    path = db_path or "data/hedgefund.db"
    _ensure_agent_opinions_table(path)

    payload_json = json.dumps(opinion, default=str)
    created_at = datetime.now().isoformat()

    with sqlite3.connect(path) as conn:
        cur = conn.execute(
            "INSERT INTO agent_opinions (ticker, created_at, payload_json) "
            "VALUES (?, ?, ?)",
            (ticker, created_at, payload_json),
        )
        conn.commit()
        row_id = cur.lastrowid

    logger.info("portfolio_db.save_agent_opinion | %s | row=%s", ticker, row_id)
    return int(row_id or 0)


class PortfolioDB:
    """Portfolio tracking and analytics backed by SQLite."""

    def __init__(self, db_path: str = "data/hedgefund.db"):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Positions: current holdings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL UNIQUE,
                    shares REAL NOT NULL DEFAULT 0,
                    avg_entry_price REAL NOT NULL DEFAULT 0,
                    sector TEXT,
                    thesis TEXT,
                    conviction TEXT CHECK(conviction IN ('HIGH', 'MEDIUM', 'LOW')),
                    position_type TEXT DEFAULT 'LONG'
                        CHECK(position_type IN ('LONG', 'SHORT')),
                    opened_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

            # Trades: full trade history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('BUY', 'SELL', 'SHORT', 'COVER')),
                    shares REAL NOT NULL,
                    price REAL NOT NULL,
                    fees REAL DEFAULT 0,
                    notes TEXT,
                    strategy TEXT,
                    executed_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

            # Watchlist: tickers being monitored
            conn.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL UNIQUE,
                    target_entry REAL,
                    target_exit REAL,
                    stop_loss REAL,
                    thesis TEXT,
                    priority TEXT DEFAULT 'MEDIUM'
                        CHECK(priority IN ('HIGH', 'MEDIUM', 'LOW')),
                    added_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

            # Daily snapshots: historical portfolio values for P&L tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL DEFAULT 0,
                    invested REAL NOT NULL DEFAULT 0,
                    daily_pnl REAL DEFAULT 0,
                    daily_pnl_pct REAL DEFAULT 0,
                    positions_json TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

            # Alerts: price/event triggers
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    alert_type TEXT NOT NULL
                        CHECK(alert_type IN (
                            'PRICE_ABOVE', 'PRICE_BELOW',
                            'PCT_CHANGE', 'VOLUME_SPIKE',
                            'EARNINGS', 'FILING', 'CUSTOM'
                        )),
                    threshold REAL,
                    message TEXT,
                    triggered INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    triggered_at TEXT
                )
            """)

            # Portfolio metadata (cash balance, benchmark, etc.)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

            # Set default cash balance if not exists
            conn.execute("""
                INSERT OR IGNORE INTO portfolio_meta (key, value)
                VALUES ('cash_balance', '100000.00')
            """)
            conn.execute("""
                INSERT OR IGNORE INTO portfolio_meta (key, value)
                VALUES ('benchmark', 'SPY')
            """)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    def add_position(
        self,
        ticker: str,
        shares: float,
        entry_price: float,
        sector: Optional[str] = None,
        thesis: Optional[str] = None,
        conviction: str = "MEDIUM",
        position_type: str = "LONG",
    ) -> dict:
        """
        Add a new position or increase an existing one.

        If the position already exists, this computes the new average
        entry price using cost-basis averaging.
        """
        ticker = ticker.upper()

        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT shares, avg_entry_price FROM positions WHERE ticker = ?",
                (ticker,),
            ).fetchone()

            if existing:
                old_shares, old_price = existing
                # Weighted average entry price
                total_shares = old_shares + shares
                if total_shares > 0:
                    new_avg = (
                        (old_shares * old_price) + (shares * entry_price)
                    ) / total_shares
                else:
                    new_avg = entry_price

                conn.execute(
                    """
                    UPDATE positions
                    SET shares = ?, avg_entry_price = ?, updated_at = datetime('now')
                    WHERE ticker = ?
                    """,
                    (total_shares, new_avg, ticker),
                )

                # Also record the trade
                self.record_trade(ticker, "BUY", shares, entry_price)

                return {
                    "ticker": ticker,
                    "action": "INCREASED",
                    "total_shares": total_shares,
                    "avg_entry_price": round(new_avg, 4),
                }
            else:
                conn.execute(
                    """
                    INSERT INTO positions
                        (ticker, shares, avg_entry_price, sector, thesis,
                         conviction, position_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ticker, shares, entry_price, sector, thesis,
                     conviction, position_type),
                )

                self.record_trade(ticker, "BUY", shares, entry_price)

                return {
                    "ticker": ticker,
                    "action": "OPENED",
                    "shares": shares,
                    "entry_price": entry_price,
                }

    def reduce_position(self, ticker: str, shares: float, price: float) -> dict:
        """Sell shares from an existing position."""
        ticker = ticker.upper()

        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT shares, avg_entry_price FROM positions WHERE ticker = ?",
                (ticker,),
            ).fetchone()

            if not existing:
                raise ValueError(f"No position in {ticker}")

            old_shares, entry_price = existing
            if shares > old_shares:
                raise ValueError(
                    f"Cannot sell {shares} shares of {ticker}, only hold {old_shares}"
                )

            new_shares = old_shares - shares
            realized_pnl = (price - entry_price) * shares

            if new_shares <= 0:
                conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
                action = "CLOSED"
            else:
                conn.execute(
                    "UPDATE positions SET shares = ?, updated_at = datetime('now') WHERE ticker = ?",
                    (new_shares, ticker),
                )
                action = "REDUCED"

            self.record_trade(ticker, "SELL", shares, price,
                              notes=f"Realized P&L: ${realized_pnl:,.2f}")

            # Update cash
            cash = self._get_meta_float("cash_balance")
            self._set_meta("cash_balance", str(cash + (shares * price)))

            return {
                "ticker": ticker,
                "action": action,
                "shares_sold": shares,
                "remaining_shares": new_shares,
                "sell_price": price,
                "realized_pnl": round(realized_pnl, 2),
            }

    def get_positions(self) -> list[dict]:
        """Get all current positions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT ticker, shares, avg_entry_price, sector, thesis,
                       conviction, position_type, opened_at, updated_at
                FROM positions
                ORDER BY shares * avg_entry_price DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_position(self, ticker: str) -> Optional[dict]:
        """Get a single position."""
        ticker = ticker.upper()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM positions WHERE ticker = ?", (ticker,)
            ).fetchone()
            return dict(row) if row else None

    def update_thesis(self, ticker: str, thesis: str, conviction: Optional[str] = None):
        """Update investment thesis and conviction for a position."""
        ticker = ticker.upper()
        with sqlite3.connect(self.db_path) as conn:
            if conviction:
                conn.execute(
                    "UPDATE positions SET thesis = ?, conviction = ?, updated_at = datetime('now') WHERE ticker = ?",
                    (thesis, conviction, ticker),
                )
            else:
                conn.execute(
                    "UPDATE positions SET thesis = ?, updated_at = datetime('now') WHERE ticker = ?",
                    (thesis, ticker),
                )

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------
    def record_trade(
        self,
        ticker: str,
        action: str,
        shares: float,
        price: float,
        fees: float = 0,
        notes: Optional[str] = None,
        strategy: Optional[str] = None,
    ):
        """Record a trade in the trade log."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO trades (ticker, action, shares, price, fees, notes, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ticker.upper(), action, shares, price, fees, notes, strategy),
            )

    def get_trade_history(
        self,
        ticker: Optional[str] = None,
        limit: int = 50,
        start_date: Optional[str] = None,
    ) -> list[dict]:
        """Get trade history, optionally filtered by ticker."""
        sql = "SELECT * FROM trades WHERE 1=1"
        params: list = []

        if ticker:
            sql += " AND ticker = ?"
            params.append(ticker.upper())
        if start_date:
            sql += " AND executed_at >= ?"
            params.append(start_date)

        sql += " ORDER BY executed_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Portfolio Snapshot (with live prices)
    # ------------------------------------------------------------------
    def get_portfolio_snapshot(self, market_data=None) -> dict:
        """
        Generate a complete portfolio snapshot with live P&L.

        Args:
            market_data: Optional MarketData instance for live prices.
                         If None, uses stored entry prices only.

        Returns:
            Dict with: positions (with P&L), totals, allocation breakdown
        """
        positions = self.get_positions()
        cash = self._get_meta_float("cash_balance")

        enriched = []
        total_invested = 0
        total_market_value = 0
        total_unrealized_pnl = 0

        for pos in positions:
            ticker = pos["ticker"]
            shares = pos["shares"]
            entry = pos["avg_entry_price"]
            cost_basis = shares * entry

            # Get live price if market_data available
            if market_data:
                try:
                    latest = market_data.get_latest_price(ticker)
                    current_price = latest["price"]
                except Exception:
                    current_price = entry
            else:
                current_price = entry

            market_value = shares * current_price
            unrealized_pnl = market_value - cost_basis
            unrealized_pct = (unrealized_pnl / cost_basis * 100) if cost_basis else 0

            total_invested += cost_basis
            total_market_value += market_value
            total_unrealized_pnl += unrealized_pnl

            enriched.append({
                **pos,
                "current_price": current_price,
                "cost_basis": round(cost_basis, 2),
                "market_value": round(market_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pct": round(unrealized_pct, 2),
            })

        total_value = total_market_value + cash

        # Allocation breakdown
        for pos in enriched:
            pos["portfolio_weight"] = round(
                pos["market_value"] / total_value * 100, 2
            ) if total_value else 0

        # Sector allocation
        sector_alloc = {}
        for pos in enriched:
            sector = pos.get("sector") or "Unknown"
            sector_alloc[sector] = sector_alloc.get(sector, 0) + pos["market_value"]

        return {
            "timestamp": datetime.now().isoformat(),
            "positions": enriched,
            "summary": {
                "total_value": round(total_value, 2),
                "cash": round(cash, 2),
                "invested": round(total_market_value, 2),
                "cost_basis": round(total_invested, 2),
                "unrealized_pnl": round(total_unrealized_pnl, 2),
                "unrealized_pct": round(
                    (total_unrealized_pnl / total_invested * 100) if total_invested else 0, 2
                ),
                "cash_pct": round(cash / total_value * 100, 2) if total_value else 100,
                "num_positions": len(enriched),
            },
            "sector_allocation": {
                k: round(v / total_value * 100, 2)
                for k, v in sector_alloc.items()
            } if total_value else {},
        }

    def save_daily_snapshot(self, market_data=None):
        """Save today's portfolio snapshot for historical tracking."""
        snap = self.get_portfolio_snapshot(market_data)
        today = datetime.now().strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            # Check if we already have today's snapshot
            existing = conn.execute(
                "SELECT id FROM portfolio_snapshots WHERE date = ?", (today,)
            ).fetchone()

            # Get yesterday's value for daily P&L
            yesterday = conn.execute(
                "SELECT total_value FROM portfolio_snapshots ORDER BY date DESC LIMIT 1"
            ).fetchone()
            prev_value = yesterday[0] if yesterday else snap["summary"]["total_value"]

            daily_pnl = snap["summary"]["total_value"] - prev_value
            daily_pct = (daily_pnl / prev_value * 100) if prev_value else 0

            if existing:
                conn.execute(
                    """
                    UPDATE portfolio_snapshots
                    SET total_value = ?, cash = ?, invested = ?,
                        daily_pnl = ?, daily_pnl_pct = ?, positions_json = ?
                    WHERE date = ?
                    """,
                    (
                        snap["summary"]["total_value"],
                        snap["summary"]["cash"],
                        snap["summary"]["invested"],
                        daily_pnl,
                        daily_pct,
                        json.dumps(snap["positions"], default=str),
                        today,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO portfolio_snapshots
                        (date, total_value, cash, invested, daily_pnl,
                         daily_pnl_pct, positions_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        today,
                        snap["summary"]["total_value"],
                        snap["summary"]["cash"],
                        snap["summary"]["invested"],
                        daily_pnl,
                        daily_pct,
                        json.dumps(snap["positions"], default=str),
                    ),
                )

    # ------------------------------------------------------------------
    # Risk Metrics
    # ------------------------------------------------------------------
    def get_risk_metrics(self, market_data=None) -> dict:
        """
        Portfolio risk analysis.

        Computes concentration risk, position sizing metrics,
        and basic risk flags.

        For VaR and Monte Carlo, those live in the quant_models module
        which uses historical return data from MarketData.

        YouTube: "Portfolio Risk Management for Quant Traders"
        https://www.youtube.com/results?search_query=portfolio+risk+management+quant

        Research: "Risk Management and Financial Institutions"
        by John Hull — the textbook used at most quant programs.
        """
        snap = self.get_portfolio_snapshot(market_data)
        positions = snap["positions"]
        summary = snap["summary"]

        if not positions:
            return {"warning": "No positions in portfolio"}

        weights = [p["portfolio_weight"] for p in positions]
        max_weight = max(weights) if weights else 0
        hhi = sum(w ** 2 for w in weights)  # Herfindahl-Hirschman Index

        # Position sizing flags
        flags = []
        for pos in positions:
            w = pos["portfolio_weight"]
            if w > 20:
                flags.append(f"CONCENTRATION: {pos['ticker']} is {w:.1f}% of portfolio")
            if pos.get("unrealized_pct", 0) < -15:
                flags.append(
                    f"DRAWDOWN: {pos['ticker']} down {pos['unrealized_pct']:.1f}%"
                )
            if pos.get("conviction") == "LOW" and w > 10:
                flags.append(
                    f"SIZING: {pos['ticker']} is {w:.1f}% but conviction is LOW"
                )

        # Cash level check
        if summary["cash_pct"] < 5:
            flags.append(f"LIQUIDITY: Cash is only {summary['cash_pct']:.1f}% of portfolio")
        elif summary["cash_pct"] > 40:
            flags.append(f"UNDERINVESTED: {summary['cash_pct']:.1f}% in cash")

        # Sector concentration
        sectors = snap.get("sector_allocation", {})
        for sector, pct in sectors.items():
            if pct > 40:
                flags.append(f"SECTOR RISK: {pct:.1f}% in {sector}")

        return {
            "num_positions": summary["num_positions"],
            "max_position_weight": round(max_weight, 2),
            "herfindahl_index": round(hhi, 2),
            "concentration": (
                "HIGH" if hhi > 3000 else "MODERATE" if hhi > 1500 else "LOW"
            ),
            "cash_pct": summary["cash_pct"],
            "total_unrealized_pnl": summary["unrealized_pnl"],
            "risk_flags": flags,
            "flag_count": len(flags),
        }

    # ------------------------------------------------------------------
    # Watchlist
    # ------------------------------------------------------------------
    def add_to_watchlist(
        self,
        ticker: str,
        target_entry: Optional[float] = None,
        target_exit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        thesis: Optional[str] = None,
        priority: str = "MEDIUM",
    ):
        """Add a ticker to the watchlist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO watchlist
                    (ticker, target_entry, target_exit, stop_loss, thesis, priority)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ticker.upper(), target_entry, target_exit, stop_loss, thesis, priority),
            )

    def get_watchlist(self) -> list[dict]:
        """Get all watchlist items."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM watchlist ORDER BY priority, added_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def remove_from_watchlist(self, ticker: str):
        """Remove a ticker from the watchlist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    def add_alert(
        self,
        ticker: str,
        alert_type: str,
        threshold: Optional[float] = None,
        message: Optional[str] = None,
    ):
        """Create a price/event alert."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO alerts (ticker, alert_type, threshold, message)
                VALUES (?, ?, ?, ?)
                """,
                (ticker.upper(), alert_type, threshold, message),
            )

    def check_alerts(self, market_data) -> list[dict]:
        """
        Check all active alerts against current prices.

        Returns list of triggered alerts.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            alerts = conn.execute(
                "SELECT * FROM alerts WHERE triggered = 0"
            ).fetchall()

        triggered = []
        for alert in alerts:
            alert = dict(alert)
            try:
                latest = market_data.get_latest_price(alert["ticker"])
                price = latest["price"]

                fire = False
                if alert["alert_type"] == "PRICE_ABOVE" and price > alert["threshold"]:
                    fire = True
                elif alert["alert_type"] == "PRICE_BELOW" and price < alert["threshold"]:
                    fire = True
                elif alert["alert_type"] == "VOLUME_SPIKE":
                    if latest.get("volume", 0) > alert["threshold"]:
                        fire = True

                if fire:
                    alert["current_price"] = price
                    triggered.append(alert)

                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute(
                            """
                            UPDATE alerts
                            SET triggered = 1, triggered_at = datetime('now')
                            WHERE id = ?
                            """,
                            (alert["id"],),
                        )
            except Exception as e:
                logger.warning(f"Alert check failed for {alert['ticker']}: {e}")

        return triggered

    # ------------------------------------------------------------------
    # Performance tracking
    # ------------------------------------------------------------------
    def get_performance_history(
        self, days: int = 30
    ) -> pd.DataFrame:
        """Get historical portfolio value series."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT date, total_value, daily_pnl, daily_pnl_pct
                FROM portfolio_snapshots
                WHERE date >= ?
                ORDER BY date
                """,
                conn,
                params=[cutoff],
            )

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        return df

    def get_realized_pnl(
        self, ticker: Optional[str] = None, start_date: Optional[str] = None
    ) -> float:
        """
        Calculate realized P&L from trade history.

        Uses FIFO (First In, First Out) matching of buys and sells.
        """
        trades = self.get_trade_history(ticker=ticker, limit=10000, start_date=start_date)

        # Group by ticker and compute realized P&L
        total_pnl = 0.0
        buy_queue: dict[str, list] = {}  # ticker -> [(shares, price), ...]

        # Process in chronological order
        for trade in reversed(trades):
            t = trade["ticker"]
            if t not in buy_queue:
                buy_queue[t] = []

            if trade["action"] == "BUY":
                buy_queue[t].append((trade["shares"], trade["price"]))
            elif trade["action"] == "SELL":
                remaining = trade["shares"]
                sell_price = trade["price"]

                while remaining > 0 and buy_queue.get(t):
                    buy_shares, buy_price = buy_queue[t][0]
                    matched = min(remaining, buy_shares)
                    total_pnl += matched * (sell_price - buy_price)
                    remaining -= matched

                    if matched >= buy_shares:
                        buy_queue[t].pop(0)
                    else:
                        buy_queue[t][0] = (buy_shares - matched, buy_price)

        return round(total_pnl, 2)

    # ------------------------------------------------------------------
    # Cash management
    # ------------------------------------------------------------------
    def get_cash(self) -> float:
        return self._get_meta_float("cash_balance")

    def set_cash(self, amount: float):
        self._set_meta("cash_balance", str(amount))

    def deposit(self, amount: float):
        cash = self.get_cash()
        self.set_cash(cash + amount)

    def withdraw(self, amount: float):
        cash = self.get_cash()
        if amount > cash:
            raise ValueError(f"Insufficient cash: have ${cash:,.2f}, need ${amount:,.2f}")
        self.set_cash(cash - amount)

    # ------------------------------------------------------------------
    # Export / reporting
    # ------------------------------------------------------------------
    def export_portfolio_summary(self, market_data=None) -> str:
        """
        Generate a text summary suitable for feeding to the LLM agent.

        This is the function your agent calls to understand the current
        portfolio state before generating analysis.
        """
        snap = self.get_portfolio_snapshot(market_data)
        risk = self.get_risk_metrics(market_data)
        s = snap["summary"]

        lines = [
            "=== PORTFOLIO SUMMARY ===",
            f"Total Value: ${s['total_value']:,.2f}",
            f"Cash: ${s['cash']:,.2f} ({s['cash_pct']:.1f}%)",
            f"Invested: ${s['invested']:,.2f}",
            f"Unrealized P&L: ${s['unrealized_pnl']:,.2f} ({s['unrealized_pct']:+.1f}%)",
            f"Positions: {s['num_positions']}",
            f"Concentration: {risk.get('concentration', 'N/A')}",
            "",
            "--- POSITIONS ---",
        ]

        for pos in snap["positions"]:
            lines.append(
                f"  {pos['ticker']:6s} | {pos['shares']:>8.1f} shares | "
                f"Entry ${pos['avg_entry_price']:.2f} | "
                f"Now ${pos['current_price']:.2f} | "
                f"P&L ${pos['unrealized_pnl']:>+10,.2f} ({pos['unrealized_pct']:>+6.1f}%) | "
                f"Weight {pos['portfolio_weight']:.1f}% | "
                f"Conviction: {pos.get('conviction', 'N/A')}"
            )

        if snap["sector_allocation"]:
            lines.append("\n--- SECTOR ALLOCATION ---")
            for sector, pct in sorted(
                snap["sector_allocation"].items(), key=lambda x: -x[1]
            ):
                lines.append(f"  {sector}: {pct:.1f}%")

        if risk.get("risk_flags"):
            lines.append("\n--- RISK FLAGS ---")
            for flag in risk["risk_flags"]:
                lines.append(f"  ⚠ {flag}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _get_meta_float(self, key: str) -> float:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM portfolio_meta WHERE key = ?", (key,)
            ).fetchone()
            return float(row[0]) if row else 0.0

    def _set_meta(self, key: str, value: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO portfolio_meta (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
                """,
                (key, value),
            )


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = PortfolioDB()

    print("=== Portfolio DB Test ===\n")

    # Start fresh for test
    db.set_cash(100_000)

    # Add some positions
    db.add_position("AAPL", shares=100, entry_price=170.00,
                     sector="Technology", thesis="AI integration in services",
                     conviction="HIGH")
    db.add_position("MSFT", shares=50, entry_price=380.00,
                     sector="Technology", thesis="Azure + Copilot growth",
                     conviction="HIGH")
    db.add_position("JPM", shares=75, entry_price=190.00,
                     sector="Financials", thesis="Net interest margin expansion",
                     conviction="MEDIUM")
    db.add_position("XOM", shares=60, entry_price=105.00,
                     sector="Energy", thesis="Free cash flow + dividend",
                     conviction="LOW")

    # Deduct cash for purchases
    total_cost = (100 * 170) + (50 * 380) + (75 * 190) + (60 * 105)
    db.set_cash(100_000 - total_cost)

    # Add watchlist items
    db.add_to_watchlist("NVDA", target_entry=850.00, thesis="Waiting for pullback",
                         priority="HIGH")
    db.add_to_watchlist("AMZN", target_entry=175.00, thesis="AWS margin expansion",
                         priority="MEDIUM")

    # Add an alert
    db.add_alert("AAPL", "PRICE_ABOVE", threshold=200.00,
                  message="AAPL hit target, consider trimming")

    # Print portfolio
    print(db.export_portfolio_summary())

    # Print watchlist
    print("\n--- WATCHLIST ---")
    for w in db.get_watchlist():
        print(f"  {w['ticker']} | Entry target: ${w['target_entry']}"
              f" | Priority: {w['priority']}")

    # Print risk metrics
    print("\n--- RISK METRICS ---")
    risk = db.get_risk_metrics()
    print(f"  Positions: {risk['num_positions']}")
    print(f"  Max weight: {risk['max_position_weight']}%")
    print(f"  HHI: {risk['herfindahl_index']}")
    print(f"  Concentration: {risk['concentration']}")
    print(f"  Flags: {risk['flag_count']}")
    for flag in risk["risk_flags"]:
        print(f"    ⚠ {flag}")