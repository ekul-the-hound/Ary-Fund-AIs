"""
Market Data Module
==================
Wraps yfinance with SQLite caching, technical indicators, and batch operations.
Designed for the hedge fund AI pipeline — feeds price/volume/fundamentals to
the LLM agent and quant models.

No API key needed. yfinance scrapes Yahoo Finance (free, ~2000 req/hr).

Usage:
    md = MarketData()
    price_df = md.get_prices("AAPL", period="1y")
    fundamentals = md.get_fundamentals("AAPL")
    technicals = md.get_technicals("AAPL")
    comparison = md.compare_stocks(["AAPL", "MSFT", "GOOGL"])
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketData:
    """Yahoo Finance data with SQLite caching and technical analysis."""

    def __init__(self, db_path: str = "data/hedgefund.db"):
        self.db_path = db_path
        self._init_db()

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fundamentals_cache (
                    ticker TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

    # ------------------------------------------------------------------
    # Price History
    # ------------------------------------------------------------------
    def get_prices(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price data.

        Args:
            ticker:    Stock symbol (e.g., "AAPL")
            period:    yfinance period string: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
            interval:  1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
            start:     Start date YYYY-MM-DD (overrides period)
            end:       End date YYYY-MM-DD
            use_cache: Check SQLite cache first (only for daily data)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
        """
        ticker = ticker.upper()

        # Try cache for daily data
        if use_cache and interval == "1d" and not start:
            cached = self._get_cached_prices(ticker, period)
            if cached is not None and len(cached) > 0:
                logger.info(f"Returning {len(cached)} cached daily bars for {ticker}")
                return cached

        # Fetch from Yahoo
        logger.info(f"Fetching {ticker} prices: period={period}, interval={interval}")
        t = yf.Ticker(ticker)

        if start:
            df = t.history(start=start, end=end, interval=interval)
        else:
            df = t.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No price data returned for {ticker}")
            return df

        # Drop dividends/stock splits columns if present
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Cache daily data
        if interval == "1d":
            self._cache_prices(ticker, df)

        return df

    def get_latest_price(self, ticker: str) -> dict:
        """
        Get the most recent price data for a ticker.

        Returns dict with: price, change, change_pct, volume, market_cap,
        day_high, day_low, fifty_two_week_high, fifty_two_week_low
        """
        ticker = ticker.upper()
        t = yf.Ticker(ticker)
        info = t.info

        price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        prev_close = info.get("previousClose", price)
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        return {
            "ticker": ticker,
            "price": price,
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": info.get("volume", 0),
            "market_cap": info.get("marketCap", 0),
            "day_high": info.get("dayHigh", 0),
            "day_low": info.get("dayLow", 0),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh", 0),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow", 0),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
        }

    # ------------------------------------------------------------------
    # Fundamentals
    # ------------------------------------------------------------------
    def get_fundamentals(self, ticker: str, use_cache: bool = True) -> dict:
        """
        Comprehensive fundamental data for a single stock.

        Returns a dict with sections: overview, valuation, financials,
        growth, dividends, analyst_estimates.

        Cache expires after 24 hours (fundamentals don't change intraday).
        """
        ticker = ticker.upper()

        if use_cache:
            cached = self._get_cached_fundamentals(ticker)
            if cached:
                return cached

        t = yf.Ticker(ticker)
        info = t.info

        fundamentals = {
            "ticker": ticker,
            "name": info.get("shortName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "overview": {
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "float_shares": info.get("floatShares"),
                "avg_volume_10d": info.get("averageDailyVolume10Day"),
                "beta": info.get("beta"),
            },
            "valuation": {
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "ev_to_revenue": info.get("enterpriseToRevenue"),
            },
            "financials": {
                "revenue": info.get("totalRevenue"),
                "gross_profit": info.get("grossProfits"),
                "ebitda": info.get("ebitda"),
                "net_income": info.get("netIncomeToCommon"),
                "free_cash_flow": info.get("freeCashflow"),
                "operating_cash_flow": info.get("operatingCashflow"),
                "total_debt": info.get("totalDebt"),
                "total_cash": info.get("totalCash"),
                "debt_to_equity": info.get("debtToEquity"),
                "current_ratio": info.get("currentRatio"),
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "gross_margin": info.get("grossMargins"),
            },
            "growth": {
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
            },
            "dividends": {
                "dividend_rate": info.get("dividendRate"),
                "dividend_yield": info.get("dividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                "ex_dividend_date": str(info.get("exDividendDate", "")),
            },
            "analyst": {
                "target_mean": info.get("targetMeanPrice"),
                "target_low": info.get("targetLowPrice"),
                "target_high": info.get("targetHighPrice"),
                "recommendation": info.get("recommendationKey"),
                "num_analysts": info.get("numberOfAnalystOpinions"),
            },
        }

        # Cache
        self._cache_fundamentals(ticker, fundamentals)
        return fundamentals

    def get_financial_statements(self, ticker: str) -> dict:
        """
        Fetch income statement, balance sheet, and cash flow statement.

        Returns dict with keys: income_statement, balance_sheet, cash_flow
        Each is a DataFrame with annual data (most recent 4 years).
        """
        ticker = ticker.upper()
        t = yf.Ticker(ticker)

        return {
            "income_statement": t.income_stmt,
            "balance_sheet": t.balance_sheet,
            "cash_flow": t.cashflow,
            "quarterly_income": t.quarterly_income_stmt,
            "quarterly_balance": t.quarterly_balance_sheet,
            "quarterly_cashflow": t.quarterly_cashflow,
        }

    # ------------------------------------------------------------------
    # Technical Indicators
    # ------------------------------------------------------------------
    def get_technicals(self, ticker: str, period: str = "1y") -> dict:
        """
        Calculate common technical indicators.

        Returns dict with DataFrames/values for: sma, ema, rsi, macd,
        bollinger bands, average true range, and a summary signal.

        These are the indicators most quant models and the LLM agent
        will reference when generating trade signals.

        Fun fact: RSI was invented by J. Welles Wilder Jr. in 1978.
        His book "New Concepts in Technical Trading Systems" also
        introduced ATR, ADX, and the Parabolic SAR — still widely
        used 45+ years later.

        YouTube deep-dive: "RSI Divergence Trading Strategy"
        https://www.youtube.com/results?search_query=RSI+divergence+trading+strategy
        """
        df = self.get_prices(ticker, period=period)
        if df.empty:
            return {}

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # --- Moving Averages ---
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        sma_200 = close.rolling(window=200).mean()
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()

        # --- RSI (Relative Strength Index) ---
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # --- MACD (Moving Average Convergence Divergence) ---
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - signal_line

        # --- Bollinger Bands ---
        bb_mid = sma_20
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_mid + (bb_std * 2)
        bb_lower = bb_mid - (bb_std * 2)
        bb_width = (bb_upper - bb_lower) / bb_mid  # Normalized width

        # --- ATR (Average True Range) ---
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()

        # --- Volume analysis ---
        vol_sma = df["Volume"].rolling(window=20).mean()
        relative_volume = df["Volume"] / vol_sma

        # --- Summary signal ---
        latest = close.iloc[-1]
        signal = self._generate_tech_signal(
            latest, sma_50.iloc[-1], sma_200.iloc[-1],
            rsi.iloc[-1], macd_histogram.iloc[-1],
            bb_upper.iloc[-1], bb_lower.iloc[-1],
        )

        return {
            "ticker": ticker.upper(),
            "latest_close": float(latest),
            "sma_20": float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None,
            "sma_50": float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None,
            "sma_200": float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else None,
            "rsi_14": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None,
            "macd": {
                "macd_line": float(macd_line.iloc[-1]),
                "signal_line": float(signal_line.iloc[-1]),
                "histogram": float(macd_histogram.iloc[-1]),
            },
            "bollinger": {
                "upper": float(bb_upper.iloc[-1]),
                "middle": float(bb_mid.iloc[-1]),
                "lower": float(bb_lower.iloc[-1]),
                "width": float(bb_width.iloc[-1]) if not pd.isna(bb_width.iloc[-1]) else None,
            },
            "atr_14": float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else None,
            "relative_volume": float(relative_volume.iloc[-1]) if not pd.isna(relative_volume.iloc[-1]) else None,
            "signal": signal,
            # Full series for charting
            "_series": {
                "close": close,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "sma_200": sma_200,
                "rsi": rsi,
                "macd_line": macd_line,
                "signal_line": signal_line,
                "macd_histogram": macd_histogram,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
            },
        }

    # ------------------------------------------------------------------
    # Multi-stock comparison
    # ------------------------------------------------------------------
    def compare_stocks(
        self, tickers: list[str], period: str = "1y"
    ) -> pd.DataFrame:
        """
        Compare multiple stocks' performance over a period.

        Returns DataFrame with normalized returns (starting at 100)
        for easy visual comparison. This is what you'd feed into
        the Streamlit dashboard's comparison chart.
        """
        frames = {}
        for ticker in tickers:
            df = self.get_prices(ticker, period=period)
            if not df.empty:
                # Normalize to 100 at start
                frames[ticker] = (df["Close"] / df["Close"].iloc[0]) * 100

        if not frames:
            return pd.DataFrame()

        return pd.DataFrame(frames)

    def get_correlation_matrix(
        self, tickers: list[str], period: str = "1y"
    ) -> pd.DataFrame:
        """
        Compute return correlation matrix for a list of stocks.

        Correlation is the backbone of portfolio construction.
        Low correlation between holdings = better diversification.

        Research article on this:
        "The Shrinkage Approach to Portfolio Optimization"
        by Ledoit & Wolf (2004) — still the gold standard for
        covariance estimation in quantitative finance.
        """
        returns = {}
        for ticker in tickers:
            df = self.get_prices(ticker, period=period)
            if not df.empty:
                returns[ticker] = df["Close"].pct_change().dropna()

        if not returns:
            return pd.DataFrame()

        return pd.DataFrame(returns).corr()

    # ------------------------------------------------------------------
    # Sector / screening
    # ------------------------------------------------------------------
    def screen_stocks(
        self,
        tickers: list[str],
        min_market_cap: Optional[float] = None,
        max_pe: Optional[float] = None,
        min_dividend_yield: Optional[float] = None,
        min_revenue_growth: Optional[float] = None,
    ) -> list[dict]:
        """
        Simple stock screener across a list of tickers.

        Returns list of stocks passing all filters, with key metrics.
        Use this to narrow a watchlist before deeper analysis.
        """
        results = []
        for ticker in tickers:
            try:
                f = self.get_fundamentals(ticker)
                overview = f.get("overview", {})
                val = f.get("valuation", {})
                div = f.get("dividends", {})
                growth = f.get("growth", {})

                # Apply filters
                mc = overview.get("market_cap") or 0
                if min_market_cap and mc < min_market_cap:
                    continue

                pe = val.get("trailing_pe")
                if max_pe and pe and pe > max_pe:
                    continue

                dy = div.get("dividend_yield") or 0
                if min_dividend_yield and dy < min_dividend_yield:
                    continue

                rg = growth.get("revenue_growth") or 0
                if min_revenue_growth and rg < min_revenue_growth:
                    continue

                results.append({
                    "ticker": ticker,
                    "name": f.get("name", ""),
                    "sector": f.get("sector", ""),
                    "market_cap": mc,
                    "pe_ratio": pe,
                    "dividend_yield": dy,
                    "revenue_growth": rg,
                })
            except Exception as e:
                logger.warning(f"Screening failed for {ticker}: {e}")

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _generate_tech_signal(
        self, price, sma50, sma200, rsi, macd_hist, bb_upper, bb_lower
    ) -> dict:
        """Generate a simple technical signal summary."""
        signals = []
        bias = 0  # -2 to +2 scale

        # Trend: Golden/Death cross
        if pd.notna(sma50) and pd.notna(sma200):
            if sma50 > sma200:
                signals.append("Golden cross (SMA50 > SMA200) — bullish trend")
                bias += 1
            else:
                signals.append("Death cross (SMA50 < SMA200) — bearish trend")
                bias -= 1

        # RSI
        if pd.notna(rsi):
            if rsi > 70:
                signals.append(f"RSI {rsi:.1f} — overbought territory")
                bias -= 1
            elif rsi < 30:
                signals.append(f"RSI {rsi:.1f} — oversold territory")
                bias += 1
            else:
                signals.append(f"RSI {rsi:.1f} — neutral")

        # MACD
        if pd.notna(macd_hist):
            if macd_hist > 0:
                signals.append("MACD histogram positive — bullish momentum")
                bias += 0.5
            else:
                signals.append("MACD histogram negative — bearish momentum")
                bias -= 0.5

        # Bollinger position
        if pd.notna(bb_upper) and pd.notna(bb_lower):
            if price > bb_upper:
                signals.append("Price above upper Bollinger Band — potentially overextended")
                bias -= 0.5
            elif price < bb_lower:
                signals.append("Price below lower Bollinger Band — potentially oversold")
                bias += 0.5

        # Overall
        if bias >= 1.5:
            overall = "BULLISH"
        elif bias >= 0.5:
            overall = "SLIGHTLY_BULLISH"
        elif bias <= -1.5:
            overall = "BEARISH"
        elif bias <= -0.5:
            overall = "SLIGHTLY_BEARISH"
        else:
            overall = "NEUTRAL"

        return {
            "overall": overall,
            "score": round(bias, 1),
            "signals": signals,
        }

    def _get_cached_prices(self, ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Check if we have recent cached prices."""
        # Determine how far back to look
        days_map = {
            "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
            "2y": 730, "5y": 1825,
        }
        days = days_map.get(period)
        if not days:
            return None

        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT date, open as "Open", high as "High", low as "Low",
                       close as "Close", volume as "Volume"
                FROM price_history
                WHERE ticker = ? AND date >= ?
                ORDER BY date
                """,
                conn,
                params=[ticker, cutoff],
            )

        if df.empty:
            return None

        # Only use cache if recent (within 1 trading day)
        latest_cached = df["date"].max()
        if (datetime.now() - datetime.strptime(latest_cached, "%Y-%m-%d")).days > 3:
            return None

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    def _cache_prices(self, ticker: str, df: pd.DataFrame):
        """Cache daily price data to SQLite."""
        if df.empty:
            return

        rows = []
        for date, row in df.iterrows():
            rows.append((
                ticker,
                date.strftime("%Y-%m-%d"),
                float(row.get("Open", 0)),
                float(row.get("High", 0)),
                float(row.get("Low", 0)),
                float(row.get("Close", 0)),
                float(row.get("Close", 0)),  # adj_close
                int(row.get("Volume", 0)),
            ))

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO price_history
                    (ticker, date, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def _get_cached_fundamentals(self, ticker: str) -> Optional[dict]:
        """Get cached fundamentals if < 24 hours old."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT data_json, fetched_at FROM fundamentals_cache WHERE ticker = ?",
                (ticker,),
            ).fetchone()

        if not row:
            return None

        fetched = datetime.fromisoformat(row[1])
        if (datetime.now() - fetched).total_seconds() > 86400:
            return None

        return json.loads(row[0])

    def _cache_fundamentals(self, ticker: str, data: dict):
        """Cache fundamentals data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fundamentals_cache (ticker, data_json, fetched_at)
                VALUES (?, ?, datetime('now'))
                """,
                (ticker, json.dumps(data, default=str)),
            )


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    md = MarketData()

    print("=== Market Data Test ===\n")

    # Latest price
    latest = md.get_latest_price("AAPL")
    print(f"AAPL: ${latest['price']:.2f} ({latest['change_pct']:+.2f}%)")
    print(f"  Market Cap: ${latest['market_cap']:,.0f}")
    print(f"  P/E: {latest['pe_ratio']}")

    # Technicals
    tech = md.get_technicals("AAPL")
    if tech:
        print(f"\nTechnicals:")
        print(f"  RSI(14): {tech['rsi_14']:.1f}")
        print(f"  SMA(50): ${tech['sma_50']:.2f}")
        print(f"  Signal: {tech['signal']['overall']} ({tech['signal']['score']:+.1f})")
        for s in tech["signal"]["signals"]:
            print(f"    • {s}")

    # Fundamentals
    f = md.get_fundamentals("AAPL")
    print(f"\nFundamentals:")
    print(f"  Sector: {f['sector']}")
    print(f"  P/E: {f['valuation']['trailing_pe']}")
    print(f"  Profit Margin: {f['financials']['profit_margin']}")
    print(f"  Revenue Growth: {f['growth']['revenue_growth']}")
