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

    def __init__(self, db_path: str = "data/hedgefund.db", registry=None):
        self.db_path = db_path
        self._registry = registry
        self._init_db()
        self._init_extended_db()
        self._register_sources()

    # ------------------------------------------------------------------
    # Registry plumbing
    # ------------------------------------------------------------------
    @property
    def registry(self):
        if self._registry is None:
            try:
                from data.data_registry import get_default_registry
                self._registry = get_default_registry(self.db_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("market_data | could not load data_registry: %s", e)
        return self._registry

    def _register_sources(self) -> None:
        reg = self.registry
        if reg is None:
            return
        try:
            reg.register_source("yfinance", "price", "hourly", base_priority=2,
                                notes="Yahoo Finance prices/options/info")
            reg.register_source("finra",    "short", "daily", base_priority=1,
                                notes="FINRA short sale volume")
            reg.register_source("ibkr_scrape", "borrow", "hourly", base_priority=3,
                                notes="IBKR shortable shares (scrape, brittle)")
            reg.register_source("issuer_csv", "etf", "daily", base_priority=2,
                                notes="ETF issuer holdings CSVs")
        except Exception as e:  # noqa: BLE001
            logger.debug("market_data | source registration skipped: %s", e)

    def _init_extended_db(self) -> None:
        """Raw tables for options, short interest, borrow, ETF holdings."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS options_chain (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    expiry TEXT NOT NULL,
                    option_type TEXT CHECK(option_type IN ('call','put')),
                    strike REAL NOT NULL,
                    last_price REAL,
                    bid REAL, ask REAL,
                    volume INTEGER, open_interest INTEGER,
                    implied_volatility REAL,
                    in_the_money INTEGER,
                    UNIQUE(ticker, expiry, option_type, strike, fetched_at)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_opt_ticker ON options_chain(ticker, expiry)")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS short_interest (
                    ticker TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    short_interest REAL,
                    short_pct_float REAL,
                    days_to_cover REAL,
                    avg_volume REAL,
                    source TEXT,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (ticker, as_of, source)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS borrow_data (
                    ticker TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    borrow_fee_bps REAL,
                    shares_available REAL,
                    source TEXT,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (ticker, as_of, source)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS etf_holdings (
                    etf_ticker TEXT NOT NULL,
                    holding_ticker TEXT NOT NULL,
                    weight REAL,
                    shares REAL,
                    market_value REAL,
                    as_of TEXT NOT NULL,
                    source TEXT,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (etf_ticker, holding_ticker, as_of)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_etfh_holding ON etf_holdings(holding_ticker)")

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


    # ==================================================================
    # === EXTENDED API: options, short interest, borrow, ETFs, registry sync ===
    # ==================================================================

    # ------------------------------------------------------------------
    # Options chains and derived IV / put-call / unusual activity
    # ------------------------------------------------------------------
    def get_options_chain(self, ticker: str, max_expiries: int = 4) -> dict:
        """Pull option chains for the next ``max_expiries`` expirations.

        Returns a dict with one entry per expiry containing puts and calls
        DataFrames plus computed ATM IV and put/call ratio.
        """
        ticker = ticker.upper()
        t = yf.Ticker(ticker)
        try:
            expiries = list(t.options or [])[:max_expiries]
        except Exception as e:  # noqa: BLE001
            logger.warning("options | %s | could not list expiries: %s", ticker, e)
            return {}
        if not expiries:
            return {}

        spot = self._spot_price_safe(ticker, t)
        out: dict = {"ticker": ticker, "spot": spot, "expiries": {}}

        rows_to_cache: list[tuple] = []
        for exp in expiries:
            try:
                chain = t.option_chain(exp)
            except Exception as e:  # noqa: BLE001
                logger.debug("options | %s | %s skipped: %s", ticker, exp, e)
                continue
            calls = chain.calls
            puts = chain.puts
            if calls is None or puts is None:
                continue
            atm_iv_call = self._atm_iv(calls, spot)
            atm_iv_put = self._atm_iv(puts, spot)
            atm_iv = (
                (atm_iv_call + atm_iv_put) / 2
                if atm_iv_call is not None and atm_iv_put is not None
                else (atm_iv_call or atm_iv_put)
            )
            put_vol = float(puts["volume"].fillna(0).sum()) if "volume" in puts else 0.0
            call_vol = float(calls["volume"].fillna(0).sum()) if "volume" in calls else 0.0
            put_call_ratio = (put_vol / call_vol) if call_vol > 0 else None

            # Unusual activity z-score: max(volume / open_interest)
            ua_score = self._unusual_activity_score(pd.concat([calls, puts], ignore_index=True))

            out["expiries"][exp] = {
                "atm_iv": atm_iv,
                "put_volume": put_vol,
                "call_volume": call_vol,
                "put_call_ratio": put_call_ratio,
                "unusual_activity": ua_score,
                "n_calls": len(calls),
                "n_puts": len(puts),
            }

            for df_, otype in ((calls, "call"), (puts, "put")):
                for _, r in df_.iterrows():
                    rows_to_cache.append((
                        ticker, exp, otype,
                        float(r.get("strike", 0)),
                        float(r.get("lastPrice", 0) or 0),
                        float(r.get("bid", 0) or 0),
                        float(r.get("ask", 0) or 0),
                        int(r.get("volume", 0) or 0),
                        int(r.get("openInterest", 0) or 0),
                        float(r.get("impliedVolatility", 0) or 0),
                        int(bool(r.get("inTheMoney", False))),
                    ))

        # Bulk insert with current timestamp grouped under the same fetched_at
        if rows_to_cache:
            now = datetime.now().isoformat(timespec="seconds")
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(
                    """INSERT OR IGNORE INTO options_chain
                        (ticker, expiry, option_type, strike, last_price, bid, ask,
                         volume, open_interest, implied_volatility, in_the_money, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [(*r, now) for r in rows_to_cache],
                )

        # Compute term-structure summary and write to registry
        self._write_options_aggregates(ticker, out)
        return out

    def _spot_price_safe(self, ticker: str, t=None) -> Optional[float]:
        try:
            if t is None:
                t = yf.Ticker(ticker)
            info = getattr(t, "info", {}) or {}
            return float(info.get("regularMarketPrice") or info.get("currentPrice") or 0) or None
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _atm_iv(df: pd.DataFrame, spot: Optional[float]) -> Optional[float]:
        if df is None or len(df) == 0 or spot in (None, 0):
            return None
        if "strike" not in df or "impliedVolatility" not in df:
            return None
        # Closest strike to spot
        idx = (df["strike"] - spot).abs().idxmin()
        iv = df.loc[idx, "impliedVolatility"]
        try:
            iv = float(iv)
        except (TypeError, ValueError):
            return None
        return iv if iv > 0 else None

    @staticmethod
    def _unusual_activity_score(df: pd.DataFrame) -> Optional[float]:
        """Max ratio of volume to open_interest across the chain. Above ~3
        is loosely 'unusual'."""
        if df is None or len(df) == 0:
            return None
        if "volume" not in df or "openInterest" not in df:
            return None
        oi = df["openInterest"].replace(0, np.nan)
        ratio = (df["volume"].fillna(0) / oi).replace([np.inf, -np.inf], np.nan)
        if ratio.dropna().empty:
            return None
        return float(ratio.max())

    def _write_options_aggregates(self, ticker: str, chain: dict) -> None:
        """Compute IV term structure (30/60/90d), p/c ratio, UA, and push
        into the data_registry."""
        if self.registry is None or not chain.get("expiries"):
            return

        # Find expiries closest to 30/60/90 days
        today = datetime.now().date()
        targets = {30: None, 60: None, 90: None}
        for exp_str, summary in chain["expiries"].items():
            try:
                exp_dt = datetime.strptime(exp_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            days = (exp_dt - today).days
            for tgt in list(targets.keys()):
                cur = targets[tgt]
                if cur is None or abs(days - tgt) < abs(cur[1] - tgt):
                    if summary.get("atm_iv"):
                        targets[tgt] = (summary["atm_iv"], days)

        as_of = datetime.now().isoformat(timespec="seconds")
        if targets[30]:
            self.registry.upsert_point(
                ticker, "ticker", "ticker.options.iv_30d",
                as_of=as_of, source_id="yfinance",
                value_num=float(targets[30][0]), confidence=0.85,
            )
        if targets[60]:
            self.registry.upsert_point(
                ticker, "ticker", "ticker.options.iv_60d",
                as_of=as_of, source_id="yfinance",
                value_num=float(targets[60][0]), confidence=0.85,
            )
        if targets[90]:
            self.registry.upsert_point(
                ticker, "ticker", "ticker.options.iv_90d",
                as_of=as_of, source_id="yfinance",
                value_num=float(targets[90][0]), confidence=0.85,
            )
        if targets[30] and targets[90]:
            slope = float(targets[90][0]) - float(targets[30][0])
            self.registry.upsert_point(
                ticker, "ticker", "ticker.options.iv_term_slope",
                as_of=as_of, source_id="derived",
                value_num=slope, confidence=0.85,
            )

        # Aggregate p/c ratio across expiries (volume-weighted)
        total_put = sum(e.get("put_volume", 0) for e in chain["expiries"].values())
        total_call = sum(e.get("call_volume", 0) for e in chain["expiries"].values())
        if total_call > 0:
            self.registry.upsert_point(
                ticker, "ticker", "ticker.options.put_call_ratio",
                as_of=as_of, source_id="yfinance",
                value_num=float(total_put / total_call), confidence=0.85,
            )

        # Highest UA across expiries
        ua_vals = [e.get("unusual_activity") for e in chain["expiries"].values()
                   if e.get("unusual_activity") is not None]
        if ua_vals:
            self.registry.upsert_point(
                ticker, "ticker", "ticker.options.unusual_activity",
                as_of=as_of, source_id="derived",
                value_num=float(max(ua_vals)), confidence=0.7,
            )

    # ------------------------------------------------------------------
    # Short interest
    # ------------------------------------------------------------------
    def get_short_interest(self, ticker: str) -> dict:
        """Read short interest from yfinance ``info``. FINRA's true bi-monthly
        figures would be more authoritative; this is the best free fallback.
        """
        ticker = ticker.upper()
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception as e:  # noqa: BLE001
            logger.warning("short_interest | %s | yfinance error: %s", ticker, e)
            return {}
        si = info.get("sharesShort")
        si_pct_float = info.get("shortPercentOfFloat")
        avg_vol = info.get("averageVolume10days") or info.get("averageVolume")
        days_to_cover = info.get("shortRatio")
        as_of = info.get("dateShortInterest")
        if isinstance(as_of, (int, float)):
            try:
                as_of = datetime.utcfromtimestamp(int(as_of)).strftime("%Y-%m-%d")
            except (OSError, ValueError):
                as_of = None
        if not as_of:
            as_of = datetime.now().strftime("%Y-%m-%d")

        result = {
            "ticker": ticker,
            "as_of": as_of,
            "short_interest": float(si) if si else None,
            "short_pct_float": float(si_pct_float) if si_pct_float else None,
            "days_to_cover": float(days_to_cover) if days_to_cover else None,
            "avg_volume": float(avg_vol) if avg_vol else None,
        }

        # Cache raw row
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO short_interest
                    (ticker, as_of, short_interest, short_pct_float,
                     days_to_cover, avg_volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, 'yfinance')""",
                (ticker, as_of, result["short_interest"], result["short_pct_float"],
                 result["days_to_cover"], result["avg_volume"]),
            )

        # Push to registry
        if self.registry:
            if result["short_pct_float"] is not None:
                self.registry.upsert_point(
                    ticker, "ticker", "ticker.short.interest_pct_float",
                    as_of=as_of, source_id="yfinance",
                    value_num=float(result["short_pct_float"]), confidence=0.8,
                )
            if result["days_to_cover"] is not None:
                self.registry.upsert_point(
                    ticker, "ticker", "ticker.short.days_to_cover",
                    as_of=as_of, source_id="yfinance",
                    value_num=float(result["days_to_cover"]), confidence=0.8,
                )
        return result

    # ------------------------------------------------------------------
    # Borrow data (best-effort)
    # ------------------------------------------------------------------
    def set_borrow_data(
        self,
        ticker: str,
        borrow_fee_bps: Optional[float] = None,
        shares_available: Optional[float] = None,
        source: str = "ibkr_scrape",
        as_of: Optional[str] = None,
    ) -> None:
        """Externally-supplied borrow data point. The IBKR public availability
        page (or a TWS API hook) is the typical caller. We don't try to scrape
        it ourselves here because the page format changes frequently — this
        keeps the borrow ingestion contract clean and lets the caller swap
        sources without touching this file.
        """
        ticker = ticker.upper()
        as_of = as_of or datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO borrow_data
                    (ticker, as_of, borrow_fee_bps, shares_available, source)
                   VALUES (?, ?, ?, ?, ?)""",
                (ticker, as_of, borrow_fee_bps, shares_available, source),
            )
        if self.registry:
            if borrow_fee_bps is not None:
                self.registry.upsert_point(
                    ticker, "ticker", "ticker.short.borrow_fee_bps",
                    as_of=as_of, source_id=source,
                    value_num=float(borrow_fee_bps), confidence=0.4,
                )
            if shares_available is not None:
                self.registry.upsert_point(
                    ticker, "ticker", "ticker.short.borrow_available",
                    as_of=as_of, source_id=source,
                    value_num=float(shares_available), confidence=0.4,
                )

    # ------------------------------------------------------------------
    # ETF holdings (used downstream by sector heatmap + ETF overlap)
    # ------------------------------------------------------------------
    def set_etf_holdings(
        self,
        etf_ticker: str,
        holdings: list[dict],
        as_of: Optional[str] = None,
        source: str = "issuer_csv",
    ) -> int:
        """Upsert ETF holdings rows. ``holdings`` is a list of dicts with
        keys: ``ticker``, ``weight``, optional ``shares``, ``market_value``.

        We accept holdings from any caller (issuer CSVs, yfinance funds_data,
        manual lists). This keeps the file dependency-light.
        """
        etf_ticker = etf_ticker.upper()
        as_of = as_of or datetime.now().strftime("%Y-%m-%d")
        n = 0
        with sqlite3.connect(self.db_path) as conn:
            for h in holdings:
                tk = (h.get("ticker") or "").upper()
                if not tk:
                    continue
                conn.execute(
                    """INSERT OR REPLACE INTO etf_holdings
                        (etf_ticker, holding_ticker, weight, shares, market_value,
                         as_of, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (etf_ticker, tk, h.get("weight"),
                     h.get("shares"), h.get("market_value"),
                     as_of, source),
                )
                n += 1
        return n

    def get_etf_overlap_for_ticker(self, ticker: str) -> list[dict]:
        """Return the list of ETFs that hold a given ticker, ordered by weight."""
        ticker = ticker.upper()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT etf_ticker, weight, market_value, as_of
                     FROM etf_holdings WHERE holding_ticker = ?
                  ORDER BY weight DESC NULLS LAST""",
                (ticker,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Push prices + technicals into registry
    # ------------------------------------------------------------------
    def sync_prices_to_registry(
        self, ticker: str, period: str = "6mo"
    ) -> int:
        """Pull recent prices and write the daily close + volume points
        into ``data_points``. Returns rows written."""
        ticker = ticker.upper()
        df = self.get_prices(ticker, period=period, use_cache=True)
        if df.empty or self.registry is None:
            return 0
        rows = []
        for date, r in df.iterrows():
            d = date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            for field, key in (
                ("ticker.price.close", "Close"),
                ("ticker.price.adj_close", "Close"),
                ("ticker.price.volume", "Volume"),
                ("ticker.price.high", "High"),
                ("ticker.price.low", "Low"),
                ("ticker.price.open", "Open"),
            ):
                val = r.get(key)
                if pd.isna(val) or val is None:
                    continue
                rows.append({
                    "entity_id": ticker, "entity_type": "ticker",
                    "field": field, "as_of": d, "source_id": "yfinance",
                    "value_num": float(val), "confidence": 0.9,
                })
        return self.registry.upsert_points_bulk(rows)

    def sync_technicals_to_registry(self, ticker: str) -> int:
        """Compute technicals and push the latest signal values to registry."""
        if self.registry is None:
            return 0
        tech = self.get_technicals(ticker)
        if not tech:
            return 0
        as_of = datetime.now().strftime("%Y-%m-%d")
        n = 0
        mapping = (
            ("ticker.signal.rsi_14",  tech.get("rsi_14")),
            ("ticker.signal.atr_14",  tech.get("atr_14")),
            ("ticker.signal.sma_50",  tech.get("sma_50")),
            ("ticker.signal.sma_200", tech.get("sma_200")),
            ("ticker.signal.macd_hist", tech.get("macd", {}).get("histogram")),
        )
        for field, value in mapping:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            self.registry.upsert_point(
                ticker.upper(), "ticker", field,
                as_of=as_of, source_id="derived",
                value_num=float(value), confidence=1.0,
            )
            n += 1

        # Regime label
        sig = tech.get("signal", {}).get("overall")
        if sig:
            regime = ("BULL" if "BULLISH" in sig
                      else "BEAR" if "BEARISH" in sig
                      else "CHOP")
            self.registry.upsert_point(
                ticker.upper(), "ticker", "ticker.signal.regime",
                as_of=as_of, source_id="derived",
                value_text=regime, confidence=1.0,
            )
            n += 1
        return n

    def refresh_ticker_market(self, ticker: str) -> dict[str, int]:
        """Run all extended market fetches for one ticker."""
        out: dict[str, int] = {}
        try:
            out["prices"] = self.sync_prices_to_registry(ticker, period="6mo")
        except Exception as e:  # noqa: BLE001
            logger.warning("market refresh | %s | prices failed: %s", ticker, e)
            out["prices"] = 0
        try:
            self.get_options_chain(ticker)
            out["options"] = 1
        except Exception as e:  # noqa: BLE001
            logger.warning("market refresh | %s | options failed: %s", ticker, e)
            out["options"] = 0
        try:
            self.get_short_interest(ticker)
            out["short"] = 1
        except Exception as e:  # noqa: BLE001
            logger.warning("market refresh | %s | short failed: %s", ticker, e)
            out["short"] = 0
        try:
            out["technicals"] = self.sync_technicals_to_registry(ticker)
        except Exception as e:  # noqa: BLE001
            logger.warning("market refresh | %s | technicals failed: %s", ticker, e)
            out["technicals"] = 0
        return out


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
