"""
Macro Data Module
=================
Pulls macroeconomic data from the Federal Reserve Economic Data (FRED) API.
Covers rates, inflation, employment, GDP, yield curves, and financial conditions.

Get a free API key at: https://fred.stlouisfed.org/docs/api/api_key.html
Set environment variable: FRED_API_KEY=your_key_here

The FRED API is free and generous — 120 requests per minute,
no daily limit. Has 800,000+ data series going back decades.

Fun fact: FRED is maintained by the St. Louis Fed and is the single
most-used source of economic data by quant funds. Renaissance
Technologies reportedly pulls from FRED as one of their data inputs.

Usage:
    macro = MacroData(api_key="your_key")
    dashboard = macro.get_macro_dashboard()
    yield_curve = macro.get_yield_curve()
    recession_prob = macro.get_recession_probability()
"""

import os
import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred"

# ---------------------------------------------------------------------------
# Key FRED series IDs — these are the economic indicators that matter
# for equity markets and hedge fund macro analysis
# ---------------------------------------------------------------------------
SERIES = {
    # Interest rates
    "fed_funds_rate": "FEDFUNDS",          # Federal Funds Effective Rate
    "treasury_10y": "DGS10",              # 10-Year Treasury Yield
    "treasury_2y": "DGS2",               # 2-Year Treasury Yield
    "treasury_3m": "DTB3",               # 3-Month Treasury Bill
    "treasury_30y": "DGS30",             # 30-Year Treasury Yield

    # Yield curve
    "yield_spread_10y2y": "T10Y2Y",       # 10Y-2Y spread (inversion = recession signal)
    "yield_spread_10y3m": "T10Y3M",       # 10Y-3M spread

    # Inflation
    "cpi_yoy": "CPIAUCSL",               # CPI (need to compute YoY)
    "core_cpi_yoy": "CPILFESL",           # Core CPI (ex food/energy)
    "pce_price_index": "PCEPI",           # PCE Price Index (Fed's preferred)
    "breakeven_5y": "T5YIE",             # 5-Year Breakeven Inflation
    "breakeven_10y": "T10YIE",           # 10-Year Breakeven Inflation

    # Employment
    "unemployment_rate": "UNRATE",         # Unemployment Rate
    "nonfarm_payrolls": "PAYEMS",         # Total Nonfarm Payrolls
    "initial_claims": "ICSA",            # Initial Jobless Claims (weekly)
    "continuing_claims": "CCSA",          # Continuing Jobless Claims

    # GDP & Output
    "real_gdp": "GDPC1",                 # Real GDP
    "gdp_growth": "A191RL1Q225SBEA",     # Real GDP Growth Rate (annualized)
    "industrial_production": "INDPRO",    # Industrial Production Index

    # Consumer & Business
    "consumer_sentiment": "UMCSENT",      # U of Michigan Consumer Sentiment
    "retail_sales": "RSXFS",             # Retail Sales (ex food services)
    "housing_starts": "HOUST",           # Housing Starts

    # Financial conditions
    "vix": "VIXCLS",                     # VIX (CBOE Volatility Index)
    "financial_stress": "STLFSI2",        # St. Louis Financial Stress Index
    "corporate_spread_baa": "BAA10Y",     # Baa Corporate Bond Spread
    "sp500": "SP500",                    # S&P 500

    # Money supply
    "m2": "M2SL",                        # M2 Money Supply

    # Recession indicator
    "recession_prob": "RECPROUSM156N",    # Smoothed Recession Probabilities
    "sahm_rule": "SAHMREALTIME",         # Sahm Rule Recession Indicator
}


class MacroData:
    """FRED macroeconomic data with caching and derived indicators."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        db_path: str = "data/hedgefund.db",
        registry=None,
    ):
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "No FRED API key set. Get one free at "
                "https://fred.stlouisfed.org/docs/api/api_key.html "
                "and set FRED_API_KEY environment variable."
            )
        self.db_path = db_path
        self._registry = registry
        self._init_db()
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
                logger.warning("macro_data | could not load data_registry: %s", e)
        return self._registry

    def _register_sources(self) -> None:
        reg = self.registry
        if reg is None:
            return
        try:
            reg.register_source("fred", "macro", "daily", base_priority=1,
                                notes="St. Louis Fed FRED API")
            reg.register_source("cboe", "macro", "daily", base_priority=2,
                                notes="CBOE indices via yfinance proxy")
        except Exception as e:  # noqa: BLE001
            logger.debug("macro_data | source registration skipped: %s", e)

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_data (
                    series_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    value REAL,
                    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
                    PRIMARY KEY (series_id, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_cache_meta (
                    series_id TEXT PRIMARY KEY,
                    last_fetched TEXT NOT NULL
                )
            """)

    # ------------------------------------------------------------------
    # Core data fetching
    # ------------------------------------------------------------------
    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None,
        use_cache: bool = True,
        cache_hours: int = 6,
    ) -> pd.DataFrame:
        """
        Fetch a single FRED data series.

        Args:
            series_id:   FRED series ID (e.g., "FEDFUNDS", "UNRATE")
            start_date:  YYYY-MM-DD
            end_date:    YYYY-MM-DD
            frequency:   d=daily, w=weekly, m=monthly, q=quarterly, a=annual
            use_cache:   Check SQLite cache first
            cache_hours: How long cache is valid

        Returns:
            DataFrame with DatetimeIndex and 'value' column
        """
        if not self.api_key:
            raise ValueError("FRED API key required. Set FRED_API_KEY env var.")

        # Check cache
        if use_cache:
            cached = self._get_cached_series(series_id, cache_hours)
            if cached is not None and len(cached) > 0:
                df = cached
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                if len(df) > 0:
                    logger.info(f"Returning cached {series_id}: {len(df)} obs")
                    return df

        # Fetch from FRED API
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
        }
        if start_date:
            params["observation_start"] = start_date
        if end_date:
            params["observation_end"] = end_date
        if frequency:
            params["frequency"] = frequency

        url = f"{FRED_BASE}/series/observations"
        logger.info(f"Fetching FRED series: {series_id}")

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        observations = data.get("observations", [])
        if not observations:
            logger.warning(f"No observations for {series_id}")
            return pd.DataFrame(columns=["value"])

        rows = []
        for obs in observations:
            val = obs["value"]
            if val == ".":  # FRED uses "." for missing
                continue
            rows.append({"date": obs["date"], "value": float(val)})

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)

        # Cache
        self._cache_series(series_id, df)
        return df

    def get_series_latest(self, series_id: str) -> Optional[float]:
        """Get the most recent observation for a series."""
        df = self.get_series(series_id)
        if df.empty:
            return None
        return float(df["value"].iloc[-1])

    # ------------------------------------------------------------------
    # Derived / composite indicators
    # ------------------------------------------------------------------
    def get_macro_dashboard(self) -> dict:
        """
        One-call macro snapshot — the economic indicators that drive
        equity markets and hedge fund positioning.

        Returns a dict ready to feed to the LLM agent for macro analysis.
        """
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "interest_rates": {},
            "inflation": {},
            "employment": {},
            "growth": {},
            "financial_conditions": {},
            "recession_signals": {},
        }

        # --- Interest Rates ---
        rate_series = {
            "fed_funds": "FEDFUNDS",
            "treasury_2y": "DGS2",
            "treasury_10y": "DGS10",
            "treasury_30y": "DGS30",
        }
        for name, sid in rate_series.items():
            try:
                val = self.get_series_latest(sid)
                dashboard["interest_rates"][name] = val
            except Exception as e:
                logger.warning(f"Could not fetch {name}: {e}")

        # --- Yield Curve ---
        try:
            spread = self.get_series_latest("T10Y2Y")
            dashboard["interest_rates"]["yield_spread_10y2y"] = spread
            dashboard["interest_rates"]["yield_curve_inverted"] = (
                spread is not None and spread < 0
            )
        except Exception:
            pass

        # --- Inflation ---
        try:
            # Get CPI and compute YoY change
            cpi = self.get_series("CPIAUCSL")
            if not cpi.empty and len(cpi) > 12:
                yoy = (cpi["value"].iloc[-1] / cpi["value"].iloc[-13] - 1) * 100
                dashboard["inflation"]["cpi_yoy_pct"] = round(yoy, 2)

            be5 = self.get_series_latest("T5YIE")
            dashboard["inflation"]["breakeven_5y"] = be5
        except Exception as e:
            logger.warning(f"Inflation data error: {e}")

        # --- Employment ---
        try:
            dashboard["employment"]["unemployment_rate"] = self.get_series_latest("UNRATE")
            dashboard["employment"]["initial_claims"] = self.get_series_latest("ICSA")
        except Exception as e:
            logger.warning(f"Employment data error: {e}")

        # --- Growth ---
        try:
            dashboard["growth"]["gdp_growth_annualized"] = self.get_series_latest(
                "A191RL1Q225SBEA"
            )
            dashboard["growth"]["consumer_sentiment"] = self.get_series_latest("UMCSENT")
        except Exception as e:
            logger.warning(f"Growth data error: {e}")

        # --- Financial Conditions ---
        try:
            dashboard["financial_conditions"]["vix"] = self.get_series_latest("VIXCLS")
            dashboard["financial_conditions"]["financial_stress_index"] = (
                self.get_series_latest("STLFSI2")
            )
            dashboard["financial_conditions"]["baa_corporate_spread"] = (
                self.get_series_latest("BAA10Y")
            )
        except Exception as e:
            logger.warning(f"Financial conditions error: {e}")

        # --- Recession Signals ---
        try:
            dashboard["recession_signals"]["recession_probability"] = (
                self.get_series_latest("RECPROUSM156N")
            )
            dashboard["recession_signals"]["sahm_rule"] = self.get_series_latest(
                "SAHMREALTIME"
            )
            # Sahm Rule: >= 0.5 historically signals recession
            sahm = dashboard["recession_signals"]["sahm_rule"]
            if sahm is not None:
                dashboard["recession_signals"]["sahm_triggered"] = sahm >= 0.50
        except Exception as e:
            logger.warning(f"Recession signals error: {e}")

        return dashboard

    def get_yield_curve(self) -> dict:
        """
        Build the current Treasury yield curve.

        The yield curve is arguably the single most important macro
        indicator for equity markets. An inverted curve (short rates
        above long rates) has preceded every US recession since 1970.

        YouTube deep-dive: "Yield Curve Inversion Explained"
        https://www.youtube.com/results?search_query=yield+curve+inversion+explained

        Research: "The Yield Curve as a Predictor of U.S. Recessions"
        by Arturo Estrella & Frederic Mishkin (Federal Reserve Bank of NY)
        """
        maturities = {
            "3m": "DTB3",
            "6m": "DTB6",
            "1y": "DGS1",
            "2y": "DGS2",
            "3y": "DGS3",
            "5y": "DGS5",
            "7y": "DGS7",
            "10y": "DGS10",
            "20y": "DGS20",
            "30y": "DGS30",
        }

        curve = {}
        for label, series_id in maturities.items():
            try:
                val = self.get_series_latest(series_id)
                if val is not None:
                    curve[label] = val
            except Exception:
                pass

        # Analyze shape
        vals = list(curve.values())
        if len(vals) >= 2:
            is_inverted = vals[0] > vals[-1]
            spread_3m_10y = curve.get("10y", 0) - curve.get("3m", 0)
            spread_2y_10y = curve.get("10y", 0) - curve.get("2y", 0)
        else:
            is_inverted = None
            spread_3m_10y = None
            spread_2y_10y = None

        return {
            "curve": curve,
            "is_inverted": is_inverted,
            "spread_3m_10y": round(spread_3m_10y, 3) if spread_3m_10y else None,
            "spread_2y_10y": round(spread_2y_10y, 3) if spread_2y_10y else None,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }

    def get_recession_probability(self) -> dict:
        """
        Multi-signal recession probability assessment.

        Combines:
        1. FRED's smoothed recession probability model
        2. Sahm Rule (unemployment trigger)
        3. Yield curve inversion duration
        4. Financial stress index

        This is what your LLM agent should reference when generating
        macro risk assessments for the portfolio.
        """
        signals = {}

        # 1. FRED recession probability model
        try:
            prob = self.get_series_latest("RECPROUSM156N")
            signals["fred_model_probability"] = prob
        except Exception:
            signals["fred_model_probability"] = None

        # 2. Sahm Rule
        try:
            sahm = self.get_series_latest("SAHMREALTIME")
            signals["sahm_rule_value"] = sahm
            signals["sahm_triggered"] = sahm >= 0.50 if sahm else None
        except Exception:
            signals["sahm_rule_value"] = None

        # 3. Yield curve
        try:
            spread = self.get_series("T10Y2Y")
            if not spread.empty:
                current_spread = float(spread["value"].iloc[-1])
                signals["yield_spread_10y2y"] = current_spread
                signals["yield_curve_inverted"] = current_spread < 0

                # How long has it been inverted (if it is)?
                inverted_mask = spread["value"] < 0
                if inverted_mask.iloc[-1]:
                    # Find when inversion started
                    last_positive = spread[~inverted_mask].index[-1]
                    days_inverted = (spread.index[-1] - last_positive).days
                    signals["days_inverted"] = days_inverted
                else:
                    signals["days_inverted"] = 0
        except Exception:
            pass

        # 4. Financial stress
        try:
            stress = self.get_series_latest("STLFSI2")
            signals["financial_stress_index"] = stress
            # Above 0 = above-average stress, >1 = notable stress
            signals["elevated_stress"] = stress > 1.0 if stress else None
        except Exception:
            pass

        # Composite assessment
        red_flags = sum([
            signals.get("fred_model_probability", 0) or 0 > 30,
            signals.get("sahm_triggered", False),
            signals.get("yield_curve_inverted", False),
            signals.get("elevated_stress", False),
        ])

        if red_flags >= 3:
            signals["composite_assessment"] = "HIGH_RISK"
        elif red_flags >= 2:
            signals["composite_assessment"] = "ELEVATED_RISK"
        elif red_flags >= 1:
            signals["composite_assessment"] = "MODERATE_RISK"
        else:
            signals["composite_assessment"] = "LOW_RISK"

        signals["red_flags_count"] = red_flags

        return signals

    # ------------------------------------------------------------------
    # Historical comparisons
    # ------------------------------------------------------------------
    def get_rate_cycle_context(self) -> dict:
        """
        Where are we in the Fed rate cycle?

        Returns current rate, direction (hiking/cutting/paused),
        and comparison to historical levels.
        """
        try:
            ff = self.get_series("FEDFUNDS", start_date="2020-01-01")
            if ff.empty:
                return {}

            current = float(ff["value"].iloc[-1])
            prev_month = float(ff["value"].iloc[-2]) if len(ff) > 1 else current
            peak = float(ff["value"].max())
            trough = float(ff["value"].min())

            # Determine direction
            if current > prev_month:
                direction = "HIKING"
            elif current < prev_month:
                direction = "CUTTING"
            else:
                direction = "PAUSED"

            return {
                "current_rate": current,
                "direction": direction,
                "cycle_peak": peak,
                "cycle_trough": trough,
                "distance_from_peak": round(current - peak, 2),
                "since_2020_data_points": len(ff),
            }
        except Exception as e:
            logger.error(f"Rate cycle error: {e}")
            return {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _get_cached_series(
        self, series_id: str, cache_hours: int
    ) -> Optional[pd.DataFrame]:
        """Get cached series if fresh enough."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT last_fetched FROM macro_cache_meta WHERE series_id = ?",
                (series_id,),
            ).fetchone()

            if not row:
                return None

            fetched = datetime.fromisoformat(row[0])
            if (datetime.now() - fetched).total_seconds() > cache_hours * 3600:
                return None

            df = pd.read_sql_query(
                "SELECT date, value FROM macro_data WHERE series_id = ? ORDER BY date",
                conn,
                params=[series_id],
            )

        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    def _cache_series(self, series_id: str, df: pd.DataFrame):
        """Cache a FRED series to SQLite."""
        if df.empty:
            return

        with sqlite3.connect(self.db_path) as conn:
            for date, row in df.iterrows():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO macro_data (series_id, date, value)
                    VALUES (?, ?, ?)
                    """,
                    (series_id, date.strftime("%Y-%m-%d"), float(row["value"])),
                )
            conn.execute(
                """
                INSERT OR REPLACE INTO macro_cache_meta (series_id, last_fetched)
                VALUES (?, ?)
                """,
                (series_id, datetime.now().isoformat()),
            )
    # ==================================================================
    # === EXTENDED API: VIX term structure, credit spreads, sentiment ===
    # ==================================================================

    EXTENDED_FRED_MAP: dict[str, tuple[str, float]] = {
        "VIXCLS":              ("global.vix",                 1.0),
        "VXVCLS":              ("global.vix_3m",              1.0),
        "BAMLH0A0HYM2":        ("global.hy_oas",              1.0),
        "BAMLC0A0CM":          ("global.ig_oas",              1.0),
        "RECPROUSM156N":       ("global.recession_prob",      1.0),
        "UMCSENT":             ("global.consumer_sentiment",  1.0),
        "STLFSI4":             ("global.financial_stress",    1.0),
        "T10Y2Y":              ("global.yield_curve_2y10y",   1.0),
    }

    def get_vix_term_structure(self) -> dict:
        """Fetch VIX, VIX3M and compute term-structure slope. Ratio < 1
        signals stress (backwardation).
        """
        out = {"as_of": datetime.now().strftime("%Y-%m-%d"),
               "vix": None, "vix_3m": None, "ratio_3m_1m": None}
        try:
            vix = self.get_series_latest("VIXCLS")
            vix3m = self.get_series_latest("VXVCLS")
        except Exception as e:  # noqa: BLE001
            logger.warning("macro vix | failed: %s", e)
            return out
        out["vix"] = vix
        out["vix_3m"] = vix3m
        if vix and vix3m and vix > 0:
            out["ratio_3m_1m"] = float(vix3m) / float(vix)
        if self.registry:
            today = out["as_of"]
            if vix is not None:
                self.registry.upsert_point(
                    "global", "global", "global.vix",
                    as_of=today, source_id="fred",
                    value_num=float(vix), confidence=1.0,
                )
            if vix3m is not None:
                self.registry.upsert_point(
                    "global", "global", "global.vix_3m",
                    as_of=today, source_id="fred",
                    value_num=float(vix3m), confidence=1.0,
                )
            if out["ratio_3m_1m"] is not None:
                self.registry.upsert_point(
                    "global", "global", "global.vix_term_3m_1m",
                    as_of=today, source_id="derived",
                    value_num=float(out["ratio_3m_1m"]), confidence=1.0,
                )
        return out

    def get_credit_spreads(self) -> dict:
        """Fetch HY OAS and IG OAS (used as CDS proxy in stress scoring)."""
        out = {"as_of": datetime.now().strftime("%Y-%m-%d")}
        for fred_id, (field, conf) in [
            ("BAMLH0A0HYM2", ("global.hy_oas", 1.0)),
            ("BAMLC0A0CM",   ("global.ig_oas", 1.0)),
        ]:
            try:
                val = self.get_series_latest(fred_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("macro credit | %s failed: %s", fred_id, e)
                continue
            out[field] = val
            if self.registry and val is not None:
                self.registry.upsert_point(
                    "global", "global", field,
                    as_of=out["as_of"], source_id="fred",
                    value_num=float(val), confidence=conf,
                )
        return out

    def get_consumer_sentiment(self) -> dict:
        """U-Mich sentiment + financial stress index (with FRED series fallbacks)."""
        out = {"as_of": datetime.now().strftime("%Y-%m-%d")}
        try:
            cs = self.get_series_latest("UMCSENT")
            out["consumer_sentiment"] = cs
            if self.registry and cs is not None:
                self.registry.upsert_point(
                    "global", "global", "global.consumer_sentiment",
                    as_of=out["as_of"], source_id="fred",
                    value_num=float(cs), confidence=1.0,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("macro sentiment | UMCSENT failed: %s", e)
        for sid in ("STLFSI4", "STLFSI3", "STLFSI2"):
            try:
                fs = self.get_series_latest(sid)
            except Exception:  # noqa: BLE001
                continue
            if fs is not None:
                out["financial_stress"] = fs
                if self.registry:
                    self.registry.upsert_point(
                        "global", "global", "global.financial_stress",
                        as_of=out["as_of"], source_id="fred",
                        value_num=float(fs), confidence=1.0,
                    )
                break
        return out

    def sync_macro_to_registry(self) -> int:
        """Walk EXTENDED_FRED_MAP and push every series to the registry."""
        if self.registry is None:
            return 0
        today = datetime.now().strftime("%Y-%m-%d")
        n = 0
        for fred_id, (field, conf) in self.EXTENDED_FRED_MAP.items():
            try:
                val = self.get_series_latest(fred_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("sync_macro | %s failed: %s", fred_id, e)
                continue
            if val is None:
                continue
            self.registry.upsert_point(
                "global", "global", field,
                as_of=today, source_id="fred",
                value_num=float(val), confidence=conf,
            )
            n += 1
        try:
            self.get_vix_term_structure()
        except Exception:
            pass
        return n

    def refresh_macro(self) -> dict[str, int]:
        """High-level orchestrator. Returns rows written per section."""
        out: dict[str, int] = {}
        try:
            out["fred_canonical"] = self.sync_macro_to_registry()
        except Exception as e:  # noqa: BLE001
            logger.warning("refresh_macro | fred sync failed: %s", e)
            out["fred_canonical"] = 0
        try:
            self.get_credit_spreads(); out["credit"] = 1
        except Exception as e:  # noqa: BLE001
            logger.warning("refresh_macro | credit failed: %s", e); out["credit"] = 0
        try:
            self.get_consumer_sentiment(); out["sentiment"] = 1
        except Exception as e:  # noqa: BLE001
            logger.warning("refresh_macro | sentiment failed: %s", e); out["sentiment"] = 0
        return out


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("Set FRED_API_KEY environment variable first!")
        print("Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        exit(1)

    macro = MacroData(api_key=api_key)

    print("=== Macro Data Test ===\n")

    # Dashboard
    print("Fetching macro dashboard (this makes ~15 API calls, takes ~10s)...")
    dash = macro.get_macro_dashboard()

    print("\nInterest Rates:")
    for k, v in dash["interest_rates"].items():
        print(f"  {k}: {v}")

    print("\nInflation:")
    for k, v in dash["inflation"].items():
        print(f"  {k}: {v}")

    print("\nEmployment:")
    for k, v in dash["employment"].items():
        print(f"  {k}: {v}")

    print("\nFinancial Conditions:")
    for k, v in dash["financial_conditions"].items():
        print(f"  {k}: {v}")

    print("\nRecession Signals:")
    for k, v in dash["recession_signals"].items():
        print(f"  {k}: {v}")

    # Yield curve
    print("\n--- Yield Curve ---")
    yc = macro.get_yield_curve()
    for mat, rate in yc["curve"].items():
        print(f"  {mat}: {rate}%")
    print(f"  Inverted: {yc['is_inverted']}")
    print(f"  10Y-2Y Spread: {yc['spread_2y_10y']}%")
