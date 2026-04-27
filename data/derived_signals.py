"""
derived_signals.py
==================
Pure-Python computation layer. Reads from the SQLite DB (raw tables +
``data_points``), computes derived signals, writes back into the registry.

Strict rules
------------
* No external network I/O. If you need a new market data point, fetch it
  in ``market_data.py`` first.
* No side effects beyond writes to the local DB.
* Deterministic: same inputs in the DB -> same numbers out. Tests can
  construct synthetic price tables and assert exact values.
* Optional dependencies (statsmodels, scipy) are guarded behind try/except
  so the module imports without them.

What it computes
----------------
1. Technical regime signals: 30-day realized vol, 252-day drawdown
2. Relative strength vs sector ETFs: 20/60/120-day return spread
3. Sector heatmap: 5/20/60-day returns for the 11 SPDR sector ETFs
4. Factor exposures (Fama-French 5 + momentum) when an FF dataset has
   been pre-loaded into ``factor_returns``; else exits gracefully.
5. Composite risk scores combining macro stress, sanctions pressure,
   commodity sensitivity, energy crisis, supply chain.

Public entry points
-------------------
    >>> from data.derived_signals import DerivedSignals
    >>> ds = DerivedSignals()
    >>> ds.recompute_for("AAPL")
    >>> ds.recompute_sector_heatmap()
    >>> ds.recompute_global_risk_pulse()
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default sector ETFs (SPDR) used for relative-strength comparisons.
SECTOR_ETFS: dict[str, str] = {
    "XLK":  "Technology",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLE":  "Energy",
    "XLP":  "Consumer Staples",
    "XLY":  "Consumer Discretionary",
    "XLU":  "Utilities",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLC":  "Communication Services",
}


# ----------------------------------------------------------------------
# Class
# ----------------------------------------------------------------------


class DerivedSignals:
    """Pure-Python compute layer over the SQLite DB."""

    def __init__(self, db_path: str = "data/hedgefund.db", registry=None):
        self.db_path = db_path
        self._registry = registry
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @property
    def registry(self):
        if self._registry is None:
            try:
                from data.data_registry import get_default_registry
                self._registry = get_default_registry(self.db_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("derived_signals | registry load failed: %s", e)
        return self._registry

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS technical_signals (
                    ticker TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    rsi_14 REAL, atr_14 REAL,
                    sma_50 REAL, sma_200 REAL,
                    realized_vol_30d REAL, drawdown_252d REAL,
                    PRIMARY KEY (ticker, as_of)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_exposures (
                    ticker TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    beta_market REAL, beta_smb REAL, beta_hml REAL,
                    beta_mom REAL, beta_qmj REAL,
                    PRIMARY KEY (ticker, as_of)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_scores (
                    ticker TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    macro_stress REAL,
                    supply_chain REAL,
                    sanctions_pressure REAL,
                    commodity_sensitivity REAL,
                    energy_crisis REAL,
                    PRIMARY KEY (ticker, as_of)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relative_strength (
                    ticker TEXT NOT NULL,
                    sector_etf TEXT NOT NULL,
                    as_of TEXT NOT NULL,
                    spread_20d REAL, spread_60d REAL, spread_120d REAL,
                    PRIMARY KEY (ticker, sector_etf, as_of)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS factor_returns (
                    factor TEXT NOT NULL,
                    date TEXT NOT NULL,
                    value REAL,
                    PRIMARY KEY (factor, date)
                )
            """)

    # ------------------------------------------------------------------
    # Pure helpers (also used by tests)
    # ------------------------------------------------------------------
    @staticmethod
    def realized_vol(returns: pd.Series, window: int = 30, annualize: bool = True) -> Optional[float]:
        """Std dev of returns over a window. Annualized by sqrt(252) by default."""
        if returns is None or len(returns) < window:
            return None
        s = returns.dropna().tail(window)
        if len(s) < window:
            return None
        v = float(s.std(ddof=0))
        if annualize:
            v *= math.sqrt(252)
        return v

    @staticmethod
    def drawdown_from_high(close: pd.Series, window: int = 252) -> Optional[float]:
        """Current drawdown from the rolling-window peak."""
        if close is None or len(close) == 0:
            return None
        s = close.dropna().tail(window)
        if len(s) < 2:
            return None
        peak = s.max()
        if peak <= 0:
            return None
        return float((s.iloc[-1] - peak) / peak)

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> Optional[float]:
        if close is None or len(close) < period + 1:
            return None
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        last = (100 - (100 / (1 + rs))).iloc[-1]
        return None if pd.isna(last) else float(last)

    @staticmethod
    def factor_betas(stock_returns: pd.Series, factor_returns: pd.DataFrame) -> dict[str, float]:
        """OLS regression of stock returns on a DataFrame of factor returns.
        Uses numpy lstsq -- no scipy dependency. Returns {factor_name: beta}.
        """
        if stock_returns is None or len(stock_returns) < 30:
            return {}
        # Align indices
        df = pd.concat([stock_returns.rename("y"), factor_returns], axis=1).dropna()
        if len(df) < 30:
            return {}
        y = df["y"].values
        X_cols = [c for c in df.columns if c != "y"]
        X = df[X_cols].values
        # Add intercept column
        X_aug = np.column_stack([np.ones(len(X)), X])
        try:
            beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        except np.linalg.LinAlgError:
            return {}
        # Skip the intercept (beta[0])
        return {f"beta_{X_cols[i]}": float(beta[i + 1]) for i in range(len(X_cols))}

    # ------------------------------------------------------------------
    # Read prices from SQLite
    # ------------------------------------------------------------------
    def _load_prices(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """Load price history for ``ticker`` from the ``price_history`` table."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """SELECT date, close FROM price_history
                    WHERE ticker = ? AND date >= ? ORDER BY date""",
                conn, params=[ticker.upper(), cutoff],
            )
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Per-ticker recompute
    # ------------------------------------------------------------------
    def recompute_for(self, ticker: str) -> dict[str, float]:
        """Recompute all per-ticker derived signals. Returns the values written."""
        ticker = ticker.upper()
        prices = self._load_prices(ticker, days=400)
        if prices.empty:
            logger.info("derived | %s | no price history available", ticker)
            return {}
        close = prices["close"]
        rets = close.pct_change()

        rv30 = self.realized_vol(rets, window=30)
        dd = self.drawdown_from_high(close, window=252)
        rsi14 = self.rsi(close, 14)
        sma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
        sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None

        as_of = close.index[-1].strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO technical_signals
                    (ticker, as_of, rsi_14, atr_14, sma_50, sma_200,
                     realized_vol_30d, drawdown_252d)
                    VALUES (?, ?, ?, NULL, ?, ?, ?, ?)""",
                (ticker, as_of, rsi14, sma50, sma200, rv30, dd),
            )

        # Push canonical fields to the registry
        if self.registry:
            for field, val in (
                ("ticker.signal.rsi_14", rsi14),
                ("ticker.signal.sma_50", sma50),
                ("ticker.signal.sma_200", sma200),
                ("ticker.signal.realized_vol_30d", rv30),
                ("ticker.signal.drawdown", dd),
            ):
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                self.registry.upsert_point(
                    ticker, "ticker", field,
                    as_of=as_of, source_id="derived",
                    value_num=float(val), confidence=1.0,
                )
            # Regime
            regime = self._classify_regime(close.iloc[-1], sma50, sma200, dd)
            if regime:
                self.registry.upsert_point(
                    ticker, "ticker", "ticker.signal.regime",
                    as_of=as_of, source_id="derived",
                    value_text=regime, confidence=1.0,
                )

        # Relative strength vs sector ETFs
        rs_results = self.recompute_relative_strength(ticker)

        return {
            "rsi_14": rsi14, "sma_50": sma50, "sma_200": sma200,
            "realized_vol_30d": rv30, "drawdown_252d": dd,
            "rs_pairs": len(rs_results),
        }

    @staticmethod
    def _classify_regime(price, sma50, sma200, drawdown) -> Optional[str]:
        """Lightweight rule-based regime label.

        BULL : price > sma50 > sma200 AND drawdown > -0.10
        BEAR : price < sma50 < sma200 OR  drawdown < -0.20
        else : CHOP
        """
        if price is None:
            return None
        if (sma50 and sma200 and price > sma50 > sma200
                and (drawdown is None or drawdown > -0.10)):
            return "BULL"
        if (sma50 and sma200 and price < sma50 < sma200) or (drawdown is not None and drawdown < -0.20):
            return "BEAR"
        return "CHOP"

    # ------------------------------------------------------------------
    # Relative strength vs sector ETFs
    # ------------------------------------------------------------------
    def recompute_relative_strength(self, ticker: str) -> list[dict]:
        """Compute return spread vs each available sector ETF over 20/60/120d.

        We compare against every sector ETF that has price data in
        ``price_history``. Caller can later select the canonical sector
        from fundamentals and pick the relevant pair.
        """
        ticker = ticker.upper()
        tk_prices = self._load_prices(ticker, days=400)
        if tk_prices.empty:
            return []
        results = []
        as_of = tk_prices.index[-1].strftime("%Y-%m-%d")
        for etf in SECTOR_ETFS:
            etf_prices = self._load_prices(etf, days=400)
            if etf_prices.empty:
                continue
            spreads = {}
            for window in (20, 60, 120):
                tk_ret = self._return_over(tk_prices["close"], window)
                etf_ret = self._return_over(etf_prices["close"], window)
                if tk_ret is None or etf_ret is None:
                    spreads[window] = None
                else:
                    spreads[window] = float(tk_ret - etf_ret)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO relative_strength
                        (ticker, sector_etf, as_of, spread_20d, spread_60d, spread_120d)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                    (ticker, etf, as_of, spreads[20], spreads[60], spreads[120]),
                )
            if self.registry and spreads[60] is not None:
                # Use 60-day as the canonical RS-vs-sector field
                self.registry.upsert_point(
                    ticker, "ticker", "ticker.signal.rs_vs_sector_60d",
                    as_of=as_of, source_id="derived",
                    value_num=spreads[60], confidence=0.9,
                )
            results.append({"sector_etf": etf, **spreads})
        return results

    @staticmethod
    def _return_over(close: pd.Series, days: int) -> Optional[float]:
        if close is None or len(close) <= days:
            return None
        return float(close.iloc[-1] / close.iloc[-days - 1] - 1)

    # ------------------------------------------------------------------
    # Sector heatmap (no ticker — refreshes the SECTOR_ETFS dataset)
    # ------------------------------------------------------------------
    def recompute_sector_heatmap(self) -> int:
        """Compute and write 5/20/60-day returns for each SPDR sector ETF."""
        if self.registry is None:
            return 0
        n = 0
        as_of = datetime.now().strftime("%Y-%m-%d")
        for etf in SECTOR_ETFS:
            etf_prices = self._load_prices(etf, days=120)
            if etf_prices.empty:
                continue
            close = etf_prices["close"]
            for d, field in ((5, "sector.return_5d"),
                             (20, "sector.return_20d"),
                             (60, "sector.return_60d")):
                ret = self._return_over(close, d)
                if ret is None:
                    continue
                self.registry.upsert_point(
                    etf, "sector", field,
                    as_of=as_of, source_id="derived",
                    value_num=float(ret), confidence=1.0,
                )
                n += 1
        return n

    # ------------------------------------------------------------------
    # Factor exposures
    # ------------------------------------------------------------------
    def recompute_factor_exposures(self, ticker: str, lookback_days: int = 180) -> dict:
        """OLS regression of stock returns on factor returns previously
        loaded into ``factor_returns``. Caller is responsible for keeping
        that table populated (e.g. from Ken French's data library).
        """
        ticker = ticker.upper()
        # Load factor returns
        with sqlite3.connect(self.db_path) as conn:
            fdf = pd.read_sql_query(
                "SELECT factor, date, value FROM factor_returns ORDER BY date",
                conn,
            )
        if fdf.empty:
            return {}
        fdf["date"] = pd.to_datetime(fdf["date"])
        factor_wide = fdf.pivot(index="date", columns="factor", values="value")
        # Align stock returns
        prices = self._load_prices(ticker, days=lookback_days * 2)
        if prices.empty:
            return {}
        rets = prices["close"].pct_change()
        rets.index = pd.to_datetime(rets.index)
        common = rets.index.intersection(factor_wide.index)
        if len(common) < 30:
            return {}
        rets = rets.loc[common]
        factor_wide = factor_wide.loc[common]
        betas = self.factor_betas(rets, factor_wide)
        if not betas:
            return {}
        as_of = common[-1].strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO factor_exposures
                    (ticker, as_of, beta_market, beta_smb, beta_hml, beta_mom, beta_qmj)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (ticker, as_of,
                 betas.get("beta_market") or betas.get("beta_mkt") or betas.get("beta_Mkt-RF"),
                 betas.get("beta_smb") or betas.get("beta_SMB"),
                 betas.get("beta_hml") or betas.get("beta_HML"),
                 betas.get("beta_mom") or betas.get("beta_MOM") or betas.get("beta_UMD"),
                 betas.get("beta_qmj") or betas.get("beta_QMJ")),
            )
        if self.registry:
            mapping = {
                "beta_market": "ticker.factor.beta_market",
                "beta_smb":    "ticker.factor.beta_smb",
                "beta_hml":    "ticker.factor.beta_hml",
                "beta_mom":    "ticker.factor.beta_mom",
                "beta_qmj":    "ticker.factor.beta_qmj",
            }
            for k, field in mapping.items():
                # Try both name conventions
                v = (betas.get(k) or betas.get(f"beta_{k.split('_')[1].upper()}")
                     or betas.get(f"beta_Mkt-RF") if k == "beta_market" else None)
                if v is None:
                    continue
                self.registry.upsert_point(
                    ticker, "ticker", field,
                    as_of=as_of, source_id="derived",
                    value_num=float(v), confidence=0.9,
                )
        return betas

    # ------------------------------------------------------------------
    # Composite risk scores
    # ------------------------------------------------------------------
    def recompute_risk_scores(self, ticker: str) -> dict[str, float]:
        """Combine macro / sanctions / commodity inputs into 0..1 scores.

        Inputs are pulled from the registry. Each component is normalized
        using a fixed reasonable scale; missing inputs are skipped. The
        final score for each component is a weighted average of available
        sub-components.

        These are decision-aiding heuristics, NOT calibrated probabilities.
        """
        if self.registry is None:
            return {}
        reg = self.registry
        ticker = ticker.upper()
        as_of = datetime.now().strftime("%Y-%m-%d")

        # ---- Macro stress ------------------------------------------------
        # Pull global indicators
        vix = reg.latest_value("global", "global.vix") or 0.0
        vix_term = reg.latest_value("global", "global.vix_term_3m_1m") or 1.0
        hy_oas = reg.latest_value("global", "global.hy_oas") or 0.0
        recession_p = reg.latest_value("global", "global.recession_prob") or 0.0
        # Normalize each: VIX < 15 calm, > 35 crisis
        n_vix = _clip01((float(vix) - 15) / 20)
        # Term ratio < 1 = stress (backwardation)
        n_term = _clip01((1.0 - float(vix_term)) * 2.0)
        # HY OAS: < 3% calm, > 8% crisis
        n_hy = _clip01((float(hy_oas) - 3) / 5)
        n_rec = _clip01(float(recession_p) * 1.5)
        macro_stress = _avg(n_vix, n_term, n_hy, n_rec)

        # ---- Sanctions pressure (placeholder per-ticker linkage) --------
        sanc7 = reg.latest_value("global", "global.sanctions_added_7d") or 0
        sanctions = _clip01(float(sanc7) / 50.0)

        # ---- Commodity sensitivity (placeholder; default 0.5 if oil>$80) -
        # We can't reliably infer per-ticker commodity exposure without
        # the supply-chain disclosure feature. Use a modest default that
        # reflects market-wide commodity stress.
        oil = reg.latest_value("WTI", "commodity.spot_usd")
        commodity = _clip01((float(oil) - 70) / 50) if oil else 0.0

        # ---- Energy crisis -----------------------------------------------
        nat = reg.latest_value("NG", "commodity.spot_usd")
        energy = _clip01((float(nat) - 3) / 5) if nat else 0.0

        # ---- Supply chain ------------------------------------------------
        bdiy = reg.latest_value("global", "freight.bdiy")
        # BDIY range: 1000 calm, 4000 crisis (very rough)
        supply = _clip01((float(bdiy) - 1500) / 3000) if bdiy else macro_stress * 0.5

        scores = {
            "macro_stress": round(macro_stress, 3),
            "supply_chain": round(supply, 3),
            "sanctions_pressure": round(sanctions, 3),
            "commodity_sensitivity": round(commodity, 3),
            "energy_crisis": round(energy, 3),
        }

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO risk_scores
                    (ticker, as_of, macro_stress, supply_chain, sanctions_pressure,
                     commodity_sensitivity, energy_crisis)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (ticker, as_of, scores["macro_stress"], scores["supply_chain"],
                 scores["sanctions_pressure"], scores["commodity_sensitivity"],
                 scores["energy_crisis"]),
            )

        # Push to registry as canonical fields
        for field, val in (
            ("ticker.risk.macro_stress_score",     scores["macro_stress"]),
            ("ticker.risk.supply_chain_score",     scores["supply_chain"]),
            ("ticker.risk.sanctions_pressure",     scores["sanctions_pressure"]),
            ("ticker.risk.commodity_sensitivity",  scores["commodity_sensitivity"]),
            ("ticker.risk.energy_crisis_score",    scores["energy_crisis"]),
        ):
            reg.upsert_point(
                ticker, "ticker", field,
                as_of=as_of, source_id="derived",
                value_num=float(val), confidence=0.6,
            )
        return scores

    # ------------------------------------------------------------------
    # Global pulse (no ticker)
    # ------------------------------------------------------------------
    def recompute_global_risk_pulse(self) -> dict[str, float]:
        """Same composite calc as recompute_risk_scores but for 'global'.
        Useful as a tape-temperature gauge in the dashboard."""
        return self.recompute_risk_scores("__GLOBAL__")


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------

def _clip01(x: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _avg(*vals: float) -> float:
    finite = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(sum(finite) / len(finite)) if finite else 0.0


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ds = DerivedSignals()
    print("Schema OK")
