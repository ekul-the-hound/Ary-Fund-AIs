"""
data/openbb_provider.py
=======================

Thin adapter exposing OpenBB Platform data to ARY QUANT, alongside the
existing Tiingo / FMP / Finnhub providers in ``data/providers.py``.

Scope (the gaps ARY QUANT doesn't already cover):
    * Economy      — GDP, CPI, unemployment  (OECD / EconDB / FRED)
    * Fixed income — treasury rates + full yield curve (Federal Reserve)
    * Commodities  — spot prices (FRED)
    * Currency/FX  — historical FX pairs (yfinance)

Design rules (same contract as providers.py):
    * PURE ADAPTER: no SQLite writes, no UI, no Ollama. Fetch -> DataFrame.
    * DEFENSIVE: every public function returns an EMPTY, correctly-shaped
      DataFrame on any failure (openbb missing, no network, provider error)
      and logs the reason. Nothing here can crash a caller.
    * LAZY: ``openbb`` is a heavy import (several seconds on first ever
      import while it builds its static assets). It is imported on first
      use, never at module import time, so importing this file is free.
    * KEYS: on first use, FRED/FMP keys from your ``.env`` (already loaded
      by ``load_dotenv()`` elsewhere) are pushed into OpenBB's runtime
      credentials, so its FRED/FMP-backed endpoints work without touching
      ``~/.openbb_platform/user_settings.json``. Keyless providers
      (federal_reserve, oecd, econdb, yfinance) work with no keys at all.

INSTALL (PowerShell, from project root, hedgefund_ai active) — see the
constraint-file rationale in the module docstring bottom:

    pip freeze > pip_freeze_pre_openbb.txt
    Set-Content constraints_openbb.txt "numpy<2"
    pip install openbb -c constraints_openbb.txt
    python -c "import numpy; print('numpy', numpy.__version__)"
    python -c "from openbb import obb; print('openbb import ok')"

The freeze file is your rollback: if anything misbehaves,
    pip uninstall openbb (and the openbb-* packages it lists)
or rebuild the venv from pip_freeze_pre_openbb.txt.

SELF-TEST (after install):
    python data\\openbb_provider.py
prints a one-line status per data group.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("ary_quant.data.openbb_provider")

# ----------------------------------------------------------------------
# Lazy OpenBB handle
# ----------------------------------------------------------------------
_obb: Any = None
_obb_failed: bool = False


def _get_obb() -> Optional[Any]:
    """Import openbb on first use; wire .env keys; None if unavailable.

    The import is stateful and slow the first time (it discovers installed
    extensions and builds the ``obb`` interface). We do it exactly once and
    cache the handle. If openbb isn't installed, we log once and every
    public function degrades to its empty shape.
    """
    global _obb, _obb_failed
    if _obb is not None:
        return _obb
    if _obb_failed:
        return None
    try:
        from openbb import obb  # noqa: PLC0415 (deliberate lazy import)
    except Exception as e:  # noqa: BLE001
        logger.warning("openbb not available (%s) — OpenBB-backed data "
                       "will return empty results. Install with: "
                       "pip install openbb -c constraints_openbb.txt", e)
        _obb_failed = True
        return None

    # Push existing .env keys into OpenBB's runtime credentials so its
    # FRED/FMP-backed endpoints authenticate. Missing keys are fine —
    # the keyless providers below don't need them.
    try:
        fred = os.getenv("FRED_API_KEY")
        fmp = os.getenv("FMP_API_KEY")
        if fred:
            obb.user.credentials.fred_api_key = fred
        if fmp:
            obb.user.credentials.fmp_api_key = fmp
    except Exception as e:  # noqa: BLE001
        logger.warning("could not set openbb credentials: %s", e)

    _obb = obb
    return _obb


def _to_df(result: Any) -> pd.DataFrame:
    """OBBject -> DataFrame, tolerating API-shape drift."""
    for attr in ("to_dataframe", "to_df"):
        fn = getattr(result, attr, None)
        if callable(fn):
            try:
                df = fn()
                return df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
            except Exception:  # noqa: BLE001
                continue
    try:
        return pd.DataFrame(result.results)  # raw fallback
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def _safe_call(label: str, fn, *args, **kwargs) -> pd.DataFrame:
    """Run one obb endpoint call defensively; empty DataFrame on failure."""
    obb = _get_obb()
    if obb is None:
        return pd.DataFrame()
    try:
        return _to_df(fn(obb)(*args, **kwargs))
    except Exception as e:  # noqa: BLE001
        logger.warning("openbb %s failed: %s", label, e)
        return pd.DataFrame()


# ======================================================================
# Economy — GDP, CPI, unemployment
# ======================================================================
def gdp_real(country: str = "united_states",
             start_date: Optional[str] = None) -> pd.DataFrame:
    """Real GDP series (OECD, keyless). Columns typically date/value."""
    return _safe_call(
        "economy.gdp.real",
        lambda o: o.economy.gdp.real,
        country=country, start_date=start_date, provider="oecd",
    )


def cpi(country: str = "united_states",
        start_date: Optional[str] = None) -> pd.DataFrame:
    """CPI series. Uses FRED (your key) with OECD as the mental fallback —
    if this returns empty and you have no FRED key set, try provider='oecd'
    via `cpi_oecd`."""
    return _safe_call(
        "economy.cpi(fred)",
        lambda o: o.economy.cpi,
        country=country, start_date=start_date, provider="fred",
    )


def cpi_oecd(country: str = "united_states",
             start_date: Optional[str] = None) -> pd.DataFrame:
    """CPI via OECD — keyless alternative to `cpi`."""
    return _safe_call(
        "economy.cpi(oecd)",
        lambda o: o.economy.cpi,
        country=country, start_date=start_date, provider="oecd",
    )


def unemployment(country: str = "united_states",
                 start_date: Optional[str] = None) -> pd.DataFrame:
    """Unemployment rate series (OECD, keyless)."""
    return _safe_call(
        "economy.unemployment",
        lambda o: o.economy.unemployment,
        country=country, start_date=start_date, provider="oecd",
    )


# ======================================================================
# Fixed income — treasury rates / yield curve
# ======================================================================
def treasury_rates(start_date: Optional[str] = None) -> pd.DataFrame:
    """Daily US treasury rates across maturities (Federal Reserve,
    keyless). One row per date, one column per tenor."""
    return _safe_call(
        "fixedincome.government.treasury_rates",
        lambda o: o.fixedincome.government.treasury_rates,
        start_date=start_date, provider="federal_reserve",
    )


def yield_curve(date: Optional[str] = None) -> pd.DataFrame:
    """US yield curve snapshot for a date (latest if None), Federal
    Reserve, keyless. Natural input for quant/yield_curve_3d.py."""
    kwargs: dict[str, Any] = {"provider": "federal_reserve"}
    if date:
        kwargs["date"] = date
    return _safe_call(
        "fixedincome.government.yield_curve",
        lambda o: o.fixedincome.government.yield_curve,
        **kwargs,
    )


# ======================================================================
# Commodities — spot prices
# ======================================================================
def commodity_spot(commodity: str = "all",
                   start_date: Optional[str] = None) -> pd.DataFrame:
    """Commodity spot price series (FRED-backed; needs FRED_API_KEY,
    which you have). `commodity` accepts e.g. 'wti', 'brent', 'gold',
    'natural_gas', or 'all'."""
    return _safe_call(
        "commodity.price.spot",
        lambda o: o.commodity.price.spot,
        commodity=commodity, start_date=start_date, provider="fred",
    )


# ======================================================================
# Currency / FX
# ======================================================================
def fx_historical(pair: str = "EURUSD",
                  start_date: Optional[str] = None) -> pd.DataFrame:
    """Daily historical FX for a pair like 'EURUSD' (yfinance, keyless).
    Returns OHLC-style rows; empty DataFrame if the pair is unknown."""
    return _safe_call(
        "currency.price.historical",
        lambda o: o.currency.price.historical,
        symbol=pair, start_date=start_date, provider="yfinance",
    )


__all__ = [
    "gdp_real", "cpi", "cpi_oecd", "unemployment",
    "treasury_rates", "yield_curve",
    "commodity_spot", "fx_historical",
]


# ======================================================================
# Self-test — python data\openbb_provider.py
# ======================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:  # load .env when run directly (dual-import-era convenience)
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass

    checks = [
        ("GDP (real, OECD)", lambda: gdp_real(start_date="2020-01-01")),
        ("CPI (FRED)", lambda: cpi(start_date="2024-01-01")),
        ("Unemployment (OECD)", lambda: unemployment(start_date="2024-01-01")),
        ("Treasury rates (Fed)", lambda: treasury_rates(start_date="2025-01-01")),
        ("Yield curve (Fed)", lambda: yield_curve()),
        ("Commodity spot (FRED)", lambda: commodity_spot("wti", "2025-01-01")),
        ("FX EURUSD (yfinance)", lambda: fx_historical("EURUSD", "2025-06-01")),
    ]
    print("=== openbb_provider self-test ===")
    for name, fn in checks:
        try:
            df = fn()
            status = f"OK — {len(df)} rows" if not df.empty else "EMPTY"
        except Exception as e:  # noqa: BLE001
            status = f"ERROR — {e}"
        print(f"{name:26s}: {status}")
    print("(EMPTY on a keyless endpoint usually means openbb isn't "
          "installed or no network; EMPTY on CPI/commodity means "
          "FRED_API_KEY didn't load.)")
