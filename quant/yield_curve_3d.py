"""
3D Yield Curve — historical Treasury surface evolution.

Builds the time × maturity × yield surface from historical FRED Treasury
data.  At each point in time the curve is INTERPOLATED (quadratic) across
maturities so the resulting surface is smooth, and per-timestamp shape
diagnostics (inversion, slope, curvature) flag regime shifts.

Input data
----------
A ``pd.DataFrame`` of historical Treasury yields, columns labelled by
maturity (e.g. "3m", "2y", "10y", "30y") and DatetimeIndex of dates.
This matches the schema of ``MacroData.get_series`` joined across the
canonical FRED series (DTB3, DGS2, DGS5, DGS10, DGS30, etc.).

Pipeline
--------
1. Convert maturity labels → months.
2. For each row (date), interpolate the curve onto a fine maturity grid.
3. Build the surface arrays X (maturity), Y (time), Z (yield).
4. Compute per-date diagnostics:
       - inversion flag (3M > 10Y)
       - 10Y - 2Y spread
       - level / slope / curvature (Nelson-Siegel-flavored shape factors)
5. Optionally align macro shock dates onto the surface for annotation.

What the user learns from it
----------------------------
- The CURRENT curve shape — normal / flat / inverted.
- How the curve has evolved through past Fed cycles.
- Whether the recent dynamics look like a bull-steepener, bear-flattener,
  inversion, or twist — each carries different implications for equity
  positioning and credit spreads.
- The level / slope / curvature decomposition gives clean per-date
  factor exposures so a portfolio's rate sensitivity can be projected.

Reference
---------
quant-traderr-lab / Yield Curve / Yield Curve Pipeline.py
The reference uses yfinance Treasury proxies (^IRX, ^FVX, ^TNX, ^TYX);
this version uses FRED directly through the project's MacroData layer
because FRED yields are higher quality and already cached locally.

Design
------
Pure functions, structured dict returns, no I/O.  The data-loading
helper is split out so the pure surface builder can be tested with
synthetic data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── maturity label → months ─────────────────────────────────────────

_MATURITY_MONTHS = {
    "1m": 1, "3m": 3, "6m": 6,
    "1y": 12, "2y": 24, "3y": 36, "5y": 60,
    "7y": 84, "10y": 120, "20y": 240, "30y": 360,
}


def _maturity_to_months(label: str) -> Optional[int]:
    """Convert a maturity label (e.g. '10y', '3m') to integer months."""
    return _MATURITY_MONTHS.get(label.lower())


# ── core surface builder ────────────────────────────────────────────

def build_yield_surface(
    yields: pd.DataFrame,
    interp_points: int = 50,
    max_maturity_months: int = 360,
    interp_kind: str = "quadratic",
    macro_shocks: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Build the 3D yield-curve surface from historical Treasury data.

    Parameters
    ----------
    yields : pd.DataFrame
        DatetimeIndex × maturity columns (column names = labels like
        "3m", "2y", "10y").  Values are yields in percent (FRED convention).
    interp_points : int, default 50
        Number of points along the maturity axis after interpolation.
    max_maturity_months : int, default 360
        Upper bound of the maturity grid.
    interp_kind : str, default "quadratic"
        Passed to ``scipy.interpolate.interp1d``.  Falls back to "linear"
        if the row has too few maturities for the requested order.
    macro_shocks : dict, optional
        Mapping ``{name: date_string}``.  Each shock date is snapped to
        the nearest available row index for annotation.

    Returns
    -------
    dict with keys:
        available : bool
        X : np.ndarray (T, M)             maturity grid (months)
        Y : np.ndarray (T, M)             time-step grid
        Z : np.ndarray (T, M)             yields (%)
        dates : pd.DatetimeIndex (T,)
        maturities_grid : np.ndarray (M,) maturities used
        actual_maturities : np.ndarray    raw maturity points (months)
        column_labels : list[str]         the label order
        diagnostics : pd.DataFrame
            cols [is_inverted, spread_2s10s, spread_3m10y,
                  level, slope, curvature]
        shock_indices : dict[name -> row_idx]
        latest_curve : pd.Series          most recent curve
    """
    from scipy.interpolate import interp1d

    if not isinstance(yields, pd.DataFrame) or yields.empty:
        return {"available": False, "reason": "yields DataFrame is empty."}

    # 1. Resolve maturity columns → months (drop columns we don't recognise)
    col_to_months: Dict[str, int] = {}
    for col in yields.columns:
        m = _maturity_to_months(str(col))
        if m is not None:
            col_to_months[col] = m
    if len(col_to_months) < 3:
        return {
            "available": False,
            "reason": ("Need >= 3 recognisable maturity columns "
                       "(e.g. '3m', '2y', '10y')."),
        }

    # Sort columns by maturity ascending
    sorted_cols = sorted(col_to_months.keys(), key=lambda c: col_to_months[c])
    actual_mats = np.array([col_to_months[c] for c in sorted_cols], dtype=float)

    df = yields[sorted_cols].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Forward-fill small gaps; drop fully-empty rows
    df = df.ffill().dropna(how="any")
    if len(df) < 5:
        return {
            "available": False,
            "reason": "Need >= 5 valid rows after ffill/dropna.",
        }

    # 2. Maturity grid for interpolation
    M = max(int(interp_points), 5)
    mat_grid = np.linspace(
        actual_mats.min(), min(max_maturity_months, actual_mats.max()), M,
    )
    T = len(df)

    Z = np.zeros((T, M))
    bond_vals = df.values  # (T, n_actual)

    # If quadratic is impossible (n < 3), fall back gracefully
    use_kind = interp_kind
    if len(actual_mats) < 3 and interp_kind == "quadratic":
        use_kind = "linear"

    for i in range(T):
        f = interp1d(
            actual_mats, bond_vals[i],
            kind=use_kind, fill_value="extrapolate", assume_sorted=True,
        )
        Z[i, :] = f(mat_grid)

    time_steps = np.arange(T)
    X, Y = np.meshgrid(mat_grid, time_steps)

    # 3. Per-date diagnostics
    diags = []
    for i in range(T):
        row = pd.Series(bond_vals[i], index=sorted_cols)

        short = row.iloc[0]
        long = row.iloc[-1]
        is_inverted = bool(short > long) if pd.notna(short) and pd.notna(long) else False

        s_2s10s = (row.get("10y", np.nan) - row.get("2y", np.nan))
        s_3m10y = (row.get("10y", np.nan) - row.get("3m", np.nan))

        # Nelson-Siegel-style level/slope/curvature factor proxies:
        #   level     = mean of (long-end yields)
        #   slope     = long - short
        #   curvature = 2 * mid - short - long
        level = float(row.iloc[-3:].mean()) if len(row) >= 3 else float(row.mean())
        slope = float(long - short) if pd.notna(long) and pd.notna(short) else np.nan

        if len(row) >= 5:
            mid = float(row.iloc[len(row) // 2])
            curvature = float(2.0 * mid - short - long)
        else:
            curvature = np.nan

        diags.append({
            "is_inverted": is_inverted,
            "spread_2s10s": float(s_2s10s) if pd.notna(s_2s10s) else np.nan,
            "spread_3m10y": float(s_3m10y) if pd.notna(s_3m10y) else np.nan,
            "level": level,
            "slope": slope,
            "curvature": curvature,
        })
    diagnostics = pd.DataFrame(diags, index=df.index)

    # 4. Macro shocks
    shock_indices: Dict[str, int] = {}
    if macro_shocks:
        for name, date_str in macro_shocks.items():
            try:
                target = pd.to_datetime(date_str)
                pos = df.index.get_indexer([target], method="nearest")[0]
                if pos >= 0:
                    delta = abs((df.index[pos] - target).days)
                    if delta < 90:  # tighter snap than the reference
                        shock_indices[name] = int(pos)
            except Exception:
                continue

    return {
        "available": True,
        "X": X,
        "Y": Y,
        "Z": Z,
        "dates": df.index,
        "maturities_grid": mat_grid,
        "actual_maturities": actual_mats,
        "column_labels": sorted_cols,
        "diagnostics": diagnostics,
        "shock_indices": shock_indices,
        "latest_curve": df.iloc[-1],
        "params": {
            "interp_points": int(interp_points),
            "interp_kind": use_kind,
            "max_maturity_months": int(max_maturity_months),
            "n_dates": int(T),
            "n_maturities": int(len(actual_mats)),
        },
    }


# ── helper: assemble FRED yields into the expected shape ────────────

def fetch_treasury_panel_from_fred(
    macro_data,
    start_date: str = "2014-01-01",
    end_date: Optional[str] = None,
    maturities: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fetch a multi-maturity Treasury panel using the project's MacroData.

    Parameters
    ----------
    macro_data : MacroData
        An initialised ``data.macro_data.MacroData`` instance.
    start_date, end_date : str
        Date bounds in YYYY-MM-DD form.
    maturities : list[str], optional
        Subset of labels to fetch.  Defaults to a broad coverage.

    Returns
    -------
    pd.DataFrame  with DatetimeIndex and one column per maturity label.
        Columns are named "3m", "2y", "10y", etc. so they can be passed
        directly to ``build_yield_surface``.
    """
    if maturities is None:
        maturities = ["3m", "6m", "1y", "2y", "3y", "5y", "7y",
                      "10y", "20y", "30y"]

    label_to_fred = {
        "1m": "DGS1MO", "3m": "DTB3", "6m": "DTB6",
        "1y": "DGS1", "2y": "DGS2", "3y": "DGS3", "5y": "DGS5",
        "7y": "DGS7", "10y": "DGS10", "20y": "DGS20", "30y": "DGS30",
    }

    series_data: Dict[str, pd.Series] = {}
    for label in maturities:
        fred_id = label_to_fred.get(label)
        if not fred_id:
            continue
        try:
            df = macro_data.get_series(
                fred_id,
                start_date=start_date,
                end_date=end_date,
            )
            if df is not None and not df.empty and "value" in df.columns:
                series_data[label] = df["value"].astype(float)
        except Exception:
            continue

    if not series_data:
        return pd.DataFrame()

    panel = pd.concat(series_data, axis=1).sort_index()
    panel.columns = list(series_data.keys())
    return panel
