"""
fix_screener_derived.py
======================

Fill the screener's DERIVED-RATIO and PER-SHARE columns (#8): FCF Margin,
FCF Yield, ROIC, Cash/Share, Book/Share, Sales/Share, FCF/Share, Div/Share.

These columns already exist in the screener's schema but were never populated
(they showed "None"), because ``_fetch_fundamentals_one`` only mapped the raw
yfinance fields, not the ratios/per-share values derived from them.

WHAT THIS DOES
--------------
Rewrites the return dict of ``_fetch_fundamentals_one`` to ALSO compute, from
fields it already fetches (no extra network/data calls):

    fcf_margin   = free_cash_flow / revenue        (%)
    fcf_yield    = free_cash_flow / market_cap      (%)
    roic (proxy) = net_income / (market_cap + total_debt)  (%)   [approximation]
    cash_per_share  = total_cash / shares_outstanding
    book_per_share  = price_proxy / price_to_book   (price ÷ P/B)
    sales_per_share = revenue / shares_outstanding
    fcf_per_share   = free_cash_flow / shares_outstanding
    div_per_share   = div_yield% * price_proxy / 100

where price_proxy = market_cap / shares_outstanding.

ROIC is explicitly a PROXY (uses market cap as an equity stand-in because book
equity isn't in the fundamentals payload) — good for relative screening, not a
substitute for a from-filings ROIC.

Because these are pure arithmetic on already-fetched data, this adds ZERO
extra data calls and no render-time cost.

This does NOT touch _FUNDAMENTALS_LAZY_LIMIT or the skip-check — it only
rewrites the _fetch_fundamentals_one return block, so prior fixes are
preserved.

SAFETY
------
* Targets ui/screener.py.
* Backs up to ui/screener.py.bak before writing.
* Idempotent: detects the new fcf_margin computation and does nothing on re-run.
* Verifies ast.parse before saving.

Usage (from project root, venv active):
    python fix_screener_derived.py
"""
from __future__ import annotations

import ast
import shutil
import sys
from pathlib import Path

TARGET = Path("ui") / "screener.py"

# The current return block of _fetch_fundamentals_one (must match exactly).
OLD = '''    return {
        "symbol":           symbol,
        "name":             f.get("name") or symbol,
        "sector":           f.get("sector") or "—",
        # Overview
        "market_cap":       _num(ov.get("market_cap")),
        "beta":             _num(ov.get("beta")),
        # Valuation
        "pe":               _num(val.get("trailing_pe")),
        "forward_pe":       _num(val.get("forward_pe")),
        "peg":              _num(val.get("peg_ratio")),
        "ps":               _num(val.get("price_to_sales")),
        "pb":               _num(val.get("price_to_book")),
        "ev_ebitda":        _num(val.get("ev_to_ebitda")),
        # Financials
        "revenue":          _num(fin.get("revenue")),
        "gross_profit":     _num(fin.get("gross_profit")),
        "ebitda":           _num(fin.get("ebitda")),
        "net_income":       _num(fin.get("net_income")),
        "fcf":              _num(fin.get("free_cash_flow")),
        "op_cash_flow":     _num(fin.get("operating_cash_flow")),
        "total_debt":       _num(fin.get("total_debt")),
        # yfinance returns debtToEquity as a percent already (e.g. 60.5),
        # not a decimal — keep as-is for display.
        "debt_to_equity":   _num(fin.get("debt_to_equity")),
        "current_ratio":    _num(fin.get("current_ratio")),
        "roe":              _pct(fin.get("return_on_equity")),
        "roa":              _pct(fin.get("return_on_assets")),
        "profit_margin":    _pct(fin.get("profit_margin")),
        "gross_margin":     _pct(fin.get("gross_margin")),
        "op_margin":        _pct(fin.get("operating_margin")),
        # Growth
        "revenue_growth":   _pct(gr.get("revenue_growth")),
        "eps_dil_growth":   _pct(gr.get("earnings_growth")),
        # Dividends
        # yfinance: dividendYield is already a percent in newer versions
        # (e.g. 0.74 = 0.74%), but historically was a decimal (0.0074).
        # We probe and normalize: a value < 1 is treated as decimal.
        "div_yield":        _div_yield_normalize(div.get("dividend_yield")),
        "div_payout":       _pct(div.get("payout_ratio")),
        "ex_div_date":      str(div.get("ex_dividend_date") or "—"),
        # Analyst
        "analyst_rating":   _normalize_recommendation(an.get("recommendation")),
    }'''

NEW = '''    # --- Derived ratios + per-share (computed from already-fetched fields) ---
    _shares_out = _num(ov.get("shares_outstanding"))
    _revenue = _num(fin.get("revenue"))
    _fcf = _num(fin.get("free_cash_flow"))
    _net_income = _num(fin.get("net_income"))
    _total_debt = _num(fin.get("total_debt"))
    _mktcap = _num(ov.get("market_cap"))
    _pb = _num(val.get("price_to_book"))

    def _safe_div(a: Any, b: Any) -> float:
        try:
            if a is None or b is None or pd.isna(a) or pd.isna(b) or float(b) == 0.0:
                return float("nan")
            return float(a) / float(b)
        except (TypeError, ValueError):
            return float("nan")

    _fcf_margin = _safe_div(_fcf, _revenue) * 100.0
    _fcf_yield = _safe_div(_fcf, _mktcap) * 100.0
    _invested = (_mktcap + _total_debt
                 if pd.notna(_mktcap) and pd.notna(_total_debt) else float("nan"))
    _roic = _safe_div(_net_income, _invested) * 100.0  # proxy (mkt cap as equity)
    _price_proxy = _safe_div(_mktcap, _shares_out)
    _cash_ps = _safe_div(_num(fin.get("total_cash")), _shares_out)
    _book_ps = _safe_div(_price_proxy, _pb)            # price ÷ P/B
    _sales_ps = _safe_div(_revenue, _shares_out)
    _fcf_ps = _safe_div(_fcf, _shares_out)
    _dy = _div_yield_normalize(div.get("dividend_yield"))
    _div_ps = (_dy / 100.0 * _price_proxy
               if pd.notna(_dy) and pd.notna(_price_proxy) else float("nan"))

    return {
        "symbol":           symbol,
        "name":             f.get("name") or symbol,
        "sector":           f.get("sector") or "—",
        # Overview
        "market_cap":       _num(ov.get("market_cap")),
        "beta":             _num(ov.get("beta")),
        # Valuation
        "pe":               _num(val.get("trailing_pe")),
        "forward_pe":       _num(val.get("forward_pe")),
        "peg":              _num(val.get("peg_ratio")),
        "ps":               _num(val.get("price_to_sales")),
        "pb":               _num(val.get("price_to_book")),
        "ev_ebitda":        _num(val.get("ev_to_ebitda")),
        # Financials
        "revenue":          _num(fin.get("revenue")),
        "gross_profit":     _num(fin.get("gross_profit")),
        "ebitda":           _num(fin.get("ebitda")),
        "net_income":       _num(fin.get("net_income")),
        "fcf":              _num(fin.get("free_cash_flow")),
        "op_cash_flow":     _num(fin.get("operating_cash_flow")),
        "total_debt":       _num(fin.get("total_debt")),
        # yfinance returns debtToEquity as a percent already (e.g. 60.5),
        # not a decimal — keep as-is for display.
        "debt_to_equity":   _num(fin.get("debt_to_equity")),
        "current_ratio":    _num(fin.get("current_ratio")),
        "roe":              _pct(fin.get("return_on_equity")),
        "roa":              _pct(fin.get("return_on_assets")),
        "profit_margin":    _pct(fin.get("profit_margin")),
        "gross_margin":     _pct(fin.get("gross_margin")),
        "op_margin":        _pct(fin.get("operating_margin")),
        # Derived ratios (computed above)
        "fcf_margin":       _fcf_margin,
        "fcf_yield":        _fcf_yield,
        "roic":             _roic,
        # Per-share (computed above)
        "cash_per_share":   _cash_ps,
        "book_per_share":   _book_ps,
        "sales_per_share":  _sales_ps,
        "fcf_per_share":    _fcf_ps,
        "div_per_share":    _div_ps,
        # Growth
        "revenue_growth":   _pct(gr.get("revenue_growth")),
        "eps_dil_growth":   _pct(gr.get("earnings_growth")),
        # Dividends
        # yfinance: dividendYield is already a percent in newer versions
        # (e.g. 0.74 = 0.74%), but historically was a decimal (0.0074).
        # We probe and normalize: a value < 1 is treated as decimal.
        "div_yield":        _div_yield_normalize(div.get("dividend_yield")),
        "div_payout":       _pct(div.get("payout_ratio")),
        "ex_div_date":      str(div.get("ex_dividend_date") or "—"),
        # Analyst
        "analyst_rating":   _normalize_recommendation(an.get("recommendation")),
    }'''


def _fail(msg: str) -> None:
    print(f"[fix_screener_derived] ABORT: {msg}")
    sys.exit(1)


def main() -> None:
    if not TARGET.exists():
        _fail(f"{TARGET} not found. Run from the project root (D:\\\\Ary Fund) "
              "with the venv active.")

    src = TARGET.read_text(encoding="utf-8")

    if '_fcf_margin = _safe_div(_fcf, _revenue)' in src:
        print("[fix_screener_derived] Already applied — derived ratios present. "
              "Nothing to do.")
        return

    if OLD not in src:
        _fail("could not find the exact _fetch_fundamentals_one return block. "
              "The file may differ (whitespace/quotes/prior edits). Not editing "
              "blindly.")

    src = src.replace(OLD, NEW, 1)

    try:
        ast.parse(src)
    except SyntaxError as e:
        _fail(f"patched file does not parse ({e}); not saving.")

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding="utf-8")

    print("[fix_screener_derived] SUCCESS")
    print(f"  • Backed up original to {backup}")
    print("  • Added derived ratios (FCF margin, FCF yield, ROIC proxy) and")
    print("    per-share values (cash/book/sales/fcf/div per share) to")
    print("    _fetch_fundamentals_one — computed from already-fetched fields.")
    print()
    print("Fully restart Streamlit (clears the _build_screener_frame cache),")
    print("then check Profitability (FCF Margin, ROIC) and Per share. Those")
    print("columns should now fill instead of None.")
    print()
    print("NOTE: ROIC is a PROXY (net income / (market cap + debt)) because")
    print("book equity isn't in the fundamentals payload. Good for relative")
    print("screening; not a from-filings ROIC.")


if __name__ == "__main__":
    main()

# D:\Ary Fund\fix_screener_derived.py
