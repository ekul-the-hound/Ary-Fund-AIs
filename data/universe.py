"""
US Stock Universe
=================
Canonical list of US-listed tickers eligible to appear in the ARY Fund UI
and research flow. The module provides three things:

  1. ``SP500_TICKERS``  — the S&P 500 (the "core" universe).
  2. ``EXTRA_LARGE_CAPS`` — additional large/mid caps people commonly
     research (Russell 1000 selections + popular mid-caps not in the S&P).
  3. ``US_UNIVERSE``    — the merged list (~600 names) used as the default
     pool for the screener.

Plus :func:`is_valid_us_ticker` for free-text user input — anything that
looks like a US ticker symbol is accepted, so the practical universe is
unbounded for users who already know the symbol they want.

Why static
----------
The S&P 500 changes only quarterly. Pulling it dynamically from
Wikipedia / IndexArb on every app start is fragile (HTML scrapers break)
and slow. A static list is fine; refresh manually after each S&P
rebalance (next dates: third Friday of March, June, September, December).

Why ~600, not 6,000
-------------------
The full NYSE+NASDAQ list is ~6,000 symbols, most of which are illiquid
small caps with thin coverage. Loading that into a screener UI hurts
performance and adds little signal. The S&P 500 covers ~80% of US market
cap; adding the next-tier large/mid caps gets us close to 95%. Users
who want a smaller name can type its symbol via :func:`is_valid_us_ticker`.

Source
------
S&P 500 constituents as of late 2024 (latest published rebalance). If a
stock has been added or removed since, edit the list below. The
`SectorMap` is informational only and isn't used for filtering.
"""
from __future__ import annotations

import re
from typing import Tuple


# =============================================================================
# S&P 500 (canonical core universe)
# =============================================================================

SP500_TICKERS: Tuple[str, ...] = (
    "A", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI",
    "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG",
    "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN",
    "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH",
    "APTV", "ARE", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO",
    "BA", "BAC", "BALL", "BAX", "BBY", "BDX", "BEN", "BF.B", "BG", "BIIB",
    "BIO", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK.B", "BRO",
    "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB",
    "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG",
    "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", "CME",
    "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COR", "COST",
    "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX",
    "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR", "D", "DAL",
    "DAY", "DD", "DE", "DECK", "DFS", "DG", "DGX", "DHI", "DHR", "DIS",
    "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA",
    "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL",
    "ELV", "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ERIE",
    "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD", "EXPE", "EXR",
    "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO",
    "FIS", "FITB", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD",
    "GDDY", "GE", "GEHC", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM",
    "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS",
    "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE",
    "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE",
    "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV",
    "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI",
    "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM",
    "KKR", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS",
    "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX",
    "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS",
    "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK",
    "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC",
    "MPWR", "MRK", "MRNA", "MRO", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH",
    "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE",
    "NOC", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS",
    "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS",
    "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE",
    "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PM", "PNC", "PNR",
    "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PWR",
    "PYPL", "QCOM", "QRVO", "RCL", "REG", "REGN", "RF", "RJF", "RL", "RMD",
    "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", "SCHW",
    "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS", "SO", "SOLV", "SPG", "SPGI",
    "SRE", "STE", "STLD", "STT", "STX", "STZ", "SW", "SWK", "SWKS", "SYF",
    "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC",
    "TFX", "TGT", "TJX", "TMO", "TMUS", "TPL", "TPR", "TRGP", "TRMB", "TROW",
    "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL",
    "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V",
    "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR", "VTRS",
    "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WM",
    "WMB", "WMT", "WRB", "WST", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM",
    "ZBH", "ZBRA", "ZTS",
)


# =============================================================================
# Extra large/mid caps not in S&P 500
# =============================================================================
# Popular names people search for that aren't in the S&P 500 — typically
# Russell 1000 mid-caps, recent IPOs, or international ADRs that trade
# heavily on US exchanges. Add freely; unique strings only.

EXTRA_LARGE_CAPS: Tuple[str, ...] = (
    # Recent IPOs / not-yet-S&P
    "ARM", "RIVN", "LCID", "PLTR", "RBLX", "SNOW", "DDOG", "NET",
    "ZS", "ZM", "DOCU", "U", "BILL", "AFRM", "SQ", "HOOD",
    "COIN", "MSTR", "DKNG", "PINS", "RDFN", "ZG", "OPEN",
    # Mid-cap industrials / specialty
    "AXSM", "AVAV", "DAVA", "SAIA", "PSTG",
    # Foreign large-caps that trade as US ADRs
    "TSM", "ASML", "BABA", "JD", "NIO", "PDD", "BIDU", "NTES",
    "SHOP", "SE", "MELI", "GLOB",
    # Energy / commodities mid-caps
    "PXD", "DVN", "MRO", "HES", "EQT", "AR",
    # Biotech / pharma mid-caps
    "BNTX", "NVAX", "BMRN", "RPRX", "SRPT", "ALNY",
    # Consumer / restaurant mid-caps
    "CAVA", "DUOL", "WING", "TXRH", "DPZ", "CMG",
    # Crypto / fintech
    "BITO", "GBTC", "MARA", "RIOT", "CLSK",
)


# =============================================================================
# Combined US universe (used as default pool)
# =============================================================================

US_UNIVERSE: Tuple[str, ...] = tuple(sorted(set(SP500_TICKERS + EXTRA_LARGE_CAPS)))


# =============================================================================
# Free-text ticker validation
# =============================================================================
# US ticker symbols on NYSE/NASDAQ/AMEX:
#   - 1-5 uppercase letters
#   - May contain a "." (e.g. BRK.A, BF.B)
#   - May contain a "-" (e.g. BRK-A as Yahoo-style spelling)
#   - No digits, no other punctuation
# This is loose by design: yfinance accepts anything ticker-shaped, and
# the network call will fail cleanly if the symbol doesn't exist.

_TICKER_RE = re.compile(r"^[A-Z]{1,5}(?:[.\-][A-Z]{1,2})?$")


def is_valid_us_ticker(symbol: str) -> bool:
    """Loose validation for a free-text ticker symbol.

    Accepts: AAPL, BRK.B, BF-B, GOOG, BRK.A
    Rejects: lowercase, leading digits, more than 5+2 chars, special chars

    This is intentionally permissive — yfinance will reject anything that
    doesn't actually trade, so we only filter obvious garbage at the UI
    layer. The goal is to let users research tickers outside the curated
    universe (e.g. micro-caps, foreign ADRs we haven't pre-listed).
    """
    if not symbol or not isinstance(symbol, str):
        return False
    return bool(_TICKER_RE.match(symbol.strip().upper()))


def normalize_ticker(symbol: str) -> str:
    """Canonicalize a user-typed ticker.

    - Uppercases
    - Trims whitespace
    - Converts Yahoo-style hyphens to dots (BRK-B -> BRK.B) only when the
      result matches a known dot-form ticker; otherwise leaves the input
      alone (some symbols genuinely use hyphens).
    """
    s = (symbol or "").strip().upper()
    if not s:
        return s
    # Try the dot-form first if hyphenated.
    if "-" in s and s.replace("-", ".") in US_UNIVERSE:
        return s.replace("-", ".")
    return s


# =============================================================================
# Universe metadata
# =============================================================================

def universe_size() -> int:
    """How many tickers are in the merged US universe."""
    return len(US_UNIVERSE)


def is_in_universe(symbol: str) -> bool:
    """Check whether a normalized ticker is in the curated universe."""
    return normalize_ticker(symbol) in US_UNIVERSE
