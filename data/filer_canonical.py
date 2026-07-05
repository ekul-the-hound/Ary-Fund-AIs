"""
data/filer_canonical.py
===========================
Collapse the many spellings of an institutional filer into ONE canonical node.

The same institution appears across 13D/13G headers and 13F feeds spelled every
which way — "VANGUARD GROUP INC", "The Vanguard Group", "Vanguard Group Inc.",
and (from 13F, which stores only a CIK) "CIK 102909". Without normalization the
money-flow graph renders these as separate hubs. This module maps all of them
to a single canonical name so 13D/13G and 13F edges land on the same node.

Public API
----------
    canonical_name(raw_name)  -> "Vanguard Group" | None
    canonical_from_cik(cik)   -> "Vanguard Group" | None   (major filers only)
    MAJOR_FILERS              -> [(cik, canonical_name), ...] for 13F ingest
"""
from __future__ import annotations

import re
from typing import Optional

# --- major institutional 13F filers: (CIK, canonical name) ----------------
# CIKs for the index giants + big banks are high-confidence; the rest are
# best-effort. A wrong CIK simply ingests nothing (handled upstream). Edit
# freely — this list is what backfill_13f.py iterates over.
MAJOR_FILERS: list[tuple[str, str]] = [
    ("102909",  "Vanguard Group"),
    ("1364742", "BlackRock"),
    ("93751",   "State Street"),
    ("1067983", "Berkshire Hathaway"),
    ("315066",  "Fidelity (FMR)"),
    ("1214717", "Geode Capital"),
    ("895421",  "Morgan Stanley"),
    ("886982",  "Goldman Sachs"),
    ("19617",   "JPMorgan"),
    ("70858",   "Bank of America"),
    ("72971",   "Wells Fargo"),
    ("73124",   "Northern Trust"),
    ("1390777", "BNY Mellon"),
    ("316709",  "Charles Schwab"),
    ("914208",  "Invesco"),
    ("354204",  "Dimensional (DFA)"),
    ("1352280", "Norges Bank"),
    ("902219",  "Wellington Mgmt"),
    ("80255",   "T. Rowe Price"),
    ("1423053", "Citadel"),
    ("1037389", "Renaissance Tech"),
    ("1273087", "Millennium"),
    ("1350694", "Bridgewater"),
    ("1603466", "Point72"),
]
CIK_TO_NAME: dict[str, str] = {str(int(c)): n for c, n in MAJOR_FILERS}

# --- substring aliases: first match wins (order matters) ------------------
# Keyed on an UPPERCASE substring that appears in the raw filer name.
_ALIASES: list[tuple[str, str]] = [
    ("VANGUARD", "Vanguard Group"),
    ("BLACKROCK", "BlackRock"),
    ("STATE STREET", "State Street"),
    ("BERKSHIRE HATHAWAY", "Berkshire Hathaway"),
    ("GEODE", "Geode Capital"),
    ("FMR", "Fidelity (FMR)"), ("FIDELITY", "Fidelity (FMR)"),
    ("MORGAN STANLEY", "Morgan Stanley"),
    ("GOLDMAN SACHS", "Goldman Sachs"),
    ("JPMORGAN", "JPMorgan"), ("J P MORGAN", "JPMorgan"), ("JP MORGAN", "JPMorgan"),
    ("WELLINGTON", "Wellington Mgmt"),
    ("T ROWE PRICE", "T. Rowe Price"), ("PRICE T ROWE", "T. Rowe Price"),
    ("CAPITAL RESEARCH", "Capital Group"), ("CAPITAL WORLD", "Capital Group"),
    ("CAPITAL INTERNATIONAL", "Capital Group"),
    ("NORTHERN TRUST", "Northern Trust"),
    ("MELLON", "BNY Mellon"), ("BANK OF NEW YORK", "BNY Mellon"),
    ("BANK OF AMERICA", "Bank of America"),
    ("WELLS FARGO", "Wells Fargo"),
    ("SCHWAB", "Charles Schwab"),
    ("DIMENSIONAL", "Dimensional (DFA)"),
    ("INVESCO", "Invesco"),
    ("NORGES", "Norges Bank"),
    ("DEUTSCHE BANK", "Deutsche Bank"),
    ("CREDIT SUISSE", "Credit Suisse"),
    ("BARCLAYS", "Barclays"),
    ("CITADEL", "Citadel"),
    ("RENAISSANCE", "Renaissance Tech"),
    ("TWO SIGMA", "Two Sigma"),
    ("MILLENNIUM", "Millennium"),
    ("BRIDGEWATER", "Bridgewater"),
    ("POINT72", "Point72"),
    ("DE SHAW", "D.E. Shaw"), ("D E SHAW", "D.E. Shaw"),
    ("SUSQUEHANNA", "Susquehanna"),
    ("MAGELLAN ASSET", "Magellan Asset Mgmt"),
    ("LEGAL & GENERAL", "Legal & General"), ("LEGAL AND GENERAL", "Legal & General"),
    ("PRUDENTIAL", "Prudential"),
]

# Conservative trailing legal-suffix stripper for the long tail. Deliberately
# does NOT strip identity words (CAPITAL, PARTNERS, MANAGEMENT, ASSET, ADVISORS)
# to avoid merging distinct small firms.
_SUFFIX = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|LLC|L L C|LP|L P|LLP|LTD|LIMITED|"
    r"PLC|THE|AG|SA|N A|NA|SE)\b", re.I)
_STATECODE = re.compile(r"/[A-Z]{2}/?")


def canonical_name(raw_name) -> Optional[str]:
    """Normalize a raw filer name to a single canonical form. None if empty."""
    if not raw_name:
        return None
    up = _STATECODE.sub(" ", str(raw_name).upper())
    up = re.sub(r"[.,]", " ", up)
    up = re.sub(r"\s+", " ", up).strip()
    if not up:
        return None
    for sub, canon in _ALIASES:
        if sub in up:
            return canon
    core = _SUFFIX.sub(" ", up)
    core = re.sub(r"\s+", " ", core).strip()
    if not core:                      # name was only legal suffixes
        core = up
    return core.title()


def name_for_cik(cik) -> Optional[str]:
    """Canonical name for a known major filer CIK, else None."""
    if cik in (None, ""):
        return None
    try:
        key = str(int(str(cik).strip()))
    except (TypeError, ValueError):
        return None
    return CIK_TO_NAME.get(key)


# Backwards-compatible alias.
canonical_from_cik = name_for_cik


# D:\Ary Fund\data\filer_canonical.py
