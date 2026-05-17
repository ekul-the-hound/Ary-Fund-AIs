"""
data/factor_returns_loader.py
=============================

Populates the ``factor_returns`` table from Ken French's daily data
library so :meth:`DerivedSignals.recompute_factor_exposures` has
something to regress against.

Datasets pulled (all daily, US research factors):

    * ``F-F_Research_Data_Factors_daily`` -> ``Mkt-RF``, ``SMB``, ``HML``,
      ``RF``
    * ``F-F_Momentum_Factor_daily``       -> ``Mom`` (stored as ``MOM``)

Stored as ``(factor, date, value)`` rows in the existing
``factor_returns`` schema created by ``DerivedSignals._init_db``::

    CREATE TABLE factor_returns (
        factor TEXT NOT NULL,
        date   TEXT NOT NULL,
        value  REAL,
        PRIMARY KEY (factor, date)
    )

Important convention notes
--------------------------
- Ken French publishes returns in **percent** (e.g. ``2.96`` means 2.96%).
  This loader divides by 100 so values are stored as decimals — matching
  the ``pct_change()`` returns that ``recompute_factor_exposures``
  regresses them against. Mixing percent and decimal returns is one of
  the classic silent-failure modes of factor regressions; this guarantees
  unit consistency.
- Factor names are stored using Ken French's published casing
  (``Mkt-RF``, ``SMB``, ``HML``, ``RF``, ``MOM``). The ``betas``
  resolution logic in ``derived_signals.recompute_factor_exposures``
  already handles both ``beta_Mkt-RF`` and ``beta_market`` spellings, so
  no rename is needed.
- The function is idempotent: ``INSERT OR REPLACE`` is used so repeated
  runs on overlapping date ranges just refresh the values without
  duplicating rows.

CLI
---
    python -m data.factor_returns_loader --backfill          # full history
    python -m data.factor_returns_loader --since 2020-01-01  # from a date
    python -m data.factor_returns_loader --recent 90         # last 90 days
    python -m data.factor_returns_loader --status            # summary only
"""
from __future__ import annotations

import argparse
import io
import logging
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
KEN_FRENCH_BASE = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
)

# Ken French daily datasets we ingest. Each entry: (dataset_name,
# zip_filename, csv_filename, column_renames). ``column_renames`` maps
# the CSV's column names to the canonical factor names we store in the
# ``factor_returns`` table. We store the canonical names exactly as Ken
# French publishes them, so ``recompute_factor_exposures`` can resolve
# them via either the long or short alias.
@dataclass(frozen=True)
class FFDataset:
    name: str
    zip_filename: str
    csv_filename: str
    factor_columns: tuple[str, ...]  # canonical factor names to keep


DATASETS: tuple[FFDataset, ...] = (
    FFDataset(
        name="F-F_Research_Data_Factors_daily",
        zip_filename="F-F_Research_Data_Factors_daily_CSV.zip",
        csv_filename="F-F_Research_Data_Factors_daily.CSV",
        factor_columns=("Mkt-RF", "SMB", "HML", "RF"),
    ),
    FFDataset(
        name="F-F_Momentum_Factor_daily",
        zip_filename="F-F_Momentum_Factor_daily_CSV.zip",
        csv_filename="F-F_Momentum_Factor_daily.CSV",
        # File ships the column as "Mom"; we promote to "MOM" so the
        # canonical factor name matches the rest of the table.
        factor_columns=("Mom",),
    ),
)


# Default DB location matches DerivedSignals / RefreshScheduler defaults.
DEFAULT_DB_PATH = "data/hedgefund.db"


# Sentinel values Ken French uses for missing data.
_MISSING_SENTINELS = (-99.99, -999, -99)


HTTP_TIMEOUT = 60
USER_AGENT = (
    "hedgefund-ai-research/0.1 "
    "(+factor_returns_loader; for academic factor regression)"
)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def load_all(
    db_path: str = DEFAULT_DB_PATH,
    since: Optional[str] = None,
    until: Optional[str] = None,
    datasets: Iterable[FFDataset] = DATASETS,
) -> dict[str, int]:
    """Download every configured Ken French daily file and upsert rows.

    Parameters
    ----------
    db_path:
        SQLite path. The ``factor_returns`` table is created if missing.
    since, until:
        Optional ``YYYY-MM-DD`` strings to filter the rows that get
        written. Filtering happens after the parse, so the full file is
        still downloaded.
    datasets:
        Override the default set (mostly useful in tests).

    Returns
    -------
    dict[str, int]
        ``{factor: rows_written}`` — total rows upserted per factor
        across all datasets.
    """
    _ensure_table(db_path)
    written: dict[str, int] = {}
    for ds in datasets:
        try:
            df = fetch_dataset(ds)
        except Exception as e:  # noqa: BLE001
            logger.error("factor_returns_loader | fetch %s failed: %s", ds.name, e)
            continue
        if df.empty:
            logger.warning("factor_returns_loader | %s parsed empty", ds.name)
            continue
        if since:
            df = df[df["date"] >= since]
        if until:
            df = df[df["date"] <= until]
        if df.empty:
            logger.info(
                "factor_returns_loader | %s no rows in window %s..%s",
                ds.name, since, until,
            )
            continue
        n = upsert_rows(db_path, df)
        for factor, sub in df.groupby("factor"):
            written[factor] = written.get(factor, 0) + len(sub)
        logger.info(
            "factor_returns_loader | %s upserted %d rows (factors=%s, dates=%s..%s)",
            ds.name, n, sorted(df["factor"].unique()),
            df["date"].min(), df["date"].max(),
        )
    return written


def fetch_dataset(ds: FFDataset) -> pd.DataFrame:
    """Download one Ken French zip and return a long-format DataFrame.

    Output columns: ``factor``, ``date`` (YYYY-MM-DD), ``value``
    (decimal returns — already divided by 100). Rows containing Ken
    French's missing-data sentinels are dropped.
    """
    url = f"{KEN_FRENCH_BASE}/{ds.zip_filename}"
    logger.info("factor_returns_loader | downloading %s", url)
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return parse_zip_bytes(resp.content, ds)


def parse_zip_bytes(zip_bytes: bytes, ds: FFDataset) -> pd.DataFrame:
    """Parse the in-memory Ken French zip into a long DataFrame.

    Split out from ``fetch_dataset`` so tests can feed canned bytes
    without hitting the network.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # The .CSV filename in some archives is case-sensitive on Linux
        # filesystems. Resolve case-insensitively so a future rename
        # (e.g. .csv vs .CSV) doesn't break ingestion.
        names = zf.namelist()
        target = _resolve_csv_name(names, ds.csv_filename)
        with zf.open(target) as fh:
            raw_text = fh.read().decode("utf-8", errors="replace")
    return _parse_csv_text(raw_text, ds)


def upsert_rows(db_path: str, df: pd.DataFrame) -> int:
    """Insert-or-replace each (factor, date, value) row. Returns row count."""
    if df.empty:
        return 0
    rows = [
        (str(r.factor), str(r.date), float(r.value))
        for r in df.itertuples(index=False)
    ]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO factor_returns (factor, date, value) "
            "VALUES (?, ?, ?)",
            rows,
        )
    return len(rows)


def status(db_path: str = DEFAULT_DB_PATH) -> dict[str, dict]:
    """Per-factor row count + earliest/latest date present in the DB."""
    _ensure_table(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT factor, COUNT(*), MIN(date), MAX(date)
            FROM factor_returns
            GROUP BY factor
            ORDER BY factor
            """
        ).fetchall()
    return {
        factor: {"rows": int(n), "min_date": mn, "max_date": mx}
        for factor, n, mn, mx in rows
    }


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _ensure_table(db_path: str) -> None:
    """Create ``factor_returns`` if a fresh DB hasn't been touched yet.

    DerivedSignals._init_db creates the same schema; we duplicate the
    DDL here so the loader can run standalone (e.g. on first install
    before DerivedSignals has ever been instantiated).
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS factor_returns (
                factor TEXT NOT NULL,
                date   TEXT NOT NULL,
                value  REAL,
                PRIMARY KEY (factor, date)
            )
            """
        )


def _resolve_csv_name(names: list[str], target: str) -> str:
    target_low = target.lower()
    for n in names:
        if n.lower() == target_low:
            return n
    # Some archives use a slightly different filename; fall back to the
    # first .CSV in the zip.
    for n in names:
        if n.lower().endswith(".csv"):
            return n
    raise FileNotFoundError(
        f"No CSV found in archive (looking for {target}, got {names})"
    )


def _parse_csv_text(text: str, ds: FFDataset) -> pd.DataFrame:
    """Parse a Ken French daily CSV blob into a long DataFrame.

    The files have this rough shape::

        This file was created by ...
        <blank line>
        ,Mkt-RF,SMB,HML,RF
        19260701,0.10,-0.24,-0.28,0.009
        19260702,0.45,-0.32,-0.08,0.009
        ...
        20251231,0.12,-0.05,0.02,0.018

         Copyright 2025 Eugene F. Fama and Kenneth R. French
                                                                 (footer)

    Strategy:
      1. Locate the header line containing one of the canonical factor
         columns. That's the start of the data block.
      2. Read until the first row whose first cell isn't an 8-digit date
         (or until EOF). That's the end.
      3. Drop sentinel-coded missing rows and convert percent -> decimal.
    """
    lines = text.splitlines()
    header_idx = _find_header_index(lines, ds.factor_columns)
    if header_idx is None:
        raise ValueError(
            f"Could not find header row containing {ds.factor_columns} in CSV"
        )
    # Scan forward to find where the data block ends — the first line
    # that isn't a YYYYMMDD-prefixed row.
    data_lines: list[str] = []
    for line in lines[header_idx + 1:]:
        stripped = line.strip()
        if not stripped:
            # blank line ends the daily block (annual table or footer)
            break
        first_cell = stripped.split(",", 1)[0].strip()
        if not _looks_like_yyyymmdd(first_cell):
            break
        data_lines.append(line)

    if not data_lines:
        return pd.DataFrame(columns=["factor", "date", "value"])

    header_line = lines[header_idx]
    csv_blob = header_line + "\n" + "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_blob))

    # First column is unlabeled (the YYYYMMDD date). Pandas names it
    # "Unnamed: 0" or similar — rename to a known label.
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "yyyymmdd"})
    df["yyyymmdd"] = df["yyyymmdd"].astype(str).str.strip()
    df = df[df["yyyymmdd"].str.match(r"^\d{8}$")]
    df["date"] = pd.to_datetime(df["yyyymmdd"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

    # Promote "Mom" -> "MOM" so the canonical name is consistent.
    df = df.rename(columns={"Mom": "MOM"})
    canonical = tuple("MOM" if c == "Mom" else c for c in ds.factor_columns)

    keep = [c for c in canonical if c in df.columns]
    if not keep:
        raise ValueError(
            f"None of {canonical} present after parse; got {list(df.columns)}"
        )

    long = df[["date", *keep]].melt(
        id_vars="date", var_name="factor", value_name="value"
    )

    # Drop sentinels and NaNs, then convert percent -> decimal.
    long = long[~long["value"].isin(_MISSING_SENTINELS)]
    long = long.dropna(subset=["value"])
    long["value"] = long["value"].astype(float) / 100.0
    return long.reset_index(drop=True)


def _find_header_index(lines: list[str], wanted: tuple[str, ...]) -> Optional[int]:
    """Locate the header line containing the desired factor columns."""
    # Use lower-case, comma-split tokens for tolerant matching.
    wanted_low = {w.lower() for w in wanted}
    # Map "Mom" -> also "mom" because the momentum file labels the
    # column lowercase. Same intent as the rename in _parse_csv_text.
    wanted_low |= {"mom" if w == "MOM" else w.lower() for w in wanted}
    for i, line in enumerate(lines):
        if "," not in line:
            continue
        cols = {c.strip().lower() for c in line.split(",")}
        if wanted_low & cols:
            return i
    return None


def _looks_like_yyyymmdd(token: str) -> bool:
    return len(token) == 8 and token.isdigit()


# ----------------------------------------------------------------------
# CLI runner
# ----------------------------------------------------------------------
def _print_status(stats: dict[str, dict]) -> None:
    if not stats:
        print("factor_returns: empty")
        return
    print(f"{'factor':<10s} {'rows':>8s}  {'min_date':<12s} {'max_date':<12s}")
    print("-" * 46)
    for factor, info in stats.items():
        print(
            f"{factor:<10s} {info['rows']:>8d}  "
            f"{info['min_date'] or '-':<12s} {info['max_date'] or '-':<12s}"
        )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite path")
    p.add_argument("--backfill", action="store_true",
                   help="Pull full history (default if no window is given).")
    p.add_argument("--since", help="ISO date — only ingest rows >= this date.")
    p.add_argument("--until", help="ISO date — only ingest rows <= this date.")
    p.add_argument("--recent", type=int, default=None,
                   help="Shorthand: only keep the last N days of data.")
    p.add_argument("--status", action="store_true",
                   help="Print row counts per factor and exit.")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.status:
        _print_status(status(args.db))
        return 0

    since = args.since
    if args.recent and not since:
        since = (datetime.now() - timedelta(days=args.recent)).strftime("%Y-%m-%d")

    written = load_all(db_path=args.db, since=since, until=args.until)
    if not written:
        print("factor_returns_loader: nothing written")
        return 1
    for factor, n in sorted(written.items()):
        print(f"  {factor:<10s} rows_upserted={n}")
    print("---")
    _print_status(status(args.db))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
