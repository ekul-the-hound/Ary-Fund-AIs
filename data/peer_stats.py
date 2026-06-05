"""
Peer Statistics
===============
Computes real per-sector, per-metric peer distributions from the actual
investable universe, so the sector-relative z-scores in
``agent/risk_scanner.py`` are measured against *this* universe rather than
the coarse hard-coded broad-market defaults baked into that module.

Why this matters
----------------
``risk_scanner`` already z-scores fundamentals against a peer cohort, but
when no real peer stats are supplied it falls back to ``_SECTOR_DEFAULTS``
â€” values its own comment describes as "coarse â€” broad-market averages."
That means a REIT's leverage is compared against a rough guess at "typical
real-estate leverage," not against the leverage of the REITs you actually
track. This module closes that gap: it aggregates the real mean / std / n
for each (sector, metric) pair from the universe and hands them to
``compute_risk_flags`` through the ``peer_stats`` kwarg that module already
exposes. **No change to risk_scanner is required** â€” we feed the channel it
already reads.

Consistency principle
---------------------
The z-scored metrics include *derived* ratios (``debt_ebitda``,
``fcf_yield``, ``roic``, ``cash_conversion``, ``net_debt_ebitda``) that are
NOT stored as single registry fields â€” they are computed inside
``filing_analyzer.extract_key_metrics_for_agent``. To avoid two divergent
definitions of the same ratio, peer stats are computed from **metric
snapshots produced by that same function**, not by re-deriving the ratios
here. A ``get_metrics_fn`` is injected so callers (and tests) control where
those snapshots come from; the default path goes through the pipeline.

Performance
-----------
Computing stats means gathering one metric snapshot per universe ticker
(~600), which is slow on a cold cache. Results are therefore cached to a
human-readable JSON file with a timestamp and reused for ``max_age_hours``
(default 24). The expensive recompute happens at most once a day.
"""
from __future__ import annotations

import json
import logging
import os
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# Metrics we compute peer stats for. This list MUST stay in sync with
# ``risk_scanner._METRIC_DIRECTIONS`` â€” those are the only metrics the
# scanner z-scores, so computing stats for anything else is wasted work.
Z_SCORED_METRICS: tuple[str, ...] = (
    "debt_ebitda",
    "net_debt_ebitda",
    "interest_coverage",
    "fcf_yield",
    "roic",
    "operating_margin",
    "gross_margin",
    "cash_conversion",
    "risk_factor_count",
)

# Minimum valid peers before a (sector, metric) stat is trustworthy.
# Below this we return None and the scanner falls back to its defaults.
_MIN_PEERS = 2

_CACHE_FILENAME = "peer_stats_cache.json"


# ---------------------------------------------------------------------------
# Universe & sector helpers
# ---------------------------------------------------------------------------

def get_universe_tickers(config: Any = None, tickers: Optional[List[str]] = None) -> List[str]:
    """Return the ticker list for peer-stats computation.

    Defaults to the full ``universe.US_UNIVERSE``. ``config.WATCHLIST`` is
    intentionally ignored: peer-relative scoring needs a real per-sector
    distribution, and scoping to the handful of names being analyzed
    collapses every sector to 0-few peers. An explicit ``tickers`` list may
    be injected (used by tests) to override the universe source.
    """
    if tickers is not None:
        return list(tickers)
    try:
        from data.universe import US_UNIVERSE
        return list(US_UNIVERSE)
    except Exception:  # noqa: BLE001
        try:
            from universe import US_UNIVERSE  # flat-layout fallback
            return list(US_UNIVERSE)
        except Exception:  # noqa: BLE001
            return []


def _sector_of(metrics: Dict[str, Any]) -> Optional[str]:
    """Normalized (lowercased, stripped) sector tag, or None."""
    s = metrics.get("sector") if isinstance(metrics, dict) else None
    if isinstance(s, str) and s.strip():
        return s.strip().lower()
    return None


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def compute_all_sector_peer_stats(
    get_metrics_fn: Callable[[str], Optional[Dict[str, Any]]],
    tickers: List[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate peer stats across every sector and z-scored metric.

    Parameters
    ----------
    get_metrics_fn:
        ``ticker -> metric-snapshot dict`` (the same shape
        ``extract_key_metrics_for_agent`` returns, i.e. carrying ``sector``
        and the derived ratios). Returning ``None`` for a ticker skips it.
    tickers:
        The universe to aggregate over.

    Returns
    -------
    ``{sector: {metric: {"mean": float, "std": float, "n": int}}}``

    A (sector, metric) entry is omitted when fewer than ``_MIN_PEERS``
    tickers have a usable value. Sample standard deviation (ddof=1) is
    used â€” these are samples of a population, not the population itself.
    """
    # First pass: bucket raw values by sector -> metric -> [values].
    buckets: Dict[str, Dict[str, List[float]]] = {}
    n_seen = 0
    for tk in tickers:
        try:
            m = get_metrics_fn(tk)
        except Exception as e:  # noqa: BLE001 â€” one bad ticker must not abort the run
            logger.debug("peer_stats | %s | metrics fetch failed: %s", tk, e)
            continue
        if not isinstance(m, dict):
            continue
        sector = _sector_of(m)
        if sector is None:
            continue
        n_seen += 1
        sector_bucket = buckets.setdefault(sector, {})
        for metric in Z_SCORED_METRICS:
            v = _as_float(m.get(metric))
            if v is not None:
                sector_bucket.setdefault(metric, []).append(v)

    # Second pass: reduce each list to mean/std/n.
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sector, metric_map in buckets.items():
        for metric, values in metric_map.items():
            if len(values) < _MIN_PEERS:
                continue
            mean = statistics.fmean(values)
            std = statistics.stdev(values)  # sample std, ddof=1
            out.setdefault(sector, {})[metric] = {
                "mean": float(mean),
                "std": float(std),
                "n": int(len(values)),
            }

    logger.info(
        "peer_stats | computed stats for %d sector(s) from %d ticker(s)",
        len(out), n_seen,
    )
    return out


def peer_stats_for_sector(
    all_stats: Dict[str, Dict[str, Dict[str, float]]],
    sector: Optional[str],
) -> Dict[str, Dict[str, float]]:
    """Pull the ``{metric: {mean, std, n}}`` block for one sector.

    Returns ``{}`` when the sector is unknown or absent â€” which makes the
    scanner fall back to its own sector defaults, exactly as if no real
    stats existed for that sector.
    """
    if not sector:
        return {}
    return all_stats.get(sector.strip().lower(), {})


# ---------------------------------------------------------------------------
# Caching (JSON, human-readable)
# ---------------------------------------------------------------------------

def _cache_path(data_dir: str) -> str:
    return os.path.join(data_dir, _CACHE_FILENAME)


def cache_peer_stats(
    peer_stats: Dict[str, Dict[str, Dict[str, float]]],
    data_dir: str = "data",
) -> None:
    """Write peer stats to ``data/peer_stats_cache.json`` with a timestamp.

    Best-effort: a write failure logs a warning and is swallowed (a missing
    cache simply forces a recompute next time; it never breaks a run).
    """
    payload = {
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "computed_at_epoch": time.time(),
        "stats": peer_stats,
    }
    try:
        os.makedirs(data_dir, exist_ok=True)
        with open(_cache_path(data_dir), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        logger.info("peer_stats | cached to %s", _cache_path(data_dir))
    except Exception as e:  # noqa: BLE001
        logger.warning("peer_stats | cache write failed (non-fatal): %s", e)


def load_peer_stats_cache(
    max_age_hours: float = 24.0,
    data_dir: str = "data",
) -> Optional[Dict[str, Dict[str, Dict[str, float]]]]:
    """Load cached peer stats if present and younger than ``max_age_hours``.

    Returns the ``stats`` dict on a fresh hit, or ``None`` when the cache is
    missing, unreadable, malformed, or stale (so the caller recomputes).
    """
    path = _cache_path(data_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:  # noqa: BLE001
        logger.warning("peer_stats | cache read failed: %s", e)
        return None

    epoch = payload.get("computed_at_epoch")
    if not isinstance(epoch, (int, float)):
        return None
    age_hours = (time.time() - epoch) / 3600.0
    # >= (not >) so that max_age_hours=0.0 means "any cache is stale" — a
    # just-written cache has age ~0.0, and 0.0 > 0.0 is False, which would
    # wrongly serve it. The boundary case (age exactly == max_age) is also
    # correctly treated as stale.
    if age_hours >= max_age_hours:
        logger.info("peer_stats | cache stale (%.1fh > %.1fh)", age_hours, max_age_hours)
        return None

    stats = payload.get("stats")
    return stats if isinstance(stats, dict) else None


# ---------------------------------------------------------------------------
# Top-level convenience: load-or-compute
# ---------------------------------------------------------------------------

def get_or_compute_peer_stats(
    get_metrics_fn: Callable[[str], Optional[Dict[str, Any]]],
    config: Any = None,
    data_dir: str = "data",
    max_age_hours: float = 24.0,
    force: bool = False,
    tickers: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return peer stats, using the disk cache when it is fresh.

    This is the function the pipeline calls. On a fresh cache it is cheap
    (one JSON read); otherwise it recomputes over the universe and rewrites
    the cache. ``force=True`` bypasses the cache (useful for a manual
    refresh).
    """
    # Minimum sectors a healthy cache must contain. A degenerate cache (e.g.
    # 1 sector from a prior run that only saw watchlist tech names) is treated
    # as invalid and forces a recompute, regardless of age. This self-heals
    # the poisoning failure mode where a thin cache stayed "fresh" for 24h.
    _MIN_HEALTHY_SECTORS = 5

    if not force:
        cached = load_peer_stats_cache(max_age_hours=max_age_hours, data_dir=data_dir)
        if cached is not None and (tickers is not None or len(cached) >= _MIN_HEALTHY_SECTORS):
            logger.info("peer_stats | using cached stats (%d sectors)", len(cached))
            return cached
        if cached is not None:
            logger.info(
                "peer_stats | cache has only %d sector(s) (< %d) — recomputing",
                len(cached), _MIN_HEALTHY_SECTORS,
            )

    universe = get_universe_tickers(config, tickers=tickers)
    stats = compute_all_sector_peer_stats(get_metrics_fn, universe)
    cache_peer_stats(stats, data_dir=data_dir)
    return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_float(x: Any) -> Optional[float]:
    """Coerce to float, rejecting None/NaN/inf and non-numerics."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return v