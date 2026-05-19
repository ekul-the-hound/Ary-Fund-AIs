"""
rag/learning/scorer.py
======================
Composite quality scoring for theses considered for self-indexing.

The pitfall this exists to avoid
--------------------------------
A thesis can score 0.9 from the LLM reviewer because the prose is
well-organized, even if the prediction turned out wrong. If we
index by review score alone, we teach the system "be clear" not
"be right." The cumulative effect over months is a corpus of
clearly-written but unreliable reasoning.

The fix: combine review score with realized P&L when available.

Composite score formula
-----------------------
    base = (w_review * review + w_outcome * outcome) / (w_review + w_outcome)
    quality = base * age_decay

Where:
    review   = LLM-judged review score in [0, 1]
    outcome  = P&L-derived score in [0, 1] (or None if position open)
    age_decay = half-life decay; 1.0 for fresh theses, →0 for old ones
    weights  = (w_review, w_outcome), default (0.3, 0.6)

Why multiplicative age, not additive?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An additive formula treats age as a small bonus — a great thesis
from 2022 would still score high. Multiplicative treats age as a
trust discount: stale theses lose value proportionally to how stale
they are, no matter how good they looked at the time. The latter
matches how investment reasoning actually ages.

When the position is still open, ``outcome`` is None and we fall
back to review-only with explicit warning in the result dict. Don't
trust review-only quality scores for indexing — they're the failure
mode we're trying to avoid.

The P&L → outcome mapping
-------------------------
Raw return percentages aren't directly comparable across positions
(a +5% on a 6-month thesis is great; a +5% on a 5-year thesis is
poor). We normalize against:

    1. Time held (per-year IRR is the comparison unit)
    2. The position's beta-adjusted benchmark (e.g. SPY for equities)

For Phase 4 simplicity, we use raw annualized return mapped through
a sigmoid centered at +5% (the rough "barely beat the market" line).
Returns above +20% saturate near 1.0; below -10% saturate near 0.0.

Calibration matters but not enormously
--------------------------------------
The exact (w_review, w_outcome) weights and sigmoid shape don't have
to be optimal. The architectural decision — combining review with
outcome at ALL — is the order-of-magnitude improvement. Refinements
are a smaller effect.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Tunable parameters
# ----------------------------------------------------------------------

# Weights for the three components of the composite score.
# Default emphasizes outcome over review because outcome is the
# ground truth we wish we had at indexing time. Review is the proxy
# we use when outcome isn't available yet.
DEFAULT_WEIGHTS = {
    "review": 0.3,
    "outcome": 0.6,
    "age": 0.1,
}

# Sigmoid parameters for return → outcome score
SIGMOID_CENTER_ANNUAL_RETURN = 0.05    # +5% annualized = midpoint (0.5)
SIGMOID_SCALE = 12.0                    # higher = steeper transition

# Age decay: half-life in days. A thesis loses half its weight every
# AGE_HALF_LIFE_DAYS. Defaults to 730 (2 years) — a thesis from a year
# ago carries ~71% of its original weight, a 4-year-old thesis ~25%.
# Two years matches the rough "useful lifetime" of a fund thesis;
# beyond that the world has moved on enough that even a great thesis
# is mostly archeology.
AGE_HALF_LIFE_DAYS = 730.0


# ----------------------------------------------------------------------
# Result dataclass
# ----------------------------------------------------------------------

@dataclass
class QualityScore:
    """Output of ``score_thesis``. Carries the composite plus components.

    The components are exposed so callers can debug *why* a thesis
    scored the way it did. ``warnings`` flags risky situations (no
    P&L, very old, very new) without forcing the caller to inspect
    every field.
    """
    composite: float           # final score in [0, 1]
    review: float              # input review score, passed through
    outcome: Optional[float]   # P&L-derived, None if position open
    age_decay: float           # multiplier in [0, 1] from age
    warnings: list[str]
    components: dict           # raw inputs, for audit logs

    @property
    def is_high_quality(self) -> bool:
        """True iff the thesis clears the indexing bar AND has no warnings."""
        return self.composite >= 0.7 and not self.warnings


# ----------------------------------------------------------------------
# Core scorer
# ----------------------------------------------------------------------

def score_thesis(
    thesis: dict,
    realized_pnl: Optional[dict] = None,
    weights: Optional[dict] = None,
    as_of: Optional[datetime] = None,
) -> QualityScore:
    """Compute a composite quality score for one thesis.

    Parameters
    ----------
    thesis:
        A row from your ``portfolio_db.get_thesis_history``. Expected
        keys: ``id``, ``ticker``, ``created_at``, ``score`` (review
        score in [0, 1]), ``stance`` (optional: 'bull'|'bear'|'neutral').
    realized_pnl:
        Optional dict with keys ``return_pct`` (the position's
        cumulative return, e.g. 0.15 for +15%), ``days_held``, and
        ``benchmark_return_pct``. Pass ``None`` if the position is
        still open or you don't have P&L data.
    weights:
        Override of ``DEFAULT_WEIGHTS``. Useful for sensitivity
        analysis but rarely needed in production.
    as_of:
        Reference time for age calculation. Defaults to now. Pass
        explicitly in tests for deterministic output.

    Returns
    -------
    QualityScore with the composite and all components.
    """
    weights = weights or DEFAULT_WEIGHTS
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        logger.warning("scorer | weights sum to %.3f, expected 1.0",
                       sum(weights.values()))
    as_of = as_of or datetime.now(timezone.utc)
    warnings: list[str] = []

    # ---- Review score ----
    review = float(thesis.get("score") or 0.0)
    if review < 0 or review > 1:
        # Some review systems output [0, 100] or [-1, 1]. Clip to
        # the expected range and flag.
        warnings.append(f"review_out_of_range_{review}")
        review = max(0.0, min(1.0, review))

    # ---- Outcome score (from realized P&L) ----
    outcome: Optional[float] = None
    if realized_pnl is not None:
        return_pct = realized_pnl.get("return_pct")
        days_held = realized_pnl.get("days_held")
        bench_pct = realized_pnl.get("benchmark_return_pct")
        if return_pct is not None and days_held and days_held > 0:
            # Annualize the return
            try:
                annual = (1.0 + float(return_pct)) ** (365.0 / float(days_held)) - 1.0
            except (ValueError, OverflowError):
                annual = float(return_pct)
            # Subtract benchmark if provided — outperformance is
            # what we actually want to reward
            if bench_pct is not None and days_held:
                try:
                    bench_annual = (1.0 + float(bench_pct)) ** (365.0 / float(days_held)) - 1.0
                    annual = annual - bench_annual
                except (ValueError, OverflowError):
                    pass
            # Sigmoid: maps annualized return to [0, 1]
            x = (annual - SIGMOID_CENTER_ANNUAL_RETURN) * SIGMOID_SCALE
            # Clamp x to avoid math.exp overflow on extreme values
            x = max(-50.0, min(50.0, x))
            outcome = 1.0 / (1.0 + math.exp(-x))
        else:
            warnings.append("incomplete_pnl_data")
    else:
        warnings.append("no_realized_pnl")

    # ---- Age decay ----
    created_at = thesis.get("created_at")
    age_decay = 1.0
    if created_at:
        try:
            # Accept both datetime and ISO string
            if isinstance(created_at, str):
                # Strip trailing Z if present (UTC marker)
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            else:
                dt = created_at
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            days_old = max(0.0, (as_of - dt).total_seconds() / 86400.0)
            age_decay = 0.5 ** (days_old / AGE_HALF_LIFE_DAYS)
        except Exception as e:  # noqa: BLE001
            logger.debug("scorer | bad created_at %r: %s", created_at, e)
            warnings.append("bad_created_at")
    else:
        warnings.append("no_created_at")

    # ---- Composite ----
    # If outcome is unknown, redistribute its weight to review.
    # Without this, theses on open positions would always score low.
    # With it, they score by review alone — which is still flagged
    # via the no_realized_pnl warning.
    if outcome is None:
        w = dict(weights)
        w["review"] = w["review"] + w["outcome"]
        w["outcome"] = 0.0
        outcome_contrib = 0.0
    else:
        # Normalize the non-age weights to sum to 1 since age_decay
        # is applied as a multiplier outside.
        w = dict(weights)
        outcome_contrib = w["outcome"] * outcome

    # The base score is the weighted average of review + outcome,
    # normalized by their combined weight (so it lives in [0, 1]).
    non_age_weight = w["review"] + w["outcome"]
    if non_age_weight <= 0:
        non_age_weight = 1.0  # defensive
    base = (w["review"] * review + outcome_contrib) / non_age_weight

    # Age decay multiplies the base score. A 4-year-old thesis with
    # decay=0.06 loses 94% of its quality, regardless of how good the
    # review and outcome looked at the time.
    composite = base * age_decay
    composite = max(0.0, min(1.0, composite))

    return QualityScore(
        composite=composite,
        review=review,
        outcome=outcome,
        age_decay=age_decay,
        warnings=warnings,
        components={
            "weights": w,
            "annualized_return": (annual if outcome is not None else None),
            "days_old": (
                (as_of - datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                 .replace(tzinfo=timezone.utc) if isinstance(created_at, str) else
                 as_of - created_at.replace(tzinfo=timezone.utc) if created_at and getattr(created_at, "tzinfo", None) is None else
                 (as_of - created_at) if created_at else None
                ).total_seconds() / 86400.0 if created_at else None
            ),
        },
    )
