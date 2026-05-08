"""
Sandpile model (Bak-Tang-Wiesenfeld) — Self-Organized Criticality.

Maps market stress to a 2-D cellular automaton.  Each cell holds
"grains".  When a cell exceeds the CRITICAL_MASS (= 4), it topples:
4 grains are subtracted and 1 grain is sent to each of its 4
neighbours.  Open boundaries let grains fall off the edges (the system
"dissipates" stress to the outside world).

Connection to markets:
    grain drops  ←  daily |return| (or vol)
    topples      →  micro-avalanches
    avalanche    →  cascading sell-off / regime break

The defining feature of SOC is that AVALANCHE SIZE is roughly
power-law distributed once the system is critical:

    P(s) ~ s^(-τ)

That same power-law is observed in real market drawdowns.  The
sandpile lets you:
    1. Predict structural fragility independent of the trigger size.
       A small shock to an over-loaded grid produces a huge avalanche.
    2. Compare avalanche distribution shape against observed crashes.
    3. Measure stress-avalanche correlation:  LOW correlation means
       the system is in an "endogenous instability" regime — small
       triggers produce big effects.  HIGH correlation means
       exogenous-shock-driven dynamics.

Reference
---------
quant-traderr-lab / Sandpile Model / Sandpile Pipeline.py
2-D BTW lattice, daily stress as grains, avalanche-size analysis.

Design
------
Pure functions, structured dict returns, no I/O.
Vectorised NumPy lattice updates for speed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _clean_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)
    s = prices.dropna()
    s = s[s > 0]
    return s


# ── BTW lattice ────────────────────────────────────────────────────

class _BTWLattice:
    """Bak-Tang-Wiesenfeld 2-D sandpile with open boundaries."""

    def __init__(self, N: int, critical_mass: int, rng: np.random.Generator):
        self.N = N
        self.critical_mass = critical_mass
        self.grid = np.zeros((N, N), dtype=np.int32)
        self.rng = rng

    def add_sand(self, n_grains: int) -> None:
        if n_grains <= 0:
            return
        xs = self.rng.integers(0, self.N, n_grains)
        ys = self.rng.integers(0, self.N, n_grains)
        np.add.at(self.grid, (xs, ys), 1)

    def relax(self, max_substeps: int = 200) -> int:
        """Topple unstable sites until the grid is stable.

        Returns the avalanche size (total topples this step).
        """
        avalanche = 0
        cm = self.critical_mass
        for _ in range(max_substeps):
            unstable = self.grid >= cm
            if not unstable.any():
                break
            avalanche += int(unstable.sum())

            # Topple
            self.grid[unstable] -= 4
            mask = unstable.astype(np.int32)

            # 4-neighbour distribution (open boundaries)
            self.grid[:-1, :] += mask[1:, :]
            self.grid[1:, :]  += mask[:-1, :]
            self.grid[:, :-1] += mask[:, 1:]
            self.grid[:, 1:]  += mask[:, :-1]

        return avalanche

    @property
    def energy(self) -> int:
        return int(self.grid.sum())


# ── public API ──────────────────────────────────────────────────────

def run_sandpile(
    prices: pd.Series,
    grid_size: int = 50,
    critical_mass: int = 4,
    grain_scale: int = 1500,
    critical_threshold: int = 50,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Drive a BTW sandpile with daily |return| stress and record dynamics.

    Parameters
    ----------
    prices : pd.Series
        Historical price series with a DatetimeIndex (or any Index).
    grid_size : int
        N for the N×N lattice.
    critical_mass : int
        Toppling threshold (4 = canonical 2-D BTW).
    grain_scale : int
        |return| × grain_scale = grains dropped per step.
    critical_threshold : int
        Avalanche size above which an event is flagged "critical".
    random_state : int

    Returns
    -------
    dict with keys:
        available : bool
        timeline : pd.DataFrame
            cols [date, input_stress, grains, avalanche_size, energy]
        avalanche_distribution : dict
            sizes, frequencies, log_sizes, log_freqs (for log-log plot)
        power_law_fit : dict
            tau (exponent), R2, n_bins
        critical_events : list of dict
            top critical events by size
        criticality_ratio : float
            fraction of steps with avalanche > threshold
        stress_avalanche_corr : float
        regime_label : str       "endogenous_SOC" / "exogenous_driven"
        final_grid : np.ndarray  (snapshot of the lattice at the end)
    """
    prices = _clean_prices(prices)
    if len(prices) < 30:
        return {
            "available": False,
            "reason": f"Need >= 30 bars; got {len(prices)}.",
        }

    returns = prices.pct_change().fillna(0.0)
    stress = returns.abs()

    rng = np.random.default_rng(random_state)
    sim = _BTWLattice(N=grid_size, critical_mass=critical_mass, rng=rng)

    # Vectorise grain counts
    grains = np.maximum((stress * grain_scale).astype(int).values, 1)

    records = []
    for i, date in enumerate(prices.index):
        sim.add_sand(int(grains[i]))
        avalanche_size = sim.relax()
        records.append({
            "date": date,
            "input_stress": float(stress.iloc[i]),
            "grains": int(grains[i]),
            "avalanche_size": int(avalanche_size),
            "energy": sim.energy,
        })
    timeline = pd.DataFrame(records)

    # Avalanche-size distribution (excluding zero-size events)
    sizes = timeline["avalanche_size"][timeline["avalanche_size"] > 0]
    if len(sizes) > 0:
        unique, counts = np.unique(sizes, return_counts=True)
        # Log-binning for power-law plot
        if len(unique) >= 5:
            log_min = np.log10(unique.min())
            log_max = np.log10(unique.max())
            bins = np.unique(np.logspace(log_min, log_max, 25).astype(int))
            hist, edges = np.histogram(sizes, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            valid = hist > 0
            avalanche_distribution = {
                "sizes": unique,
                "frequencies": counts,
                "binned_centers": centers[valid],
                "binned_freqs": hist[valid],
            }
        else:
            avalanche_distribution = {
                "sizes": unique,
                "frequencies": counts,
                "binned_centers": unique.astype(float),
                "binned_freqs": counts,
            }
    else:
        avalanche_distribution = {
            "sizes": np.array([]), "frequencies": np.array([]),
            "binned_centers": np.array([]), "binned_freqs": np.array([]),
        }

    # Power-law fit  P(s) ~ s^(-tau)
    pl_fit = {"tau": 0.0, "R2": 0.0, "n_bins": 0, "fit_succeeded": False}
    centers = avalanche_distribution["binned_centers"]
    freqs = avalanche_distribution["binned_freqs"]
    if len(centers) >= 4:
        try:
            log_x = np.log(centers)
            log_y = np.log(freqs)
            slope, intercept = np.polyfit(log_x, log_y, 1)
            y_pred = slope * log_x + intercept
            ss_res = np.sum((log_y - y_pred) ** 2)
            ss_tot = np.sum((log_y - log_y.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            pl_fit = {
                "tau": float(-slope),
                "R2": float(r2),
                "n_bins": int(len(centers)),
                "fit_succeeded": True,
            }
        except Exception:
            pass

    # Critical events
    critical_mask = timeline["avalanche_size"] > critical_threshold
    crit_df = (
        timeline[critical_mask]
        .sort_values("avalanche_size", ascending=False)
        .head(10)
    )
    critical_events = crit_df.to_dict(orient="records")
    criticality_ratio = float(critical_mask.mean())

    # Stress-avalanche correlation
    if timeline["avalanche_size"].std() > 0:
        corr = float(
            timeline["input_stress"].corr(timeline["avalanche_size"])
        )
    else:
        corr = 0.0

    if corr < 0.3:
        regime_label = "endogenous_SOC"
    elif corr < 0.6:
        regime_label = "mixed"
    else:
        regime_label = "exogenous_driven"

    return {
        "available": True,
        "timeline": timeline,
        "avalanche_distribution": avalanche_distribution,
        "power_law_fit": pl_fit,
        "critical_events": critical_events,
        "criticality_ratio": criticality_ratio,
        "stress_avalanche_corr": corr,
        "regime_label": regime_label,
        "final_grid": sim.grid.copy(),
        "params": {
            "grid_size": int(grid_size),
            "critical_mass": int(critical_mass),
            "grain_scale": int(grain_scale),
            "critical_threshold": int(critical_threshold),
            "n_steps": int(len(timeline)),
        },
    }