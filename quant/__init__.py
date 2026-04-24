"""
quant: Quantitative models.

Modules:
    volatility          — returns, rolling / realized / GARCH vol
    var_es              — Value at Risk and Expected Shortfall
    regime              — market regime detection (rule-based)
    regime_hmm          — market regime detection (Gaussian HMM)
    hurst               — Hurst exponent (persistence / mean-reversion)
    ou_process          — Ornstein-Uhlenbeck mean-reversion fit
    gbm                 — Geometric Brownian Motion Monte Carlo
    position_sizing     — vol-target + combined sizing reports
    kelly               — Kelly criterion (discrete, continuous, multi-asset, dd-capped)
    black_scholes       — European option pricing, Greeks, implied vol
    poisson             — Poisson / Hawkes intensity, jump detection
    avellaneda_stoikov  — optimal market-making quotes

All public functions are pure (no I/O, no global state) and return
structured outputs that are easy to test, log, plot, or feed into an agent.
"""

from quant.volatility import (
    compute_returns,
    rolling_volatility,
    realized_volatility,
    annualize_volatility,
    forecast_volatility,
    garch_forecast,
)
from quant.var_es import (
    historical_var,
    historical_es,
    parametric_var,
    monte_carlo_var,
    var_es_report,
)
from quant.regime import detect_regime
from quant.regime_hmm import fit_hmm_regime
from quant.hurst import hurst_exponent, rolling_hurst
from quant.ou_process import fit_ou_process
from quant.gbm import estimate_gbm_params, simulate_gbm, gbm_from_prices
from quant.position_sizing import (
    kelly_fraction,
    volatility_target_size,
    position_sizing_report,
)
from quant.kelly import (
    kelly_discrete,
    kelly_continuous,
    kelly_multi_asset,
    kelly_with_drawdown_cap,
    kelly_report,
)
from quant.black_scholes import (
    bs_price,
    bs_greeks,
    implied_vol,
)
from quant.poisson import (
    poisson_mle,
    exponential_decay_intensity,
    hawkes_mle,
    simulate_poisson,
    simulate_hawkes,
    detect_jumps,
)
from quant.avellaneda_stoikov import (
    reservation_price,
    optimal_spread,
    compute_quotes,
    calibrate_k_from_fills,
)

__all__ = [
    # volatility
    "compute_returns",
    "rolling_volatility",
    "realized_volatility",
    "annualize_volatility",
    "forecast_volatility",
    "garch_forecast",
    # var_es
    "historical_var",
    "historical_es",
    "parametric_var",
    "monte_carlo_var",
    "var_es_report",
    # regime
    "detect_regime",
    "fit_hmm_regime",
    # persistence & mean reversion
    "hurst_exponent",
    "rolling_hurst",
    "fit_ou_process",
    # simulation
    "estimate_gbm_params",
    "simulate_gbm",
    "gbm_from_prices",
    # position_sizing (legacy sizing helpers)
    "kelly_fraction",
    "volatility_target_size",
    "position_sizing_report",
    # kelly (full)
    "kelly_discrete",
    "kelly_continuous",
    "kelly_multi_asset",
    "kelly_with_drawdown_cap",
    "kelly_report",
    # black_scholes
    "bs_price",
    "bs_greeks",
    "implied_vol",
    # poisson
    "poisson_mle",
    "exponential_decay_intensity",
    "hawkes_mle",
    "simulate_poisson",
    "simulate_hawkes",
    "detect_jumps",
    # avellaneda_stoikov
    "reservation_price",
    "optimal_spread",
    "compute_quotes",
    "calibrate_k_from_fills",
]

__version__ = "0.3.0"