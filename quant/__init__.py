"""
quant/ — quantitative finance building blocks for the research system.

Modules:
    volatility       — returns, rolling / realized / GARCH volatility
    var_es           — Value at Risk and Expected Shortfall
    regime           — market regime detection
    position_sizing  — Kelly, vol-target, and combined sizing reports

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
from quant.position_sizing import (
    kelly_fraction,
    volatility_target_size,
    position_sizing_report,
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
    # position_sizing
    "kelly_fraction",
    "volatility_target_size",
    "position_sizing_report",
]

__version__ = "0.1.0"
