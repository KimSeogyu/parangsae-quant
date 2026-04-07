"""Inter-component signal value types.

These frozen dataclasses define the contract between components.
Actors publish these as signal values; Strategy receives and pattern-matches them.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlphaScore:
    """Published by Alpha Models. Consumed by Portfolio Construction."""

    name: str
    scores: dict[str, float]  # {instrument_id_str: alpha_value}


@dataclass(frozen=True)
class RiskState:
    """Published by Risk Model. Consumed by Portfolio Construction.

    risk_scale: multiplicative exposure scalar (0.05-1.5) from all four layers.
    """

    risk_scale: float  # combined multiplicative scale
    regime_scale: float  # BTC regime component
    vol_scale: float  # volatility targeting component
    corr_scale: float  # correlation monitor component
    dd_scale: float  # drawdown scaling component
    drawdown: float  # current drawdown fraction
    portfolio_vol: float  # realized portfolio vol (annualized)
    avg_correlation: float  # average pairwise correlation of top coins
    volatility: dict[str, float] = field(default_factory=dict)  # per-symbol vol


@dataclass(frozen=True)
class UniverseState:
    """Published by Universe Model. Consumed by Portfolio Construction."""

    current: frozenset[str]  # instrument_id strings in universe
    added: tuple[str, ...]
    removed: tuple[str, ...]
