"""Inter-component signal value types.

These frozen dataclasses define the contract between components.
Actors publish these as signal values; Strategy receives and pattern-matches them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AlphaScore:
    """Published by Alpha Models. Consumed by Portfolio Construction."""

    name: str
    scores: dict[str, float]  # {instrument_id_str: alpha_value}


@dataclass(frozen=True)
class RiskState:
    """Published by Risk Model. Consumed by Portfolio Construction."""

    volatility: dict[str, float]  # {instrument_id_str: rolling_vol}
    drawdown: float
    total_exposure: float
    risk_halt: bool


@dataclass(frozen=True)
class UniverseState:
    """Published by Universe Model. Consumed by Portfolio Construction."""

    current: frozenset[str]  # instrument_id strings in universe
    added: tuple[str, ...]
    removed: tuple[str, ...]
