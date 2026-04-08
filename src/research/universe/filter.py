"""Crypto-native universe filter.

PRD Section 2: Universe Rules
- Only native crypto risk assets (exclude stablecoins, tokenized gold/bonds/stocks, etc.)
- 180+ day listing history
- Top 30 by median 24h quote volume (30d)
- Funding/OI coverage >= 95%
- Weekly reconstitution
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.research.config import UniverseSettings


# Default excluded symbols (stablecoins, tokenized assets, leveraged tokens)
DEFAULT_EXCLUDED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR", "3L", "3S", "5L", "5S")

DEFAULT_EXCLUDED_BASES = frozenset(
    {
        "USDC", "USDT", "DAI", "TUSD", "BUSD", "FDUSD", "USDP", "PYUSD", "GUSD",
        "PAXG", "XAUT", "TGOLD",
        "ONDO",
        "AEUR",
    }
)


def is_native_crypto(
    symbol: str,
    base_currency: str,
    excluded_categories: list[str] | None = None,
    excluded_symbols: list[str] | None = None,
) -> bool:
    """Check if a symbol is a native crypto risk asset.

    Rejects stablecoins, tokenized gold/bonds/stocks, leveraged/inverse tokens.
    """
    if excluded_symbols and symbol in excluded_symbols:
        return False

    if base_currency in DEFAULT_EXCLUDED_BASES:
        return False

    for suffix in DEFAULT_EXCLUDED_SUFFIXES:
        if base_currency.endswith(suffix):
            return False

    return True


def filter_by_listing_age(
    symbols: list[str],
    listing_dates: dict[str, datetime],
    reference_date: datetime,
    min_days: int = 180,
) -> list[str]:
    """Keep only symbols listed for at least min_days."""
    result = []
    for sym in symbols:
        if sym not in listing_dates:
            continue
        age = (reference_date - listing_dates[sym]).days
        if age >= min_days:
            result.append(sym)
    return result


def filter_by_data_coverage(
    symbols: list[str],
    coverage: dict[str, float],
    min_coverage: float = 0.95,
) -> list[str]:
    """Keep symbols with funding/OI data coverage >= threshold."""
    return [s for s in symbols if coverage.get(s, 0.0) >= min_coverage]


def rank_by_liquidity(
    symbols: list[str],
    median_volumes: dict[str, float],
    top_n: int = 30,
) -> list[str]:
    """Rank symbols by median 24h quote volume and keep top N."""
    ranked = sorted(symbols, key=lambda s: median_volumes.get(s, 0.0), reverse=True)
    return ranked[:top_n]


def filter_by_spread(
    symbols: list[str],
    median_spreads: dict[str, float],
    max_spread_bps: float = 20.0,
) -> list[str]:
    """Remove symbols with excessively wide spreads."""
    return [s for s in symbols if median_spreads.get(s, 0.0) <= max_spread_bps]


def build_universe(
    all_symbols: list[str],
    base_currencies: dict[str, str],
    listing_dates: dict[str, datetime],
    median_volumes: dict[str, float],
    data_coverage: dict[str, float],
    median_spreads: dict[str, float],
    reference_date: datetime,
    settings: UniverseSettings,
) -> list[str]:
    """Apply all universe filters in sequence. Returns final universe list.

    Filter chain:
    1. Native crypto check (exclude stablecoins, tokenized, leveraged)
    2. Listing age >= 180 days
    3. Funding/OI coverage >= 95%
    4. Spread filter (exclude excessive spreads)
    5. Liquidity rank (top 30)
    """
    # Step 1: Native crypto filter
    native = [
        s for s in all_symbols
        if is_native_crypto(
            s,
            base_currencies.get(s, ""),
            settings.exclusion.categories,
            settings.exclusion.symbols,
        )
    ]

    # Step 2: Listing age
    aged = filter_by_listing_age(
        native, listing_dates, reference_date, settings.inclusion.min_listing_days
    )

    # Step 3: Data coverage
    covered = filter_by_data_coverage(
        aged, data_coverage, settings.inclusion.min_funding_oi_coverage
    )

    # Step 4: Spread filter
    tight = filter_by_spread(
        covered, median_spreads, settings.inclusion.max_spread_bps
    )

    # Step 5: Liquidity rank
    universe = rank_by_liquidity(
        tight, median_volumes, settings.liquidity.top_n
    )

    return universe


def compute_reconstitution_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    day_of_week: str = "monday",
    hour_utc: int = 0,
) -> list[pd.Timestamp]:
    """Generate weekly reconstitution timestamps between start and end."""
    day_map = {
        "monday": "W-MON", "tuesday": "W-TUE", "wednesday": "W-WED",
        "thursday": "W-THU", "friday": "W-FRI", "saturday": "W-SAT",
        "sunday": "W-SUN",
    }
    freq = day_map.get(day_of_week.lower(), "W-MON")
    dates = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    return [d.replace(hour=hour_utc, minute=0, second=0) for d in dates]


def get_symbol_sector(
    symbol: str,
    base_currency: str,
    sector_map: dict[str, list[str]],
) -> str:
    """Look up sector for a symbol from the sector map."""
    for sector, members in sector_map.items():
        if base_currency in members:
            return sector
    return "other"
