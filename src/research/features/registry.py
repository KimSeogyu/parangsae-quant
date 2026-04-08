"""Feature registry mapping manifest.csv entries to computation functions.

Provides a central lookup for all 40 MVP features defined in the PRD.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    block: str
    function: str
    params: dict[str, str]
    description: str


def load_manifest(path: str | Path = "features/manifest.csv") -> list[FeatureSpec]:
    """Load feature manifest from CSV."""
    specs = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            params = {}
            if row.get("params"):
                for pair in row["params"].split(","):
                    pair = pair.strip()
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        params[k.strip()] = v.strip()
            specs.append(
                FeatureSpec(
                    name=row["name"],
                    block=row["block"],
                    function=row["function"],
                    params=params,
                    description=row.get("description", ""),
                )
            )
    return specs


def get_features_by_block(
    manifest: list[FeatureSpec],
    block: str,
) -> list[FeatureSpec]:
    """Filter manifest by block name."""
    return [f for f in manifest if f.block == block]


FEATURE_BLOCKS = ["relative", "trend", "liquidity", "derivatives", "risk"]
