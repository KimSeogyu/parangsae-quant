from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

from src.config.models import Settings

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _substitute_env_vars(value: str) -> str:
    def replacer(match: re.Match) -> str:
        return os.environ.get(match.group(1), "")
    return _ENV_PATTERN.sub(replacer, value)


def _walk_and_substitute(obj: object) -> object:
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_substitute(item) for item in obj]
    return obj


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = _walk_and_substitute(raw)
    return Settings(**raw)
