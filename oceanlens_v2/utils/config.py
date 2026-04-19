"""Small YAML config loader with attribute access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigNode(dict):
    """Dictionary with recursive attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _to_config_node(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigNode({key: _to_config_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_config_node(item) for item in value]
    return value


def load_config(path: str | Path) -> ConfigNode:
    """Load a YAML config file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return _to_config_node(data)

