"""Load YAML configs with `inherit:` and apply command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

# Overrides under these sections do not change the experiment itself.
SAFE_OVERRIDE_SECTIONS = ("machine", "logging")


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of `base` with `override` merged in (nested dicts merged)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict:
    """Load a config file, following `inherit: <file>` relative to it."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> list[str]:
    """Apply `section.key=value` overrides in place.

    Values are parsed as YAML, so `0.4`, `true`, `null` and `[1, 2]` work.
    Unknown keys raise an error, so a typo cannot silently do nothing.
    Returns the overrides that change the experiment (outside machine/logging),
    so they can be flagged in the run record.
    """
    experiment_changing = []
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"Unknown config section in override: {key!r}")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key in override: {key!r}")
        node[parts[-1]] = yaml.safe_load(raw)
        if parts[0] not in SAFE_OVERRIDE_SECTIONS:
            experiment_changing.append(item)
    return experiment_changing


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
