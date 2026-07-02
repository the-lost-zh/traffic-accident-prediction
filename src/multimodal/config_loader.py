"""Minimal YAML/JSON config loader for multimodal training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    with config_path.open("r", encoding="utf-8") as f:
        if suffix == ".json":
            data = json.load(f)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError("YAML config files require PyYAML.") from exc
            data = yaml.safe_load(f) or {}
        else:
            raise ValueError(f"Unsupported config file type: {config_path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return data


def flatten_cli_config(config: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        if key in {"training", "model", "split", "preprocessing"} and isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value
    if "hidden_dims" in flattened and isinstance(flattened["hidden_dims"], list):
        flattened["hidden_dims"] = ",".join(str(item) for item in flattened["hidden_dims"])
    return flattened
