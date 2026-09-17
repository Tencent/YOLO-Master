"""Final A2 acceptance configuration loaded from the repository default YAML."""

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import DEFAULT_CFG_PATH, YAML

A2_CONFIG_KEYS = {
    "stal_area_threshold",
    "stal_min_candidates",
    "a2_model",
    "a2_epochs",
    "a2_imgsz",
    "a2_batch",
    "a2_seed",
    "a2_patience",
    "a2_wandb_project",
}


def load_a2_config(path: str | Path = DEFAULT_CFG_PATH) -> dict[str, int | float | str]:
    """Load and validate the frozen A2 protocol from a default YAML file."""
    config = YAML.load(path)
    missing = sorted(A2_CONFIG_KEYS.difference(config))
    if missing:
        raise KeyError(f"default.yaml is missing A2 settings: {', '.join(missing)}")
    values = {
        "area_threshold": float(config["stal_area_threshold"]),
        "min_candidates": int(config["stal_min_candidates"]),
        "model": str(config["a2_model"]),
        "epochs": int(config["a2_epochs"]),
        "imgsz": int(config["a2_imgsz"]),
        "batch": int(config["a2_batch"]),
        "seed": int(config["a2_seed"]),
        "patience": int(config["a2_patience"]),
        "wandb_project": str(config["a2_wandb_project"]),
    }
    if values["area_threshold"] != 16.0 or values["min_candidates"] != 4:
        raise ValueError("A2 final code only supports default.yaml settings stal_area_threshold=16 and min=4")
    if values["model"] != "v0.1-N":
        raise ValueError("A2 final protocol requires default.yaml a2_model=v0.1-N")
    return values


A2_CONFIG = load_a2_config()
