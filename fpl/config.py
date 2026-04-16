"""
YAML config loader for the fpl-solver project.

Loads config.yaml from the project root, applies sensible defaults for optional
fields, validates required fields (team_id, free_transfers), and provides helpers
to extract solver params and player overrides in the format expected by the solver.
Supports merging auto-detected values (current_gw, chips_used) from the FPL API.

Local overrides: if config.local.yaml exists next to config.yaml, its values are
deep-merged on top — letting users customize team_id, chips, fixture_overrides etc.
without touching the tracked config.yaml.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default values for optional config sections
_DEFAULT_SOLVER = {
    "planning_horizon": "rest_of_season",
    "min_hist_games": 7,
    "sub_probability": 0.10,
    "first_gw_transfer_penalty": -1,
    "time_limit_per_scenario": 15,
    "max_scenarios": 100,
}

_DEFAULT_TRANSFER_TOPUP = {
    "enabled": True,
    "trigger_gw": 15,
    "transfer_count": 5,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge override dict into base dict.

    For nested dicts, values are merged recursively. For all other types
    (including lists), the override value replaces the base value entirely.

    Args:
        base: Base dictionary (not mutated).
        override: Override dictionary whose values take precedence.

    Returns:
        New merged dictionary.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """
    Load config from YAML file, then apply local overrides if present.

    Looks for a matching .local.yaml file next to the base config
    (e.g. config.yaml -> config.local.yaml) and deep-merges any values
    found there on top. This lets users keep personal settings (team_id,
    chips, fixture_overrides) outside of version control.

    Args:
        config_path: Path to config file, relative to project root.

    Returns:
        Merged config dict with defaults applied.

    Raises:
        FileNotFoundError: If base config file does not exist.
        ValueError: If required fields (team_id, free_transfers) are missing.
    """
    full_path = _PROJECT_ROOT / config_path
    if not full_path.exists():
        raise FileNotFoundError(f"Config file not found: {full_path}")

    with open(full_path) as f:
        raw = yaml.safe_load(f) or {}

    local_path = full_path.parent / (full_path.stem + ".local.yaml")
    if local_path.exists():
        with open(local_path) as f:
            local_raw = yaml.safe_load(f) or {}
        raw = _deep_merge(raw, local_raw)
        overridden_keys = list(local_raw.keys())
        logger.info("Applied local overrides from %s (keys: %s)", local_path, ", ".join(overridden_keys))
    else:
        logger.debug("No local config found at %s", local_path)

    config = _apply_defaults(raw)
    _validate_required(config)
    logger.info("Loaded config from %s", full_path)
    return config


def _apply_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    """Merge raw config with defaults for optional sections."""
    config = dict(raw)

    if "solver" not in config:
        config["solver"] = {}
    config["solver"] = {**_DEFAULT_SOLVER, **config["solver"]}

    if "transfer_topup" not in config:
        config["transfer_topup"] = {}
    config["transfer_topup"] = {**_DEFAULT_TRANSFER_TOPUP, **config["transfer_topup"]}

    if "chips" not in config:
        config["chips"] = {}

    for key in ("non_playing", "forced_lineup", "points_multiplier",
                "excluded_players", "extra_players", "fixture_overrides"):
        if key not in config:
            config[key] = []

    return config


def _validate_required(config: dict[str, Any]) -> None:
    """Validate that required fields are present."""
    if "team_id" not in config:
        raise ValueError("Required field 'team_id' is missing from config")
    if "free_transfers" not in config:
        raise ValueError("Required field 'free_transfers' is missing from config")


def merge_api_values(config: dict[str, Any], current_gw: int | None = None, chips: dict[str, Any] | None = None) -> None:
    """
    Merge auto-detected values from FPL API into config (in-place).

    Args:
        config: Config dict to update.
        current_gw: Current gameweek from API, if detected.
        chips: Chip usage from API, e.g. {"wildcards_used": 1, "free_hits_used": 1, ...}
    """
    if current_gw is not None and "current_gw" not in config:
        config["current_gw"] = current_gw
        logger.debug("Merged current_gw=%s from API", current_gw)

    if chips and "chips" in config:
        for key, value in chips.items():
            if key not in config["chips"] or config["chips"][key] is None:
                config["chips"][key] = value
        logger.debug("Merged chips from API: %s", chips)


def get_solver_params(config: dict[str, Any]) -> dict[str, Any]:
    """
    Extract solver parameters from config.

    Args:
        config: Full config dict from load_config.

    Returns:
        Solver params dict.
    """
    return dict(config.get("solver", _DEFAULT_SOLVER))


def get_player_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """
    Convert YAML player override format to tuple format used by the solver.

    Args:
        config: Full config dict from load_config.

    Returns:
        Dict with keys: non_playing, forced_lineup, points_multiplier,
        excluded_players, extra_players. List fields use (player_id, gw_list)
        or (player_id, multiplier) tuples as appropriate.
    """
    def to_non_playing_tuples(items: list) -> list[tuple[int, list[int]]]:
        result = []
        for item in items:
            pid = item.get("player") or item.get("id")
            gws = item.get("gameweeks", [])
            if pid is not None:
                result.append((int(pid), list(gws)))
        return result

    def to_points_multiplier_tuples(items: list) -> list[tuple[int, float]]:
        result = []
        for item in items:
            pid = item.get("player") or item.get("id")
            mult = item.get("multiplier", 1.0)
            if pid is not None:
                result.append((int(pid), float(mult)))
        return result

    return {
        "non_playing": to_non_playing_tuples(config.get("non_playing") or []),
        "forced_lineup": to_non_playing_tuples(config.get("forced_lineup") or []),
        "points_multiplier": to_points_multiplier_tuples(config.get("points_multiplier") or []),
        "excluded_players": [int(p) for p in (config.get("excluded_players") or [])],
        "extra_players": [int(p) for p in (config.get("extra_players") or [])],
    }
