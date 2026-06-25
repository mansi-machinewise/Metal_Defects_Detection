"""
src/utils/config.py
====================
Configuration management.

Loads config.yaml and merges environment variable overrides.
Returns a nested dict-like object accessible with dot notation.

Design decisions:
  - Single source of truth: config.yaml
  - .env overrides specific values (useful in Docker / CI)
  - Config is loaded once and cached
  - No magic — every override is explicit and logged
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config Path
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "config.yaml"
_cached_config: dict | None = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(config_path: Path | str | None = None) -> dict:
    """
    Load configuration from YAML file and apply environment variable overrides.

    The function is idempotent — repeated calls return the same cached dict
    unless force=True is used.

    Args:
        config_path: Override default config.yaml path.

    Returns:
        Merged configuration dictionary.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    path = Path(config_path) if config_path else _CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Ensure config/config.yaml exists in the project root."
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f)

    logger.info("Config loaded from: %s", path)

    # Apply environment variable overrides
    _apply_env_overrides(cfg)

    # Ensure output directories exist
    _ensure_output_dirs(cfg)

    _cached_config = cfg
    return cfg


def _apply_env_overrides(cfg: dict) -> None:
    """
    Apply .env / environment variable overrides to the config dict.
    Only overrides keys that are explicitly set in environment.
    """
    overrides: dict[str, tuple[list[str], Any]] = {
        # ENV_VAR: (config_path_as_list, cast_function)
        "DATASET_ROOT":         (["dataset", "root"], str),
        "DATASET_YAML":         (["dataset", "yaml"], str),
        "TRAINING_MODEL":       (["training", "model"], str),
        "TRAINING_EPOCHS":      (["training", "epochs"], int),
        "TRAINING_BATCH_SIZE":  (["training", "batch_size"], int),
        "TRAINING_IMAGE_SIZE":  (["training", "image_size"], int),
        "TRAINING_WORKERS":     (["training", "workers"], int),
        "TRAINING_AMP":         (["training", "amp"], lambda v: v.lower() == "true"),
        "TRAINING_DEVICE":      (["inference", "device"], str),
        "MODEL_PATH":           (["inference", "model_path"], str),
        "CONFIDENCE_THRESHOLD": (["inference", "confidence_threshold"], float),
        "BAD_THRESHOLD":        (["inference", "bad_threshold"], float),
        "LOG_LEVEL":            (["logging", "level"], str),
        "LOG_DIR":              (["logging", "log_dir"], str),
    }

    for env_key, (cfg_path, cast) in overrides.items():
        env_val = os.getenv(env_key)
        if env_val is not None:
            try:
                casted = cast(env_val)
                # Navigate and set nested key
                node = cfg
                for key in cfg_path[:-1]:
                    node = node.setdefault(key, {})
                node[cfg_path[-1]] = casted
                logger.debug("Config override: %s → %s = %r", env_key, ".".join(cfg_path), casted)
            except (ValueError, TypeError) as e:
                logger.warning("Failed to apply env override %s=%s: %s", env_key, env_val, e)


def _ensure_output_dirs(cfg: dict) -> None:
    """Create output directories defined in config if they don't exist."""
    dirs_to_create = [
        cfg.get("training", {}).get("project", "outputs/runs"),
        cfg.get("evaluation", {}).get("output_dir", "outputs/reports"),
        cfg.get("evaluation", {}).get("predictions_dir", "outputs/predictions"),
        cfg.get("logging", {}).get("log_dir", "outputs/logs"),
    ]
    for d in dirs_to_create:
        Path(d).mkdir(parents=True, exist_ok=True)


def get(key_path: str, default: Any = None) -> Any:
    """
    Convenience accessor with dot-notation path.

    Example:
        get("training.batch_size")   → 4
        get("inference.device")      → "auto"

    Args:
        key_path: Dot-separated path into the config dict.
        default:  Value to return if path does not exist.
    """
    cfg = load_config()
    keys = key_path.split(".")
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node