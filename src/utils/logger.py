"""
src/utils/logger.py
===================
Centralized logging configuration.

Every module in this project imports `get_logger(__name__)` from here.
This ensures:
  - Consistent format across all modules
  - Single point to change log level / format
  - File + console handlers configured once
  - No duplicate handlers on repeated imports
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Internal state — prevent handler duplication on re-import
# ---------------------------------------------------------------------------
_configured = False
_root_logger_name = "metal_defect"


def _load_log_config() -> dict:
    """Load logging section from config.yaml with safe fallbacks."""
    config_path = Path(__file__).parents[2] / "config" / "config.yaml"
    defaults = {
        "level": "INFO",
        "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "log_dir": "outputs/logs",
        "log_file": "metal_defect.log",
    }
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        return {**defaults, **cfg.get("logging", {})}
    except Exception:
        return defaults


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure root logger for the project.
    Safe to call multiple times — only configures once.

    Args:
        level: Override log level (e.g. "DEBUG"). Falls back to
               LOG_LEVEL env var, then config.yaml, then INFO.
    """
    global _configured
    if _configured:
        return

    cfg = _load_log_config()

    # Resolve log level (priority: arg > env var > config)
    resolved_level = (
        level
        or os.getenv("LOG_LEVEL")
        or cfg["level"]
    ).upper()

    numeric_level = getattr(logging, resolved_level, logging.INFO)

    formatter = logging.Formatter(
        fmt=cfg["format"],
        datefmt=cfg["date_format"],
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # File handler
    log_dir = Path(os.getenv("LOG_DIR") or cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / cfg["log_file"]

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    # Configure project root logger
    logger = logging.getLogger(_root_logger_name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if somehow called twice
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("ultralytics", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True

    logger.info("=" * 70)
    logger.info("Logging initialized")
    logger.info("  Level    : %s", resolved_level)
    logger.info("  Log file : %s", log_file)
    logger.info("=" * 70)


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger namespaced under the project root.

    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        A Logger instance inheriting project-level handlers.
    """
    setup_logging()
    # Strip top-level package prefix for cleaner names
    short_name = name.replace("src.", "").replace("scripts.", "")
    return logging.getLogger(f"{_root_logger_name}.{short_name}")