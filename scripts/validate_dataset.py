#!/usr/bin/env python3
"""
scripts/validate_dataset.py
============================
Entry-point script for dataset validation.

Run:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --dataset path/to/dataset
    python scripts/validate_dataset.py --yaml path/to/data.yaml

Exit codes:
    0 — dataset is valid, ready for training
    1 — dataset has errors, fix before training
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv
load_dotenv()

from src.data.validator import DatasetValidator
from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate YOLO-format dataset for metal defect detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use paths from config.yaml
  python scripts/validate_dataset.py

  # Override dataset root
  python scripts/validate_dataset.py --dataset dataset/

  # Override data.yaml path
  python scripts/validate_dataset.py --yaml dataset/data.yaml
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset root directory (overrides config.yaml)",
    )
    parser.add_argument(
        "--yaml",
        type=str,
        default=None,
        help="Path to data.yaml file (overrides config.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: config/config.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("METAL DEFECT DETECTION — Dataset Validation")
    logger.info("=" * 60)

    # Load and optionally patch config
    cfg = load_config(args.config)

    if args.dataset:
        cfg["dataset"]["root"] = args.dataset
        logger.info("Dataset root overridden: %s", args.dataset)

    if args.yaml:
        cfg["dataset"]["yaml"] = args.yaml
        logger.info("YAML path overridden: %s", args.yaml)

    # Run validation
    try:
        validator = DatasetValidator(config=cfg)
        report = validator.validate()
    except Exception as e:
        logger.exception("Validation failed with unexpected error: %s", e)
        return 1

    # Return exit code based on validity
    if report.is_valid:
        logger.info("\n✓  Dataset validation PASSED. Ready to train.\n")
        return 0
    else:
        logger.error("\n✗  Dataset validation FAILED. Fix errors before training.\n")
        logger.error("   Common fixes:")
        logger.error("   - Ensure data.yaml has 'nc' and 'names' fields")
        logger.error("   - Ensure train/val image directories exist")
        logger.error("   - Ensure label files match image files (same stem, .txt extension)")
        logger.error("   - Verify bounding box values are in [0, 1] range")
        logger.error("   - Check that class IDs match nc in data.yaml")
        return 1


if __name__ == "__main__":
    sys.exit(main())