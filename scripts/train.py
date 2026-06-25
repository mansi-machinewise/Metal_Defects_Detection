#!/usr/bin/env python3
"""
scripts/train.py
=================
Entry-point script for YOLOv8 training.

Always runs dataset validation before training.
Exits with code 1 if validation fails (protects against wasted GPU time).

Run:
    python scripts/train.py
    python scripts/train.py --epochs 50 --batch 2
    python scripts/train.py --model yolov8s.pt --device cpu

Exit codes:
    0 — training completed successfully
    1 — training failed or dataset invalid
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv
load_dotenv()

from src.data.validator import DatasetValidator
from src.training.trainer import DefectModelTrainer
from src.utils.config import load_config
from src.utils.device import detect_hardware, log_hardware_info
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for metal defect detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default training (uses config.yaml)
  python scripts/train.py

  # Quick test run (2 epochs, small batch)
  python scripts/train.py --epochs 2 --batch 2 --imgsz 416

  # Force CPU training
  python scripts/train.py --device cpu

  # Use a larger model if you have VRAM
  python scripts/train.py --model yolov8s.pt

  # Skip dataset validation (not recommended)
  python scripts/train.py --skip-validation
        """,
    )
    parser.add_argument("--epochs",   type=int,   default=None)
    parser.add_argument("--batch",    type=int,   default=None)
    parser.add_argument("--imgsz",    type=int,   default=None)
    parser.add_argument("--model",    type=str,   default=None, help="Base model weights (e.g. yolov8n.pt)")
    parser.add_argument("--device",   type=str,   default=None, help="Device: auto|cpu|cuda|cuda:0")
    parser.add_argument("--workers",  type=int,   default=None)
    parser.add_argument("--name",     type=str,   default=None, help="Run name (overrides config)")
    parser.add_argument("--config",   type=str,   default=None)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip dataset validation (use only if you're sure dataset is valid)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("METAL DEFECT DETECTION — Training Pipeline")
    logger.info("=" * 60)

    cfg = load_config(args.config)

    # ------------------------------------------------------------------
    # Step 1: Hardware detection
    # ------------------------------------------------------------------
    hw = detect_hardware()
    log_hardware_info(hw)

    # ------------------------------------------------------------------
    # Step 2: Dataset validation (unless skipped)
    # ------------------------------------------------------------------
    if not args.skip_validation:
        logger.info("Running dataset validation before training ...")
        try:
            validator = DatasetValidator(config=cfg)
            report = validator.validate()
        except Exception as e:
            logger.exception("Dataset validation crashed: %s", e)
            return 1

        if not report.is_valid:
            logger.error("Dataset validation failed. Fix errors before training.")
            logger.error("Run: python scripts/validate_dataset.py  for details.")
            return 1
        logger.info("Dataset validation passed. Proceeding to training.\n")
    else:
        logger.warning("Dataset validation SKIPPED by user request.")

    # ------------------------------------------------------------------
    # Step 3: Build training overrides from CLI args
    # ------------------------------------------------------------------
    overrides: dict = {}
    if args.epochs  is not None: overrides["epochs"]  = args.epochs
    if args.batch   is not None: overrides["batch"]   = args.batch
    if args.imgsz   is not None: overrides["imgsz"]   = args.imgsz
    if args.device  is not None: overrides["device"]  = args.device
    if args.workers is not None: overrides["workers"] = args.workers
    if args.name    is not None: overrides["name"]    = args.name

    if args.model is not None:
        cfg["training"]["model"] = args.model
        logger.info("Model overridden via CLI: %s", args.model)

    # ------------------------------------------------------------------
    # Step 4: Train
    # ------------------------------------------------------------------
    try:
        trainer = DefectModelTrainer(config=cfg, override_params=overrides)
        result = trainer.train()
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except RuntimeError as e:
        logger.exception("Training runtime error: %s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected training error: %s", e)
        return 1

    # ------------------------------------------------------------------
    # Step 5: Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("  Duration      : %.1f minutes", result.elapsed_minutes)
    logger.info("  Device used   : %s", result.device_used)
    logger.info("  Run directory : %s", result.run_dir)
    logger.info("  Best model    : %s", result.best_model_path)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Evaluate model:")
    logger.info("       python scripts/evaluate.py")
    logger.info("  2. Update config.yaml — set inference.model_path to:")
    logger.info("       %s", result.best_model_path)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())