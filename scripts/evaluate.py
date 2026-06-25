#!/usr/bin/env python3
"""
scripts/evaluate.py
====================
Evaluation and inference pipeline.

Actions performed:
  1. Load trained model
  2. Run inference on test images (or val if test not available)
  3. Save annotated predictions
  4. Generate summary report (JSON + text)
  5. Log per-image and aggregate metrics

Run:
    python scripts/evaluate.py
    python scripts/evaluate.py --model outputs/runs/defect_detection_v1/weights/best.pt
    python scripts/evaluate.py --images path/to/test/images --max 50
    python scripts/evaluate.py --model best.pt --conf 0.5

Exit codes:
    0 — evaluation completed
    1 — model not found or inference failed
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv
load_dotenv()

import cv2

from src.inference.detector import DefectDetector
from src.inference.visualizer import DefectVisualizer
from src.utils.config import load_config
from src.utils.device import detect_hardware, log_hardware_info
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLOv8 model on test images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use model and paths from config.yaml
  python scripts/evaluate.py

  # Specify model and image directory
  python scripts/evaluate.py --model best.pt --images dataset/images/test

  # Run on val split instead
  python scripts/evaluate.py --images dataset/images/val

  # Limit to 20 images
  python scripts/evaluate.py --max 20

  # Higher confidence threshold
  python scripts/evaluate.py --conf 0.6
        """,
    )
    parser.add_argument("--model",   type=str, default=None, help="Path to .pt weights")
    parser.add_argument("--images",  type=str, default=None, help="Directory of test images")
    parser.add_argument("--output",  type=str, default=None, help="Output directory for results")
    parser.add_argument("--max",     type=int, default=None, help="Max images to process")
    parser.add_argument("--conf",    type=float, default=None, help="Confidence threshold override")
    parser.add_argument("--device",  type=str, default=None)
    parser.add_argument("--config",  type=str, default=None)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated images (faster)",
    )
    return parser.parse_args()


def find_test_images(cfg: dict, images_dir: str | None) -> list[Path]:
    """Locate test images from argument or config."""
    image_exts = set(cfg["dataset"].get("image_extensions", [".jpg", ".jpeg", ".png"]))

    if images_dir:
        d = Path(images_dir)
    else:
        # Try test split, then val
        dataset_root = Path(cfg["dataset"]["root"])
        splits = cfg["dataset"].get("splits", {})
        test_dir = dataset_root / splits.get("test", "images/test")
        val_dir  = dataset_root / splits.get("val",  "images/val")

        if test_dir.exists() and any(test_dir.iterdir()):
            d = test_dir
            logger.info("Using test split: %s", d)
        elif val_dir.exists():
            d = val_dir
            logger.warning("Test split not found. Using val split: %s", d)
        else:
            raise FileNotFoundError(
                "No test or val image directory found. "
                "Specify --images path/to/images"
            )

    if not d.exists():
        raise FileNotFoundError(f"Image directory not found: {d}")

    images = sorted([
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in image_exts
    ])
    logger.info("Found %d images in: %s", len(images), d)
    return images


def run_evaluation(
    detector: DefectDetector,
    visualizer: DefectVisualizer,
    images: list[Path],
    output_dir: Path,
    max_images: int | None,
    save_images: bool,
) -> dict:
    """
    Run inference on all images and collect statistics.

    Returns:
        Summary dict with aggregate metrics.
    """
    if max_images:
        images = images[:max_images]
        logger.info("Limiting to %d images.", max_images)

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_results = []
    good_count = bad_count = 0
    total_defects = 0
    failed = 0
    all_ms = []

    for i, img_path in enumerate(images, 1):
        logger.info("[%d/%d] Processing: %s", i, len(images), img_path.name)

        try:
            result = detector.predict(img_path)
        except Exception as e:
            logger.error("  Failed: %s", e)
            failed += 1
            continue

        # Tally
        if result.status == "GOOD":
            good_count += 1
        else:
            bad_count += 1
        total_defects += result.defect_count
        all_ms.append(result.inference_ms)

        pred_results.append(result.to_dict())

        # Save annotated image
        if save_images:
            out_img = output_dir / f"{img_path.stem}_pred{img_path.suffix}"
            try:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is not None:
                    annotated = visualizer.draw(img_bgr, result)
                    cv2.imwrite(str(out_img), annotated)
            except Exception as e:
                logger.warning("  Could not save annotated image: %s", e)

    # Aggregate statistics
    n = len(pred_results)
    summary = {
        "total_images": n,
        "good_count": good_count,
        "bad_count": bad_count,
        "failed": failed,
        "total_defects_detected": total_defects,
        "avg_defects_per_image": round(total_defects / n, 2) if n > 0 else 0,
        "good_rate_pct": round(good_count / n * 100, 1) if n > 0 else 0,
        "bad_rate_pct": round(bad_count / n * 100, 1) if n > 0 else 0,
        "avg_inference_ms": round(sum(all_ms) / len(all_ms), 1) if all_ms else 0,
        "min_inference_ms": round(min(all_ms), 1) if all_ms else 0,
        "max_inference_ms": round(max(all_ms), 1) if all_ms else 0,
        "model_path": str(detector.model_path),
        "device_used": detector.device,
        "confidence_threshold": detector.conf_threshold,
        "bad_threshold": detector.bad_threshold,
        "predictions": pred_results,
    }
    return summary


def save_report(summary: dict, output_dir: Path) -> None:
    """Save JSON report and human-readable text summary."""
    # JSON
    json_path = output_dir / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("JSON report saved: %s", json_path)

    # Text summary
    txt_path = output_dir / "evaluation_summary.txt"
    lines = [
        "=" * 60,
        "METAL DEFECT DETECTION — EVALUATION REPORT",
        "=" * 60,
        f"  Model              : {summary['model_path']}",
        f"  Device             : {summary['device_used']}",
        f"  Confidence Thresh  : {summary['confidence_threshold']}",
        f"  BAD Threshold      : {summary['bad_threshold']}",
        "",
        "  RESULTS",
        f"  Total images       : {summary['total_images']}",
        f"  GOOD               : {summary['good_count']}  ({summary['good_rate_pct']}%)",
        f"  BAD                : {summary['bad_count']}   ({summary['bad_rate_pct']}%)",
        f"  Failed             : {summary['failed']}",
        "",
        "  DEFECTS",
        f"  Total detected     : {summary['total_defects_detected']}",
        f"  Avg per image      : {summary['avg_defects_per_image']}",
        "",
        "  PERFORMANCE",
        f"  Avg inference      : {summary['avg_inference_ms']} ms",
        f"  Min inference      : {summary['min_inference_ms']} ms",
        f"  Max inference      : {summary['max_inference_ms']} ms",
        "=" * 60,
    ]
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    # Also print to console
    for line in lines:
        logger.info(line)

    logger.info("Text report saved: %s", txt_path)


def main() -> int:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("METAL DEFECT DETECTION — Evaluation Pipeline")
    logger.info("=" * 60)

    cfg = load_config(args.config)
    hw = detect_hardware()
    log_hardware_info(hw)

    # Resolve output dir
    output_dir = Path(args.output or cfg["evaluation"]["predictions_dir"])
    reports_dir = Path(cfg["evaluation"]["output_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Initialise detector
    # ------------------------------------------------------------------
    detector_kwargs: dict = {}
    if args.model:  detector_kwargs["model_path"] = args.model
    if args.device: detector_kwargs["device"] = args.device
    if args.conf:
        cfg["inference"]["confidence_threshold"] = args.conf
        cfg["inference"]["bad_threshold"] = args.conf

    try:
        detector = DefectDetector(config=cfg, **detector_kwargs)
        detector.load_model()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("")
        logger.error("Train the model first:")
        logger.error("  python scripts/train.py")
        return 1
    except Exception as e:
        logger.exception("Failed to initialize detector: %s", e)
        return 1

    visualizer = DefectVisualizer()

    # ------------------------------------------------------------------
    # Find images
    # ------------------------------------------------------------------
    try:
        images = find_test_images(cfg, args.images)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    if not images:
        logger.error("No images found to evaluate.")
        return 1

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    logger.info("Starting evaluation on %d images ...", len(images))
    start = time.time()

    try:
        summary = run_evaluation(
            detector=detector,
            visualizer=visualizer,
            images=images,
            output_dir=output_dir,
            max_images=args.max,
            save_images=not args.no_save,
        )
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        return 1

    elapsed = time.time() - start
    summary["total_eval_seconds"] = round(elapsed, 1)

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    save_report(summary, reports_dir)

    if not args.no_save:
        logger.info("Annotated images saved to: %s", output_dir)

    logger.info("Evaluation completed in %.1f seconds.", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())