#!/usr/bin/env python3
"""
scripts/run_camera.py
=======================
Entry-point script for live camera defect detection.

Run:
    python scripts/run_camera.py
    python scripts/run_camera.py --source 1
    python scripts/run_camera.py --source rtsp://192.168.1.10/stream
    python scripts/run_camera.py --model outputs/runs/defect_detection_v1/weights/best.pt

Controls (while the window is focused):
    q — quit
    s — save current annotated frame to outputs/predictions/live_captures/

Exit codes:
    0 — session ended normally (user pressed quit)
    1 — camera or model failed to initialize
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv
load_dotenv()

from src.camera.live_runner import LiveDetectionRunner
from src.inference.detector import DefectDetector
from src.utils.config import load_config
from src.utils.device import detect_hardware, log_hardware_info
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live metal defect detection from a camera feed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default webcam, settings from config.yaml
  python scripts/run_camera.py

  # Use a different webcam index
  python scripts/run_camera.py --source 1

  # Use an IP camera
  python scripts/run_camera.py --source rtsp://192.168.1.10:554/stream

  # Use a specific model
  python scripts/run_camera.py --model outputs/runs/defect_detection_v1/weights/best.pt

  # Run inference every 2nd frame (smoother feed on slow hardware)
  python scripts/run_camera.py --skip-frames 2
        """,
    )
    parser.add_argument("--source",      type=str, default=None, help="Camera index (e.g. 0, 1) or IP camera URL")
    parser.add_argument("--model",       type=str, default=None, help="Path to .pt weights")
    parser.add_argument("--width",       type=int, default=None)
    parser.add_argument("--height",      type=int, default=None)
    parser.add_argument("--skip-frames", type=int, default=None, help="Run inference every Nth frame")
    parser.add_argument("--conf",        type=float, default=None, help="Confidence threshold override")
    parser.add_argument("--device",      type=str, default=None)
    parser.add_argument("--config",      type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("METAL DEFECT DETECTION — Live Camera Pipeline")
    logger.info("=" * 60)

    cfg = load_config(args.config)
    hw = detect_hardware()
    log_hardware_info(hw)

    # Apply CLI overrides to camera config
    if args.source is not None:
        # Numeric strings become webcam indices; anything else stays a URL string.
        source = int(args.source) if args.source.isdigit() else args.source
        cfg["camera"]["source"] = source
        logger.info("Camera source overridden: %s", source)

    if args.width:  cfg["camera"]["width"] = args.width
    if args.height: cfg["camera"]["height"] = args.height
    if args.skip_frames: cfg["camera"]["inference_every_n_frames"] = args.skip_frames

    if args.conf:
        cfg["inference"]["confidence_threshold"] = args.conf
        cfg["inference"]["bad_threshold"] = args.conf

    # ------------------------------------------------------------------
    # Initialize detector
    # ------------------------------------------------------------------
    detector_kwargs: dict = {}
    if args.model:  detector_kwargs["model_path"] = args.model
    if args.device: detector_kwargs["device"] = args.device

    try:
        detector = DefectDetector(config=cfg, **detector_kwargs)
    except Exception as e:
        logger.exception("Failed to initialize detector: %s", e)
        return 1

    # ------------------------------------------------------------------
    # Run live detection
    # ------------------------------------------------------------------
    runner = LiveDetectionRunner(config=cfg, detector=detector)

    try:
        runner.run()
    except RuntimeError as e:
        logger.error(str(e))
        logger.error("")
        logger.error("Common fixes:")
        logger.error("  - Ensure no other application is using the camera")
        logger.error("  - Try a different --source index (0, 1, 2 ...)")
        logger.error("  - For IP cameras, verify the URL and network connectivity")
        return 1
    except Exception as e:
        logger.exception("Unexpected error during live detection: %s", e)
        return 1

    logger.info("Live detection session ended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())