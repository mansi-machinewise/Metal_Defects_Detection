"""
src/inference/detector.py
==========================
Inference engine for metal defect detection.

Design decisions:
  - Model is loaded ONCE in __init__ and reused for all predictions.
    This avoids the ~1–3 second reload penalty per inference call.
  - Returns a structured DetectionResult instead of raw Ultralytics objects.
    This decouples downstream code from the Ultralytics API.
  - GOOD/BAD decision is encapsulated here with configurable thresholds.
  - GPU inference when available; CPU fallback transparent to caller.

Usage:
    detector = DefectDetector()
    result = detector.predict("path/to/image.jpg")
    print(result.status)      # "GOOD" or "BAD"
    print(result.defects)     # list of Defect objects
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from src.utils.config import load_config
from src.utils.device import (
    clear_gpu_cache,
    detect_hardware,
    log_gpu_memory,
    select_device,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Defect:
    """Single detected defect."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]   # x1, y1, x2, y2 (pixel coords)
    bbox_xywhn: tuple[float, float, float, float]  # normalized cx, cy, w, h

    @property
    def area_pixels(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class DetectionResult:
    """
    Complete inference result for one image.

    Attributes:
        image_path:  Source image path (or "live_frame" for camera).
        status:      "GOOD" or "BAD".
        defects:     List of detected defects above confidence threshold.
        image_shape: (height, width, channels) of the input image.
        model_path:  Which model weights were used.
        device_used: "cuda:0" or "cpu".
        inference_ms: Time taken for inference in milliseconds.
    """
    image_path: str
    status: str                               # "GOOD" | "BAD"
    defects: list[Defect] = field(default_factory=list)
    image_shape: tuple[int, int, int] = (0, 0, 3)
    model_path: str = ""
    device_used: str = ""
    inference_ms: float = 0.0

    @property
    def defect_count(self) -> int:
        return len(self.defects)

    @property
    def is_good(self) -> bool:
        return self.status == "GOOD"

    @property
    def max_confidence(self) -> float:
        return max((d.confidence for d in self.defects), default=0.0)

    @property
    def dominant_defect(self) -> Optional[Defect]:
        return max(self.defects, key=lambda d: d.confidence, default=None)

    def to_dict(self) -> dict:
        """Serialize to plain dict (for JSON / MongoDB)."""
        return {
            "image_path": self.image_path,
            "status": self.status,
            "defect_count": self.defect_count,
            "max_confidence": round(self.max_confidence, 4),
            "dominant_defect": self.dominant_defect.class_name if self.dominant_defect else None,
            "device_used": self.device_used,
            "inference_ms": round(self.inference_ms, 1),
            "defects": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox_xyxy": [round(v, 1) for v in d.bbox_xyxy],
                }
                for d in self.defects
            ],
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DefectDetector:
    """
    Metal defect detector using YOLOv8.

    The model is loaded once on construction and reused.
    This class is NOT thread-safe by default — for multi-threaded use,
    create one instance per thread.

    Args:
        model_path: Path to .pt weights. Defaults to config value.
        config:     Loaded config dict. Loaded automatically if None.
        device:     Override device string. Defaults to config/auto.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: dict | None = None,
        device: str | None = None,
    ):
        self.cfg = config or load_config()
        self.inf_cfg = self.cfg["inference"]

        # Resolve model path
        _model_path = model_path or self.inf_cfg.get("model_path", "")
        self.model_path = Path(_model_path)

        # Resolve device
        hw = detect_hardware()
        device_pref = device or self.inf_cfg.get("device", "auto")
        self.device = select_device(device_pref)

        # Thresholds from config
        self.conf_threshold: float = float(self.inf_cfg.get("confidence_threshold", 0.45))
        self.iou_threshold: float = float(self.inf_cfg.get("iou_threshold", 0.45))
        self.bad_threshold: float = float(self.inf_cfg.get("bad_threshold", 0.45))
        self.max_allowed_defects: int = int(self.inf_cfg.get("max_allowed_defects", 0))
        self.image_size: int = int(self.inf_cfg.get("image_size", 640))
        self.max_det: int = int(self.inf_cfg.get("max_detections", 50))

        # Class names — from config, may be overridden by model metadata
        self.class_names: list[str] = self.cfg.get("classes", {}).get("names", [])

        # Model — loaded lazily on first predict() call
        self._model = None

        logger.info(
            "DefectDetector initialized | device=%s | conf=%.2f | bad_threshold=%.2f",
            self.device, self.conf_threshold, self.bad_threshold,
        )

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Explicitly load model into memory.
        Called automatically by predict() if not already loaded.
        """
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.model_path}\n"
                "Train the model first (run scripts/train.py) or update "
                "inference.model_path in config.yaml."
            )

        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError("Ultralytics not installed.") from e

        logger.info("Loading model: %s", self.model_path)
        log_gpu_memory("before_model_load")

        self._model = YOLO(str(self.model_path))

        # Update class names from model if available
        if hasattr(self._model, "names") and self._model.names:
            self.class_names = list(self._model.names.values())
            logger.info("Class names from model: %s", self.class_names)

        log_gpu_memory("after_model_load")
        logger.info("Model loaded successfully. Classes: %d", len(self.class_names))

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        save_annotated: bool = False,
        output_path: Optional[Path] = None,
    ) -> DetectionResult:
        """
        Run defect detection on an image.

        Args:
            source:          Image path (str/Path) or numpy array (BGR, HWC).
            save_annotated:  Save annotated image with bounding boxes.
            output_path:     Where to save annotated image (optional).

        Returns:
            DetectionResult with status, defects, and metadata.
        """
        self.load_model()

        # Resolve image path string for logging
        if isinstance(source, (str, Path)):
            image_path_str = str(source)
        else:
            image_path_str = "live_frame"

        logger.debug("Running inference on: %s", image_path_str)

        # Get image shape
        try:
            if isinstance(source, np.ndarray):
                img_shape = source.shape
            else:
                img = cv2.imread(str(source))
                if img is None:
                    raise ValueError(f"Could not read image: {source}")
                img_shape = img.shape
        except Exception as e:
            logger.error("Failed to read image: %s", e)
            raise

        # Run inference
        import time
        start = time.perf_counter()

        try:
            raw_results = self._model.predict(
                source=source,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                device=self.device,
                max_det=self.max_det,
                verbose=False,
                save=save_annotated,
                project=str(output_path.parent) if output_path else None,
                name=output_path.stem if output_path else None,
            )
        except Exception as e:
            logger.exception("Inference failed for %s: %s", image_path_str, e)
            raise

        inference_ms = (time.perf_counter() - start) * 1000

        # Parse results
        defects = self._parse_results(raw_results)

        # GOOD / BAD decision
        status = self._decide_status(defects)

        result = DetectionResult(
            image_path=image_path_str,
            status=status,
            defects=defects,
            image_shape=img_shape,
            model_path=str(self.model_path),
            device_used=self.device,
            inference_ms=inference_ms,
        )

        logger.info(
            "Inference done | status=%-4s | defects=%d | max_conf=%.3f | %.1f ms",
            result.status, result.defect_count, result.max_confidence, inference_ms,
        )
        return result

    def predict_batch(
        self,
        sources: list[Union[str, Path]],
    ) -> list[DetectionResult]:
        """
        Run inference on a list of images.

        Args:
            sources: List of image paths.

        Returns:
            List of DetectionResult (same order as input).
        """
        results = []
        for i, src in enumerate(sources, 1):
            logger.debug("Batch inference [%d/%d]: %s", i, len(sources), src)
            try:
                results.append(self.predict(src))
            except Exception as e:
                logger.error("Skipping %s due to error: %s", src, e)
        return results

    def unload_model(self) -> None:
        """Release model from memory and clear GPU cache."""
        if self._model is not None:
            del self._model
            self._model = None
            clear_gpu_cache()
            logger.info("Model unloaded and GPU cache cleared.")

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _parse_results(self, raw_results: list) -> list[Defect]:
        """Convert Ultralytics result objects to our Defect data class."""
        defects: list[Defect] = []

        for result in raw_results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Get normalized xywh
                if box.xywhn is not None:
                    cx, cy, w, h = box.xywhn[0].tolist()
                else:
                    img_h, img_w = result.orig_shape
                    cx = ((x1 + x2) / 2) / img_w
                    cy = ((y1 + y2) / 2) / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h

                class_name = (
                    self.class_names[cls_id]
                    if cls_id < len(self.class_names)
                    else f"class_{cls_id}"
                )

                defects.append(Defect(
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_xywhn=(cx, cy, w, h),
                ))

        return defects

    def _decide_status(self, defects: list[Defect]) -> str:
        """
        Determine GOOD or BAD based on business rules.

        Rules (checked in order):
          1. If zero defects detected → GOOD
          2. If any defect has confidence >= bad_threshold → BAD
          3. If defect count > max_allowed_defects → BAD
          4. Otherwise → GOOD
        """
        if not defects:
            return "GOOD"

        # Rule: any high-confidence defect makes it BAD
        if any(d.confidence >= self.bad_threshold for d in defects):
            return "BAD"

        # Rule: too many defects
        if len(defects) > self.max_allowed_defects:
            return "BAD"

        return "GOOD"