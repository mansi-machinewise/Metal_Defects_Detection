"""
api/services/inference_service.py
===================================
Inference service layer.

Wraps the existing DefectDetector from Phase 1 for use inside FastAPI.

Design decisions:
  - Model is loaded ONCE when the service is first used (lazy singleton).
    FastAPI's lifespan event triggers this at server startup so the first
    request doesn't pay the model-load penalty.
  - Accepts raw image bytes (from the HTTP upload) and returns a structured
    InspectionResult — no FastAPI/HTTP concerns leak into this layer.
  - Converts the annotated OpenCV image to a base64 PNG so the frontend
    can display it directly in an <img> tag without a second HTTP request.
  - Thread-safe for single-worker use. If you later run multiple uvicorn
    workers, each worker process gets its own model instance automatically.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.inference.detector import DefectDetector, DetectionResult
from src.inference.visualizer import DefectVisualizer
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result schema returned to the API layer
# ---------------------------------------------------------------------------

@dataclass
class DefectDetail:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]   # [x1, y1, x2, y2] pixel coords


@dataclass
class InspectionResult:
    """
    Structured result returned from InferenceService.inspect().
    The API route converts this to a JSON response.
    """
    status: str                                  # "GOOD" or "BAD"
    confidence: float                            # highest defect confidence (0–1)
    defect_type: str                             # dominant defect class name
    defect_count: int
    defects: list[DefectDetail] = field(default_factory=list)
    annotated_image_b64: str = ""               # base64-encoded PNG of annotated image
    inference_ms: float = 0.0
    model_path: str = ""
    device_used: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "confidence": round(self.confidence * 100, 2),   # percentage for frontend
            "defect_type": self.defect_type,
            "defect_count": self.defect_count,
            "defects": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": round(d.confidence * 100, 2),
                    "bbox": d.bbox,
                }
                for d in self.defects
            ],
            "annotated_image": self.annotated_image_b64,
            "inference_ms": round(self.inference_ms, 1),
            "model_path": self.model_path,
            "device_used": self.device_used,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class InferenceService:
    """
    Singleton inference service — holds one loaded DefectDetector instance
    for the lifetime of the FastAPI server process.
    """

    _instance: Optional["InferenceService"] = None

    def __init__(self):
        self.cfg = load_config()
        self._detector: Optional[DefectDetector] = None
        self._visualizer = DefectVisualizer()
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "InferenceService":
        """Return the singleton instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the model. Called once at server startup."""
        if self._loaded:
            return
        logger.info("InferenceService: loading model ...")
        self._detector = DefectDetector(config=self.cfg)
        self._detector.load_model()
        self._loaded = True
        logger.info("InferenceService: model ready.")

    def unload(self) -> None:
        """Release model from memory. Called at server shutdown."""
        if self._detector:
            self._detector.unload_model()
            self._detector = None
        self._loaded = False
        logger.info("InferenceService: model unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Core Inference
    # ------------------------------------------------------------------

    def inspect(self, image_bytes: bytes, filename: str = "upload") -> InspectionResult:
        """
        Run defect detection on raw image bytes.

        Args:
            image_bytes: Raw bytes of the uploaded image file.
            filename:    Original filename (for logging only).

        Returns:
            InspectionResult with status, defects, and annotated image.

        Raises:
            RuntimeError: If the model is not loaded or image is invalid.
        """
        if not self._loaded or self._detector is None:
            raise RuntimeError(
                "Model not loaded. Call InferenceService.load() at server startup."
            )

        # Decode bytes → numpy BGR array (OpenCV format)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ValueError(
                f"Could not decode image: {filename}. "
                "Ensure the file is a valid JPG, PNG, or BMP."
            )

        logger.info("Inspecting: %s (%dx%d)", filename, img_bgr.shape[1], img_bgr.shape[0])

        # Run detection
        detection: DetectionResult = self._detector.predict(img_bgr)

        # Draw annotations onto a copy of the image
        annotated_bgr = self._visualizer.draw(img_bgr, detection)

        # Encode annotated image to base64 PNG for the frontend
        annotated_b64 = self._encode_image_b64(annotated_bgr)

        # Build result
        defect_details = [
            DefectDetail(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=list(d.bbox_xyxy),
            )
            for d in detection.defects
        ]

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        result = InspectionResult(
            status=detection.status,
            confidence=1.0 if detection.status == "GOOD" and detection.max_confidence == 0 else detection.max_confidence,
            defect_type=detection.dominant_defect.class_name if detection.dominant_defect else "No Defect",
            defect_count=detection.defect_count,
            defects=defect_details,
            annotated_image_b64=annotated_b64,
            inference_ms=detection.inference_ms,
            model_path=str(self._detector.model_path),
            device_used=self._detector.device,
            timestamp=timestamp,
        )

        logger.info(
            "Result: status=%s | defects=%d | confidence=%.2f%% | %.1fms",
            result.status, result.defect_count,
            result.confidence * 100, result.inference_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_image_b64(self, img_bgr: np.ndarray) -> str:
        """Encode a BGR OpenCV image to a base64 PNG data URL."""
        # Convert BGR → RGB for PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)

        b64_str = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"