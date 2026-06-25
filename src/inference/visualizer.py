"""
src/inference/visualizer.py
============================
Visualization module for detection results.

Responsibilities:
  - Draw bounding boxes with class labels and confidence scores
  - Color-coded by defect type
  - GOOD/BAD status banner overlay
  - Save annotated images
  - Generate grid summaries for batch results

Design decisions:
  - Pure OpenCV/NumPy — no matplotlib dependency for speed
  - Decoupled from detector — takes DetectionResult, not model outputs
  - Configurable colors and text size
  - Works on any BGR numpy array (from file or camera frame)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.inference.detector import DetectionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Color Palette — BGR format (OpenCV convention)
# ---------------------------------------------------------------------------

DEFECT_COLORS = [
    (0, 165, 255),
    (0, 0, 255),
    (0, 255, 255),
    (180, 105, 255),
    (255, 0, 255),
    (128, 0, 128),
    (0, 128, 255),
]

STATUS_COLORS = {
    "GOOD": (0, 200, 0),
    "BAD":  (0, 0, 220),
}


class DefectVisualizer:
    """
    Draws detection results onto images.

    Args:
        font_scale: OpenCV font scale for labels.
        thickness:  Line thickness for boxes.
        alpha:      Transparency for status banner (0.0-1.0).
    """

    def __init__(
        self,
        font_scale: float = 0.6,
        thickness: int = 2,
        alpha: float = 0.4,
    ):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = thickness
        self.alpha = alpha

    def draw(
        self,
        image: np.ndarray,
        result: DetectionResult,
        show_status_banner: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and status on an image.

        Args:
            image:              BGR numpy array (will NOT be modified in-place).
            result:             DetectionResult from DefectDetector.
            show_status_banner: Overlay GOOD/BAD banner at top.

        Returns:
            New BGR numpy array with annotations drawn.
        """
        canvas = image.copy()

        for defect in result.defects:
            self._draw_box(canvas, defect)

        if show_status_banner:
            self._draw_status_banner(canvas, result)

        self._draw_info_text(canvas, result)
        return canvas

    def draw_and_save(
        self,
        image_path: "Path | str",
        result: DetectionResult,
        output_path: "Path | str",
        show_status_banner: bool = True,
    ) -> Path:
        """Load image, draw result, save annotated image."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        annotated = self.draw(img, result, show_status_banner)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        logger.debug("Annotated image saved: %s", output_path)
        return output_path

    def create_summary_grid(
        self,
        results: "list[tuple[np.ndarray, DetectionResult]]",
        cols: int = 3,
        cell_size: "tuple[int, int]" = (320, 240),
    ) -> np.ndarray:
        """Create a grid image summarizing multiple detections."""
        if not results:
            return np.zeros((240, 320, 3), dtype=np.uint8)

        cells = []
        for img, result in results:
            annotated = self.draw(img, result, show_status_banner=True)
            cell = cv2.resize(annotated, cell_size)
            cells.append(cell)

        rows = math.ceil(len(cells) / cols)
        cell_w, cell_h = cell_size

        while len(cells) < rows * cols:
            cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

        grid_rows = []
        for r in range(rows):
            row_cells = cells[r * cols : (r + 1) * cols]
            grid_rows.append(np.hstack(row_cells))

        return np.vstack(grid_rows)

    def _draw_box(self, canvas: np.ndarray, defect) -> None:
        x1, y1, x2, y2 = [int(v) for v in defect.bbox_xyxy]
        color = DEFECT_COLORS[defect.class_id % len(DEFECT_COLORS)]

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, self.thickness)

        label = f"{defect.class_name} {defect.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)

        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(
            canvas,
            (x1, label_y - th - baseline - 4),
            (x1 + tw + 4, label_y),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            canvas, label,
            (x1 + 2, label_y - baseline - 2),
            self.font, self.font_scale,
            (255, 255, 255),
            self.thickness, cv2.LINE_AA,
        )

    def _draw_status_banner(self, canvas: np.ndarray, result: DetectionResult) -> None:
        h, w = canvas.shape[:2]
        banner_h = max(40, int(h * 0.08))
        color = STATUS_COLORS.get(result.status, (128, 128, 128))

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), color, cv2.FILLED)
        cv2.addWeighted(overlay, self.alpha, canvas, 1 - self.alpha, 0, canvas)

        text = f"  {result.status}  |  Defects: {result.defect_count}"
        if result.dominant_defect:
            text += f"  |  {result.dominant_defect.class_name} ({result.max_confidence:.2f})"

        font_scale = max(0.5, banner_h / 60)
        cv2.putText(
            canvas, text,
            (8, banner_h - 10),
            self.font, font_scale,
            (255, 255, 255),
            2, cv2.LINE_AA,
        )

    def _draw_info_text(self, canvas: np.ndarray, result: DetectionResult) -> None:
        h = canvas.shape[0]
        text = f"{result.inference_ms:.0f}ms | {result.device_used}"
        cv2.putText(
            canvas, text,
            (8, h - 10),
            self.font, 0.45,
            (200, 200, 200),
            1, cv2.LINE_AA,
        )