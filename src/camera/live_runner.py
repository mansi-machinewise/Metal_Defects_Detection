"""
src/camera/live_runner.py
===========================
Live camera defect detection runner.

Ties together CameraStream, DefectDetector, and DefectVisualizer into the
Phase 2 workflow:

    Camera → Frame Capture → Preprocessing → YOLOv8 Detection
    → GOOD/BAD Decision → Live Visualization

Design decisions:
  - Reuses DefectDetector and DefectVisualizer from Phase 1 unchanged —
    camera integration adds a capture/display loop around them, it does
    not duplicate inference or drawing logic.
  - Inference throttling (inference_every_n_frames) keeps the displayed
    feed responsive even when YOLOv8 inference can't keep up with the
    camera's native frame rate (especially relevant on CPU-only setups).
  - All resources (camera, OpenCV windows) are released in a `finally`
    block so Ctrl+C or unexpected errors never leave the camera locked.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2

from src.camera.stream import CameraConfig, CameraStream
from src.inference.detector import DefectDetector, DetectionResult
from src.inference.visualizer import DefectVisualizer
from src.utils.config import load_config
from src.utils.device import clear_gpu_cache, log_ram_usage
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LiveDetectionRunner:
    """
    Runs continuous defect detection on a live camera feed and displays
    the annotated result in an OpenCV window.

    Args:
        config:   Loaded config dict. Loaded automatically if None.
        detector: A pre-initialized DefectDetector. Created automatically if None
                  (useful to pass one in if the model is already loaded elsewhere).
    """

    def __init__(
        self,
        config: dict | None = None,
        detector: Optional[DefectDetector] = None,
    ):
        self.cfg = config or load_config()
        self.cam_cfg = self.cfg["camera"]

        self.detector = detector or DefectDetector(config=self.cfg)
        self.visualizer = DefectVisualizer()

        self.camera_config = CameraConfig(
            source=self.cam_cfg.get("source", 0),
            width=self.cam_cfg.get("width", 1280),
            height=self.cam_cfg.get("height", 720),
            fps_target=self.cam_cfg.get("fps_target", 30),
            buffer_size=self.cam_cfg.get("buffer_size", 1),
            reconnect_delay_sec=self.cam_cfg.get("reconnect_delay_sec", 2.0),
            max_reconnect_attempts=self.cam_cfg.get("max_reconnect_attempts", 5),
        )

        self.inference_every_n = max(1, int(self.cam_cfg.get("inference_every_n_frames", 1)))
        self.window_name = self.cam_cfg.get("window_name", "Metal Defect Detection — Live Feed")
        self.display_width = int(self.cam_cfg.get("display_width", 960))
        self.quit_key = self.cam_cfg.get("quit_key", "q")
        self.save_key = self.cam_cfg.get("save_key", "s")

        self.save_dir = Path(self.cfg["evaluation"]["predictions_dir"]) / "live_captures"

        # Running stats
        self._frames_processed = 0
        self._defects_seen = 0
        self._bad_count = 0
        self._good_count = 0
        self._last_result: Optional[DetectionResult] = None

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the live detection loop. Blocks until the user quits
        (press 'q' in the window) or the camera disconnects permanently.
        """
        logger.info("=" * 60)
        logger.info("LIVE CAMERA DETECTION — STARTING")
        logger.info("=" * 60)
        logger.info("  Source            : %s", self.camera_config.source)
        logger.info("  Resolution        : %dx%d", self.camera_config.width, self.camera_config.height)
        logger.info("  Inference every   : %d frame(s)", self.inference_every_n)
        logger.info("  Controls          : '%s' = quit | '%s' = save frame", self.quit_key, self.save_key)
        logger.info("=" * 60)

        # Load model once before entering the loop
        self.detector.load_model()

        camera = CameraStream(self.camera_config)

        try:
            camera.start()
            self._wait_for_first_frame(camera)
            self._loop(camera)
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C).")
        except Exception as e:
            logger.exception("Live detection loop failed: %s", e)
            raise
        finally:
            self._cleanup(camera)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _wait_for_first_frame(self, camera: CameraStream, timeout_sec: float = 10.0) -> None:
        """Block briefly until the camera produces its first frame."""
        start = time.time()
        while camera.read() is None:
            if time.time() - start > timeout_sec:
                raise RuntimeError(
                    f"No frame received from camera within {timeout_sec}s. "
                    "Check the camera connection and source configuration."
                )
            time.sleep(0.1)
        logger.info("First frame received. Starting display loop.")

    def _loop(self, camera: CameraStream) -> None:
        """Main capture → detect → display loop."""
        frame_index = 0
        loop_start = time.time()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while camera.is_running():
            frame = camera.read()
            if frame is None:
                continue

            frame_index += 1
            run_inference = (frame_index % self.inference_every_n == 0)

            if run_inference:
                try:
                    result = self.detector.predict(frame)
                    self._last_result = result
                    self._update_stats(result)
                except Exception as e:
                    logger.error("Inference failed on frame %d: %s", frame_index, e)
                    continue

            # Always draw using the most recent result, even on throttled
            # frames — keeps boxes visible between inference calls instead
            # of flickering off every other frame.
            if self._last_result is not None:
                display_frame = self.visualizer.draw(frame, self._last_result)
            else:
                display_frame = frame

            self._draw_overlay(display_frame, camera)

            display_frame = self._resize_for_display(display_frame)
            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(self.quit_key):
                logger.info("Quit key pressed. Stopping.")
                break
            elif key == ord(self.save_key):
                self._save_current_frame(display_frame)

            self._frames_processed += 1

        elapsed = time.time() - loop_start
        logger.info(
            "Session summary | frames_processed=%d | GOOD=%d | BAD=%d | "
            "defects_seen=%d | duration=%.1fs | avg_fps=%.1f",
            self._frames_processed, self._good_count, self._bad_count,
            self._defects_seen, elapsed,
            self._frames_processed / elapsed if elapsed > 0 else 0,
        )

    def _update_stats(self, result: DetectionResult) -> None:
        """Track running session statistics."""
        if result.status == "GOOD":
            self._good_count += 1
        else:
            self._bad_count += 1
        self._defects_seen += result.defect_count

    def _draw_overlay(self, frame, camera: CameraStream) -> None:
        """Draw camera FPS and session stats onto the frame (separate from detection visuals)."""
        h, w = frame.shape[:2]
        text = f"Camera FPS: {camera.fps:.1f} | GOOD: {self._good_count} | BAD: {self._bad_count}"
        cv2.putText(
            frame, text,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (220, 220, 220), 1, cv2.LINE_AA,
        )

    def _resize_for_display(self, frame):
        """Resize the frame for display only — inference always used the full-res frame."""
        h, w = frame.shape[:2]
        if w <= self.display_width:
            return frame
        scale = self.display_width / w
        new_size = (self.display_width, int(h * scale))
        return cv2.resize(frame, new_size)

    def _save_current_frame(self, frame) -> None:
        """Save the currently displayed annotated frame to disk."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = self.save_dir / f"capture_{timestamp}.jpg"
        cv2.imwrite(str(out_path), frame)
        logger.info("Frame saved: %s", out_path)

    def _cleanup(self, camera: CameraStream) -> None:
        """Release all resources. Always called, even on error or interrupt."""
        logger.info("Cleaning up resources ...")
        camera.stop()
        cv2.destroyAllWindows()
        # On some platforms a single destroyAllWindows() call doesn't
        # immediately close the window; a short waitKey nudges the event loop.
        cv2.waitKey(1)
        self.detector.unload_model()
        clear_gpu_cache()
        log_ram_usage("after_live_session")
        logger.info("Cleanup complete.")