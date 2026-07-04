"""
src/camera/stream.py
=====================
Camera frame capture module.

Design decisions:
  - Capture runs on a dedicated background thread. OpenCV's VideoCapture.read()
    blocks until a new frame arrives; if inference happens on the same thread,
    the displayed feed stutters every time a frame takes longer to process
    than the camera's frame interval. Decoupling capture from processing keeps
    the feed smooth and lets inference run at whatever pace the hardware allows.
  - Always exposes only the LATEST frame (not a queue of all frames). For a
    real-time inspection feed, processing stale frames is wasted work — we
    want the newest frame, always.
  - Source is generic: an integer (webcam index) or a string (RTSP/HTTP URL
    for an IP camera). No code changes needed to switch between them later.
  - Resource cleanup is explicit and idempotent — release() is safe to call
    multiple times, and stop() always releases the camera.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CameraConfig:
    """Configuration for a camera source."""
    source: Union[int, str] = 0          # 0 = default webcam, or RTSP/HTTP URL string
    width: int = 1280
    height: int = 720
    fps_target: int = 30
    buffer_size: int = 1                 # Keep OpenCV's internal buffer small to reduce latency
    reconnect_delay_sec: float = 2.0     # Wait time before retrying a dropped IP camera connection
    max_reconnect_attempts: int = 5


class CameraStream:
    """
    Threaded camera frame reader.

    Usage:
        cam = CameraStream(CameraConfig(source=0))
        cam.start()
        while True:
            frame = cam.read()
            if frame is None:
                continue
            ...
        cam.stop()

    Or as a context manager:
        with CameraStream(CameraConfig(source=0)) as cam:
            frame = cam.read()
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running = False
        self._frame_count = 0
        self._dropped_count = 0
        self._last_fps_check = time.time()
        self._fps_frame_count = 0
        self._current_fps = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "CameraStream":
        """Open the camera and start the background capture thread."""
        if self._running:
            logger.warning("CameraStream already running.")
            return self

        self._open_capture()

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        logger.info(
            "Camera stream started | source=%s | target=%dx%d @ %dfps",
            self.config.source, self.config.width, self.config.height, self.config.fps_target,
        )
        return self

    def stop(self) -> None:
        """Stop the capture thread and release the camera. Safe to call multiple times."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._release_capture()
        logger.info(
            "Camera stream stopped | frames captured=%d | frames dropped=%d",
            self._frame_count, self._dropped_count,
        )

    def __enter__(self) -> "CameraStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def read(self) -> Optional[np.ndarray]:
        """
        Get the most recently captured frame.

        Returns:
            BGR numpy array, or None if no frame is available yet.
        """
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        """Current measured capture FPS (updated roughly once per second)."""
        return round(self._current_fps, 1)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    # ------------------------------------------------------------------
    # Internal — Capture Thread
    # ------------------------------------------------------------------

    def _open_capture(self) -> None:
        """Open the VideoCapture device with configured properties."""
        cap = cv2.VideoCapture(self.config.source)

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera source: {self.config.source}\n"
                "Check that the camera is connected and not in use by another application."
            )

        # Apply settings — not all backends honor every property, hence the checks.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps_target)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            "Camera opened | requested=%dx%d@%d | actual=%dx%d@%.0f",
            self.config.width, self.config.height, self.config.fps_target,
            actual_w, actual_h, actual_fps,
        )

        self._cap = cap

    def _release_capture(self) -> None:
        """Release the OpenCV VideoCapture handle. Idempotent."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _capture_loop(self) -> None:
        """
        Background thread loop — continuously reads frames and stores
        only the latest one. Handles reconnection for IP camera sources.
        """
        reconnect_attempts = 0

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if reconnect_attempts >= self.config.max_reconnect_attempts:
                    logger.error(
                        "Max reconnect attempts (%d) reached. Stopping capture.",
                        self.config.max_reconnect_attempts,
                    )
                    self._running = False
                    break

                reconnect_attempts += 1
                logger.warning(
                    "Camera disconnected. Reconnect attempt %d/%d in %.1fs ...",
                    reconnect_attempts, self.config.max_reconnect_attempts,
                    self.config.reconnect_delay_sec,
                )
                time.sleep(self.config.reconnect_delay_sec)
                try:
                    self._open_capture()
                    reconnect_attempts = 0
                except RuntimeError as e:
                    logger.error("Reconnect failed: %s", e)
                continue

            ret, frame = self._cap.read()

            if not ret or frame is None:
                self._dropped_count += 1
                logger.debug("Frame read failed (dropped=%d).", self._dropped_count)
                # For webcams this is usually transient; for IP cameras it may
                # indicate a dropped connection — let the reconnect logic above handle it.
                time.sleep(0.01)
                continue

            with self._lock:
                self._latest_frame = frame
            self._frame_count += 1
            self._update_fps()

        self._release_capture()

    def _update_fps(self) -> None:
        """Update the rolling FPS measurement, refreshed roughly once per second."""
        self._fps_frame_count += 1
        now = time.time()
        elapsed = now - self._last_fps_check
        if elapsed >= 1.0:
            self._current_fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._last_fps_check = now