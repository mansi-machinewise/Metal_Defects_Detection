"""
src/training/trainer.py
========================
YOLOv8 training module.

Responsibilities:
  - Hardware detection before training
  - Recommend and apply memory-safe hyperparameters
  - Configure and launch Ultralytics YOLOv8 training
  - Monitor GPU/RAM during training
  - Save and return path to best model
  - Generate post-training evaluation results

Design decisions:
  - Wraps Ultralytics YOLO — never reimplements what Ultralytics provides
  - Training config driven entirely by config.yaml / .env
  - AMP forced ON for 4 GB GPU — disabling it will likely cause OOM
  - cache=False by default — avoids loading entire dataset into RAM
  - Singleton model load — model loaded once, not per-call
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from src.utils.config import load_config
from src.utils.device import (
    HardwareInfo,
    clear_gpu_cache,
    detect_hardware,
    log_gpu_memory,
    log_hardware_info,
    log_ram_usage,
    recommend_training_config,
    select_device,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DefectModelTrainer:
    """
    Trains a YOLOv8 model for metal defect detection.

    Args:
        config: Loaded config dict. Loaded automatically if None.
        override_params: Dict of training params that override config.yaml.
                         Useful for CLI arguments or sweep configs.

    Example:
        trainer = DefectModelTrainer()
        result = trainer.train()
        best_model_path = result.best_model_path
    """

    def __init__(
        self,
        config: dict | None = None,
        override_params: dict | None = None,
    ):
        self.cfg = config or load_config()
        self.train_cfg = self.cfg["training"]
        self.dataset_yaml = Path(self.cfg["dataset"]["yaml"])
        self.override_params = override_params or {}

        # Hardware — detected once, referenced throughout
        self.hw: HardwareInfo = detect_hardware()
        self.device: str = ""

        # Will be set after training completes
        self.best_model_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def train(self) -> "TrainingResult":
        """
        Execute the full training pipeline.

        Returns:
            TrainingResult with paths to artifacts and summary metrics.

        Raises:
            FileNotFoundError: If data.yaml or base model not found.
            RuntimeError:      If training fails unexpectedly.
        """
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE STARTED")
        logger.info("=" * 60)

        # 1. Log hardware
        log_hardware_info(self.hw)

        # 2. Resolve device
        device_pref = self.override_params.get(
            "device", self.train_cfg.get("device", "auto")
        )
        self.device = select_device(device_pref)

        # 3. Build training parameters
        params = self._build_training_params()

        # 4. Validate dataset yaml exists
        if not self.dataset_yaml.exists():
            raise FileNotFoundError(
                f"Dataset YAML not found: {self.dataset_yaml}\n"
                "Ensure the dataset path is correct in config.yaml."
            )

        # 5. Log final config
        self._log_training_config(params)

        # 6. Import YOLO here to defer heavy import (allows unit testing without torch)
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as e:
            raise RuntimeError(
                "Ultralytics not installed. Run: pip install ultralytics"
            ) from e

        # 7. Load base model
        model_name = self.train_cfg["model"]
        logger.info("Loading base model: %s", model_name)
        clear_gpu_cache()
        log_gpu_memory("before_model_load")
        log_ram_usage("before_model_load")

        try:
            model = YOLO(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e

        log_gpu_memory("after_model_load")

        # 8. Train
        logger.info("Starting training — this may take a while ...")
        start_time = time.time()

        try:
            results = model.train(**params)
        except Exception as e:
            logger.exception("Training failed: %s", e)
            raise RuntimeError(f"Training failed: {e}") from e

        elapsed = time.time() - start_time
        logger.info("Training completed in %.1f minutes.", elapsed / 60)

        # 9. Locate best model
        run_dir = Path(params["project"]) / params["name"]
        best_weights = run_dir / "weights" / "best.pt"
        last_weights = run_dir / "weights" / "last.pt"

        if best_weights.exists():
            self.best_model_path = best_weights
            logger.info("Best model saved to: %s", best_weights)
        elif last_weights.exists():
            self.best_model_path = last_weights
            logger.warning(
                "best.pt not found. Using last.pt: %s", last_weights
            )
        else:
            logger.error("No model weights found in: %s", run_dir / "weights")

        log_gpu_memory("after_training")
        clear_gpu_cache()

        return TrainingResult(
            run_dir=run_dir,
            best_model_path=self.best_model_path,
            elapsed_seconds=elapsed,
            device_used=self.device,
            ultralytics_results=results,
        )

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _build_training_params(self) -> dict:
        """
        Build the full dict of params to pass to model.train().

        Priority (highest → lowest):
          1. self.override_params (CLI / test overrides)
          2. Hardware recommendations
          3. config.yaml training section
        """
        hw_recs = recommend_training_config(self.hw)
        logger.info("Hardware recommendations: %s", hw_recs)

        # Base from config
        aug = self.train_cfg.get("augmentation", {})
        params: dict = {
            "data":         str(self.dataset_yaml),
            "epochs":       self.train_cfg["epochs"],
            "batch":        self.train_cfg["batch_size"],
            "imgsz":        self.train_cfg["image_size"],
            "workers":      self.train_cfg["workers"],
            "amp":          self.train_cfg["amp"],
            "cache":        self.train_cfg["cache"],
            "patience":     self.train_cfg["patience"],
            "save_period":  self.train_cfg["save_period"],
            "exist_ok":     self.train_cfg["exist_ok"],
            "project":      self.train_cfg["project"],
            "name":         self.train_cfg["run_name"],
            "device":       self.device,
            "optimizer":    self.train_cfg["optimizer"],
            "lr0":          self.train_cfg["lr0"],
            "lrf":          self.train_cfg["lrf"],
            "momentum":     self.train_cfg["momentum"],
            "weight_decay": self.train_cfg["weight_decay"],
            "warmup_epochs":self.train_cfg["warmup_epochs"],
            # Augmentation
            "hsv_h":    aug.get("hsv_h", 0.015),
            "hsv_s":    aug.get("hsv_s", 0.7),
            "hsv_v":    aug.get("hsv_v", 0.4),
            "degrees":  aug.get("degrees", 10),
            "translate":aug.get("translate", 0.1),
            "scale":    aug.get("scale", 0.5),
            "flipud":   aug.get("flipud", 0.0),
            "fliplr":   aug.get("fliplr", 0.5),
            "mosaic":   aug.get("mosaic", 1.0),
            "mixup":    aug.get("mixup", 0.0),
            "verbose":  True,
        }

        # Apply hardware recommendations (only when safer than config)
        if hw_recs["batch_size"] < params["batch"]:
            logger.warning(
                "Reducing batch_size from %d → %d based on available VRAM.",
                params["batch"], hw_recs["batch_size"],
            )
            params["batch"] = hw_recs["batch_size"]

        if hw_recs["image_size"] < params["imgsz"]:
            logger.warning(
                "Reducing imgsz from %d → %d based on available VRAM.",
                params["imgsz"], hw_recs["image_size"],
            )
            params["imgsz"] = hw_recs["image_size"]

        # AMP must be disabled on CPU
        if self.device == "cpu":
            params["amp"] = False
            logger.info("AMP disabled (CPU training).")

        # Apply manual overrides last (highest priority)
        params.update(self.override_params)

        return params

    def _log_training_config(self, params: dict) -> None:
        """Log the final training configuration."""
        logger.info("-" * 60)
        logger.info("TRAINING CONFIGURATION")
        logger.info("-" * 60)
        key_params = [
            "data", "epochs", "batch", "imgsz", "workers",
            "amp", "cache", "device", "optimizer", "lr0",
            "patience", "project", "name",
        ]
        for k in key_params:
            logger.info("  %-15s : %s", k, params.get(k, "N/A"))
        logger.info("-" * 60)


# ---------------------------------------------------------------------------
# Result Container
# ---------------------------------------------------------------------------

class TrainingResult:
    """Holds training outputs and provides convenient accessors."""

    def __init__(
        self,
        run_dir: Path,
        best_model_path: Optional[Path],
        elapsed_seconds: float,
        device_used: str,
        ultralytics_results: object,
    ):
        self.run_dir = run_dir
        self.best_model_path = best_model_path
        self.elapsed_seconds = elapsed_seconds
        self.device_used = device_used
        self.ultralytics_results = ultralytics_results

    @property
    def weights_dir(self) -> Path:
        return self.run_dir / "weights"

    @property
    def elapsed_minutes(self) -> float:
        return round(self.elapsed_seconds / 60, 1)

    def __repr__(self) -> str:
        return (
            f"TrainingResult("
            f"best={self.best_model_path}, "
            f"elapsed={self.elapsed_minutes}min, "
            f"device={self.device_used})"
        )