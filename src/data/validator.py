"""
src/data/validator.py
======================
Dataset validation module.

Validates a YOLO-format dataset before training begins.
Catches common problems early to avoid wasting GPU time on a corrupt dataset.

Checks performed:
  1. data.yaml exists and is valid
  2. Required splits (train/val/test) exist
  3. Minimum sample counts per split
  4. Every image has a corresponding label file
  5. Label files are valid YOLO format
  6. Class IDs are within range
  7. Bounding box values are within [0, 1]
  8. Images can be opened (not corrupt)
  9. Class distribution report
  10. Statistics summary

Usage:
    from src.data.validator import DatasetValidator
    validator = DatasetValidator(config)
    report = validator.validate()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import yaml

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class SplitStats:
    """Statistics for a single dataset split."""
    name: str
    image_count: int = 0
    label_count: int = 0
    missing_labels: list[str] = field(default_factory=list)
    corrupt_images: list[str] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    annotation_count: int = 0
    class_distribution: dict[int, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return (
            len(self.missing_labels) == 0
            and len(self.corrupt_images) == 0
            and len(self.invalid_labels) == 0
            and self.image_count > 0
        )


@dataclass
class ValidationReport:
    """Full dataset validation report."""
    dataset_root: str
    yaml_valid: bool = False
    num_classes: int = 0
    class_names: list[str] = field(default_factory=list)
    splits: dict[str, SplitStats] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (
            self.yaml_valid
            and len(self.errors) == 0
            and all(s.is_valid for s in self.splits.values() if s.image_count > 0)
        )

    @property
    def total_images(self) -> int:
        return sum(s.image_count for s in self.splits.values())


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DatasetValidator:
    """
    Validates a YOLO-format dataset.

    Args:
        config: Loaded config dict (from load_config()).
                If None, loads config automatically.
    """

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        self.dataset_cfg = self.cfg["dataset"]
        self.dataset_root = Path(self.dataset_cfg["root"])
        self.yaml_path = Path(self.dataset_cfg["yaml"])
        self.image_exts = set(self.dataset_cfg.get("image_extensions", [".jpg", ".jpeg", ".png"]))
        self.min_samples = self.dataset_cfg.get("min_samples", {})
        self._data_yaml: dict = {}

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def validate(self) -> ValidationReport:
        """
        Run full dataset validation.

        Returns:
            ValidationReport with detailed results and pass/fail status.
        """
        logger.info("=" * 60)
        logger.info("DATASET VALIDATION STARTED")
        logger.info("  Dataset Root : %s", self.dataset_root.resolve())
        logger.info("  YAML Path    : %s", self.yaml_path.resolve())
        logger.info("=" * 60)

        report = ValidationReport(dataset_root=str(self.dataset_root.resolve()))

        # Step 1: Validate YAML
        self._validate_yaml(report)
        if not report.yaml_valid:
            logger.error("YAML validation failed. Aborting further checks.")
            return report

        # Step 2: Validate each split
        for split_name in ["train", "val", "test"]:
            self._validate_split(report, split_name)

        # Step 3: Final summary
        self._log_report(report)
        return report

    # ------------------------------------------------------------------
    # YAML Validation
    # ------------------------------------------------------------------

    def _validate_yaml(self, report: ValidationReport) -> None:
        """Validate data.yaml structure and content."""
        logger.info("Checking data.yaml ...")

        if not self.yaml_path.exists():
            msg = f"data.yaml not found at: {self.yaml_path}"
            report.errors.append(msg)
            logger.error(msg)
            return

        try:
            with open(self.yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._data_yaml = data
        except yaml.YAMLError as e:
            msg = f"Failed to parse data.yaml: {e}"
            report.errors.append(msg)
            logger.error(msg)
            return

        # Required fields
        required_fields = ["nc", "names"]
        for field_name in required_fields:
            if field_name not in data:
                msg = f"data.yaml missing required field: '{field_name}'"
                report.errors.append(msg)
                logger.error(msg)

        # Validate class count consistency
        if "nc" in data and "names" in data:
            nc = data["nc"]
            names = data["names"]
            if isinstance(names, dict):
                names = list(names.values())
            if len(names) != nc:
                msg = (
                    f"data.yaml mismatch: nc={nc} but {len(names)} class names defined."
                )
                report.warnings.append(msg)
                logger.warning(msg)
            report.num_classes = nc
            report.class_names = names if isinstance(names, list) else list(names.values())

        # Check path fields exist
        for path_key in ["train", "val", "test"]:
            if path_key in data:
                p = Path(data[path_key])
                if not p.is_absolute():
                    # Relative to yaml location
                    p = self.yaml_path.parent / p
                if not p.exists():
                    msg = f"data.yaml path for '{path_key}' not found: {p}"
                    report.warnings.append(msg)
                    logger.warning(msg)
            # test split is optional — don't error
            elif path_key == "train" or path_key == "val":
                msg = f"data.yaml missing '{path_key}' path field."
                report.warnings.append(msg)
                logger.warning(msg)

        report.yaml_valid = len([e for e in report.errors if "data.yaml" in e]) == 0
        logger.info(
            "data.yaml: ✓  nc=%d  classes=%s",
            report.num_classes,
            report.class_names,
        )

    # ------------------------------------------------------------------
    # Split Validation
    # ------------------------------------------------------------------

    def _validate_split(self, report: ValidationReport, split_name: str) -> None:
        """Validate a single dataset split (train/val/test)."""
        # Resolve image directory
        images_dir = self._resolve_split_path(split_name)

        if images_dir is None or not images_dir.exists():
            msg = f"Split '{split_name}' image directory not found."
            if split_name == "test":
                report.warnings.append(msg)
                logger.warning(msg)
            else:
                report.errors.append(msg)
                logger.error(msg)
            return

        labels_dir = self._get_labels_dir(images_dir)
        stats = SplitStats(name=split_name)
        logger.info("Validating split '%s': %s", split_name, images_dir)

        # Collect images
        image_files = sorted([
            f for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.image_exts
        ])
        stats.image_count = len(image_files)

        if stats.image_count == 0:
            msg = f"Split '{split_name}': no images found in {images_dir}"
            report.errors.append(msg)
            logger.error(msg)
            report.splits[split_name] = stats
            return

        # Check minimum sample count
        min_required = self.min_samples.get(split_name, 0)
        if stats.image_count < min_required:
            msg = (
                f"Split '{split_name}': only {stats.image_count} images "
                f"(minimum required: {min_required})"
            )
            report.warnings.append(msg)
            logger.warning(msg)

        # Validate each image + label
        for img_path in image_files:
            self._validate_image(img_path, labels_dir, stats, report)

        report.splits[split_name] = stats

        logger.info(
            "  %-6s → images: %d | labels: %d | annotations: %d | "
            "missing: %d | corrupt: %d | invalid: %d",
            split_name,
            stats.image_count,
            stats.label_count,
            stats.annotation_count,
            len(stats.missing_labels),
            len(stats.corrupt_images),
            len(stats.invalid_labels),
        )

    def _validate_image(
        self,
        img_path: Path,
        labels_dir: Path,
        stats: SplitStats,
        report: ValidationReport,
    ) -> None:
        """Validate a single image + its label file."""
        # Check label file exists
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            stats.missing_labels.append(str(img_path.name))
            return

        stats.label_count += 1

        # Validate label content
        try:
            with open(label_path, "r") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]

            if not lines:
                # Empty label file = background image (valid in YOLO)
                return

            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    stats.invalid_labels.append(f"{label_path.name}:{line}")
                    continue

                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])

                # Class ID range check
                if cls_id < 0 or cls_id >= report.num_classes:
                    msg = (
                        f"Invalid class ID {cls_id} in {label_path.name} "
                        f"(nc={report.num_classes})"
                    )
                    stats.invalid_labels.append(msg)
                    continue

                # Bounding box range check
                if not all(0.0 <= v <= 1.0 for v in [cx, cy, w, h]):
                    msg = f"BBox out of [0,1] range in {label_path.name}: {line}"
                    stats.invalid_labels.append(msg)
                    continue

                stats.annotation_count += 1
                stats.class_distribution[cls_id] = (
                    stats.class_distribution.get(cls_id, 0) + 1
                )

        except Exception as e:
            stats.invalid_labels.append(f"{label_path.name}: {e}")

        # Quick image integrity check (open header only)
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                stats.corrupt_images.append(str(img_path.name))
        except Exception:
            stats.corrupt_images.append(str(img_path.name))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_split_path(self, split_name: str) -> Optional[Path]:
        """Resolve image directory for a split from yaml or config."""
        # Try data.yaml first
        if split_name in self._data_yaml:
            raw = self._data_yaml[split_name]
            p = Path(raw)
            if not p.is_absolute():
                p = self.yaml_path.parent / p
            if p.exists():
                return p

        # Fall back to config.yaml split paths
        split_rel = self.dataset_cfg.get("splits", {}).get(split_name, "")
        if split_rel:
            p = self.dataset_root / split_rel
            if p.exists():
                return p

        return None

    def _get_labels_dir(self, images_dir: Path) -> Path:
        """Derive labels directory from images directory (YOLO convention)."""
        # Standard: replace /images/ with /labels/
        parts = images_dir.parts
        new_parts = ["labels" if p == "images" else p for p in parts]
        labels_dir = Path(*new_parts)
        return labels_dir

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _log_report(self, report: ValidationReport) -> None:
        """Log final validation summary."""
        sep = "=" * 60
        logger.info(sep)
        logger.info("DATASET VALIDATION REPORT")
        logger.info(sep)
        logger.info("  Dataset Root  : %s", report.dataset_root)
        logger.info("  YAML Valid    : %s", report.yaml_valid)
        logger.info("  Classes       : %d → %s", report.num_classes, report.class_names)
        logger.info("  Total Images  : %d", report.total_images)
        logger.info("")

        for split_name, stats in report.splits.items():
            status = "✓ OK" if stats.is_valid else "✗ ISSUES"
            logger.info("  Split '%s' [%s]", split_name, status)
            logger.info("    Images       : %d", stats.image_count)
            logger.info("    Labels       : %d", stats.label_count)
            logger.info("    Annotations  : %d", stats.annotation_count)

            if stats.class_distribution and report.class_names:
                logger.info("    Class dist.  :")
                for cls_id, count in sorted(stats.class_distribution.items()):
                    name = (
                        report.class_names[cls_id]
                        if cls_id < len(report.class_names)
                        else f"class_{cls_id}"
                    )
                    logger.info("      [%d] %-25s : %d", cls_id, name, count)

            if stats.missing_labels:
                logger.warning("    Missing labels : %d", len(stats.missing_labels))
                for m in stats.missing_labels[:5]:
                    logger.warning("      - %s", m)
                if len(stats.missing_labels) > 5:
                    logger.warning("      ... and %d more", len(stats.missing_labels) - 5)

            if stats.corrupt_images:
                logger.error("    Corrupt images : %d", len(stats.corrupt_images))
                for c in stats.corrupt_images[:5]:
                    logger.error("      - %s", c)

            if stats.invalid_labels:
                logger.error("    Invalid labels : %d", len(stats.invalid_labels))
                for inv in stats.invalid_labels[:5]:
                    logger.error("      - %s", inv)

        logger.info("")
        if report.warnings:
            logger.info("  WARNINGS (%d):", len(report.warnings))
            for w in report.warnings:
                logger.warning("    ⚠  %s", w)

        if report.errors:
            logger.info("  ERRORS (%d):", len(report.errors))
            for e in report.errors:
                logger.error("    ✗  %s", e)

        logger.info("")
        final = "✓  DATASET IS VALID — Ready for training." if report.is_valid else "✗  DATASET HAS ISSUES — Fix errors before training."
        logger.info("  RESULT: %s", final)
        logger.info(sep)