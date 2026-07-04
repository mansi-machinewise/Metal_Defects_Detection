"""
scripts/relabel_and_merge.py
=============================
Relabels your existing dataset (19 classes → 1 class)
and merges it into the new merged dataset folder.

Run this AFTER merge_datasets.py.

Usage:
    python scripts/relabel_and_merge.py
      --existing  dataset/
      --merged    dataset_merged/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def relabel_and_copy(
    existing_root: Path,
    merged_root: Path,
    prefix: str = "orig",
):
    """
    For each image in the existing dataset:
      - Copy image to merged dataset
      - Relabel: change all class IDs to 0 (single class)
      - Empty label files stay empty (good images)
    """
    counter = [0]
    stats = {"total": 0, "good": 0, "bad": 0}

    for split in ["train", "valid", "test"]:
        src_img_dir = existing_root / split / "images"
        src_lbl_dir = existing_root / split / "labels"
        dst_img_dir = merged_root / split / "images"
        dst_lbl_dir = merged_root / split / "labels"

        if not src_img_dir.exists():
            # Try alternative split name "val"
            if split == "valid":
                src_img_dir = existing_root / "val" / "images"
                src_lbl_dir = existing_root / "val" / "labels"
            if not src_img_dir.exists():
                print(f"  Skipping split '{split}' (not found)")
                continue

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = [f for f in src_img_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

        print(f"  {split}: {len(images)} images")

        for img_path in tqdm(images, desc=f"  {prefix}/{split}", leave=False):
            idx = counter[0]
            counter[0] += 1
            stem = f"{prefix}_{idx:05d}"
            suffix = img_path.suffix.lower()

            # Copy image
            dst_img = dst_img_dir / f"{stem}{suffix}"
            shutil.copy2(img_path, dst_img)

            # Relabel: read existing label, replace all class IDs with 0
            src_lbl = src_lbl_dir / (img_path.stem + ".txt")
            dst_lbl = dst_lbl_dir / f"{stem}.txt"

            if src_lbl.exists():
                lines = src_lbl.read_text().strip().splitlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = "0"   # all classes → defect
                        new_lines.append(" ".join(parts))
                dst_lbl.write_text("\n".join(new_lines))

                if new_lines:
                    stats["bad"] += 1
                else:
                    stats["good"] += 1
            else:
                # No label file — create empty (treat as good)
                dst_lbl.write_text("")
                stats["good"] += 1

            stats["total"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Relabel existing dataset and merge into converted dataset."
    )
    parser.add_argument("--existing", type=str, default="dataset",
                        help="Path to your existing dataset (default: dataset/)")
    parser.add_argument("--merged",   type=str, default="dataset_merged",
                        help="Path to merged output dataset (default: dataset_merged/)")
    args = parser.parse_args()

    existing = Path(args.existing)
    merged   = Path(args.merged)

    print("=" * 60)
    print("Relabel Existing Dataset + Merge")
    print("=" * 60)
    print(f"  Existing dataset : {existing.resolve()}")
    print(f"  Merged output    : {merged.resolve()}")
    print()

    if not existing.exists():
        print(f"ERROR: Existing dataset not found: {existing}")
        sys.exit(1)

    if not merged.exists():
        print(f"ERROR: Merged dataset folder not found: {merged}")
        print("Run merge_datasets.py first.")
        sys.exit(1)

    print("Relabeling and copying existing dataset ...")
    stats = relabel_and_copy(existing, merged, prefix="orig")

    # Count final totals
    total_train = sum(1 for f in (merged / "train" / "images").iterdir() if f.is_file())
    total_valid = sum(1 for f in (merged / "valid" / "images").iterdir() if f.is_file())
    total_test  = sum(1 for f in (merged / "test"  / "images").iterdir() if f.is_file())

    # Count good vs bad in merged dataset
    total_good = total_bad = 0
    for split in ["train", "valid", "test"]:
        for lbl in (merged / split / "labels").iterdir():
            if lbl.is_file():
                if lbl.read_text().strip() == "":
                    total_good += 1
                else:
                    total_bad += 1

    print()
    print("=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)
    print(f"  From existing dataset : {stats['total']} images added")
    print()
    print(f"  FINAL MERGED DATASET")
    print(f"  Train  : {total_train}")
    print(f"  Valid  : {total_valid}")
    print(f"  Test   : {total_test}")
    print(f"  Total  : {total_train + total_valid + total_test}")
    print()
    print(f"  Good (clean/no defect) : {total_good}")
    print(f"  Bad  (has defects)     : {total_bad}")
    print()
    print("Next steps:")
    print("  1. Update config/config.yaml:")
    print("       dataset.root: dataset_merged")
    print("       dataset.yaml: dataset_merged/data.yaml")
    print("       classes.names: [defect]")
    print("       training.model: yolov8s.pt")
    print("       training.run_name: defect_detection_v2")
    print("  2. python scripts/validate_dataset.py")
    print("  3. python scripts/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()