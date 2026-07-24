# scripts/preprocess_clahe.py
import cv2
from pathlib import Path

def apply_clahe(img_path, output_path):
    img = cv2.imread(str(img_path))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    cv2.imwrite(str(output_path), result)

for split in ["train", "valid", "test"]:
    img_dir = Path(f"dataset_multiclass/{split}/images")
    for img_path in img_dir.glob("*.jpg"):
        apply_clahe(img_path, img_path)  # overwrite in place

print("CLAHE preprocessing done")