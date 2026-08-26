"""
evaluate_hybrid.py
===================
Evaluate the full hybrid cascade:
YOLO11m + EfficientNetB0 on the test set.
Reports per-class accuracy, precision, recall, F1.
"""

import os
import sys
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms, models
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ---- Config ---------------------------------------------------------------
TEST_IMG_DIR   = r"D:\Metal_Defects\dataset_multiclass\test\images"
TEST_LBL_DIR   = r"D:\Metal_Defects\dataset_multiclass\test\labels"
YOLO_MODEL     = r"D:\Metal_Defects\outputs\runs\NEU_Metal_yolo11m_960_v1-4\weights\best.pt"
CNN_MODEL      = r"D:\Metal_Defects\outputs\classifier\best_classifier.pth"
OUTPUT_DIR     = r"D:\Metal_Defects\outputs\reports"
NUM_CLASSES    = 6
CONF_THRESHOLD = 0.35
IOU_THRESHOLD  = 0.7
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load EfficientNetB0 --------------------------------------------------
print("Loading EfficientNetB2 classifier...")
cnn = models.efficientnet_b2(weights=None)
cnn.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(cnn.classifier[1].in_features, NUM_CLASSES),
)
cnn.load_state_dict(torch.load(CNN_MODEL, map_location=DEVICE))
cnn.eval()
cnn = cnn.to(DEVICE)

# ---- Load YOLO ------------------------------------------------------------
print("Loading YOLO11m...")
yolo = YOLO(YOLO_MODEL)

# ---- Transform for CNN ----------------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def classify_crop(img_bgr, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0, 0.0
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = transform(crop_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = cnn(tensor)
        prob = torch.softmax(out, dim=1)
        conf, idx = prob.max(dim=1)
    return idx.item(), conf.item()

# ---- Evaluate -------------------------------------------------------------
print(f"\nEvaluating on test set: {TEST_IMG_DIR}")

y_true = []
y_pred_yolo = []
y_pred_hybrid = []

img_files = sorted(Path(TEST_IMG_DIR).glob("*.jpg"))
print(f"Found {len(img_files)} test images\n")

for i, img_path in enumerate(img_files):
    lbl_path = Path(TEST_LBL_DIR) / (img_path.stem + ".txt")
    if not lbl_path.exists():
        continue

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue

    H, W = img_bgr.shape[:2]

    # Read ground truth
    with open(lbl_path) as f:
        gt_lines = f.readlines()

    gt_classes = []
    gt_boxes = []
    for line in gt_lines:
        parts = line.strip().split()
        if len(parts) == 5:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw/2) * W)
            y1 = int((cy - bh/2) * H)
            x2 = int((cx + bw/2) * W)
            y2 = int((cy + bh/2) * H)
            gt_classes.append(cls)
            gt_boxes.append([x1, y1, x2, y2])

    if not gt_classes:
        continue

    # YOLO prediction
    results = yolo(img_bgr, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    pred_boxes = []
    pred_classes_yolo = []
    pred_classes_hybrid = []

    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        for box in boxes:
            bbox = box.xyxy[0].tolist()
            cls_yolo = int(box.cls[0].item())
            pred_boxes.append(bbox)
            pred_classes_yolo.append(cls_yolo)

            # CNN refinement
            cls_hybrid, _ = classify_crop(img_bgr, bbox)
            pred_classes_hybrid.append(cls_hybrid)

    # Match GT to predictions (simple: use dominant class per image)
    for gt_cls in gt_classes:
        y_true.append(gt_cls)
        if pred_classes_yolo:
            y_pred_yolo.append(max(set(pred_classes_yolo), key=pred_classes_yolo.count))
            y_pred_hybrid.append(max(set(pred_classes_hybrid), key=pred_classes_hybrid.count))
        else:
            y_pred_yolo.append(-1)
            y_pred_hybrid.append(-1)

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(img_files)} images...")

# ---- Results --------------------------------------------------------------
print("\n" + "="*60)
print("YOLO11m ONLY — Classification Report")
print("="*60)
print(classification_report(
    y_true, y_pred_yolo,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    zero_division=0
))

print("\n" + "="*60)
print("YOLO11m + EfficientNetB0 (Hybrid) — Classification Report")
print("="*60)
print(classification_report(
    y_true, y_pred_hybrid,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    zero_division=0
))

# ---- Confusion Matrix -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

cm_yolo = confusion_matrix(y_true, y_pred_yolo, labels=list(range(NUM_CLASSES)))
sns.heatmap(cm_yolo, annot=True, fmt='d', xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES, ax=axes[0], cmap='Blues')
axes[0].set_title('YOLO11m Only')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

cm_hybrid = confusion_matrix(y_true, y_pred_hybrid, labels=list(range(NUM_CLASSES)))
sns.heatmap(cm_hybrid, annot=True, fmt='d', xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES, ax=axes[1], cmap='Greens')
axes[1].set_title('YOLO11m + EfficientNetB0 (Hybrid)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_confusion_matrix.png'), dpi=150)
print(f"\nConfusion matrix saved to: {OUTPUT_DIR}\\hybrid_confusion_matrix.png")