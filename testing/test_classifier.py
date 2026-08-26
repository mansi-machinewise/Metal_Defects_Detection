"""
test_classifier.py
===================
Evaluate EfficientNetB0 classifier accuracy on test set.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---- Config ---------------------------------------------------------------
DATA_DIR   = r"D:\Metal_Defects\dataset_crops"
MODEL_PATH  = r"D:\Metal_Defects\outputs\classifier\best_classifier.pth"
NUM_CLASSES = 6
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

# ---- Transform ------------------------------------------------------------
test_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---- Dataset --------------------------------------------------------------
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

print(f"Test samples: {len(test_dataset)}")
print(f"Device: {DEVICE}")

# ---- Model ----------------------------------------------------------------
model = models.efficientnet_b2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model = model.to(DEVICE)

# ---- Evaluate -------------------------------------------------------------
correct = 0
total = 0
class_correct = [0] * NUM_CLASSES
class_total   = [0] * NUM_CLASSES

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        for i in range(len(labels)):
            class_correct[labels[i]] += (preds[i] == labels[i]).item()
            class_total[labels[i]]   += 1

print(f"\n{'='*50}")
print(f"Overall Accuracy: {correct/total*100:.2f}%")
print(f"{'='*50}")
print(f"\nPer-class Accuracy:")
for i, name in enumerate(CLASS_NAMES):
    acc = class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0
    print(f"  {name:<20} {acc:.2f}%  ({class_correct[i]}/{class_total[i]})")