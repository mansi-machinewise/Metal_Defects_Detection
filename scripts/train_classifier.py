"""
train_classifier.py
====================
Train EfficientNetB2 on cropped defect regions.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---- Config ---------------------------------------------------------------
DATA_DIR    = r"D:\Metal_Defects\Metal_Defects_Detection\dataset_crops"
OUTPUT_DIR  = r"D:\Metal_Defects\Metal_Defects_Detection\outputs\classifier"
BATCH_SIZE  = 32          # was 4 — too small for stable BatchNorm
EPOCHS      = 50
LR_HEAD     = 1e-3        # head learning rate
LR_BACKBONE = 1e-4        # backbone learning rate — 10x lower
NUM_CLASSES = 6
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Transforms -----------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---- Datasets -------------------------------------------------------------
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=val_transform)

# WeightedRandomSampler only — no class weights in loss (don't double-penalise)
class_counts  = [len([x for x in train_dataset.targets if x == i]) for i in range(NUM_CLASSES)]
sample_weights = [1.0 / class_counts[label] for label in train_dataset.targets]
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0, pin_memory=True)

print(f"Classes: {train_dataset.classes}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Device: {DEVICE}")

# ---- Model ----------------------------------------------------------------
model = models.efficientnet_b2(weights="IMAGENET1K_V1")
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES),
)
model = model.to(DEVICE)

# ---- Loss — no class weights (sampler handles balance), reduced smoothing --
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)  # was 0.1 + weighted

# ---- Optimizer — different LR for backbone vs head ------------------------
backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
head_params     = [p for n, p in model.named_parameters() if "classifier" in n]

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": 0.0005},
    {"params": head_params,     "lr": LR_HEAD,     "weight_decay": 0.0005},
])
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---- Training loop --------------------------------------------------------
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_correct = 0.0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item() * imgs.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()

    scheduler.step()

    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            val_loss    += criterion(outputs, labels).item() * imgs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_dataset) * 100
    val_acc   = val_correct   / len(val_dataset)   * 100
    print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss/len(train_dataset):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_classifier.pth"))
        print(f"Best model saved — Val Acc: {val_acc:.2f}%")

print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2f}%")
print(f"Model saved to: {OUTPUT_DIR}\\best_classifier.pth")