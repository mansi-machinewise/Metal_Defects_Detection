"""
train_classifier.py
====================
Train EfficientNetB0 on cropped defect regions.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

# ---- Config ---------------------------------------------------------------
DATA_DIR   = r"D:\Metal_Defects\dataset_crops"
OUTPUT_DIR  = r"D:\Metal_Defects\outputs\classifier"
BATCH_SIZE  = 4
EPOCHS      = 50
LR          = 0.001
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
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---- Datasets -------------------------------------------------------------
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=val_transform)

from torch.utils.data import DataLoader, WeightedRandomSampler

class_counts_auto = [len([x for x in train_dataset.targets if x == i]) for i in range(NUM_CLASSES)]
sample_weights = [1.0 / class_counts_auto[label] for label in train_dataset.targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,   num_workers=0, pin_memory=True)

print(f"Classes: {train_dataset.classes}")
print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
print(f"Device: {DEVICE}")

# ---- Model ----------------------------------------------------------------
model = models.efficientnet_b2(weights="IMAGENET1K_V1")

# Replace classifier head
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES),
)

model = model.to(DEVICE)

# ---- Class weights for imbalance ------------------------------------------
class_counts = [len([x for x in train_dataset.targets if x == i]) for i in range(NUM_CLASSES)]
total = sum(class_counts)
weights = torch.tensor([total / c for c in class_counts], dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

# ---- Optimizer ------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0005)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---- Training loop --------------------------------------------------------
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # Train
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

    # Validate
    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss    += loss.item() * imgs.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = train_correct / len(train_dataset) * 100
    val_acc   = val_correct   / len(val_dataset)   * 100
    print(f"Epoch {epoch:3d}/{EPOCHS} | Train Loss: {train_loss/len(train_dataset):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_classifier.pth"))
        print(f"  ✅ Best model saved — Val Acc: {val_acc:.2f}%")

print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2f}%")
print(f"Model saved to: {OUTPUT_DIR}\\best_classifier.pth")