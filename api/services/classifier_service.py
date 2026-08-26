"""
api/services/classifier_service.py
=====================================
EfficientNetB0 classifier service.
Takes a cropped defect region and returns the refined class name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = [
    'crazing', 'inclusion', 'patches',
    'pitted_surface', 'rolled-in_scale', 'scratches'
]

TRANSFORM = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


class ClassifierService:
    """
    Singleton EfficientNetB0 classifier.
    Refines YOLO's class prediction on each cropped defect region.
    """

    _instance = None

    def __init__(self):
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "ClassifierService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self, model_path: str) -> None:
        """Load EfficientNetB0 weights."""
        if self._loaded:
            return
        logger.info("ClassifierService: loading EfficientNetB0 ...")
        model = models.efficientnet_b2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES)),
        )
        model.load_state_dict(torch.load(model_path, map_location=self._device))
        model.eval()
        self._model = model.to(self._device)
        self._loaded = True
        logger.info("ClassifierService: EfficientNetB0 ready.")

    def classify(self, img_bgr: np.ndarray, bbox: list) -> tuple[str, float]:
        if not self._loaded:
            raise RuntimeError("ClassifierService not loaded.")

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)

        crop_bgr = img_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            return "unknown", 0.0

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(crop_rgb)

        # TTA augmentations
        tta_transforms = [
            TRANSFORM,  # original
            transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ColorJitter(brightness=0.5),  # brighter
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ColorJitter(brightness=0.2),  # darker
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.Grayscale(num_output_channels=3),  # grayscale
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        ]

        TEMPERATURE = 1.2

        all_probs = []
        with torch.no_grad():
            for tta_transform in tta_transforms:
                tensor = tta_transform(pil_img).unsqueeze(0).to(self._device)
                outputs = self._model(tensor)
                probs = torch.softmax(outputs / TEMPERATURE, dim=1)
                all_probs.append(probs)

        # Average probabilities across all augmentations
        avg_probs = torch.stack(all_probs).mean(dim=0)
        conf, idx = avg_probs.max(dim=1)

        return CLASS_NAMES[idx.item()], conf.item()

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        logger.info("ClassifierService: unloaded.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded