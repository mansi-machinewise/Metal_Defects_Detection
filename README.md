# AI Metal Defect Detection System

An end-to-end industrial visual quality inspection system that detects surface defects on metal components using a hybrid deep learning pipeline — YOLO11m for detection and EfficientNetB2 for classification.

---

## Table of Contents

- [Overview](#overview)
- [Defect Classes](#defect-classes)
- [System Architecture](#system-architecture)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Frontend Pages](#frontend-pages)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Test Suite](#test-suite)

---

## Overview

The system takes a surface image of a metal component and outputs:

- **GOOD** — no defects detected
- **BAD** — one or more defects found, with bounding boxes, class labels, and confidence scores

Two inspection modes are supported:

| Mode | Description |
|------|-------------|
| Single Image | Upload one image, view annotated result with bounding boxes |
| Batch Folder | Upload a full folder, get results for all images with side-by-side viewer and CSV export |

---

## Defect Classes

| Class | Description |
|-------|-------------|
| `crazing` | Fine network of surface cracks |
| `inclusion` | Foreign material embedded in the surface |
| `patches` | Irregular surface discolouration or roughness |
| `pitted_surface` | Small pits or holes in the surface |
| `rolled-in_scale` | Oxide scale pressed into the surface during rolling |
| `scratches` | Linear surface scratches |

---

## System Architecture

```
Input Image
     │
     ▼
 YOLO11m ──────────────────── Detects bounding boxes + class
     │
     ▼
 EfficientNetB2 ─────────────  Classifies each detected crop
     │
     ▼
 Hybrid Decision ────────────  Final class + confidence per box
     │
     ▼
 FastAPI Response ───────────  JSON with status, detections, annotated image
     │
     ▼
 Frontend Viewer ────────────  Original + annotated image, bounding boxes, result cards
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended — tested on NVIDIA with 4GB+ VRAM)

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd Metal_Defects
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```
torch
torchvision
ultralytics
fastapi
uvicorn
python-multipart
opencv-python
scikit-learn
seaborn
matplotlib
numpy
pillow
pyyaml
```

### 3. Verify GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Running the Project

### Start the backend

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`.

### Open the frontend

Open `frontend/home.html` in your browser using Live Server (VS Code extension) or any static file server:

```bash
# Using Python's built-in server from the frontend directory
cd frontend
python -m http.server 5500
```

Then visit `http://localhost:5500/home.html`.

> **Note:** The frontend must be served (not opened as `file:///`) because browser security blocks fetch requests from file:// origins.

---

## Frontend Pages

| Page | File | Description |
|------|------|-------------|
| Home | `home.html` | Landing page with Single Image and Batch Inspect buttons |
| Upload | `upload.html` | Drag-and-drop single image upload with preview |
| Dashboard | `dashboard.html` | Single image result — annotated image, bounding boxes in lightbox, result cards |
| Batch | `batch.html` | Folder upload + batch results in one page — sidebar image list, viewer panel, Prev/Next navigation, CSV export |

### Single Image Flow

```
home.html → upload.html → dashboard.html
```

### Batch Flow

```
home.html → batch.html (upload view) → batch.html (results view)
```

> Both batch views live in the same file to avoid losing File objects across page navigation.

---

## API Reference

### `POST /predict`

Inspect a single image.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Image file (JPG, PNG, BMP, WEBP) |

**Response:**

```json
{
  "status": "BAD",
  "confidence": 0.91,
  "defect_type": "crazing",
  "annotated_image": "<base64-encoded-jpeg>",
  "defects": [
    {
      "class_name": "crazing",
      "confidence": 0.91,
      "bbox": [42, 38, 180, 160]
    }
  ]
}
```

---

### `POST /predict/batch`

Inspect multiple images at once.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `files` | File[] | Up to 500 image files |

**Response:**

```json
{
  "total": 50,
  "processed": 50,
  "failed": 0,
  "good_count": 32,
  "bad_count": 18,
  "defect_totals": {
    "crazing": 5,
    "inclusion": 8,
    "scratches": 5
  },
  "total_time_ms": 4823.1,
  "results": [
    {
      "filename": "image_001.jpg",
      "status": "BAD",
      "defect_count": 2,
      "avg_conf": 89.5,
      "defects": ["crazing", "inclusion"],
      "summary": { "crazing": 1, "inclusion": 1 },
      "detections": [
        {
          "bbox": [42, 38, 180, 160],
          "class_name": "crazing",
          "yolo_conf": 91.0,
          "cnn_conf": 88.5
        }
      ],
      "time_ms": 96.2
    }
  ],
  "errors": []
}
```

---

### `POST /predict/batch/csv`

Same as `/predict/batch` but returns a downloadable CSV file.

---

## Model Details

### YOLO11m — Detection

| Parameter | Value |
|-----------|-------|
| Architecture | YOLO11m |
| Input size | 960 × 960 |
| Classes | 6 |
| Optimizer | AdamW |
| Epochs | 200 |

### EfficientNetB2 — Classification

| Parameter | Value |
|-----------|-------|
| Architecture | EfficientNetB2 |
| Input size | 224 × 224 |
| Classes | 6 |
| Pretrained backbone | ImageNet |

### Hybrid Decision Logic

1. YOLO detects bounding boxes and predicts class + confidence
2. Each detected crop is passed to EfficientNetB2
3. CNN output is remapped to YOLO's class index space
4. Final class is determined by highest confidence between the two models

---

## Test Suite

Run individual tests from the project root with the virtual environment active and the FastAPI server running:

```bash
# Full hybrid evaluation on test set
python testing/evaluate_hybrid.py

# Speed benchmark
python testing/test_speed.py

# Robustness (noise, blur, brightness)
python testing/test_robustness.py

# Edge cases (black image, white image, tiny image)
python testing/test_edge_cases.py

# Compression robustness
python testing/test_compression.py

# Stress test (360 images)
python testing/test_stress.py

# Confidence distribution
python testing/test_confidence.py

# False positive analysis
python testing/test_false_positives.py
```
