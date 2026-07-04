"""
api/routes/inspection.py
=========================
Inspection API routes.

Exposes:
  POST /api/inspect        — upload an image, get defect detection results
  GET  /api/health         — confirm server and model are ready
  GET  /api/model-info     — return model metadata
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from api.services.inference_service import InferenceService
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["inspection"])

# Allowed upload MIME types — matches what the frontend enforces
ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/jpg",
}

# Max upload size: 20 MB
MAX_FILE_SIZE = 20 * 1024 * 1024


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@router.get("/health")
async def health_check():
    """
    Returns server and model status.
    The frontend can call this on load to confirm the backend is reachable.
    """
    service = InferenceService.get_instance()
    return {
        "status": "ok",
        "model_loaded": service.is_loaded,
        "message": "Metal Defect Detection API is running.",
    }


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

@router.get("/model-info")
async def model_info():
    """Return basic model metadata."""
    from src.utils.config import load_config
    cfg = load_config()
    return {
        "model_path": cfg["inference"]["model_path"],
        "confidence_threshold": cfg["inference"]["confidence_threshold"],
        "bad_threshold": cfg["inference"]["bad_threshold"],
        "device": cfg["inference"]["device"],
        "classes": cfg["classes"]["names"],
    }


# ---------------------------------------------------------------------------
# Main inspection endpoint
# ---------------------------------------------------------------------------

@router.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):
    """
    Upload a metal component image and receive defect detection results.

    Request:
        multipart/form-data with field 'file' containing an image.

    Response JSON:
        {
            "status":        "GOOD" | "BAD",
            "confidence":    98.74,          // percentage
            "defect_type":   "crack",
            "defect_count":  2,
            "defects":       [...],
            "annotated_image": "data:image/png;base64,...",
            "inference_ms":  120.5,
            "timestamp":     "2026-01-01 12:00:00"
        }
    """
    # Validate file type
    content_type = file.content_type or ""
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type: '{content_type}'. "
                "Please upload a JPG, JPEG, PNG, or BMP image."
            ),
        )

    # Read file bytes
    image_bytes = await file.read()

    # Validate file size
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE // (1024*1024)} MB.",
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # Run inference
    service = InferenceService.get_instance()

    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet. Please try again in a moment.",
        )

    try:
        result = service.inspect(image_bytes, filename=file.filename or "upload")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Inspection failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inspection failed due to an internal error. Check server logs.",
        )

    return JSONResponse(content=result.to_dict())