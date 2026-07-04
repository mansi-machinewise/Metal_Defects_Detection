"""
api/main.py
============
FastAPI application entry point.

Responsibilities:
  - Create the FastAPI app instance
  - Configure CORS so the browser frontend can call the API
  - Mount the frontend folder as static files (serves home.html etc.)
  - Load the YOLOv8 model at startup via lifespan event
  - Unload the model cleanly at shutdown
  - Register all API routers
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.inspection import router as inspection_router
from api.services.inference_service import InferenceService
from src.utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Path to the frontend folder (relative to project root)
FRONTEND_DIR = Path(__file__).parents[1] / "frontend"


# ---------------------------------------------------------------------------
# Lifespan — model load/unload tied to server lifetime
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Code before `yield` runs at startup.
    Code after `yield` runs at shutdown.
    """
    logger.info("=" * 60)
    logger.info("Metal Defect Detection API — Starting Up")
    logger.info("=" * 60)

    service = InferenceService.get_instance()
    try:
        service.load()
        logger.info("Model loaded successfully. API is ready.")
    except Exception as e:
        logger.error("Failed to load model at startup: %s", e)
        logger.error("Check that inference.model_path in config.yaml points to best.pt")

    yield  # Server is running — handle requests

    logger.info("Shutting down — releasing model ...")
    service.unload()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Metal Defect Detection API",
    description="YOLOv8-powered industrial metal surface defect detection.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the frontend (running as a local file or on any origin)
# to call the API. Tighten this in production.
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

app.include_router(inspection_router)

# ---------------------------------------------------------------------------
# Serve frontend as static files
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    logger.info("Frontend served from: %s", FRONTEND_DIR)
else:
    logger.warning(
        "Frontend directory not found at: %s\n"
        "Create a 'frontend/' folder inside the project root and place "
        "home.html, upload.html, dashboard.html inside it.",
        FRONTEND_DIR,
    )