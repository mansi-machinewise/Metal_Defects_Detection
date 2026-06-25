"""
src/utils/device.py
====================
Hardware detection and device selection utility.

Responsibilities:
  - Detect CUDA availability
  - Report GPU name, VRAM, CUDA version
  - Auto-select the best available device
  - Provide memory monitoring helpers
  - Log all hardware info at startup
  - Gracefully fall back to CPU

Used by: trainer.py, detector.py, and any future GPU-dependent module.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import psutil
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Snapshot of a single GPU's properties."""
    index: int
    name: str
    total_vram_gb: float
    free_vram_gb: float
    cuda_capability: tuple[int, int]


@dataclass
class HardwareInfo:
    """Full system hardware snapshot."""
    cuda_available: bool
    cuda_version: Optional[str]
    torch_version: str
    device_name: str          # e.g. "cuda:0" or "cpu"
    gpus: list[GPUInfo] = field(default_factory=list)
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0


# ---------------------------------------------------------------------------
# Core Detection Functions
# ---------------------------------------------------------------------------

def _get_ram_info() -> tuple[float, float]:
    """Return (total_gb, available_gb) for system RAM."""
    mem = psutil.virtual_memory()
    return round(mem.total / 1e9, 2), round(mem.available / 1e9, 2)


def _get_gpu_info() -> list[GPUInfo]:
    """
    Collect per-GPU info using torch.cuda.
    Returns empty list if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return []

    gpus: list[GPUInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        # Free memory requires setting the device first
        torch.cuda.set_device(i)
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        gpus.append(
            GPUInfo(
                index=i,
                name=props.name,
                total_vram_gb=round(total_bytes / 1e9, 2),
                free_vram_gb=round(free_bytes / 1e9, 2),
                cuda_capability=(props.major, props.minor),
            )
        )
    return gpus


def detect_hardware() -> HardwareInfo:
    """
    Perform full hardware detection and return a HardwareInfo snapshot.
    This is the single entry-point for hardware info — call it once at startup.
    """
    cuda_available = torch.cuda.is_available()
    gpus = _get_gpu_info()
    ram_total, ram_available = _get_ram_info()

    if cuda_available and gpus:
        device_name = "cuda:0"
        cuda_version = torch.version.cuda or "unknown"
    else:
        device_name = "cpu"
        cuda_version = None

    return HardwareInfo(
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        torch_version=torch.__version__,
        device_name=device_name,
        gpus=gpus,
        ram_total_gb=ram_total,
        ram_available_gb=ram_available,
    )


def log_hardware_info(hw: HardwareInfo) -> None:
    """Pretty-print hardware info to the project logger."""
    sep = "=" * 60
    logger.info(sep)
    logger.info("HARDWARE CONFIGURATION")
    logger.info(sep)
    logger.info("  PyTorch Version   : %s", hw.torch_version)
    logger.info("  CUDA Available    : %s", hw.cuda_available)

    if hw.cuda_available:
        logger.info("  CUDA Version      : %s", hw.cuda_version)
        for gpu in hw.gpus:
            logger.info("  GPU [%d]           : %s", gpu.index, gpu.name)
            logger.info(
                "  VRAM Total        : %.2f GB", gpu.total_vram_gb
            )
            logger.info(
                "  VRAM Free         : %.2f GB  ← available for model",
                gpu.free_vram_gb,
            )
            logger.info(
                "  CUDA Capability   : %d.%d",
                gpu.cuda_capability[0],
                gpu.cuda_capability[1],
            )
    else:
        logger.warning("  ⚠  No CUDA-capable GPU detected.")
        logger.warning("     Training and inference will run on CPU.")
        logger.warning("     Expect significantly slower performance.")

    logger.info("  Selected Device   : %s", hw.device_name)
    logger.info("  RAM Total         : %.2f GB", hw.ram_total_gb)
    logger.info("  RAM Available     : %.2f GB", hw.ram_available_gb)
    logger.info(sep)


def select_device(preference: str = "auto") -> str:
    """
    Resolve the device string to use for torch / ultralytics.

    Args:
        preference: "auto" | "cpu" | "cuda" | "cuda:0" | "0"

    Returns:
        Resolved device string ready for torch.device() or ultralytics.
    """
    preference = str(preference).strip().lower()

    if preference == "cpu":
        logger.info("Device forced to CPU by configuration.")
        return "cpu"

    if preference in ("auto", "cuda", "cuda:0", "0"):
        if torch.cuda.is_available():
            idx = 0 if preference in ("auto", "cuda") else int(preference.replace("cuda:", ""))
            device_str = f"cuda:{idx}"
            logger.info("GPU selected: %s", device_str)
            return device_str
        else:
            if preference != "cpu":
                warnings.warn(
                    "GPU requested but CUDA is not available. Falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                logger.warning("GPU not available — falling back to CPU.")
            return "cpu"

    # Numeric index e.g. "1"
    if preference.isdigit():
        idx = int(preference)
        if torch.cuda.is_available() and idx < torch.cuda.device_count():
            return f"cuda:{idx}"
        logger.warning("GPU index %s not available — falling back to CPU.", preference)
        return "cpu"

    logger.warning("Unknown device preference '%s' — defaulting to CPU.", preference)
    return "cpu"


# ---------------------------------------------------------------------------
# Memory Monitoring Helpers
# ---------------------------------------------------------------------------

def log_gpu_memory(tag: str = "") -> None:
    """Log current GPU memory usage (only if CUDA is available)."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    label = f"[{tag}] " if tag else ""
    logger.debug(
        "%sGPU Memory — Allocated: %.3f GB | Reserved: %.3f GB",
        label, allocated, reserved,
    )


def log_ram_usage(tag: str = "") -> None:
    """Log current system RAM usage."""
    mem = psutil.virtual_memory()
    used_gb = (mem.total - mem.available) / 1e9
    label = f"[{tag}] " if tag else ""
    logger.debug(
        "%sRAM Usage — %.2f GB / %.2f GB (%.1f%%)",
        label, used_gb, mem.total / 1e9, mem.percent,
    )


def clear_gpu_cache() -> None:
    """Empty PyTorch CUDA cache. Call between heavy operations if needed."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared.")


def recommend_training_config(hw: HardwareInfo) -> dict:
    """
    Suggest training hyperparameters based on detected hardware.
    Returns a dict of recommendations to log / optionally apply.
    """
    if not hw.cuda_available or not hw.gpus:
        return {
            "device": "cpu",
            "batch_size": 2,
            "image_size": 416,
            "workers": 1,
            "amp": False,
            "note": "CPU-only mode. Use small batch and image size.",
        }

    vram = hw.gpus[0].free_vram_gb

    if vram < 2:
        rec = {"batch_size": 2, "image_size": 416, "amp": True}
        note = "Very low VRAM (<2 GB). Using minimum viable settings."
    elif vram < 4:
        rec = {"batch_size": 2, "image_size": 512, "amp": True}
        note = "Low VRAM (2–4 GB). AMP enabled, conservative batch."
    elif vram < 6:
        rec = {"batch_size": 4, "image_size": 640, "amp": True}
        note = "Medium VRAM (4–6 GB). Balanced settings."
    else:
        rec = {"batch_size": 8, "image_size": 640, "amp": True}
        note = "Sufficient VRAM. Standard settings."

    return {
        "device": "cuda:0",
        "workers": min(2, os.cpu_count() or 1),
        "note": note,
        **rec,
    }