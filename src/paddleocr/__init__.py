"""Pure PyTorch & Hugging Face Transformers implementation of PaddleOCR services.

Provides ``PaddleOCR``, ``PaddleOCRVL``, and ``PPStructureV3`` with zero PaddlePaddle
dependencies and native PyTorch device support (``mps``, ``cuda``, ``cpu``).
"""

from __future__ import annotations

from .device import DeviceLike, DTypeLike, resolve_device, resolve_dtype
from .ocr import PaddleOCR
from .results import LayoutBlock, OCRResult, StructureResult, TableResult, TextBox, VLResult
from .structure import PPStructureV3
from .vl import PaddleOCRVL

__version__ = "0.1.0"

__all__ = [
    "DTypeLike",
    "DeviceLike",
    "LayoutBlock",
    "OCRResult",
    "PPStructureV3",
    "PaddleOCR",
    "PaddleOCRVL",
    "StructureResult",
    "TableResult",
    "TextBox",
    "VLResult",
    "__version__",
    "resolve_device",
    "resolve_dtype",
]
