"""PyTorch device and dtype resolution utilities with first-class MPS and CUDA support."""

from __future__ import annotations

import torch

type DeviceLike = str | torch.device | None
type DTypeLike = str | torch.dtype | None

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_device(device: DeviceLike = None) -> torch.device:
    """Resolve a user-supplied device specification into a native ``torch.device``.

    When ``device`` is ``None`` or ``"auto"``, automatically selects ``cuda`` if available,
    then ``mps`` (Apple Silicon), and falls back to ``cpu``.
    """
    if isinstance(device, torch.device):
        return device

    if device is None or device.strip().lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    normalized = device.strip()
    lower = normalized.lower()
    if lower == "gpu":
        normalized = "cuda"
    elif lower.startswith("gpu:"):
        normalized = f"cuda:{normalized[4:]}"

    return torch.device(normalized)


def resolve_dtype(
    dtype: DTypeLike = None,
    *,
    device: torch.device,
    is_vlm: bool = False,
) -> torch.dtype:
    """Resolve a user-supplied dtype specification into a ``torch.dtype``.

    When ``dtype`` is ``None`` or ``"auto"``:
    - Vision CNN / DETR models default to ``torch.float32`` for numerical precision
      and complete operator compatibility on ``mps`` and ``cpu``.
    - Vision-Language Models (VLMs) default to ``torch.bfloat16`` on accelerated
      devices (``cuda``, ``mps``) and ``torch.float32`` on ``cpu``.
    """
    if isinstance(dtype, torch.dtype):
        return dtype

    if isinstance(dtype, str) and dtype.strip().lower() != "auto":
        key = dtype.strip().lower()
        if key not in _DTYPE_MAP:
            supported = ", ".join(sorted(_DTYPE_MAP))
            msg = f"Unsupported dtype '{dtype}'. Supported values: {supported}, 'auto'."
            raise ValueError(msg)
        return _DTYPE_MAP[key]

    if is_vlm and device.type in {"cuda", "mps"}:
        return torch.bfloat16
    return torch.float32
