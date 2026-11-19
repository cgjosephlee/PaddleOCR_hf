"""Image and PDF input loading and geometric cropping utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import pypdfium2 as pdfium  # pyright: ignore[reportMissingTypeStubs]
import pypdfium2.raw as pdfium_c  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from PIL import Image

type SingleInput = str | Path | Image.Image | NDArray[np.uint8]
type ImageInput = SingleInput | Sequence[SingleInput]
type BBox = tuple[float, float, float, float]
type Point = tuple[float, float]
type Polygon = list[Point]

SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})
SUPPORTED_INPUT_SUFFIXES = SUPPORTED_IMAGE_SUFFIXES | {".pdf"}


@dataclass(slots=True)
class PageImage:
    """Represents a single loaded image or rendered PDF page."""

    image: Image.Image
    source_path: str | None = None
    page_index: int | None = None


def _render_pdf_bytes(
    data: bytes,
    source_path: str | None,
    dpi: int,
    *,
    pdf_password: str | None = None,
) -> list[PageImage]:
    scale = dpi / 72.0
    try:
        doc: Any = pdfium.PdfDocument(data, password=pdf_password)
    except pdfium.PdfiumError as exc:
        if getattr(exc, "err_code", None) == pdfium_c.FPDF_ERR_PASSWORD:
            label = source_path or "<bytes>"
            if pdf_password:
                msg = f"Invalid PDF password for: {label}."
            else:
                msg = (
                    f"PDF is password-protected: {label}. "
                    "Provide `pdf_password` (or `--pdf_password` in CLI)."
                )
            raise ValueError(msg) from exc
        raise
    pages: list[PageImage] = []
    try:
        num_pages = len(doc)
        for idx in range(num_pages):
            page: Any = doc[idx]
            try:
                bitmap: Any = page.render(scale=scale)
                pil_img = cast(Image.Image, bitmap.to_pil()).convert("RGB")
                pages.append(PageImage(image=pil_img, source_path=source_path, page_index=idx))
            finally:
                page.close()
    finally:
        doc.close()
    return pages


def _load_single_input(
    item: SingleInput,
    *,
    pdf_dpi: int = 200,
    pdf_password: str | None = None,
) -> list[PageImage]:
    if isinstance(item, Image.Image):
        return [PageImage(image=item.convert("RGB"))]

    if isinstance(item, np.ndarray):
        arr = item
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr, mode="L").convert("RGB")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pil_img = Image.fromarray(arr, mode="RGB")
        elif arr.ndim == 3 and arr.shape[2] == 4:
            pil_img = Image.fromarray(arr, mode="RGBA").convert("RGB")
        else:
            msg = f"Unsupported numpy array shape: {arr.shape}. Expected (H, W) or (H, W, 3/4)."
            raise ValueError(msg)
        return [PageImage(image=pil_img)]

    raw_path = str(item)
    if raw_path.startswith(("http://", "https://")):
        response = httpx.get(raw_path, follow_redirects=True, timeout=30.0)
        response.raise_for_status()
        content = response.content
        if raw_path.lower().endswith(".pdf") or content.startswith(b"%PDF"):
            return _render_pdf_bytes(content, raw_path, pdf_dpi, pdf_password=pdf_password)
        img = Image.open(BytesIO(content)).convert("RGB")
        return [PageImage(image=img, source_path=raw_path)]

    path = Path(item)
    if not path.exists():
        msg = f"Input file not found: {path}"
        raise FileNotFoundError(msg)

    if path.is_dir():
        children = [
            child
            for child in sorted(path.iterdir())
            if child.is_file() and child.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
        ]
        if not children:
            msg = f"No supported image or PDF files found in directory: {path}"
            raise FileNotFoundError(msg)
        pages: list[PageImage] = []
        for child in children:
            pages.extend(_load_single_input(child, pdf_dpi=pdf_dpi, pdf_password=pdf_password))
        return pages

    if path.suffix.lower() == ".pdf":
        return _render_pdf_bytes(path.read_bytes(), str(path), pdf_dpi, pdf_password=pdf_password)

    img = Image.open(path).convert("RGB")
    return [PageImage(image=img, source_path=str(path))]


def load_inputs(
    inputs: ImageInput,
    *,
    pdf_dpi: int = 200,
    pdf_password: str | None = None,
) -> list[PageImage]:
    """Normalize single or batched image/PDF/URL/directory inputs into ``PageImage`` objects."""
    if isinstance(inputs, (str, Path, Image.Image, np.ndarray)):
        return _load_single_input(inputs, pdf_dpi=pdf_dpi, pdf_password=pdf_password)

    pages: list[PageImage] = []
    for item in inputs:
        pages.extend(_load_single_input(item, pdf_dpi=pdf_dpi, pdf_password=pdf_password))
    return pages


def rotate_image(image: Image.Image, angle: int) -> Image.Image:
    """Rotate an image to compensate for a detected orientation angle (0, 90, 180, 270)."""
    normalized = angle % 360
    if normalized == 90:
        return image.transpose(Image.Transpose.ROTATE_90)
    if normalized == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    if normalized == 270:
        return image.transpose(Image.Transpose.ROTATE_270)
    return image


def polygon_to_bbox(polygon: Sequence[Point]) -> BBox:
    """Convert a sequence of (x, y) points to an axis-aligned bounding box (xmin, ymin, xmax, ymax)."""
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def bbox_to_polygon(box: BBox) -> Polygon:
    """Convert an axis-aligned bounding box (xmin, ymin, xmax, ymax) to 4 clockwise corner points."""
    x1, y1, x2, y2 = box
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def crop_box(image: Image.Image, box: BBox) -> Image.Image:
    """Crop an axis-aligned bounding box (xmin, ymin, xmax, ymax) clamped to image bounds."""
    width, height = image.size
    x1 = max(0, min(math.floor(box[0]), width - 1))
    y1 = max(0, min(math.floor(box[1]), height - 1))
    x2 = max(x1 + 1, min(math.ceil(box[2]), width))
    y2 = max(y1 + 1, min(math.ceil(box[3]), height))
    return image.crop((x1, y1, x2, y2))


def crop_polygon(image: Image.Image, polygon: Sequence[Point]) -> Image.Image:
    """Crop a 4-point quadrilateral (or general polygon) from an image and rotate vertical text lines.

    When ``polygon`` has 4 points ordered [top-left, top-right, bottom-right, bottom-left],
    performs a quadrilateral transform to rectify skewed text lines. If the resulting crop has
    aspect ratio ``height / width >= 1.5``, rotates it 90 degrees counter-clockwise.
    """
    if len(polygon) != 4:
        return crop_box(image, polygon_to_bbox(polygon))

    p0, p1, p2, p3 = polygon
    top_w = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    bottom_w = math.hypot(p2[0] - p3[0], p2[1] - p3[1])
    left_h = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
    right_h = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    crop_w = max(1, round(max(top_w, bottom_w)))
    crop_h = max(1, round(max(left_h, right_h)))

    # PIL.Image.Transform.QUAD expects (ul, ll, lr, ur)
    quad_coords = (
        p0[0],
        p0[1],
        p3[0],
        p3[1],
        p2[0],
        p2[1],
        p1[0],
        p1[1],
    )
    cropped = image.transform(
        (crop_w, crop_h),
        Image.Transform.QUAD,
        data=quad_coords,
        resample=Image.Resampling.BILINEAR,
    )
    if crop_h / max(crop_w, 1) >= 1.5:
        cropped = cropped.transpose(Image.Transpose.ROTATE_90)
    return cropped


def sort_polygons_reading_order(
    items: Sequence[tuple[Polygon, float]],
    *,
    y_overlap_threshold: float = 10.0,
) -> list[tuple[Polygon, float]]:
    """Sort detected text polygons top-to-bottom and left-to-right."""
    if not items:
        return []

    sorted_by_y = sorted(items, key=lambda item: (item[0][0][1], item[0][0][0]))
    result: list[tuple[Polygon, float]] = list(sorted_by_y)
    n = len(result)
    for i in range(n - 1):
        for j in range(i, -1, -1):
            y_diff = abs(result[j + 1][0][0][1] - result[j][0][0][1])
            if y_diff < y_overlap_threshold and result[j + 1][0][0][0] < result[j][0][0][0]:
                result[j], result[j + 1] = result[j + 1], result[j]
            else:
                break
    return result


def box_intersection_over_area(inner: BBox, outer: BBox) -> float:
    """Compute the fraction of ``inner`` box area that lies inside ``outer`` box."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    inner_area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter_area / inner_area
