"""PyTorch & Hugging Face Transformers model runners and external VLM API client."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol, cast

import httpx
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForImageTextToText,
    AutoModelForObjectDetection,
    AutoModelForTableRecognition,
    AutoModelForTextRecognition,
    AutoProcessor,
)

from .io import BBox, Polygon, bbox_to_polygon, polygon_to_bbox


def _load_processor(cls: Any, model_id: str, **kwargs: Any) -> Any:
    loader: Any = cls
    return loader.from_pretrained(model_id, **kwargs)


def _load_model(cls: Any, model_id: str, *, device: torch.device, **kwargs: Any) -> Any:
    loader: Any = cls
    model: Any = loader.from_pretrained(model_id, **kwargs)
    return model.to(device).eval()


def _move_inputs_to_device(
    inputs: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    items = inputs.items() if hasattr(inputs, "items") else dict(inputs).items()
    for key, val in items:
        if isinstance(val, torch.Tensor):
            if val.is_floating_point():
                moved[str(key)] = val.to(device=device, dtype=dtype)
            else:
                moved[str(key)] = val.to(device=device)
        else:
            moved[str(key)] = val
    return moved


class ImageClassifierRunner:
    """Runner for orientation and table classification models (e.g. ``PP-LCNet``)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.processor: Any = _load_processor(
            AutoImageProcessor,
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.model: Any = _load_model(
            AutoModelForImageClassification,
            model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    @torch.inference_mode()
    def predict_batch(self, images: Sequence[Image.Image]) -> list[tuple[str, float]]:
        if not images:
            return []
        inputs = self.processor(images=list(images), return_tensors="pt")
        moved = _move_inputs_to_device(inputs, self.device, self.dtype)
        outputs = self.model(**moved)
        raw_logits = getattr(outputs, "logits", None)
        if raw_logits is None:
            raw_logits = outputs.last_hidden_state
        logits = cast(torch.Tensor, raw_logits)
        probs = torch.softmax(logits.float(), dim=-1)
        scores, pred_ids = probs.max(dim=-1)
        id2label: dict[Any, str] = getattr(self.model.config, "id2label", {})

        results: list[tuple[str, float]] = []
        for idx_tensor, score_tensor in zip(pred_ids, scores, strict=True):
            cls_id = int(idx_tensor.item())
            label = str(id2label.get(cls_id, id2label.get(str(cls_id), str(cls_id))))
            results.append((label, float(score_tensor.item())))
        return results

    def predict(self, image: Image.Image) -> tuple[str, float]:
        return self.predict_batch([image])[0]


class DocumentUnwarpingRunner:
    """Runner for geometric document unwarping/rectification models (e.g. ``UVDoc``)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.processor: Any = _load_processor(
            AutoImageProcessor,
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.model: Any = _load_model(
            AutoModel,
            model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Image.Image:
        inputs: Any = self.processor(images=image, return_tensors="pt")
        pixel_values = cast(torch.Tensor, inputs["pixel_values"]).to(
            device=self.device,
            dtype=self.dtype,
        )
        outputs: Any = self.model(pixel_values=pixel_values)
        rectified_list: list[dict[str, torch.Tensor]] = (
            self.processor.post_process_document_rectification(
                outputs.last_hidden_state.float(),
                inputs["original_images"],
            )
        )
        # UVDocImageProcessor returns (H, W, 3) uint8 in BGR order
        bgr_tensor = rectified_list[0]["images"]
        rgb_array = bgr_tensor.flip(dims=[-1]).detach().cpu().numpy().astype(np.uint8)
        return Image.fromarray(rgb_array, mode="RGB")


class TextDetectionRunner:
    """Runner for text & seal detection models (``PP-OCRv6_*_det``, ``PP-OCRv5_*_det``)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.processor: Any = _load_processor(
            AutoImageProcessor,
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.model: Any = _load_model(
            AutoModelForObjectDetection,
            model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        *,
        threshold: float = 0.3,
        box_threshold: float = 0.6,
        unclip_ratio: float = 1.5,
        max_candidates: int = 1000,
    ) -> list[tuple[Polygon, float]]:
        inputs: Any = self.processor(images=image, return_tensors="pt")
        pixel_values = cast(torch.Tensor, inputs["pixel_values"]).to(
            device=self.device,
            dtype=self.dtype,
        )
        target_sizes = inputs.get(
            "target_sizes",
            [torch.tensor([image.height, image.width], dtype=torch.float32, device=self.device)],
        )
        outputs: Any = self.model(pixel_values=pixel_values)
        raw_results: list[dict[str, Any]] = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            box_threshold=box_threshold,
            unclip_ratio=unclip_ratio,
            max_candidates=max_candidates,
        )
        if not raw_results:
            return []

        first = raw_results[0]
        boxes_tensor = cast(torch.Tensor, first["boxes"]).detach().cpu()
        scores_tensor = cast(torch.Tensor, first["scores"]).detach().cpu()

        detections: list[tuple[Polygon, float]] = []
        for box_item, score_item in zip(boxes_tensor, scores_tensor, strict=True):
            score_val = float(score_item.item())
            if box_item.ndim == 2 and box_item.shape[1] == 2:
                poly: Polygon = [(float(pt[0].item()), float(pt[1].item())) for pt in box_item]
            else:
                flat = [float(v.item()) for v in box_item.flatten()]
                poly = bbox_to_polygon((flat[0], flat[1], flat[2], flat[3]))
            detections.append((poly, score_val))
        return detections


class TextRecognitionRunner:
    """Runner for text line recognition models (``PP-OCRv6_*_rec``, ``PP-OCRv5_*_rec``)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.processor: Any = _load_processor(
            AutoImageProcessor,
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.model: Any = _load_model(
            AutoModelForTextRecognition,
            model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    @torch.inference_mode()
    def predict_batch(
        self,
        images: Sequence[Image.Image],
        *,
        batch_size: int = 8,
    ) -> list[tuple[str, float]]:
        if not images:
            return []

        results: list[tuple[str, float]] = []
        step = max(1, batch_size)
        for start in range(0, len(images), step):
            chunk = list(images[start : start + step])
            inputs: Any = self.processor(images=chunk, return_tensors="pt")
            pixel_values = cast(torch.Tensor, inputs["pixel_values"]).to(
                device=self.device,
                dtype=self.dtype,
            )
            outputs: Any = self.model(pixel_values=pixel_values)
            decoded: list[dict[str, Any]] = self.processor.post_process_text_recognition(outputs)
            for item in decoded:
                text = str(item.get("text", ""))
                raw_score = float(item.get("score", 0.0))
                score = 0.0 if np.isnan(raw_score) else raw_score
                results.append((text, score))
        return results


@dataclass(slots=True)
class DetectedLayoutItem:
    """A single detected layout block or table cell."""

    label: str
    box: BBox
    score: float
    polygon: Polygon | None = None
    order_index: int | None = None


class LayoutDetectionRunner:
    """Runner for document layout and table cell detection (``PP-DocLayoutV3/V4``, ``RT-DETR``)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.processor: Any = _load_processor(
            AutoImageProcessor,
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.model: Any = _load_model(
            AutoModelForObjectDetection,
            model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        *,
        threshold: float = 0.4,
    ) -> list[DetectedLayoutItem]:
        inputs: Any = self.processor(images=[image], return_tensors="pt")
        moved = _move_inputs_to_device(inputs, self.device, self.dtype)
        outputs: Any = self.model(**moved)
        target_sizes = torch.tensor(
            [[image.height, image.width]],
            dtype=torch.int64,
            device=self.device,
        )
        raw_results: list[dict[str, Any]] = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )
        if not raw_results:
            return []

        res = raw_results[0]
        scores = cast(torch.Tensor, res["scores"]).detach().cpu()
        labels = cast(torch.Tensor, res["labels"]).detach().cpu()
        boxes = cast(torch.Tensor, res["boxes"]).detach().cpu()
        raw_polygons: Any = res.get("polygon_points")
        raw_orders: Any = res.get("order_seq")
        id2label: dict[Any, str] = getattr(self.model.config, "id2label", {})

        items: list[DetectedLayoutItem] = []
        count = len(scores)
        for idx in range(count):
            score_val = float(scores[idx].item())
            label_id = int(labels[idx].item())
            label_str = str(id2label.get(label_id, id2label.get(str(label_id), str(label_id))))
            b = [float(v.item()) for v in boxes[idx].flatten()]
            bbox: BBox = (b[0], b[1], b[2], b[3])

            poly: Polygon | None = None
            if raw_polygons is not None and idx < len(raw_polygons):
                poly_elem = raw_polygons[idx]
                if isinstance(poly_elem, torch.Tensor):
                    poly_arr = poly_elem.detach().cpu().numpy()
                else:
                    poly_arr = np.asarray(poly_elem)
                if poly_arr.ndim == 2 and poly_arr.shape[1] == 2 and len(poly_arr) >= 3:
                    poly = [(float(pt[0]), float(pt[1])) for pt in poly_arr]

            order_val: int | None = idx
            if isinstance(raw_orders, torch.Tensor) and idx < len(raw_orders):
                order_val = int(raw_orders[idx].item())

            items.append(
                DetectedLayoutItem(
                    label=label_str,
                    box=bbox,
                    score=score_val,
                    polygon=poly,
                    order_index=order_val,
                )
            )
        return items


class TableStructureRunner:
    """Runner for table structure recognition (``SLANeXt_wired``, ``SLANeXt_wireless``, ``SLANet_plus``)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.processor: Any = _load_processor(
            AutoImageProcessor,
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.model: Any = _load_model(
            AutoModelForTableRecognition,
            model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> tuple[list[str], float]:
        inputs: Any = self.processor(images=image, return_tensors="pt")
        pixel_values = cast(torch.Tensor, inputs["pixel_values"]).to(
            device=self.device,
            dtype=self.dtype,
        )
        outputs: Any = self.model(pixel_values=pixel_values)
        decoded: dict[str, Any] = self.processor.post_process_table_recognition(outputs)
        structure = [str(tok) for tok in decoded.get("structure", [])]
        score = float(decoded.get("structure_score", 0.0))
        return structure, score


class FormulaRecognitionRunner:
    """Runner for mathematical formula recognition (``PP-FormulaNet_plus-*``)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.processor: Any = _load_processor(
            AutoProcessor,
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.model: Any = _load_model(
            AutoModelForImageTextToText,
            model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image, *, max_new_tokens: int = 512) -> str:
        inputs: Any = self.processor(images=image, return_tensors="pt")
        moved = _move_inputs_to_device(inputs, self.device, self.dtype)
        generated_ids: Any = self.model.generate(
            **moved,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        if hasattr(self.processor, "post_process"):
            decoded: list[str] = self.processor.post_process(generated_ids)
            return str(decoded[0]).strip() if decoded else ""
        decoded_raw: list[str] = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return str(decoded_raw[0]).strip() if decoded_raw else ""


class ChartParsingRunner:
    """Runner for chart-to-table conversion (``PP-Chart2Table``)."""

    def __init__(
        self,
        model_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.processor: Any = _load_processor(
            AutoProcessor,
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.model: Any = _load_model(
            AutoModelForImageTextToText,
            model_id,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image, *, max_new_tokens: int = 512) -> str:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}],
            }
        ]
        inputs: Any = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            return_dict=True,
            return_tensors="pt",
        )
        moved = _move_inputs_to_device(inputs, self.device, self.dtype)
        generated_ids = cast(
            torch.Tensor,
            self.model.generate(
                **moved,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            ),
        )
        input_ids = cast(torch.Tensor, moved["input_ids"])
        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids, strict=True)
        ]
        decoded: list[str] = self.processor.batch_decode(trimmed, skip_special_tokens=True)
        return str(decoded[0]).strip() if decoded else ""


class VLMRecognizer(Protocol):
    """Protocol implemented by both local PyTorch VLM runner and external API VLM client."""

    def generate(
        self,
        image: Image.Image,
        prompt: str = "OCR:",
        *,
        max_new_tokens: int = 1024,
    ) -> str: ...

    def generate_batch(
        self,
        requests: Sequence[tuple[Image.Image, str]],
        *,
        max_new_tokens: int = 1024,
    ) -> list[str]: ...


class LocalVLMRunner:
    """Local PyTorch Hugging Face ``transformers`` runner for ``PaddleOCR-VL-1.6`` / ``1.5``."""

    def __init__(
        self,
        model_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.processor: Any = _load_processor(
            AutoProcessor,
            model_id,
            trust_remote_code=trust_remote_code,
        )
        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        self.model: Any = _load_model(
            AutoModelForImageTextToText,
            model_id,
            device=device,
            **load_kwargs,
        )

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        prompt: str = "OCR:",
        *,
        max_new_tokens: int = 1024,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs: Any = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        moved = _move_inputs_to_device(inputs, self.device, self.dtype)
        generated_ids = cast(
            torch.Tensor,
            self.model.generate(
                **moved,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            ),
        )
        input_ids = cast(torch.Tensor, moved["input_ids"])
        trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids, strict=True)
        ]
        decoded: list[str] = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return str(decoded[0]).strip() if decoded else ""

    def generate_batch(
        self,
        requests: Sequence[tuple[Image.Image, str]],
        *,
        max_new_tokens: int = 1024,
    ) -> list[str]:
        return [
            self.generate(img, prompt, max_new_tokens=max_new_tokens) for img, prompt in requests
        ]


def _encode_image_base64_uri(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class ExternalVLMClient:
    """OpenAI-compatible API client for ``PaddleOCRVL`` (vLLM, SGLang, FastDeploy, MLX, OpenAI)."""

    def __init__(
        self,
        server_url: str,
        *,
        model_name: str = "PaddlePaddle/PaddleOCR-VL-1.6",
        api_key: str | None = None,
        timeout: float = 60.0,
        max_concurrency: int = 8,
        http_client: httpx.Client | None = None,
    ) -> None:
        base = server_url.strip().rstrip("/")
        if not base.endswith("/v1") and not base.endswith("/chat/completions"):
            base = f"{base}/v1"
        self.endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_concurrency = max(1, max_concurrency)
        self._client = http_client

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(
        self,
        image: Image.Image,
        prompt: str = "OCR:",
        *,
        max_new_tokens: int = 1024,
    ) -> str:
        image_uri = _encode_image_base64_uri(image)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }
        if self._client is not None:
            response = self._client.post(
                self.endpoint,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        else:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint,
                    json=payload,
                    headers=self._headers(),
                )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        choices = cast(list[dict[str, Any]], data.get("choices", []))
        if not choices:
            return ""
        message = cast(dict[str, Any], choices[0].get("message", {}))
        return str(message.get("content", "")).strip()

    def generate_batch(
        self,
        requests: Sequence[tuple[Image.Image, str]],
        *,
        max_new_tokens: int = 1024,
    ) -> list[str]:
        if not requests:
            return []
        if len(requests) == 1 or self.max_concurrency == 1:
            return [
                self.generate(img, prompt, max_new_tokens=max_new_tokens)
                for img, prompt in requests
            ]

        def _worker(item: tuple[Image.Image, str]) -> str:
            return self.generate(item[0], item[1], max_new_tokens=max_new_tokens)

        workers = min(self.max_concurrency, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_worker, requests))


def parse_orientation_angle(label: str) -> int:
    """Extract integer rotation angle (0, 90, 180, 270) from a classifier label string."""
    digits = "".join(ch for ch in label if ch.isdigit())
    if digits in {"90", "180", "270"}:
        return int(digits)
    return 0


__all__ = [
    "ChartParsingRunner",
    "DetectedLayoutItem",
    "DocumentUnwarpingRunner",
    "ExternalVLMClient",
    "FormulaRecognitionRunner",
    "ImageClassifierRunner",
    "LayoutDetectionRunner",
    "LocalVLMRunner",
    "TableStructureRunner",
    "TextDetectionRunner",
    "TextRecognitionRunner",
    "VLMRecognizer",
    "parse_orientation_angle",
    "polygon_to_bbox",
]
