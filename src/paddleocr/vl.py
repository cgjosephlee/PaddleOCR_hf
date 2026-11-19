"""Pure PyTorch & Hugging Face Transformers implementation of ``PaddleOCRVL`` (v1.6)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .device import DeviceLike, DTypeLike, resolve_device, resolve_dtype
from .io import ImageInput, crop_box, load_inputs, rotate_image
from .models import (
    DocumentUnwarpingRunner,
    ExternalVLMClient,
    ImageClassifierRunner,
    LayoutDetectionRunner,
    LocalVLMRunner,
    VLMRecognizer,
    parse_orientation_angle,
)
from .registry import (
    DEFAULT_DOC_ORI_MODEL,
    DEFAULT_LAYOUT_MODEL,
    DEFAULT_UNWARP_MODEL,
    DEFAULT_VLM_MODEL,
    VLM_PIPELINE_DEFAULTS,
    resolve_model_id,
)
from .results import LayoutBlock, VLResult, format_blocks_to_markdown, html_table_to_markdown

EXTERNAL_VLM_BACKENDS = frozenset(
    {
        "vllm-server",
        "sglang-server",
        "fastdeploy-server",
        "mlx-vlm-server",
        "llama-cpp-server",
        "openai",
        "api",
        "external",
    }
)

_SKIP_VLM_LABELS = frozenset({"image", "figure", "header_image", "footer_image"})


def _prompt_for_layout_label(label: str) -> str:
    lower = label.lower()
    if "table" in lower:
        return "Table Recognition:"
    if "formula" in lower or "equation" in lower:
        return "Formula Recognition:"
    if "chart" in lower:
        return "Chart Recognition:"
    if "seal" in lower:
        return "Seal Recognition:"
    return "OCR:"


class PaddleOCRVL:
    """Vision-Language Document Parsing pipeline powered by ``PaddleOCR-VL-1.6``.

    Supports:
    - Local PyTorch execution via Hugging Face ``transformers`` on ``mps``, ``cuda``, or ``cpu``.
    - External OpenAI-compatible API execution (``vllm-server``, ``sglang-server``, ``openai``, etc.)
      with concurrent multi-region requests.
    """

    def __init__(
        self,
        *,
        pipeline_version: str = "v1.6",
        vl_rec_model_name: str | Path | None = None,
        layout_detection_model_name: str | Path | None = None,
        doc_orientation_classify_model_name: str | Path | None = None,
        doc_unwarping_model_name: str | Path | None = None,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_layout_detection: bool = True,
        layout_threshold: float = 0.4,
        vl_rec_backend: str = "transformers",
        vl_rec_server_url: str | None = None,
        vl_rec_api_model_name: str | None = None,
        vl_rec_api_key: str | None = None,
        vl_rec_max_concurrency: int = 8,
        vl_rec_timeout: float = 60.0,
        max_new_tokens: int = 1024,
        pdf_dpi: int = 200,
        pdf_password: str | None = None,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
        attn_implementation: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.pipeline_version = pipeline_version
        self.device = resolve_device(device)
        self.vision_dtype = resolve_dtype(None, device=self.device, is_vlm=False)
        self.vlm_dtype = resolve_dtype(dtype, device=self.device, is_vlm=True)
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code

        default_vlm, default_layout = VLM_PIPELINE_DEFAULTS.get(
            pipeline_version,
            (DEFAULT_VLM_MODEL, DEFAULT_LAYOUT_MODEL),
        )
        self.vl_model_id = resolve_model_id(vl_rec_model_name, default_vlm)
        self.layout_model_id = resolve_model_id(layout_detection_model_name, default_layout)
        self.doc_ori_model_id = resolve_model_id(
            doc_orientation_classify_model_name,
            DEFAULT_DOC_ORI_MODEL,
        )
        self.unwarp_model_id = resolve_model_id(doc_unwarping_model_name, DEFAULT_UNWARP_MODEL)

        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_layout_detection = use_layout_detection
        self.layout_threshold = layout_threshold

        self.vl_rec_backend = vl_rec_backend.strip().lower()
        self.vl_rec_server_url = vl_rec_server_url
        self.vl_rec_api_model_name = vl_rec_api_model_name or self.vl_model_id
        self.vl_rec_api_key = vl_rec_api_key
        self.vl_rec_max_concurrency = vl_rec_max_concurrency
        self.vl_rec_timeout = vl_rec_timeout
        self.max_new_tokens = max_new_tokens
        self.pdf_dpi = pdf_dpi
        self.pdf_password = pdf_password

        self._doc_ori_runner: ImageClassifierRunner | None = None
        self._unwarp_runner: DocumentUnwarpingRunner | None = None
        self._layout_runner: LayoutDetectionRunner | None = None
        self._local_vlm_runner: LocalVLMRunner | None = None
        self._external_vlm_client: ExternalVLMClient | None = None

    @property
    def uses_external_api(self) -> bool:
        """Return ``True`` if VLM recognition is configured to use an external API server."""
        return self.vl_rec_backend in EXTERNAL_VLM_BACKENDS or self.vl_rec_server_url is not None

    def _get_doc_ori_runner(self) -> ImageClassifierRunner:
        if self._doc_ori_runner is None:
            self._doc_ori_runner = ImageClassifierRunner(
                self.doc_ori_model_id,
                device=self.device,
                dtype=self.vision_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._doc_ori_runner

    def _get_unwarp_runner(self) -> DocumentUnwarpingRunner:
        if self._unwarp_runner is None:
            self._unwarp_runner = DocumentUnwarpingRunner(
                self.unwarp_model_id,
                device=self.device,
                dtype=self.vision_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._unwarp_runner

    def _get_layout_runner(self, model_name_override: str | None = None) -> LayoutDetectionRunner:
        target_id = (
            resolve_model_id(model_name_override, self.layout_model_id)
            if model_name_override
            else self.layout_model_id
        )
        if self._layout_runner is None or self._layout_runner.model_id != target_id:
            self._layout_runner = LayoutDetectionRunner(
                target_id,
                device=self.device,
                dtype=self.vision_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._layout_runner

    def _get_vlm_recognizer(self) -> VLMRecognizer:
        if self.uses_external_api:
            if not self.vl_rec_server_url:
                msg = f"vl_rec_server_url must be provided when vl_rec_backend='{self.vl_rec_backend}'."
                raise ValueError(msg)
            if self._external_vlm_client is None:
                self._external_vlm_client = ExternalVLMClient(
                    self.vl_rec_server_url,
                    model_name=self.vl_rec_api_model_name,
                    api_key=self.vl_rec_api_key,
                    timeout=self.vl_rec_timeout,
                    max_concurrency=self.vl_rec_max_concurrency,
                )
            return self._external_vlm_client

        if self._local_vlm_runner is None:
            self._local_vlm_runner = LocalVLMRunner(
                self.vl_model_id,
                device=self.device,
                dtype=self.vlm_dtype,
                trust_remote_code=self.trust_remote_code,
                attn_implementation=self.attn_implementation,
            )
        return self._local_vlm_runner

    def predict(
        self,
        input: ImageInput,
        *,
        use_doc_orientation_classify: bool | None = None,
        use_doc_unwarping: bool | None = None,
        use_layout_detection: bool | None = None,
        layout_detection_model_name: str | None = None,
        layout_threshold: float | None = None,
        prompt: str = "OCR:",
        max_new_tokens: int | None = None,
        pdf_password: str | None = None,
    ) -> list[VLResult]:
        """Run document parsing using ``PaddleOCR-VL-1.6`` (local PyTorch or external API)."""
        do_doc_ori = (
            self.use_doc_orientation_classify
            if use_doc_orientation_classify is None
            else use_doc_orientation_classify
        )
        do_unwarp = self.use_doc_unwarping if use_doc_unwarping is None else use_doc_unwarping
        do_layout = (
            self.use_layout_detection if use_layout_detection is None else use_layout_detection
        )
        thresh = self.layout_threshold if layout_threshold is None else layout_threshold
        tokens_limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        effective_password = self.pdf_password if pdf_password is None else pdf_password

        pages = load_inputs(input, pdf_dpi=self.pdf_dpi, pdf_password=effective_password)
        vlm = self._get_vlm_recognizer()
        results: list[VLResult] = []

        for page in pages:
            working_img = page.image
            doc_angle: int | None = None

            if do_doc_ori:
                ori_label, _ = self._get_doc_ori_runner().predict(working_img)
                doc_angle = parse_orientation_angle(ori_label)
                if doc_angle:
                    working_img = rotate_image(working_img, doc_angle)

            if do_unwarp:
                working_img = self._get_unwarp_runner().predict(working_img)

            if not do_layout:
                full_text = vlm.generate(
                    working_img,
                    prompt=prompt,
                    max_new_tokens=tokens_limit,
                )
                w, h = working_img.size
                block = LayoutBlock(
                    label="document",
                    box=(0.0, 0.0, float(w), float(h)),
                    score=1.0,
                    content=full_text,
                    order_index=0,
                )
                results.append(
                    VLResult(
                        blocks=[block],
                        markdown_text=full_text,
                        input_path=page.source_path,
                        page_index=page.page_index,
                        doc_orientation_angle=doc_angle,
                        image=working_img,
                    )
                )
                continue

            layout_items = self._get_layout_runner(layout_detection_model_name).predict(
                working_img,
                threshold=thresh,
            )
            blocks: list[LayoutBlock] = []
            pending_indices: list[int] = []
            vlm_requests: list[tuple[Image.Image, str]] = []

            for idx, item in enumerate(layout_items):
                block = LayoutBlock(
                    label=item.label,
                    box=item.box,
                    score=item.score,
                    polygon=item.polygon,
                    order_index=item.order_index if item.order_index is not None else idx,
                )
                blocks.append(block)
                if item.label.lower() not in _SKIP_VLM_LABELS:
                    crop = crop_box(working_img, item.box)
                    task_prompt = _prompt_for_layout_label(item.label)
                    pending_indices.append(idx)
                    vlm_requests.append((crop, task_prompt))

            if vlm_requests:
                vlm_outputs = vlm.generate_batch(vlm_requests, max_new_tokens=tokens_limit)
                for block_idx, recognized in zip(pending_indices, vlm_outputs, strict=True):
                    blk = blocks[block_idx]
                    if blk.label.lower() == "table" and "<table" in recognized.lower():
                        blk.html = recognized
                        blk.content = html_table_to_markdown(recognized)
                    else:
                        blk.content = recognized

            md_text = format_blocks_to_markdown(blocks)
            results.append(
                VLResult(
                    blocks=blocks,
                    markdown_text=md_text,
                    input_path=page.source_path,
                    page_index=page.page_index,
                    doc_orientation_angle=doc_angle,
                    image=working_img,
                )
            )

        return results

    def __call__(
        self,
        input: ImageInput,
        *,
        use_doc_orientation_classify: bool | None = None,
        use_doc_unwarping: bool | None = None,
        use_layout_detection: bool | None = None,
        layout_detection_model_name: str | None = None,
        layout_threshold: float | None = None,
        prompt: str = "OCR:",
        max_new_tokens: int | None = None,
        pdf_password: str | None = None,
    ) -> list[VLResult]:
        return self.predict(
            input,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_layout_detection=use_layout_detection,
            layout_detection_model_name=layout_detection_model_name,
            layout_threshold=layout_threshold,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            pdf_password=pdf_password,
        )
