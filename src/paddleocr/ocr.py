"""Pure PyTorch & Hugging Face Transformers implementation of the ``PaddleOCR`` service."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from .device import DeviceLike, DTypeLike, resolve_device, resolve_dtype
from .io import (
    ImageInput,
    Polygon,
    crop_polygon,
    load_inputs,
    polygon_to_bbox,
    rotate_image,
    sort_polygons_reading_order,
)
from .models import (
    DocumentUnwarpingRunner,
    ImageClassifierRunner,
    TextDetectionRunner,
    TextRecognitionRunner,
    parse_orientation_angle,
)
from .registry import (
    DEFAULT_DOC_ORI_MODEL,
    DEFAULT_TEXT_DET_MODEL,
    DEFAULT_TEXT_REC_MODEL,
    DEFAULT_TEXTLINE_ORI_MODEL,
    DEFAULT_UNWARP_MODEL,
    OCR_VERSION_DEFAULTS,
    resolve_model_id,
)
from .results import OCRResult, TextBox


class PaddleOCR:
    """Two-stage general OCR pipeline powered by PP-OCRv6 and Hugging Face ``transformers``.

    Supports native PyTorch devices (``mps`` on Apple Silicon, ``cuda``, ``cpu``) with
    zero PaddlePaddle dependencies.
    """

    def __init__(
        self,
        *,
        ocr_version: str = "PP-OCRv6",
        text_detection_model_name: str | Path | None = None,
        text_recognition_model_name: str | Path | None = None,
        doc_orientation_classify_model_name: str | Path | None = None,
        doc_unwarping_model_name: str | Path | None = None,
        textline_orientation_model_name: str | Path | None = None,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        text_det_thresh: float = 0.3,
        text_det_box_thresh: float = 0.6,
        text_det_unclip_ratio: float = 1.5,
        text_rec_score_thresh: float = 0.0,
        text_recognition_batch_size: int = 8,
        pdf_dpi: int = 200,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype, device=self.device, is_vlm=False)
        self.trust_remote_code = trust_remote_code

        default_det, default_rec = OCR_VERSION_DEFAULTS.get(
            ocr_version,
            (DEFAULT_TEXT_DET_MODEL, DEFAULT_TEXT_REC_MODEL),
        )
        self.det_model_id = resolve_model_id(text_detection_model_name, default_det)
        self.rec_model_id = resolve_model_id(text_recognition_model_name, default_rec)
        self.doc_ori_model_id = resolve_model_id(
            doc_orientation_classify_model_name,
            DEFAULT_DOC_ORI_MODEL,
        )
        self.unwarp_model_id = resolve_model_id(doc_unwarping_model_name, DEFAULT_UNWARP_MODEL)
        self.textline_ori_model_id = resolve_model_id(
            textline_orientation_model_name,
            DEFAULT_TEXTLINE_ORI_MODEL,
        )

        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation

        self.text_det_thresh = text_det_thresh
        self.text_det_box_thresh = text_det_box_thresh
        self.text_det_unclip_ratio = text_det_unclip_ratio
        self.text_rec_score_thresh = text_rec_score_thresh
        self.text_recognition_batch_size = text_recognition_batch_size
        self.pdf_dpi = pdf_dpi

        self._doc_ori_runner: ImageClassifierRunner | None = None
        self._unwarp_runner: DocumentUnwarpingRunner | None = None
        self._textline_ori_runner: ImageClassifierRunner | None = None
        self._det_runner: TextDetectionRunner | None = None
        self._rec_runner: TextRecognitionRunner | None = None

    def _get_doc_ori_runner(self) -> ImageClassifierRunner:
        if self._doc_ori_runner is None:
            self._doc_ori_runner = ImageClassifierRunner(
                self.doc_ori_model_id,
                device=self.device,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._doc_ori_runner

    def _get_unwarp_runner(self) -> DocumentUnwarpingRunner:
        if self._unwarp_runner is None:
            self._unwarp_runner = DocumentUnwarpingRunner(
                self.unwarp_model_id,
                device=self.device,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._unwarp_runner

    def _get_textline_ori_runner(self) -> ImageClassifierRunner:
        if self._textline_ori_runner is None:
            self._textline_ori_runner = ImageClassifierRunner(
                self.textline_ori_model_id,
                device=self.device,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._textline_ori_runner

    def _get_det_runner(self) -> TextDetectionRunner:
        if self._det_runner is None:
            self._det_runner = TextDetectionRunner(
                self.det_model_id,
                device=self.device,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._det_runner

    def _get_rec_runner(self) -> TextRecognitionRunner:
        if self._rec_runner is None:
            self._rec_runner = TextRecognitionRunner(
                self.rec_model_id,
                device=self.device,
                dtype=self.dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._rec_runner

    @property
    def det_runner(self) -> TextDetectionRunner:
        """Lazily initialized text detection runner."""
        return self._get_det_runner()

    @property
    def rec_runner(self) -> TextRecognitionRunner:
        """Lazily initialized text recognition runner."""
        return self._get_rec_runner()

    def detect_text_polygons(
        self,
        image: Image.Image,
        *,
        text_det_thresh: float | None = None,
        text_det_box_thresh: float | None = None,
        text_det_unclip_ratio: float | None = None,
    ) -> list[tuple[Polygon, float]]:
        """Detect text polygons and confidence scores on an image."""
        det_thresh = self.text_det_thresh if text_det_thresh is None else text_det_thresh
        box_thresh = (
            self.text_det_box_thresh if text_det_box_thresh is None else text_det_box_thresh
        )
        unclip_ratio = (
            self.text_det_unclip_ratio if text_det_unclip_ratio is None else text_det_unclip_ratio
        )
        return self._get_det_runner().predict(
            image,
            threshold=det_thresh,
            box_threshold=box_thresh,
            unclip_ratio=unclip_ratio,
        )

    def recognize_text_crops(
        self,
        crops: Sequence[Image.Image],
        *,
        batch_size: int | None = None,
    ) -> list[tuple[str, float]]:
        """Recognize text strings and confidence scores from pre-cropped text line images."""
        effective_batch = self.text_recognition_batch_size if batch_size is None else batch_size
        return self._get_rec_runner().predict_batch(crops, batch_size=effective_batch)

    def extract_text_boxes(
        self,
        image: Image.Image,
        *,
        use_textline_orientation: bool | None = None,
        text_det_thresh: float | None = None,
        text_det_box_thresh: float | None = None,
        text_det_unclip_ratio: float | None = None,
        text_rec_score_thresh: float | None = None,
    ) -> list[TextBox]:
        """Run text detection, optional textline orientation correction, and recognition on an image."""
        det_thresh = self.text_det_thresh if text_det_thresh is None else text_det_thresh
        box_thresh = (
            self.text_det_box_thresh if text_det_box_thresh is None else text_det_box_thresh
        )
        unclip_ratio = (
            self.text_det_unclip_ratio if text_det_unclip_ratio is None else text_det_unclip_ratio
        )
        rec_thresh = (
            self.text_rec_score_thresh if text_rec_score_thresh is None else text_rec_score_thresh
        )
        do_textline_ori = (
            self.use_textline_orientation
            if use_textline_orientation is None
            else use_textline_orientation
        )

        raw_detections = self._get_det_runner().predict(
            image,
            threshold=det_thresh,
            box_threshold=box_thresh,
            unclip_ratio=unclip_ratio,
        )
        ordered_detections = sort_polygons_reading_order(raw_detections)
        if not ordered_detections:
            return []

        crops: list[Image.Image] = [crop_polygon(image, poly) for poly, _ in ordered_detections]

        if do_textline_ori and crops:
            ori_preds = self._get_textline_ori_runner().predict_batch(crops)
            rotated_crops: list[Image.Image] = []
            for crop_img, (ori_label, _) in zip(crops, ori_preds, strict=True):
                angle = parse_orientation_angle(ori_label)
                rotated_crops.append(rotate_image(crop_img, 180) if angle == 180 else crop_img)
            crops = rotated_crops

        rec_preds = self._get_rec_runner().predict_batch(
            crops,
            batch_size=self.text_recognition_batch_size,
        )

        boxes: list[TextBox] = []
        for (poly, det_score), (text, rec_score) in zip(
            ordered_detections,
            rec_preds,
            strict=True,
        ):
            if rec_score < rec_thresh:
                continue
            boxes.append(
                TextBox(
                    box=polygon_to_bbox(poly),
                    polygon=poly,
                    text=text,
                    score=rec_score,
                    det_score=det_score,
                )
            )
        return boxes

    def predict(
        self,
        input: ImageInput,
        *,
        use_doc_orientation_classify: bool | None = None,
        use_doc_unwarping: bool | None = None,
        use_textline_orientation: bool | None = None,
        text_det_thresh: float | None = None,
        text_det_box_thresh: float | None = None,
        text_det_unclip_ratio: float | None = None,
        text_rec_score_thresh: float | None = None,
    ) -> list[OCRResult]:
        """Run the full OCR pipeline on one or more images, URLs, or PDF documents."""
        do_doc_ori = (
            self.use_doc_orientation_classify
            if use_doc_orientation_classify is None
            else use_doc_orientation_classify
        )
        do_unwarp = self.use_doc_unwarping if use_doc_unwarping is None else use_doc_unwarping

        pages = load_inputs(input, pdf_dpi=self.pdf_dpi)
        results: list[OCRResult] = []
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

            boxes = self.extract_text_boxes(
                working_img,
                use_textline_orientation=use_textline_orientation,
                text_det_thresh=text_det_thresh,
                text_det_box_thresh=text_det_box_thresh,
                text_det_unclip_ratio=text_det_unclip_ratio,
                text_rec_score_thresh=text_rec_score_thresh,
            )
            results.append(
                OCRResult(
                    boxes=boxes,
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
        use_textline_orientation: bool | None = None,
        text_det_thresh: float | None = None,
        text_det_box_thresh: float | None = None,
        text_det_unclip_ratio: float | None = None,
        text_rec_score_thresh: float | None = None,
    ) -> list[OCRResult]:
        return self.predict(
            input,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_det_unclip_ratio=text_det_unclip_ratio,
            text_rec_score_thresh=text_rec_score_thresh,
        )
