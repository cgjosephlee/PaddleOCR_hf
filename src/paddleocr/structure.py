"""Pure PyTorch & Hugging Face Transformers implementation of ``PPStructureV3``."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .device import DeviceLike, DTypeLike, resolve_device, resolve_dtype
from .io import (
    BBox,
    ImageInput,
    box_intersection_over_area,
    crop_box,
    crop_polygon,
    load_inputs,
    rotate_image,
    sort_polygons_reading_order,
)
from .models import (
    ChartParsingRunner,
    DocumentUnwarpingRunner,
    FormulaRecognitionRunner,
    ImageClassifierRunner,
    LayoutDetectionRunner,
    TableStructureRunner,
    TextDetectionRunner,
    parse_orientation_angle,
)
from .ocr import PaddleOCR
from .registry import (
    DEFAULT_CHART_MODEL,
    DEFAULT_DOC_ORI_MODEL,
    DEFAULT_FORMULA_MODEL,
    DEFAULT_LAYOUT_MODEL,
    DEFAULT_SEAL_DET_MODEL,
    DEFAULT_TABLE_CLS_MODEL,
    DEFAULT_TEXT_DET_MODEL,
    DEFAULT_TEXT_REC_MODEL,
    DEFAULT_TEXTLINE_ORI_MODEL,
    DEFAULT_UNWARP_MODEL,
    DEFAULT_WIRED_TABLE_CELL_MODEL,
    DEFAULT_WIRED_TABLE_STRUCT_MODEL,
    DEFAULT_WIRELESS_TABLE_CELL_MODEL,
    DEFAULT_WIRELESS_TABLE_STRUCT_MODEL,
    resolve_model_id,
)
from .results import (
    LayoutBlock,
    StructureResult,
    TableResult,
    TextBox,
    assemble_table_html,
    html_table_to_markdown,
)

_FORMULA_LABELS = frozenset({"formula", "equation", "display_formula", "inline_formula"})
_IMAGE_ONLY_LABELS = frozenset({"image", "figure", "header_image", "footer_image"})


def _sort_bboxes_reading_order(boxes: list[BBox], *, y_tolerance: float = 10.0) -> list[BBox]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: (b[1], b[0]))
    n = len(ordered)
    for i in range(n - 1):
        for j in range(i, -1, -1):
            if (
                abs(ordered[j + 1][1] - ordered[j][1]) < y_tolerance
                and ordered[j + 1][0] < ordered[j][0]
            ):
                ordered[j], ordered[j + 1] = ordered[j + 1], ordered[j]
            else:
                break
    return ordered


class PPStructureV3:
    """Modular document structure analysis pipeline powered by PP-OCRv6 & Hugging Face ``transformers``.

    Supports Layout Analysis (``PP-DocLayoutV3/V4``), General OCR (``PP-OCRv6``),
    Table Recognition (``SLANeXt`` + ``RT-DETR``), Formula Recognition (``PP-FormulaNet_plus``),
    Seal Recognition (``PP-OCRv4_seal``), Chart Parsing (``PP-Chart2Table``),
    Document Orientation Classification, and Document Unwarping (``UVDoc``).

    All sub-models are lazily loaded on first use to conserve unified memory on macOS (``mps``)
    and GPU VRAM (``cuda``).
    """

    def __init__(
        self,
        *,
        layout_detection_model_name: str | Path | None = None,
        text_detection_model_name: str | Path | None = None,
        text_recognition_model_name: str | Path | None = None,
        textline_orientation_model_name: str | Path | None = None,
        doc_orientation_classify_model_name: str | Path | None = None,
        doc_unwarping_model_name: str | Path | None = None,
        table_classification_model_name: str | Path | None = None,
        wired_table_structure_recognition_model_name: str | Path | None = None,
        wireless_table_structure_recognition_model_name: str | Path | None = None,
        wired_table_cells_detection_model_name: str | Path | None = None,
        wireless_table_cells_detection_model_name: str | Path | None = None,
        formula_recognition_model_name: str | Path | None = None,
        seal_text_detection_model_name: str | Path | None = None,
        chart_recognition_model_name: str | Path | None = None,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        use_layout_detection: bool = True,
        use_table_recognition: bool = True,
        use_formula_recognition: bool = True,
        use_seal_recognition: bool = False,
        use_chart_recognition: bool = False,
        layout_threshold: float = 0.4,
        pdf_dpi: int = 200,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        self.vision_dtype = resolve_dtype(dtype, device=self.device, is_vlm=False)
        self.seq2seq_dtype = resolve_dtype(dtype, device=self.device, is_vlm=True)
        self.trust_remote_code = trust_remote_code

        self.layout_model_id = resolve_model_id(layout_detection_model_name, DEFAULT_LAYOUT_MODEL)
        self.doc_ori_model_id = resolve_model_id(
            doc_orientation_classify_model_name,
            DEFAULT_DOC_ORI_MODEL,
        )
        self.unwarp_model_id = resolve_model_id(doc_unwarping_model_name, DEFAULT_UNWARP_MODEL)
        self.table_cls_model_id = resolve_model_id(
            table_classification_model_name,
            DEFAULT_TABLE_CLS_MODEL,
        )
        self.wired_table_struct_model_id = resolve_model_id(
            wired_table_structure_recognition_model_name,
            DEFAULT_WIRED_TABLE_STRUCT_MODEL,
        )
        self.wireless_table_struct_model_id = resolve_model_id(
            wireless_table_structure_recognition_model_name,
            DEFAULT_WIRELESS_TABLE_STRUCT_MODEL,
        )
        self.wired_table_cell_model_id = resolve_model_id(
            wired_table_cells_detection_model_name,
            DEFAULT_WIRED_TABLE_CELL_MODEL,
        )
        self.wireless_table_cell_model_id = resolve_model_id(
            wireless_table_cells_detection_model_name,
            DEFAULT_WIRELESS_TABLE_CELL_MODEL,
        )
        self.formula_model_id = resolve_model_id(
            formula_recognition_model_name,
            DEFAULT_FORMULA_MODEL,
        )
        self.seal_det_model_id = resolve_model_id(
            seal_text_detection_model_name,
            DEFAULT_SEAL_DET_MODEL,
        )
        self.chart_model_id = resolve_model_id(
            chart_recognition_model_name,
            DEFAULT_CHART_MODEL,
        )

        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        self.use_layout_detection = use_layout_detection
        self.use_table_recognition = use_table_recognition
        self.use_formula_recognition = use_formula_recognition
        self.use_seal_recognition = use_seal_recognition
        self.use_chart_recognition = use_chart_recognition
        self.layout_threshold = layout_threshold
        self.pdf_dpi = pdf_dpi

        self.ocr = PaddleOCR(
            text_detection_model_name=resolve_model_id(
                text_detection_model_name,
                DEFAULT_TEXT_DET_MODEL,
            ),
            text_recognition_model_name=resolve_model_id(
                text_recognition_model_name,
                DEFAULT_TEXT_REC_MODEL,
            ),
            textline_orientation_model_name=resolve_model_id(
                textline_orientation_model_name,
                DEFAULT_TEXTLINE_ORI_MODEL,
            ),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=use_textline_orientation,
            pdf_dpi=pdf_dpi,
            device=self.device,
            dtype=self.vision_dtype,
            trust_remote_code=trust_remote_code,
        )

        self._doc_ori_runner: ImageClassifierRunner | None = None
        self._unwarp_runner: DocumentUnwarpingRunner | None = None
        self._layout_runner: LayoutDetectionRunner | None = None
        self._table_cls_runner: ImageClassifierRunner | None = None
        self._wired_table_struct_runner: TableStructureRunner | None = None
        self._wireless_table_struct_runner: TableStructureRunner | None = None
        self._wired_table_cell_runner: LayoutDetectionRunner | None = None
        self._wireless_table_cell_runner: LayoutDetectionRunner | None = None
        self._formula_runner: FormulaRecognitionRunner | None = None
        self._seal_det_runner: TextDetectionRunner | None = None
        self._chart_runner: ChartParsingRunner | None = None

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

    def _get_layout_runner(self) -> LayoutDetectionRunner:
        if self._layout_runner is None:
            self._layout_runner = LayoutDetectionRunner(
                self.layout_model_id,
                device=self.device,
                dtype=self.vision_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._layout_runner

    def _get_table_cls_runner(self) -> ImageClassifierRunner:
        if self._table_cls_runner is None:
            self._table_cls_runner = ImageClassifierRunner(
                self.table_cls_model_id,
                device=self.device,
                dtype=self.vision_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._table_cls_runner

    def _get_table_struct_runner(self, *, is_wireless: bool) -> TableStructureRunner:
        if is_wireless:
            if self._wireless_table_struct_runner is None:
                self._wireless_table_struct_runner = TableStructureRunner(
                    self.wireless_table_struct_model_id,
                    device=self.device,
                    dtype=self.vision_dtype,
                    trust_remote_code=self.trust_remote_code,
                )
            return self._wireless_table_struct_runner

        if self._wired_table_struct_runner is None:
            self._wired_table_struct_runner = TableStructureRunner(
                self.wired_table_struct_model_id,
                device=self.device,
                dtype=self.vision_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._wired_table_struct_runner

    def _get_table_cell_runner(self, *, is_wireless: bool) -> LayoutDetectionRunner:
        if is_wireless:
            if self._wireless_table_cell_runner is None:
                self._wireless_table_cell_runner = LayoutDetectionRunner(
                    self.wireless_table_cell_model_id,
                    device=self.device,
                    dtype=self.vision_dtype,
                    trust_remote_code=self.trust_remote_code,
                )
            return self._wireless_table_cell_runner

        if self._wired_table_cell_runner is None:
            self._wired_table_cell_runner = LayoutDetectionRunner(
                self.wired_table_cell_model_id,
                device=self.device,
                dtype=self.vision_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._wired_table_cell_runner

    def _get_formula_runner(self) -> FormulaRecognitionRunner:
        if self._formula_runner is None:
            self._formula_runner = FormulaRecognitionRunner(
                self.formula_model_id,
                device=self.device,
                dtype=self.seq2seq_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._formula_runner

    def _get_seal_det_runner(self) -> TextDetectionRunner:
        if self._seal_det_runner is None:
            self._seal_det_runner = TextDetectionRunner(
                self.seal_det_model_id,
                device=self.device,
                dtype=self.vision_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._seal_det_runner

    def _get_chart_runner(self) -> ChartParsingRunner:
        if self._chart_runner is None:
            self._chart_runner = ChartParsingRunner(
                self.chart_model_id,
                device=self.device,
                dtype=self.seq2seq_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        return self._chart_runner

    def _recognize_table(
        self,
        table_crop: Image.Image,
        table_box: BBox,
        local_ocr_boxes: list[TextBox],
    ) -> TableResult:
        cls_label, _ = self._get_table_cls_runner().predict(table_crop)
        is_wireless = "wireless" in cls_label.lower()
        table_type = "wireless" if is_wireless else "wired"

        struct_tokens, struct_score = self._get_table_struct_runner(
            is_wireless=is_wireless
        ).predict(table_crop)
        cell_items = self._get_table_cell_runner(is_wireless=is_wireless).predict(
            table_crop,
            threshold=0.3,
        )
        cell_boxes = _sort_bboxes_reading_order([item.box for item in cell_items])

        if not local_ocr_boxes:
            local_ocr_boxes = self.ocr.extract_text_boxes(table_crop)

        html = assemble_table_html(struct_tokens, cell_boxes, local_ocr_boxes)
        markdown = html_table_to_markdown(html)
        return TableResult(
            box=table_box,
            table_type=table_type,
            html=html,
            markdown=markdown,
            score=struct_score,
            cell_boxes=cell_boxes,
        )

    def _recognize_seal(self, seal_crop: Image.Image) -> str:
        raw_polys = self._get_seal_det_runner().predict(seal_crop)
        ordered = sort_polygons_reading_order(raw_polys)
        if not ordered:
            return ""
        crops = [crop_polygon(seal_crop, poly) for poly, _ in ordered]
        rec_results = self.ocr.recognize_text_crops(crops)
        return " ".join(text for text, _ in rec_results if text.strip())

    @staticmethod
    def _filter_and_shift_ocr_boxes(
        page_ocr_boxes: list[TextBox],
        region_box: BBox,
    ) -> tuple[list[TextBox], list[TextBox]]:
        matched_page_coords: list[TextBox] = []
        matched_local_coords: list[TextBox] = []
        rx1, ry1, _, _ = region_box

        for tb in page_ocr_boxes:
            if box_intersection_over_area(tb.box, region_box) >= 0.4:
                matched_page_coords.append(tb)
                shifted_box: BBox = (
                    max(0.0, tb.box[0] - rx1),
                    max(0.0, tb.box[1] - ry1),
                    max(0.0, tb.box[2] - rx1),
                    max(0.0, tb.box[3] - ry1),
                )
                shifted_poly = [(max(0.0, pt[0] - rx1), max(0.0, pt[1] - ry1)) for pt in tb.polygon]
                matched_local_coords.append(
                    TextBox(
                        box=shifted_box,
                        polygon=shifted_poly,
                        text=tb.text,
                        score=tb.score,
                        det_score=tb.det_score,
                    )
                )
        return matched_page_coords, matched_local_coords

    def predict(
        self,
        input: ImageInput,
        *,
        use_doc_orientation_classify: bool | None = None,
        use_doc_unwarping: bool | None = None,
        use_textline_orientation: bool | None = None,
        use_layout_detection: bool | None = None,
        use_table_recognition: bool | None = None,
        use_formula_recognition: bool | None = None,
        use_seal_recognition: bool | None = None,
        use_chart_recognition: bool | None = None,
        layout_threshold: float | None = None,
    ) -> list[StructureResult]:
        """Run the modular ``PPStructureV3`` document analysis pipeline on images or PDFs."""
        do_doc_ori = (
            self.use_doc_orientation_classify
            if use_doc_orientation_classify is None
            else use_doc_orientation_classify
        )
        do_unwarp = self.use_doc_unwarping if use_doc_unwarping is None else use_doc_unwarping
        do_textline_ori = (
            self.use_textline_orientation
            if use_textline_orientation is None
            else use_textline_orientation
        )
        do_layout = (
            self.use_layout_detection if use_layout_detection is None else use_layout_detection
        )
        do_table = (
            self.use_table_recognition if use_table_recognition is None else use_table_recognition
        )
        do_formula = (
            self.use_formula_recognition
            if use_formula_recognition is None
            else use_formula_recognition
        )
        do_seal = (
            self.use_seal_recognition if use_seal_recognition is None else use_seal_recognition
        )
        do_chart = (
            self.use_chart_recognition if use_chart_recognition is None else use_chart_recognition
        )
        thresh = self.layout_threshold if layout_threshold is None else layout_threshold

        pages = load_inputs(input, pdf_dpi=self.pdf_dpi)
        results: list[StructureResult] = []

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

            page_ocr_boxes = self.ocr.extract_text_boxes(
                working_img,
                use_textline_orientation=do_textline_ori,
            )

            if not do_layout:
                w, h = working_img.size
                text_content = "\n".join(b.text for b in page_ocr_boxes if b.text.strip())
                block = LayoutBlock(
                    label="text",
                    box=(0.0, 0.0, float(w), float(h)),
                    score=1.0,
                    content=text_content,
                    order_index=0,
                    ocr_boxes=page_ocr_boxes,
                )
                results.append(
                    StructureResult(
                        blocks=[block],
                        ocr_boxes=page_ocr_boxes,
                        input_path=page.source_path,
                        page_index=page.page_index,
                        doc_orientation_angle=doc_angle,
                        image=working_img,
                    )
                )
                continue

            layout_items = self._get_layout_runner().predict(working_img, threshold=thresh)
            blocks: list[LayoutBlock] = []
            tables: list[TableResult] = []
            formulas: list[str] = []
            seals: list[str] = []
            charts: list[str] = []

            for idx, item in enumerate(layout_items):
                label_lower = item.label.lower()
                region_crop = crop_box(working_img, item.box)
                matched_page_ocr, matched_local_ocr = self._filter_and_shift_ocr_boxes(
                    page_ocr_boxes,
                    item.box,
                )

                block = LayoutBlock(
                    label=item.label,
                    box=item.box,
                    score=item.score,
                    polygon=item.polygon,
                    order_index=item.order_index if item.order_index is not None else idx,
                    ocr_boxes=matched_page_ocr,
                )

                if "table" in label_lower and do_table:
                    table_res = self._recognize_table(region_crop, item.box, matched_local_ocr)
                    tables.append(table_res)
                    block.html = table_res.html
                    block.content = table_res.markdown
                elif label_lower in _FORMULA_LABELS and do_formula:
                    latex = self._get_formula_runner().predict(region_crop)
                    if latex:
                        formulas.append(latex)
                    block.content = latex
                elif "seal" in label_lower and do_seal:
                    seal_text = self._recognize_seal(region_crop)
                    if not seal_text and matched_page_ocr:
                        seal_text = " ".join(b.text for b in matched_page_ocr if b.text.strip())
                    if seal_text:
                        seals.append(seal_text)
                    block.content = seal_text
                elif "chart" in label_lower and do_chart:
                    chart_md = self._get_chart_runner().predict(region_crop)
                    if chart_md:
                        charts.append(chart_md)
                    block.content = chart_md
                elif label_lower not in _IMAGE_ONLY_LABELS:
                    block.content = " ".join(
                        b.text for b in matched_page_ocr if b.text.strip()
                    ).strip()

                blocks.append(block)

            results.append(
                StructureResult(
                    blocks=blocks,
                    tables=tables,
                    formulas=formulas,
                    seals=seals,
                    charts=charts,
                    ocr_boxes=page_ocr_boxes,
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
        use_layout_detection: bool | None = None,
        use_table_recognition: bool | None = None,
        use_formula_recognition: bool | None = None,
        use_seal_recognition: bool | None = None,
        use_chart_recognition: bool | None = None,
        layout_threshold: float | None = None,
    ) -> list[StructureResult]:
        return self.predict(
            input,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            use_layout_detection=use_layout_detection,
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
            use_seal_recognition=use_seal_recognition,
            use_chart_recognition=use_chart_recognition,
            layout_threshold=layout_threshold,
        )
