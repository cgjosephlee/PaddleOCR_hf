# pyright: reportPrivateUsage=false
"""Comprehensive unit tests for PaddleOCR, PaddleOCRVL, and PPStructureV3 (Mocked HF runners)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pypdfium2 as pdfium  # pyright: ignore[reportMissingTypeStubs]
import pytest
import torch
from PIL import Image

from paddleocr import (
    OCRResult,
    PaddleOCR,
    PaddleOCRVL,
    PPStructureV3,
    StructureResult,
    TextBox,
    VLResult,
    resolve_device,
    resolve_dtype,
)
from paddleocr.io import (
    BBox,
    Polygon,
    box_intersection_over_area,
    crop_box,
    crop_polygon,
    load_inputs,
    polygon_to_bbox,
    rotate_image,
    sort_polygons_reading_order,
)
from paddleocr.models import (
    DetectedLayoutItem,
    ExternalVLMClient,
    parse_orientation_angle,
)
from paddleocr.registry import (
    DEFAULT_LAYOUT_MODEL,
    DEFAULT_TEXT_DET_MODEL,
    DEFAULT_TEXT_REC_MODEL,
    DEFAULT_VLM_MODEL,
    resolve_model_id,
)
from paddleocr.results import assemble_table_html, html_table_to_markdown


class _FakeClassifierRunner:
    def __init__(self, label: str = "90", score: float = 0.99) -> None:
        self.label = label
        self.score = score
        self.calls: int = 0

    def predict(self, image: Image.Image) -> tuple[str, float]:
        self.calls += 1
        return (self.label, self.score)

    def predict_batch(self, images: Sequence[Image.Image]) -> list[tuple[str, float]]:
        self.calls += len(images)
        return [(self.label, self.score) for _ in images]


class _FakeUnwarpRunner:
    def __init__(self) -> None:
        self.calls: int = 0

    def predict(self, image: Image.Image) -> Image.Image:
        self.calls += 1
        return image.copy()


class _FakeTextDetRunner:
    def __init__(self, polygons: list[tuple[Polygon, float]]) -> None:
        self.polygons = polygons
        self.calls: int = 0

    def predict(
        self,
        image: Image.Image,
        *,
        threshold: float = 0.3,
        box_threshold: float = 0.6,
        unclip_ratio: float = 1.5,
        max_candidates: int = 1000,
    ) -> list[tuple[Polygon, float]]:
        self.calls += 1
        return list(self.polygons)


class _FakeTextRecRunner:
    def __init__(self, texts: list[tuple[str, float]]) -> None:
        self.texts = texts
        self.calls: int = 0

    def predict_batch(
        self,
        images: Sequence[Image.Image],
        *,
        batch_size: int = 8,
    ) -> list[tuple[str, float]]:
        self.calls += 1
        out: list[tuple[str, float]] = []
        for idx in range(len(images)):
            if idx < len(self.texts):
                out.append(self.texts[idx])
            else:
                out.append((f"line_{idx}", 0.95))
        return out


class _FakeLayoutRunner:
    def __init__(
        self, items: list[DetectedLayoutItem], model_id: str = DEFAULT_LAYOUT_MODEL
    ) -> None:
        self.items = items
        self.model_id = model_id
        self.calls: int = 0

    def predict(self, image: Image.Image, *, threshold: float = 0.4) -> list[DetectedLayoutItem]:
        self.calls += 1
        return list(self.items)


class _FakeTableStructRunner:
    def __init__(self, tokens: list[str], score: float = 0.98) -> None:
        self.tokens = tokens
        self.score = score
        self.calls: int = 0

    def predict(self, image: Image.Image) -> tuple[list[str], float]:
        self.calls += 1
        return (list(self.tokens), self.score)


class _FakeFormulaRunner:
    def __init__(self, latex: str = "E = mc^2") -> None:
        self.latex = latex
        self.calls: int = 0

    def predict(self, image: Image.Image) -> str:
        self.calls += 1
        return self.latex


class _FakeChartRunner:
    def __init__(self, table_md: str = "| Year | Revenue |\n| --- | --- |\n| 2026 | 100 |") -> None:
        self.table_md = table_md
        self.calls: int = 0

    def predict(self, image: Image.Image, *, max_new_tokens: int = 512) -> str:
        self.calls += 1
        return self.table_md


class _FakeVLMRecognizer:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(
        self,
        image: Image.Image,
        prompt: str = "OCR:",
        *,
        max_new_tokens: int = 1024,
    ) -> str:
        self.prompts.append(prompt)
        if "Table" in prompt:
            return "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
        if "Formula" in prompt:
            return "\\int_0^1 x^2 dx"
        return f"Recognized({prompt})"

    def generate_batch(
        self,
        requests: Sequence[tuple[Image.Image, str]],
        *,
        max_new_tokens: int = 1024,
    ) -> list[str]:
        return [
            self.generate(img, prompt, max_new_tokens=max_new_tokens) for img, prompt in requests
        ]


def test_device_and_dtype_resolution() -> None:
    cpu_dev = resolve_device("cpu")
    assert cpu_dev == torch.device("cpu")

    mps_dev = resolve_device("mps")
    assert mps_dev == torch.device("mps")

    gpu_alias = resolve_device("gpu:0")
    assert gpu_alias == torch.device("cuda:0")

    passthrough = resolve_device(torch.device("cpu"))
    assert passthrough == torch.device("cpu")

    auto_dev = resolve_device(None)
    assert auto_dev.type in {"cuda", "mps", "cpu"}

    # Vision CNN/DETR models default to float32 even on mps/cuda
    assert resolve_dtype(None, device=torch.device("mps"), is_vlm=False) == torch.float32
    # VLM models default to bfloat16 on mps/cuda and float32 on cpu
    assert resolve_dtype(None, device=torch.device("mps"), is_vlm=True) == torch.bfloat16
    assert resolve_dtype(None, device=torch.device("cpu"), is_vlm=True) == torch.float32
    assert resolve_dtype("fp16", device=torch.device("mps"), is_vlm=False) == torch.float16
    assert resolve_dtype(torch.bfloat16, device=torch.device("cpu")) == torch.bfloat16

    with pytest.raises(ValueError, match="Unsupported dtype"):
        resolve_dtype("int4_invalid", device=torch.device("cpu"))


def test_model_registry_defaults_to_ppocrv6_and_vl_1_6(tmp_path: Path) -> None:
    assert DEFAULT_TEXT_DET_MODEL == "PaddlePaddle/PP-OCRv6_medium_det_safetensors"
    assert DEFAULT_TEXT_REC_MODEL == "PaddlePaddle/PP-OCRv6_small_rec_safetensors"
    assert DEFAULT_VLM_MODEL == "PaddlePaddle/PaddleOCR-VL-1.6"

    assert (
        resolve_model_id("PP-OCRv6_small_rec", DEFAULT_TEXT_REC_MODEL)
        == "PaddlePaddle/PP-OCRv6_small_rec_safetensors"
    )
    assert (
        resolve_model_id("PaddleOCR-VL-1.6", DEFAULT_VLM_MODEL) == "PaddlePaddle/PaddleOCR-VL-1.6"
    )
    assert (
        resolve_model_id("PP-DocLayoutV4", DEFAULT_LAYOUT_MODEL)
        == "PaddlePaddle/PP-DocLayoutV4_safetensors"
    )
    assert (
        resolve_model_id("custom-org/custom-ocr", DEFAULT_TEXT_REC_MODEL) == "custom-org/custom-ocr"
    )
    local_dir = tmp_path / "local_model"
    local_dir.mkdir()
    assert resolve_model_id(local_dir, DEFAULT_TEXT_REC_MODEL) == str(local_dir)


def test_io_loading_and_geometry(tmp_path: Path) -> None:
    pil_img = Image.new("RGB", (120, 80), color=(240, 240, 240))
    img_path = tmp_path / "sample.png"
    pil_img.save(img_path)

    arr_gray = np.zeros((40, 60), dtype=np.uint8)
    arr_rgb = np.zeros((40, 60, 3), dtype=np.uint8)

    # Create a 2-page PDF via pypdfium2
    pdf_path = tmp_path / "doc.pdf"
    pdf_doc: Any = pdfium.PdfDocument.new()
    page0 = pdf_doc.new_page(200, 300)
    page0.close()
    page1 = pdf_doc.new_page(200, 300)
    page1.close()
    with pdf_path.open("wb") as f:
        pdf_doc.save(f)
    pdf_doc.close()

    loaded = load_inputs([pil_img, arr_gray, arr_rgb, img_path, pdf_path], pdf_dpi=72)
    assert len(loaded) == 6
    assert loaded[3].source_path == str(img_path)
    assert loaded[4].page_index == 0
    assert loaded[5].page_index == 1

    # Rotation & cropping
    rot90 = rotate_image(pil_img, 90)
    assert rot90.size == (80, 120)

    box_crop = crop_box(pil_img, (10.0, 10.0, 50.0, 40.0))
    assert box_crop.size == (40, 30)

    # Vertical text quad should auto-rotate 90 degrees
    vertical_poly: Polygon = [(10.0, 5.0), (25.0, 5.0), (25.0, 65.0), (10.0, 65.0)]
    vert_crop = crop_polygon(pil_img, vertical_poly)
    assert vert_crop.width >= vert_crop.height

    assert polygon_to_bbox(vertical_poly) == (10.0, 5.0, 25.0, 65.0)
    assert box_intersection_over_area((10.0, 10.0, 20.0, 20.0), (0.0, 0.0, 30.0, 30.0)) == 1.0
    assert parse_orientation_angle("180_degree") == 180
    assert parse_orientation_angle("0_degree") == 0

    # Reading order sorting
    p_right: Polygon = [(100.0, 12.0), (140.0, 12.0), (140.0, 30.0), (100.0, 30.0)]
    p_left: Polygon = [(10.0, 10.0), (50.0, 10.0), (50.0, 30.0), (10.0, 30.0)]
    p_bottom: Polygon = [(10.0, 60.0), (50.0, 60.0), (50.0, 80.0), (10.0, 80.0)]
    sorted_polys = sort_polygons_reading_order([(p_bottom, 0.9), (p_right, 0.9), (p_left, 0.9)])
    assert [item[0] for item in sorted_polys] == [p_left, p_right, p_bottom]


def test_paddleocr_pipeline_and_outputs(tmp_path: Path) -> None:
    ocr = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        device="cpu",
    )
    poly1: Polygon = [(10.0, 10.0), (90.0, 10.0), (90.0, 35.0), (10.0, 35.0)]
    poly2: Polygon = [(10.0, 50.0), (90.0, 50.0), (90.0, 75.0), (10.0, 75.0)]

    ocr._doc_ori_runner = _FakeClassifierRunner(label="90")  # type: ignore[assignment]
    ocr._unwarp_runner = _FakeUnwarpRunner()  # type: ignore[assignment]
    ocr._textline_ori_runner = _FakeClassifierRunner(label="180")  # type: ignore[assignment]
    ocr._det_runner = _FakeTextDetRunner([(poly2, 0.95), (poly1, 0.98)])  # type: ignore[assignment]
    ocr._rec_runner = _FakeTextRecRunner([("Hello PP-OCRv6", 0.99), ("PyTorch MPS", 0.97)])  # type: ignore[assignment]

    sample_img = Image.new("RGB", (200, 120), color="white")
    outputs: list[OCRResult] = ocr(sample_img)
    assert len(outputs) == 1
    res = outputs[0]

    assert res.doc_orientation_angle == 90
    assert res.rec_texts == ["Hello PP-OCRv6", "PyTorch MPS"]
    assert res["rec_texts"] == ["Hello PP-OCRv6", "PyTorch MPS"]
    assert len(res.dt_polys) == 2
    assert "Hello PP-OCRv6" in res.markdown

    json_file = res.save_to_json(tmp_path)
    md_file = res.save_to_markdown(tmp_path)
    img_file = res.save_to_img(tmp_path)
    assert json_file.exists()
    assert md_file.exists()
    assert img_file.exists()


def test_paddleocr_vl_local_and_external_api(tmp_path: Path) -> None:
    # 1. Local PyTorch VLM mode with layout detection
    vl_local = PaddleOCRVL(pipeline_version="v1.6", device="cpu")
    assert not vl_local.uses_external_api
    assert vl_local.vl_model_id == "PaddlePaddle/PaddleOCR-VL-1.6"

    layout_items = [
        DetectedLayoutItem(
            label="doc_title", box=(10.0, 10.0, 180.0, 40.0), score=0.99, order_index=0
        ),
        DetectedLayoutItem(
            label="table", box=(10.0, 50.0, 180.0, 120.0), score=0.95, order_index=1
        ),
        DetectedLayoutItem(
            label="formula", box=(10.0, 130.0, 180.0, 160.0), score=0.96, order_index=2
        ),
        DetectedLayoutItem(
            label="figure", box=(10.0, 165.0, 180.0, 195.0), score=0.90, order_index=3
        ),
    ]
    fake_vlm = _FakeVLMRecognizer()
    vl_local._layout_runner = _FakeLayoutRunner(layout_items)  # type: ignore[assignment]
    vl_local._local_vlm_runner = fake_vlm  # type: ignore[assignment]

    img = Image.new("RGB", (200, 200), color="white")
    res_list: list[VLResult] = vl_local.predict(img)
    assert len(res_list) == 1
    res = res_list[0]
    assert fake_vlm.prompts == ["OCR:", "Table Recognition:", "Formula Recognition:"]
    assert "# Recognized(OCR:)" in res.markdown
    assert "| A | B |" in res.markdown
    assert "$$" in res.markdown

    assert res.save_to_json(tmp_path / "vl.json").exists()
    assert res.save_to_markdown(tmp_path / "vl.md").exists()
    assert res.save_to_img(tmp_path / "vl.png").exists()

    # 2. External OpenAI-compatible API mode (e.g. vllm-server)
    def _mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers.get("Authorization") == "Bearer test-secret"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "External API VLM Output"}},
                ]
            },
        )

    transport = httpx.MockTransport(_mock_handler)
    mock_http = httpx.Client(transport=transport)
    ext_client = ExternalVLMClient(
        "http://127.0.0.1:8080/v1",
        model_name="PaddlePaddle/PaddleOCR-VL-1.6",
        api_key="test-secret",
        max_concurrency=2,
        http_client=mock_http,
    )

    vl_ext = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_backend="vllm-server",
        vl_rec_server_url="http://127.0.0.1:8080/v1",
        vl_rec_api_key="test-secret",
        use_layout_detection=False,
        device="cpu",
    )
    assert vl_ext.uses_external_api
    vl_ext._external_vlm_client = ext_client

    ext_results = vl_ext.predict(img)
    assert len(ext_results) == 1
    assert ext_results[0].markdown == "External API VLM Output"
    mock_http.close()


def test_ppstructurev3_full_pipeline_and_lazy_loading(tmp_path: Path) -> None:
    pipeline = PPStructureV3(
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_table_recognition=True,
        use_formula_recognition=True,
        use_seal_recognition=True,
        use_chart_recognition=True,
        device="cpu",
    )

    # Before predict(), all sub-model runners are None (lazy loading)
    assert pipeline._layout_runner is None
    assert pipeline._formula_runner is None
    assert pipeline._seal_det_runner is None
    assert pipeline._chart_runner is None

    # Inject fake runners
    pipeline._doc_ori_runner = _FakeClassifierRunner(label="0")  # type: ignore[assignment]
    pipeline._unwarp_runner = _FakeUnwarpRunner()  # type: ignore[assignment]

    poly_title: Polygon = [(10.0, 10.0), (150.0, 10.0), (150.0, 30.0), (10.0, 30.0)]
    poly_cell1: Polygon = [(15.0, 55.0), (75.0, 55.0), (75.0, 75.0), (15.0, 75.0)]
    poly_cell2: Polygon = [(85.0, 55.0), (145.0, 55.0), (145.0, 75.0), (85.0, 75.0)]
    pipeline.ocr._det_runner = _FakeTextDetRunner(  # type: ignore[assignment]
        [(poly_title, 0.99), (poly_cell1, 0.98), (poly_cell2, 0.98)]
    )
    pipeline.ocr._rec_runner = _FakeTextRecRunner(  # type: ignore[assignment]
        [("Annual Report", 0.99), ("Metric", 0.98), ("Value", 0.98)]
    )

    layout_items = [
        DetectedLayoutItem(
            label="doc_title", box=(5.0, 5.0, 160.0, 35.0), score=0.99, order_index=0
        ),
        DetectedLayoutItem(
            label="table", box=(10.0, 50.0, 160.0, 100.0), score=0.97, order_index=1
        ),
        DetectedLayoutItem(
            label="formula", box=(10.0, 110.0, 160.0, 140.0), score=0.95, order_index=2
        ),
        DetectedLayoutItem(
            label="seal", box=(10.0, 145.0, 100.0, 175.0), score=0.92, order_index=3
        ),
        DetectedLayoutItem(
            label="chart", box=(10.0, 180.0, 180.0, 230.0), score=0.94, order_index=4
        ),
    ]
    pipeline._layout_runner = _FakeLayoutRunner(layout_items)  # type: ignore[assignment]
    pipeline._table_cls_runner = _FakeClassifierRunner(label="wired_table")  # type: ignore[assignment]
    pipeline._wired_table_struct_runner = _FakeTableStructRunner(  # type: ignore[assignment]
        [
            "<html>",
            "<body>",
            "<table>",
            "<tr>",
            "<td></td>",
            "<td></td>",
            "</tr>",
            "</table>",
            "</body>",
            "</html>",
        ]
    )
    pipeline._wired_table_cell_runner = _FakeLayoutRunner(  # type: ignore[assignment]
        [
            DetectedLayoutItem(label="cell", box=(0.0, 0.0, 70.0, 40.0), score=0.95),
            DetectedLayoutItem(label="cell", box=(70.0, 0.0, 145.0, 40.0), score=0.95),
        ]
    )
    pipeline._formula_runner = _FakeFormulaRunner("a^2 + b^2 = c^2")  # type: ignore[assignment]
    seal_poly: Polygon = [(5.0, 5.0), (60.0, 5.0), (60.0, 20.0), (5.0, 20.0)]
    pipeline._seal_det_runner = _FakeTextDetRunner([(seal_poly, 0.96)])  # type: ignore[assignment]
    pipeline._chart_runner = _FakeChartRunner("| Q1 | Q2 |\n| --- | --- |\n| 10 | 20 |")  # type: ignore[assignment]

    img = Image.new("RGB", (220, 250), color="white")
    outputs: list[StructureResult] = pipeline(img)
    assert len(outputs) == 1
    res = outputs[0]

    assert len(res.tables) == 1
    assert res.tables[0].table_type == "wired"
    assert "<td>Metric</td><td>Value</td>" in res.tables[0].html
    assert res.formulas == ["a^2 + b^2 = c^2"]
    assert len(res.seals) == 1
    assert len(res.charts) == 1
    assert "# Annual Report" in res.markdown
    assert "a^2 + b^2 = c^2" in res.markdown
    assert "Chart Data" in res.markdown

    assert res.save_to_json(tmp_path).exists()
    assert res.save_to_markdown(tmp_path).exists()
    assert res.save_to_img(tmp_path).exists()


def test_table_html_and_markdown_conversion() -> None:
    tokens = ["<table>", "<tr>", "<td", ' colspan="2"', ">", "</td>", "</tr>", "</table>"]
    cells: list[BBox] = [(0.0, 0.0, 100.0, 30.0)]
    ocr_boxes = [
        TextBox(
            box=(5.0, 5.0, 95.0, 25.0),
            polygon=[(5.0, 5.0), (95.0, 5.0), (95.0, 25.0), (5.0, 25.0)],
            text="Merged Header",
            score=0.99,
        )
    ]
    html = assemble_table_html(tokens, cells, ocr_boxes)
    assert html == '<table><tr><td colspan="2">Merged Header</td></tr></table>'
    md = html_table_to_markdown(html)
    assert "Merged Header" in md
