"""Unit tests for the pure PyTorch ``paddleocr`` CLI tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from PIL import Image

import paddleocr._cli as cli
from paddleocr import OCRResult, TextBox


class _RecordingPipeline:
    last_init_kwargs: ClassVar[dict[str, Any]] = {}
    last_predict_args: ClassVar[tuple[Any, ...]] = ()
    last_predict_kwargs: ClassVar[dict[str, Any]] = {}
    results_to_return: ClassVar[list[OCRResult]] = []
    error_to_raise: ClassVar[Exception | None] = None

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_init_kwargs = kwargs

    def predict(self, input_val: Any, **kwargs: Any) -> list[OCRResult]:
        type(self).last_predict_args = (input_val,)
        type(self).last_predict_kwargs = kwargs
        error = type(self).error_to_raise
        if error is not None:
            raise error
        return list(type(self).results_to_return)


def _sample_ocr_result(text: str, *, input_path: str, page_index: int | None = None) -> OCRResult:
    return OCRResult(
        boxes=[
            TextBox(
                box=(10.0, 10.0, 90.0, 30.0),
                polygon=[(10.0, 10.0), (90.0, 10.0), (90.0, 30.0), (10.0, 30.0)],
                text=text,
                score=0.99,
            )
        ],
        input_path=input_path,
        page_index=page_index,
        image=Image.new("RGB", (120, 60), color="white"),
    )


def test_cli_ocr_flags_and_model_dir_aliases(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _RecordingPipeline.results_to_return = [
        _sample_ocr_result("Hello CLI", input_path="sample.jpg"),
    ]
    _RecordingPipeline.error_to_raise = None
    monkeypatch.setattr(cli, "PaddleOCR", _RecordingPipeline)

    code = cli.main(
        [
            "ocr",
            "-i",
            "sample.jpg",
            "--ocr_version",
            "PP-OCRv6",
            "--text_detection_model_name",
            "overridden-det",
            "--text_detection_model_dir",
            "/local/det_model",
            "--text_recognition_model_dir",
            "/local/rec_model",
            "--use_doc_orientation_classify",
            "True",
            "--use_doc_unwarping",
            "False",
            "--use_textline_orientation",
            "0",
            "--text_det_thresh",
            "0.25",
            "--text_recognition_batch_size",
            "16",
            "--pdf_dpi",
            "150",
            "--pdf_password",
            "pw123",
            "--device",
            "mps",
            "--dtype",
            "float32",
            "--engine",
            "transformers",
            "--trust_remote_code",
            "false",
        ]
    )

    assert code == 0
    init_kwargs = _RecordingPipeline.last_init_kwargs
    assert init_kwargs["ocr_version"] == "PP-OCRv6"
    assert init_kwargs["text_detection_model_name"] == "/local/det_model"
    assert init_kwargs["text_recognition_model_name"] == "/local/rec_model"
    assert init_kwargs["use_doc_orientation_classify"] is True
    assert init_kwargs["use_doc_unwarping"] is False
    assert init_kwargs["use_textline_orientation"] is False
    assert init_kwargs["text_det_thresh"] == 0.25
    assert init_kwargs["text_recognition_batch_size"] == 16
    assert init_kwargs["pdf_dpi"] == 150
    assert init_kwargs["pdf_password"] == "pw123"
    assert init_kwargs["device"] == "mps"
    assert init_kwargs["dtype"] == "float32"
    assert init_kwargs["trust_remote_code"] is False
    assert "engine" not in init_kwargs
    assert "text_detection_model_dir" not in init_kwargs

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["rec_texts"] == ["Hello CLI"]
    assert "Processed 1 item(s)" in captured.err


def test_cli_rejects_invalid_bool_and_non_transformers_engine() -> None:
    with pytest.raises(SystemExit) as bool_exc:
        cli.main(["ocr", "-i", "sample.jpg", "--use_doc_unwarping", "maybe"])
    assert bool_exc.value.code == 2

    with pytest.raises(SystemExit) as engine_exc:
        cli.main(["doc_parser", "-i", "sample.jpg", "--engine", "paddle"])
    assert engine_exc.value.code == 2


def test_cli_doc_parser_local_and_external_api(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingPipeline.results_to_return = []
    _RecordingPipeline.error_to_raise = None
    monkeypatch.setattr(cli, "PaddleOCRVL", _RecordingPipeline)

    code_local = cli.main(
        [
            "doc_parser",
            "-i",
            "doc.pdf",
            "--pipeline_version",
            "v1.6",
            "--vl_rec_backend",
            "native",
            "--use_layout_detection",
            "False",
            "--prompt",
            "Table Recognition:",
            "--max_new_tokens",
            "512",
            "--attn_implementation",
            "sdpa",
            "--pdf_password",
            "topsecret",
        ]
    )
    assert code_local == 0
    assert _RecordingPipeline.last_init_kwargs == {
        "pipeline_version": "v1.6",
        "vl_rec_backend": "native",
        "use_layout_detection": False,
        "max_new_tokens": 512,
        "attn_implementation": "sdpa",
        "pdf_password": "topsecret",
    }
    assert _RecordingPipeline.last_predict_args == ("doc.pdf",)
    assert _RecordingPipeline.last_predict_kwargs == {"prompt": "Table Recognition:"}

    code_api = cli.main(
        [
            "doc_parser",
            "-i",
            "https://example.com/demo.png",
            "--vl_rec_backend",
            "vllm-server",
            "--vl_rec_server_url",
            "http://127.0.0.1:8000/v1",
            "--vl_rec_api_model_name",
            "PaddlePaddle/PaddleOCR-VL-1.6",
            "--vl_rec_api_key",
            "secret-key",
            "--vl_rec_max_concurrency",
            "4",
            "--vl_rec_timeout",
            "45.0",
        ]
    )
    assert code_api == 0
    assert _RecordingPipeline.last_init_kwargs["vl_rec_backend"] == "vllm-server"
    assert _RecordingPipeline.last_init_kwargs["vl_rec_server_url"] == "http://127.0.0.1:8000/v1"
    assert _RecordingPipeline.last_init_kwargs["vl_rec_api_model_name"] == (
        "PaddlePaddle/PaddleOCR-VL-1.6"
    )
    assert _RecordingPipeline.last_init_kwargs["vl_rec_api_key"] == "secret-key"
    assert _RecordingPipeline.last_init_kwargs["vl_rec_max_concurrency"] == 4
    assert _RecordingPipeline.last_init_kwargs["vl_rec_timeout"] == 45.0
    assert _RecordingPipeline.last_predict_kwargs == {}


def test_cli_pp_structurev3_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingPipeline.results_to_return = []
    _RecordingPipeline.error_to_raise = None
    monkeypatch.setattr(cli, "PPStructureV3", _RecordingPipeline)

    code = cli.main(
        [
            "pp_structurev3",
            "-i",
            "report.pdf",
            "--layout_detection_model_dir",
            "/models/layout",
            "--use_table_recognition",
            "False",
            "--use_formula_recognition",
            "True",
            "--use_seal_recognition",
            "True",
            "--use_chart_recognition",
            "False",
            "--layout_threshold",
            "0.55",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    assert _RecordingPipeline.last_init_kwargs == {
        "layout_detection_model_name": "/models/layout",
        "use_table_recognition": False,
        "use_formula_recognition": True,
        "use_seal_recognition": True,
        "use_chart_recognition": False,
        "layout_threshold": 0.55,
        "device": "cpu",
    }


def test_cli_output_formats_and_saving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "PaddleOCR", _RecordingPipeline)
    _RecordingPipeline.error_to_raise = None
    _RecordingPipeline.results_to_return = [
        _sample_ocr_result("Page Zero", input_path="multi.pdf", page_index=0),
        _sample_ocr_result("Page One", input_path="multi.pdf", page_index=1),
    ]

    # 1. Markdown stdout across multiple pages + default save_all
    all_dir = tmp_path / "all_out"
    code_md = cli.main(
        [
            "ocr",
            "-i",
            "multi.pdf",
            "--output_format",
            "markdown",
            "--save_path",
            str(all_dir),
        ]
    )
    assert code_md == 0
    out_md = capsys.readouterr().out
    assert "Page Zero\n\n---\n\nPage One" in out_md
    assert (all_dir / "multi_page0.json").exists()
    assert (all_dir / "multi_page0.md").exists()
    assert (all_dir / "multi_page0.png").exists()
    assert (all_dir / "multi_page1.json").exists()

    # 2. Selective --save_formats + --output_format none
    sel_dir = tmp_path / "sel_out"
    code_sel = cli.main(
        [
            "ocr",
            "-i",
            "multi.pdf",
            "--output_format",
            "none",
            "--save_path",
            str(sel_dir),
            "--save_formats",
            "json",
            "markdown",
            "json",
        ]
    )
    assert code_sel == 0
    assert capsys.readouterr().out == ""
    assert (sel_dir / "multi_page0.json").exists()
    assert (sel_dir / "multi_page0.md").exists()
    assert not (sel_dir / "multi_page0.png").exists()

    # 3. --save_formats without --save_path should fail argument validation
    with pytest.raises(SystemExit) as exc:
        cli.main(["ocr", "-i", "multi.pdf", "--save_formats", "json"])
    assert exc.value.code == 2


def test_cli_missing_subcommand_and_runtime_error_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main([]) == 2
    assert "usage: paddleocr" in capsys.readouterr().err

    monkeypatch.setattr(cli, "PaddleOCR", _RecordingPipeline)
    _RecordingPipeline.error_to_raise = ValueError("PDF is password-protected: secret.pdf")
    code = cli.main(["ocr", "-i", "secret.pdf"])
    assert code == 1
    assert "password-protected" in capsys.readouterr().err
