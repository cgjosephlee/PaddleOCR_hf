"""Command-line interface for pure PyTorch PaddleOCR services."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from typing import Any, Protocol, cast

from . import __version__
from .ocr import PaddleOCR
from .structure import PPStructureV3
from .vl import PaddleOCRVL

_TRUE_STRINGS = frozenset({"true", "yes", "t", "y", "1"})
_FALSE_STRINGS = frozenset({"false", "no", "f", "n", "0"})
_VL_BACKEND_CHOICES = (
    "transformers",
    "native",
    "vllm-server",
    "sglang-server",
    "fastdeploy-server",
    "mlx-vlm-server",
    "llama-cpp-server",
    "openai",
    "api",
    "external",
)


class _Subparsers(Protocol):
    def add_parser(self, name: str, **kwargs: Any) -> argparse.ArgumentParser: ...


class _PipelineResult(Protocol):
    @property
    def markdown(self) -> str: ...

    def print(self) -> None: ...

    def save_to_json(self, save_path: str) -> object: ...

    def save_to_markdown(self, save_path: str) -> object: ...

    def save_to_img(self, save_path: str) -> object: ...

    def save_all(self, save_path: str) -> object: ...


def str2bool(value: str) -> bool:
    """Parse an explicit boolean CLI string (``True`` / ``False``)."""
    normalized = value.strip().lower()
    if normalized in _TRUE_STRINGS:
        return True
    if normalized in _FALSE_STRINGS:
        return False
    msg = f"Invalid boolean value: {value!r}. Expected True/False."
    raise argparse.ArgumentTypeError(msg)


def _validate_engine(value: str) -> str:
    """Ensure ``--engine`` is compatible with this pure PyTorch/Transformers package."""
    normalized = value.strip().lower()
    if normalized != "transformers":
        msg = (
            f"Unsupported engine {value!r}. "
            "paddleocr-hf only supports 'transformers' (pure PyTorch, zero PaddlePaddle dependencies)."
        )
        raise argparse.ArgumentTypeError(msg)
    return "transformers"


def _add_model_option(
    subparser: argparse.ArgumentParser,
    base_name: str,
    description: str,
) -> None:
    """Register both ``--{base_name}_name`` and ``--{base_name}_dir`` CLI options."""
    subparser.add_argument(
        f"--{base_name}_name",
        type=str,
        default=None,
        help=f"Model name, alias, or HF repo ID for {description}.",
    )
    subparser.add_argument(
        f"--{base_name}_dir",
        type=str,
        default=None,
        help=f"Local directory path for {description} (alias for --{base_name}_name).",
    )


def _resolve_model_dir_aliases(params: dict[str, Any]) -> None:
    """Map ``*_model_dir`` CLI flags onto ``*_model_name`` (directory overrides name)."""
    dir_keys = [key for key in params if key.endswith("_model_dir")]
    for dir_key in dir_keys:
        dir_val = params.pop(dir_key)
        if dir_val is not None:
            name_key = f"{dir_key.removesuffix('_model_dir')}_model_name"
            params[name_key] = dir_val


def _add_common_args(subparser: argparse.ArgumentParser) -> None:
    """Register shared I/O, device, PDF password, and output formatting flags."""
    subparser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input file path, directory path, or URL.",
    )
    subparser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Directory path to save output files.",
    )
    subparser.add_argument(
        "--save_formats",
        nargs="+",
        choices=["json", "markdown", "img"],
        default=None,
        help="Output formats to save when --save_path is specified (default: all).",
    )
    subparser.add_argument(
        "--output_format",
        type=str,
        choices=["json", "markdown", "none"],
        default="json",
        help="Format printed to stdout (default: json).",
    )
    subparser.add_argument(
        "--pdf_dpi",
        type=int,
        default=None,
        help="DPI used to render PDF pages (default: 200).",
    )
    subparser.add_argument(
        "--pdf_password",
        type=str,
        default=None,
        help="Optional password to unlock encrypted PDF inputs.",
    )
    subparser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device (e.g., mps, cuda, cuda:0, cpu). Defaults to auto-selection.",
    )
    subparser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="PyTorch dtype (e.g., float32, float16, bfloat16).",
    )
    subparser.add_argument(
        "--engine",
        type=_validate_engine,
        default="transformers",
        help="Inference engine compatibility flag (only 'transformers' is supported).",
    )
    subparser.add_argument(
        "--trust_remote_code",
        type=str2bool,
        default=None,
        help="Whether to allow executing remote Hugging Face model code (True/False).",
    )


def _register_ocr_subparser(subparsers: _Subparsers) -> None:
    subparser = subparsers.add_parser(
        "ocr",
        help="Run general OCR using PP-OCRv6 (or PP-OCRv5/v4).",
    )
    _add_common_args(subparser)

    subparser.add_argument(
        "--ocr_version",
        type=str,
        choices=["PP-OCRv6", "PP-OCRv5", "PP-OCRv4"],
        default=None,
        help="PP-OCR generation to use (default: PP-OCRv6).",
    )
    _add_model_option(subparser, "text_detection_model", "text detection")
    _add_model_option(subparser, "text_recognition_model", "text recognition")
    _add_model_option(subparser, "doc_orientation_classify_model", "document orientation")
    _add_model_option(subparser, "doc_unwarping_model", "document unwarping")
    _add_model_option(subparser, "textline_orientation_model", "text line orientation")

    subparser.add_argument(
        "--use_doc_orientation_classify",
        type=str2bool,
        default=None,
        help="Whether to run document orientation classification (True/False).",
    )
    subparser.add_argument(
        "--use_doc_unwarping",
        type=str2bool,
        default=None,
        help="Whether to run document image unwarping (True/False).",
    )
    subparser.add_argument(
        "--use_textline_orientation",
        type=str2bool,
        default=None,
        help="Whether to run text line orientation classification (True/False).",
    )
    subparser.add_argument(
        "--text_det_thresh",
        type=float,
        default=None,
        help="Pixel threshold for text detection probability map.",
    )
    subparser.add_argument(
        "--text_det_box_thresh",
        type=float,
        default=None,
        help="Box confidence threshold for text detection.",
    )
    subparser.add_argument(
        "--text_det_unclip_ratio",
        type=float,
        default=None,
        help="Polygon unclip expansion ratio for text detection.",
    )
    subparser.add_argument(
        "--text_rec_score_thresh",
        type=float,
        default=None,
        help="Minimum text recognition confidence threshold.",
    )
    subparser.add_argument(
        "--text_recognition_batch_size",
        type=int,
        default=None,
        help="Batch size for text recognition inference.",
    )
    subparser.set_defaults(pipeline_cls=PaddleOCR, predict_param_names=())


def _register_doc_parser_subparser(subparsers: _Subparsers) -> None:
    subparser = subparsers.add_parser(
        "doc_parser",
        help="Run vision-language document parsing using PaddleOCR-VL (v1.6 by default).",
    )
    _add_common_args(subparser)

    subparser.add_argument(
        "--pipeline_version",
        type=str,
        choices=["v1", "v1.5", "v1.6"],
        default=None,
        help="PaddleOCR-VL pipeline version (default: v1.6).",
    )
    _add_model_option(subparser, "vl_rec_model", "VLM recognition")
    _add_model_option(subparser, "layout_detection_model", "layout detection")
    _add_model_option(subparser, "doc_orientation_classify_model", "document orientation")
    _add_model_option(subparser, "doc_unwarping_model", "document unwarping")

    subparser.add_argument(
        "--use_doc_orientation_classify",
        type=str2bool,
        default=None,
        help="Whether to run document orientation classification (True/False).",
    )
    subparser.add_argument(
        "--use_doc_unwarping",
        type=str2bool,
        default=None,
        help="Whether to run document image unwarping (True/False).",
    )
    subparser.add_argument(
        "--use_layout_detection",
        type=str2bool,
        default=None,
        help="Whether to run layout detection before VLM recognition (True/False).",
    )
    subparser.add_argument(
        "--layout_threshold",
        type=float,
        default=None,
        help="Score threshold for layout detection.",
    )
    subparser.add_argument(
        "--vl_rec_backend",
        type=str,
        choices=list(_VL_BACKEND_CHOICES),
        default=None,
        help="Backend for VLM recognition (local 'transformers'/'native' or external server).",
    )
    subparser.add_argument(
        "--vl_rec_server_url",
        type=str,
        default=None,
        help="Base URL for an external OpenAI-compatible VLM server.",
    )
    subparser.add_argument(
        "--vl_rec_api_model_name",
        type=str,
        default=None,
        help="Model name sent to the external VLM API server.",
    )
    subparser.add_argument(
        "--vl_rec_api_key",
        type=str,
        default=None,
        help="API key for the external VLM server.",
    )
    subparser.add_argument(
        "--vl_rec_max_concurrency",
        type=int,
        default=None,
        help="Maximum concurrent requests sent to the external VLM server.",
    )
    subparser.add_argument(
        "--vl_rec_timeout",
        type=float,
        default=None,
        help="HTTP request timeout in seconds for the external VLM server.",
    )
    subparser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum tokens generated per VLM request.",
    )
    subparser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt string passed when --use_layout_detection False is set.",
    )
    subparser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Transformers attention implementation (e.g., sdpa, eager, flash_attention_2).",
    )
    subparser.set_defaults(pipeline_cls=PaddleOCRVL, predict_param_names=("prompt",))


def _register_structure_subparser(subparsers: _Subparsers) -> None:
    subparser = subparsers.add_parser(
        "pp_structurev3",
        help="Run modular document structure analysis using PPStructureV3.",
    )
    _add_common_args(subparser)

    for base_name, desc in (
        ("layout_detection_model", "layout detection"),
        ("text_detection_model", "text detection"),
        ("text_recognition_model", "text recognition"),
        ("textline_orientation_model", "text line orientation"),
        ("doc_orientation_classify_model", "document orientation"),
        ("doc_unwarping_model", "document unwarping"),
        ("table_classification_model", "table classification"),
        ("wired_table_structure_recognition_model", "wired table structure"),
        ("wireless_table_structure_recognition_model", "wireless table structure"),
        ("wired_table_cells_detection_model", "wired table cells detection"),
        ("wireless_table_cells_detection_model", "wireless table cells detection"),
        ("formula_recognition_model", "formula recognition"),
        ("seal_text_detection_model", "seal text detection"),
        ("chart_recognition_model", "chart parsing"),
    ):
        _add_model_option(subparser, base_name, desc)

    for toggle_name, help_text in (
        ("use_doc_orientation_classify", "document orientation classification"),
        ("use_doc_unwarping", "document image unwarping"),
        ("use_textline_orientation", "text line orientation classification"),
        ("use_layout_detection", "layout detection"),
        ("use_table_recognition", "table recognition"),
        ("use_formula_recognition", "formula recognition"),
        ("use_seal_recognition", "seal recognition"),
        ("use_chart_recognition", "chart recognition"),
    ):
        subparser.add_argument(
            f"--{toggle_name}",
            type=str2bool,
            default=None,
            help=f"Whether to enable {help_text} (True/False).",
        )

    subparser.add_argument(
        "--layout_threshold",
        type=float,
        default=None,
        help="Score threshold for layout detection.",
    )
    subparser.set_defaults(pipeline_cls=PPStructureV3, predict_param_names=())


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``paddleocr`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="paddleocr",
        description="Pure PyTorch & Hugging Face Transformers CLI for PaddleOCR services.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="subcommand", metavar="COMMAND")
    _register_ocr_subparser(subparsers)
    _register_doc_parser_subparser(subparsers)
    _register_structure_subparser(subparsers)
    return parser


class _StderrHandler(logging.Handler):
    """Emit log records to the active ``sys.stderr`` stream without caching stale references."""

    def emit(self, record: logging.LogRecord) -> None:
        sys.stderr.write(f"{self.format(record)}\n")
        sys.stderr.flush()


def _get_logger() -> logging.Logger:
    log = logging.getLogger("paddleocr.cli")
    if not log.handlers:
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


def _save_result(
    res: _PipelineResult,
    save_path: str,
    save_formats: Sequence[str] | None,
) -> None:
    if save_formats is None:
        res.save_all(save_path)
        return

    seen: set[str] = set()
    for fmt in save_formats:
        if fmt in seen:
            continue
        seen.add(fmt)
        if fmt == "json":
            res.save_to_json(save_path)
        elif fmt == "markdown":
            res.save_to_markdown(save_path)
        elif fmt == "img":
            res.save_to_img(save_path)


def _run_pipeline(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    log = _get_logger()
    raw_params: dict[str, Any] = vars(args).copy()

    raw_params.pop("subcommand", None)
    raw_params.pop("engine", None)
    pipeline_cls = cast(type[Any], raw_params.pop("pipeline_cls"))
    predict_param_names = cast(tuple[str, ...], raw_params.pop("predict_param_names", ()))

    input_path = cast(str, raw_params.pop("input"))
    save_path = cast(str | None, raw_params.pop("save_path"))
    save_formats = cast(list[str] | None, raw_params.pop("save_formats"))
    output_format = cast(str, raw_params.pop("output_format"))

    if save_formats is not None and save_path is None:
        parser.error("--save_formats requires --save_path to be set.")

    _resolve_model_dir_aliases(raw_params)

    predict_params: dict[str, Any] = {}
    for key in predict_param_names:
        val = raw_params.pop(key, None)
        if val is not None:
            predict_params[key] = val

    init_params = {k: v for k, v in raw_params.items() if v is not None}

    t0 = time.perf_counter()
    pipeline: Any = pipeline_cls(**init_params)
    results = cast(Sequence[_PipelineResult], pipeline.predict(input_path, **predict_params))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    log.info("Processed %d item(s) in %.1f ms", len(results), elapsed_ms)

    for idx, res in enumerate(results):
        if output_format == "json":
            res.print()
        elif output_format == "markdown":
            if idx > 0:
                print("\n---\n")
            print(res.markdown)

        if save_path is not None:
            _save_result(res, save_path, save_formats)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and execute the selected PaddleOCR pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "subcommand", None) is None:
        parser.print_usage(sys.stderr)
        return 2

    try:
        return _run_pipeline(args, parser)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _get_logger().error("%s", exc)
        return 1
