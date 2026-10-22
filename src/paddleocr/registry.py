"""Model registry mapping PaddleOCR model names to Hugging Face Hub repositories."""

from __future__ import annotations

from pathlib import Path

DEFAULT_DOC_ORI_MODEL = "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors"
DEFAULT_TEXTLINE_ORI_MODEL = "PaddlePaddle/PP-LCNet_x1_0_textline_ori_safetensors"
DEFAULT_UNWARP_MODEL = "PaddlePaddle/UVDoc_safetensors"

# PP-OCRv6 defaults (latest generation: LCNetV4 + RepLKFPN / EncoderWithLightSVTR, 50 languages)
DEFAULT_TEXT_DET_MODEL = "PaddlePaddle/PP-OCRv6_medium_det_safetensors"
DEFAULT_TEXT_REC_MODEL = "PaddlePaddle/PP-OCRv6_small_rec_safetensors"

# Layout & Document Parsing defaults
DEFAULT_LAYOUT_MODEL = "PaddlePaddle/PP-DocLayoutV3_safetensors"
DEFAULT_TABLE_CLS_MODEL = "PaddlePaddle/PP-LCNet_x1_0_table_cls_safetensors"
DEFAULT_WIRED_TABLE_STRUCT_MODEL = "PaddlePaddle/SLANeXt_wired_safetensors"
DEFAULT_WIRELESS_TABLE_STRUCT_MODEL = "PaddlePaddle/SLANeXt_wireless_safetensors"
DEFAULT_WIRED_TABLE_CELL_MODEL = "PaddlePaddle/RT-DETR-L_wired_table_cell_det_safetensors"
DEFAULT_WIRELESS_TABLE_CELL_MODEL = "PaddlePaddle/RT-DETR-L_wireless_table_cell_det_safetensors"
DEFAULT_FORMULA_MODEL = "PaddlePaddle/PP-FormulaNet_plus-L_safetensors"
DEFAULT_SEAL_DET_MODEL = "PaddlePaddle/PP-OCRv4_server_seal_det_safetensors"
DEFAULT_CHART_MODEL = "PaddlePaddle/PP-Chart2Table_safetensors"

# PaddleOCR-VL defaults (latest generation: PaddleOCR-VL-1.6)
DEFAULT_VLM_MODEL = "PaddlePaddle/PaddleOCR-VL-1.6"

MODEL_ALIASES: dict[str, str] = {
    # Orientation & Unwarping
    "PP-LCNet_x1_0_doc_ori": DEFAULT_DOC_ORI_MODEL,
    "PP-LCNet_x1_0_textline_ori": DEFAULT_TEXTLINE_ORI_MODEL,
    "PP-LCNet_x0_25_textline_ori": "PaddlePaddle/PP-LCNet_x0_25_textline_ori_safetensors",
    "UVDoc": DEFAULT_UNWARP_MODEL,
    # PP-OCRv6 Text Detection
    "PP-OCRv6_medium_det": "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
    "PP-OCRv6_server_det": "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
    "PP-OCRv6_small_det": "PaddlePaddle/PP-OCRv6_small_det_safetensors",
    "PP-OCRv6_mobile_det": "PaddlePaddle/PP-OCRv6_small_det_safetensors",
    "PP-OCRv6_tiny_det": "PaddlePaddle/PP-OCRv6_tiny_det_safetensors",
    # PP-OCRv6 Text Recognition
    "PP-OCRv6_medium_rec": "PaddlePaddle/PP-OCRv6_medium_rec_safetensors",
    "PP-OCRv6_server_rec": "PaddlePaddle/PP-OCRv6_medium_rec_safetensors",
    "PP-OCRv6_small_rec": "PaddlePaddle/PP-OCRv6_small_rec_safetensors",
    "PP-OCRv6_mobile_rec": "PaddlePaddle/PP-OCRv6_small_rec_safetensors",
    "PP-OCRv6_tiny_rec": "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors",
    # PP-OCRv5 Text Detection & Recognition
    "PP-OCRv5_server_det": "PaddlePaddle/PP-OCRv5_server_det_safetensors",
    "PP-OCRv5_mobile_det": "PaddlePaddle/PP-OCRv5_mobile_det_safetensors",
    "PP-OCRv5_server_rec": "PaddlePaddle/PP-OCRv5_server_rec_safetensors",
    "PP-OCRv5_mobile_rec": "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors",
    # PP-OCRv4 Text Detection & Recognition
    "PP-OCRv4_server_det": "PaddlePaddle/PP-OCRv4_server_det_safetensors",
    "PP-OCRv4_mobile_det": "PaddlePaddle/PP-OCRv4_mobile_det_safetensors",
    "PP-OCRv4_server_rec": "PaddlePaddle/PP-OCRv4_server_rec_safetensors",
    "PP-OCRv4_mobile_rec": "PaddlePaddle/PP-OCRv4_mobile_rec_safetensors",
    # Layout Detection
    "PP-DocLayoutV4": "PaddlePaddle/PP-DocLayoutV4_safetensors",
    "PP-DocLayoutV3": "PaddlePaddle/PP-DocLayoutV3_safetensors",
    "PP-DocLayoutV2": "PaddlePaddle/PP-DocLayoutV2_safetensors",
    "PP-DocLayout_plus-L": "PaddlePaddle/PP-DocLayout_plus-L_safetensors",
    "PP-DocLayout-L": "PaddlePaddle/PP-DocLayout-L_safetensors",
    "PP-DocLayout-M": "PaddlePaddle/PP-DocLayout-M_safetensors",
    "PP-DocLayout-S": "PaddlePaddle/PP-DocLayout-S_safetensors",
    # Table Recognition
    "PP-LCNet_x1_0_table_cls": DEFAULT_TABLE_CLS_MODEL,
    "SLANeXt_wired": DEFAULT_WIRED_TABLE_STRUCT_MODEL,
    "SLANeXt_wireless": DEFAULT_WIRELESS_TABLE_STRUCT_MODEL,
    "SLANet_plus": "PaddlePaddle/SLANet_plus_safetensors",
    "SLANet": "PaddlePaddle/SLANet_safetensors",
    "RT-DETR-L_wired_table_cell_det": DEFAULT_WIRED_TABLE_CELL_MODEL,
    "RT-DETR-L_wireless_table_cell_det": DEFAULT_WIRELESS_TABLE_CELL_MODEL,
    # Formula Recognition
    "PP-FormulaNet_plus-L": DEFAULT_FORMULA_MODEL,
    "PP-FormulaNet_plus-M": "PaddlePaddle/PP-FormulaNet_plus-M_safetensors",
    "PP-FormulaNet_plus-S": "PaddlePaddle/PP-FormulaNet_plus-S_safetensors",
    "PP-FormulaNet-L": "PaddlePaddle/PP-FormulaNet-L_safetensors",
    "PP-FormulaNet-S": "PaddlePaddle/PP-FormulaNet-S_safetensors",
    # Seal & Chart Recognition
    "PP-OCRv4_server_seal_det": DEFAULT_SEAL_DET_MODEL,
    "PP-OCRv4_mobile_seal_det": "PaddlePaddle/PP-OCRv4_mobile_seal_det_safetensors",
    "PP-Chart2Table": DEFAULT_CHART_MODEL,
    # PaddleOCR-VL Series
    "PaddleOCR-VL-1.6": "PaddlePaddle/PaddleOCR-VL-1.6",
    "PaddleOCR-VL-1.6-0.9B": "PaddlePaddle/PaddleOCR-VL-1.6",
    "PaddleOCR-VL-1.5": "PaddlePaddle/PaddleOCR-VL-1.5",
    "PaddleOCR-VL-1.5-0.9B": "PaddlePaddle/PaddleOCR-VL-1.5",
    "PaddleOCR-VL": "PaddlePaddle/PaddleOCR-VL",
    "PaddleOCR-VL-0.9B": "PaddlePaddle/PaddleOCR-VL",
}

VLM_PIPELINE_DEFAULTS: dict[str, tuple[str, str]] = {
    "v1.6": ("PaddlePaddle/PaddleOCR-VL-1.6", "PaddlePaddle/PP-DocLayoutV3_safetensors"),
    "v1.5": ("PaddlePaddle/PaddleOCR-VL-1.5", "PaddlePaddle/PP-DocLayoutV3_safetensors"),
    "v1": ("PaddlePaddle/PaddleOCR-VL", "PaddlePaddle/PP-DocLayoutV2_safetensors"),
}

OCR_VERSION_DEFAULTS: dict[str, tuple[str, str]] = {
    "PP-OCRv6": (DEFAULT_TEXT_DET_MODEL, DEFAULT_TEXT_REC_MODEL),
    "PP-OCRv5": (
        "PaddlePaddle/PP-OCRv5_server_det_safetensors",
        "PaddlePaddle/PP-OCRv5_server_rec_safetensors",
    ),
    "PP-OCRv4": (
        "PaddlePaddle/PP-OCRv4_server_det_safetensors",
        "PaddlePaddle/PP-OCRv4_server_rec_safetensors",
    ),
}


def resolve_model_id(name_or_path: str | Path | None, default: str) -> str:
    """Resolve a short PaddleOCR model alias, HF Hub ID, or local path into a loadable ID."""
    if name_or_path is None:
        return default

    raw = str(name_or_path).strip()
    if not raw:
        return default

    if Path(raw).exists():
        return raw

    if raw in MODEL_ALIASES:
        return MODEL_ALIASES[raw]

    if "/" in raw:
        return raw

    if raw.endswith("_safetensors"):
        return f"PaddlePaddle/{raw}"

    return f"PaddlePaddle/{raw}_safetensors"
