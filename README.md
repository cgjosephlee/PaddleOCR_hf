# PaddleOCR-HF

Pure PyTorch & Hugging Face `transformers` implementation of **`PaddleOCR`**, **`PaddleOCRVL`**, and **`PPStructureV3`** with **zero PaddlePaddle dependencies**.

## Highlights

- **Zero Paddle Dependencies**: No `paddlepaddle` or `paddlex` required. Powered 100% by PyTorch (`torch`, `torchvision`) and Hugging Face `transformers` (`safetensors`).
- **Latest Models by Default**:
  - **`PaddleOCR`**: Defaults to **PP-OCRv6** (`PaddlePaddle/PP-OCRv6_medium_det_safetensors` & `PaddlePaddle/PP-OCRv6_small_rec_safetensors`, supporting 50 languages).
  - **`PaddleOCRVL`**: Defaults to **PaddleOCR-VL-1.6** (`PaddlePaddle/PaddleOCR-VL-1.6` + `PaddlePaddle/PP-DocLayoutV3_safetensors`).
  - **`PPStructureV3`**: Modular layout parsing (`PP-DocLayoutV3/V4`), OCR (`PP-OCRv6`), Table Recognition (`SLANeXt` + `RT-DETR`), Formula Recognition (`PP-FormulaNet_plus`), Seal Recognition (`PP-OCRv4_seal`), Chart Parsing (`PP-Chart2Table`), Document Orientation (`PP-LCNet`), and Document Unwarping (`UVDoc`) with **lazy model loading**.
- **Native PyTorch Device Support**: Seamlessly runs on Apple Silicon (`device="mps"`), NVIDIA GPUs (`device="cuda"`), or `device="cpu"`. Automatically selects `cuda` -> `mps` -> `cpu` when `device=None`.
- **Flexible VLM Execution (`PaddleOCRVL`)**: Switch effortlessly between local PyTorch `transformers` inference and external OpenAI-compatible inference servers (`vllm-server`, `sglang-server`, `fastdeploy-server`, `mlx-vlm-server`, `llama-cpp-server`, `openai`).

---

## Installation

```bash
uv sync --managed-python
```

---

## Quick Start

### 1. `PaddleOCR` (PP-OCRv6)

```python
from paddleocr import PaddleOCR

# Automatically uses 'mps' on Apple Silicon or 'cuda' on NVIDIA GPUs
ocr = PaddleOCR(device="mps")

results = ocr.predict("sample.jpg")
for res in results:
    res.print()
    res.save_to_json("output")
    res.save_to_markdown("output")
    res.save_to_img("output")
```

### 2. `PaddleOCRVL` (PaddleOCR-VL-1.6: Local PyTorch vs. External API)

#### Local PyTorch (`transformers`)

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    device="mps",
    dtype="bfloat16",
)

results = pipeline.predict("document.pdf")
for res in results:
    res.save_to_markdown("output")
```

#### External OpenAI-Compatible API (`vLLM` / `SGLang` / `OpenAI`)

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://127.0.0.1:8000/v1",
    vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.6",
    vl_rec_api_key="EMPTY",
    device="mps",  # Local preprocessing/layout detection still runs on PyTorch device
)

results = pipeline.predict("document.png")
```

### 3. `PPStructureV3` (Full Modular Document Parsing)

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_table_recognition=True,
    use_formula_recognition=True,
    use_seal_recognition=True,
    use_chart_recognition=True,
    device="mps",
)

results = pipeline.predict("complex_document.pdf")
for res in results:
    res.save_to_markdown("output")
    res.save_to_json("output")
```

---

## Benchmark: Apple Silicon MPS vs. CPU

Benchmark evaluated on [`sample.jpg`](sample.jpg) (dense Traditional Chinese + English tax withholding statement containing **84 detected text boxes**) using **PP-OCRv6** (`PP-OCRv6_medium_det_safetensors` + `PP-OCRv6_small_rec_safetensors`, `batch_size=16`, 3 timed runs after warmup with `torch.mps.synchronize()`):

| Stage | Hugging Face `safetensors` Model | `device="mps"` | `device="cpu"` | **MPS Speedup** |
| :--- | :--- | :---: | :---: | :---: |
| **1. Text Detection** | `PaddlePaddle/PP-OCRv6_medium_det_safetensors` | **0.249 s** *(0.234s – 0.263s)* | **1.871 s** *(1.653s – 2.082s)* | **7.53x faster** |
| **2. Text Recognition (84 boxes)** | `PaddlePaddle/PP-OCRv6_small_rec_safetensors` | **1.107 s** *(steady-state **0.864 s**)* | **12.382 s** *(11.131s – 13.667s)* | **11.19x faster** *(steady-state **14.3x**)* |
| **End-to-End Pipeline** | **PP-OCRv6 (Det + Crop + Rec)** | **1.357 s** *(steady-state **1.102 s**)* | **14.256 s** | **10.50x faster** *(steady-state **12.9x**)* |

- **Accuracy Parity**: Both `mps:0` and `cpu` detect the exact same **84 text polygons** and produce identical recognition strings (100% match).
- **Reproduce the Benchmark**:
  ```bash
  uv run python benchmarks/benchmark_mps_vs_cpu.py
  ```

---

## Development & Quality Checks

```bash
# Lint & format with Ruff
uv run ruff check .
uv run ruff format --check .

# Strict type checking with Pyright
uv run pyright

# Run unit tests with Pytest
uv run pytest
```
