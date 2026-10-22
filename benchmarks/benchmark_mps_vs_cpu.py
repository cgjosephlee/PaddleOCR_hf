"""Benchmark PaddleOCR (PP-OCRv6) comparing Apple Silicon MPS vs CPU."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from paddleocr import PaddleOCR
from paddleocr.io import crop_polygon, sort_polygons_reading_order

DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "sample.jpg"


@dataclass(slots=True)
class BenchmarkMetrics:
    """Latency and accuracy metrics for a single device benchmark."""

    device: str
    det_param_device: str
    rec_param_device: str
    num_boxes: int
    load_s: float
    warmup_s: float
    det_runs_s: list[float]
    rec_runs_s: list[float]
    total_runs_s: list[float]
    texts: list[str]

    @property
    def avg_det_s(self) -> float:
        return sum(self.det_runs_s) / len(self.det_runs_s)

    @property
    def avg_rec_s(self) -> float:
        return sum(self.rec_runs_s) / len(self.rec_runs_s)

    @property
    def avg_total_s(self) -> float:
        return sum(self.total_runs_s) / len(self.total_runs_s)


def sync_device(device: torch.device) -> None:
    """Synchronize accelerator execution before taking timestamps."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run_ocr_benchmark(
    image_path: Path,
    device_name: str,
    *,
    num_runs: int = 3,
    batch_size: int = 16,
) -> BenchmarkMetrics:
    """Run warmup + timed OCR runs on the given device and return stage breakdown metrics."""
    print(f"\n{'=' * 64}")
    print(f"Benchmarking PaddleOCR (PP-OCRv6) on device='{device_name}'")
    print(f"{'=' * 64}")

    t0 = time.perf_counter()
    ocr = PaddleOCR(
        device=device_name,
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_small_rec",
        text_recognition_batch_size=batch_size,
    )
    det_runner = ocr.det_runner
    rec_runner = ocr.rec_runner
    sync_device(ocr.device)
    load_time = time.perf_counter() - t0

    det_param_dev = str(next(det_runner.model.parameters()).device)
    rec_param_dev = str(next(rec_runner.model.parameters()).device)
    print(f"Resolved device       : {ocr.device} (dtype={ocr.dtype})")
    print(f"Det model param device: {det_param_dev} ({ocr.det_model_id})")
    print(f"Rec model param device: {rec_param_dev} ({ocr.rec_model_id})")
    print(f"Model load time       : {load_time:.3f} s")

    # Warmup run
    t_warm = time.perf_counter()
    warmup_res = ocr.predict(image_path)
    sync_device(ocr.device)
    warmup_time = time.perf_counter() - t_warm
    boxes = warmup_res[0].boxes
    print(f"Warmup run time       : {warmup_time:.3f} s (detected {len(boxes)} text boxes)")

    img = Image.open(image_path).convert("RGB")
    det_times: list[float] = []
    rec_times: list[float] = []
    total_times: list[float] = []

    for i in range(num_runs):
        sync_device(ocr.device)
        t_start = time.perf_counter()

        t_d0 = time.perf_counter()
        raw_polys = ocr.detect_text_polygons(img)
        sync_device(ocr.device)
        t_d1 = time.perf_counter()

        ordered = sort_polygons_reading_order(raw_polys)
        crops = [crop_polygon(img, poly) for poly, _ in ordered]

        t_r0 = time.perf_counter()
        _ = ocr.recognize_text_crops(crops, batch_size=batch_size)
        sync_device(ocr.device)
        t_r1 = time.perf_counter()

        total_t = t_r1 - t_start
        det_t = t_d1 - t_d0
        rec_t = t_r1 - t_r0

        det_times.append(det_t)
        rec_times.append(rec_t)
        total_times.append(total_t)
        print(
            f"  Run {i + 1}/{num_runs}: total={total_t:.3f}s (det={det_t:.3f}s, rec={rec_t:.3f}s)"
        )

    metrics = BenchmarkMetrics(
        device=str(ocr.device),
        det_param_device=det_param_dev,
        rec_param_device=rec_param_dev,
        num_boxes=len(boxes),
        load_s=load_time,
        warmup_s=warmup_time,
        det_runs_s=det_times,
        rec_runs_s=rec_times,
        total_runs_s=total_times,
        texts=warmup_res[0].rec_texts,
    )
    print(
        f"Average ({num_runs} runs)  : total={metrics.avg_total_s:.3f}s "
        f"(det={metrics.avg_det_s:.3f}s, rec={metrics.avg_rec_s:.3f}s)"
    )
    print("Sample recognized texts (first 8):")
    for idx, box in enumerate(boxes[:8]):
        print(f"  [{idx + 1}] ({box.score:.3f}) {box.text}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PaddleOCR on MPS vs CPU.")
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
        help="Path to input image for benchmarking (default: sample.jpg).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs after warmup (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for text recognition (default: 16).",
    )
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available  : {torch.backends.mps.is_available()}")
    print(f"MPS built      : {torch.backends.mps.is_built()}")
    print(f"Sample image   : {args.image}")

    mps_stats = run_ocr_benchmark(
        args.image,
        "mps",
        num_runs=args.runs,
        batch_size=args.batch_size,
    )
    cpu_stats = run_ocr_benchmark(
        args.image,
        "cpu",
        num_runs=args.runs,
        batch_size=args.batch_size,
    )

    det_speedup = cpu_stats.avg_det_s / mps_stats.avg_det_s
    rec_speedup = cpu_stats.avg_rec_s / mps_stats.avg_rec_s
    total_speedup = cpu_stats.avg_total_s / mps_stats.avg_total_s

    print(f"\n{'=' * 64}")
    print(f"MPS vs CPU Summary on {args.image.name} ({mps_stats.num_boxes} text boxes)")
    print(f"{'=' * 64}")
    print(f"Exact text match: {mps_stats.texts == cpu_stats.texts}")
    print(
        f"Detection       : MPS={mps_stats.avg_det_s:.3f}s vs "
        f"CPU={cpu_stats.avg_det_s:.3f}s ({det_speedup:.2f}x speedup)"
    )
    print(
        f"Recognition     : MPS={mps_stats.avg_rec_s:.3f}s vs "
        f"CPU={cpu_stats.avg_rec_s:.3f}s ({rec_speedup:.2f}x speedup)"
    )
    print(
        f"End-to-End      : MPS={mps_stats.avg_total_s:.3f}s vs "
        f"CPU={cpu_stats.avg_total_s:.3f}s ({total_speedup:.2f}x speedup)"
    )


if __name__ == "__main__":
    main()
