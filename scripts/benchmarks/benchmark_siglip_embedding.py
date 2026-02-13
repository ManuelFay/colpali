#!/usr/bin/env python3
"""Benchmark SigLIP image embedding throughput for sequential vs batched processing."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


@dataclass
class BenchmarkResult:
    mode: str
    batch_size: int
    num_images: int
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    std_latency_s: float
    throughput_images_per_s: float


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_images(n: int, size: int) -> List[Image.Image]:
    images: List[Image.Image] = []
    for i in range(n):
        color = (i * 31 % 255, i * 57 % 255, i * 97 % 255)
        images.append(Image.new("RGB", (size, size), color=color))
    return images


def _iter_chunks(images: Sequence[Image.Image], batch_size: int) -> List[List[Image.Image]]:
    return [list(images[i : i + batch_size]) for i in range(0, len(images), batch_size)]


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(fn, device: torch.device, warmup: int, repeats: int) -> List[float]:
    for _ in range(warmup):
        fn()
    _sync_if_needed(device)

    timings: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        _sync_if_needed(device)
        timings.append(time.perf_counter() - start)
    return timings


def _stats(mode: str, batch_size: int, num_images: int, timings: Sequence[float]) -> BenchmarkResult:
    ordered = sorted(timings)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]
    mean = statistics.mean(timings)
    std = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    return BenchmarkResult(
        mode=mode,
        batch_size=batch_size,
        num_images=num_images,
        mean_latency_s=mean,
        p50_latency_s=p50,
        p95_latency_s=p95,
        std_latency_s=std,
        throughput_images_per_s=num_images / mean,
    )


def _run_sequential(images: Sequence[Image.Image], processor, model, device: torch.device) -> None:
    with torch.inference_mode():
        for image in images:
            batch = processor(images=[image], return_tensors="pt")
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            _ = model(pixel_values=pixel_values)


def _run_batched(images: Sequence[Image.Image], batch_size: int, processor, model, device: torch.device) -> None:
    with torch.inference_mode():
        for chunk in _iter_chunks(images, batch_size):
            batch = processor(images=chunk, return_tensors="pt")
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            _ = model(pixel_values=pixel_values)


def _render_markdown(results: Sequence[BenchmarkResult], args: argparse.Namespace, device_name: str) -> str:
    header = [
        "# SigLIP embedding throughput benchmark",
        "",
        f"- model: `{args.vision_model_name}`",
        f"- device: `{device_name}`",
        f"- num_images: `{args.num_images}`",
        f"- image_size: `{args.image_size}`",
        f"- batch_sizes: `{args.batch_sizes}`",
        f"- warmup: `{args.warmup}`",
        f"- repeats: `{args.repeats}`",
        "",
        "| mode | batch | mean(s) | p50(s) | p95(s) | std(s) | images/s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [
        (
            f"| {r.mode} | {r.batch_size} | {r.mean_latency_s:.4f} | {r.p50_latency_s:.4f} | "
            f"{r.p95_latency_s:.4f} | {r.std_latency_s:.4f} | {r.throughput_images_per_s:.2f} |"
        )
        for r in results
    ]
    return "\n".join(header + rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SigLIP embedding speed for sequential vs batched image processing.")
    parser.add_argument("--vision-model-name", default="google/siglip2-base-patch16-512")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="benchmark_reports/siglip")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available.")

    _seed_everything(args.seed)
    processor = AutoImageProcessor.from_pretrained(args.vision_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.vision_model_name, trust_remote_code=True).to(device).eval()
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    images = _make_images(args.num_images, args.image_size)
    results: List[BenchmarkResult] = []

    print(
        f"Loaded SigLIP model={args.vision_model_name} on device={device_name}. "
        f"num_images={args.num_images}, image_size={args.image_size}, batch_sizes={args.batch_sizes}"
    )

    print("Running sequential baseline (batch=1)...")
    sequential_timings = _time(
        lambda: _run_sequential(images, processor, model, device),
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    results.append(_stats("sequential", 1, args.num_images, sequential_timings))
    print(f"[sequential] mean={results[-1].mean_latency_s:.4f}s images/s={results[-1].throughput_images_per_s:.2f}")

    for batch_size in args.batch_sizes:
        if batch_size <= 1 or batch_size > args.num_images:
            continue
        print(f"Running batched (batch={batch_size})...")
        timings = _time(
            lambda b=batch_size: _run_batched(images, b, processor, model, device),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results.append(_stats("batched", batch_size, args.num_images, timings))
        print(f"[batched][batch={batch_size}] mean={results[-1].mean_latency_s:.4f}s images/s={results[-1].throughput_images_per_s:.2f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "siglip_embedding_benchmark.json"
    md_path = output_dir / "siglip_embedding_benchmark.md"

    payload: Dict[str, object] = {
        "config": {
            "vision_model_name": args.vision_model_name,
            "device": str(device),
            "device_name": device_name,
            "num_images": args.num_images,
            "image_size": args.image_size,
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "results": [asdict(r) for r in results],
    }
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(_render_markdown(results, args, device_name))

    print(f"\nSaved JSON report to: {json_path}")
    print(f"Saved Markdown report to: {md_path}")


if __name__ == "__main__":
    main()
