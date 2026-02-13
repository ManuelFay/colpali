#!/usr/bin/env python3
"""GPU latency benchmark for ModernVBERT document processing.

This benchmark is intended to be run before and after pipeline optimizations.
It generates JSON and Markdown reports for easy comparison.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Sequence, Tuple

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
from colpali_engine.utils.transformers_wrappers import AutoProcessorWrapper


@dataclass
class ScenarioResult:
    scenario: str
    num_docs: int
    image_size_mode: str
    batch_size: int
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    std_latency_s: float
    throughput_docs_per_s: float


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]




def _load_modernvbert_processor(model_name: str) -> ColModernVBertProcessor:
    try:
        return ColModernVBertProcessor.from_pretrained(model_name)
    except TypeError as exc:
        # Some transformers/runtime combinations fail to route args into Idefics3Processor-based
        # custom processors. Fallback: load generic AutoProcessor and re-wrap into ColModernVBertProcessor.
        auto_processor = AutoProcessorWrapper(model_name)

        image_processor = getattr(auto_processor, "image_processor", None)
        tokenizer = getattr(auto_processor, "tokenizer", None)
        if image_processor is None or tokenizer is None:
            raise RuntimeError(
                "Failed to construct ColModernVBertProcessor fallback: AutoProcessor is missing "
                "image_processor/tokenizer components."
            ) from exc

        image_seq_len = getattr(auto_processor, "image_seq_len", 64)
        chat_template = getattr(tokenizer, "chat_template", None)
        return ColModernVBertProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            image_seq_len=image_seq_len,
            chat_template=chat_template,
        )

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_images(n: int, mode: str) -> List[Image.Image]:
    if mode == "uniform":
        sizes = [(1024, 768)] * n
    elif mode == "mixed":
        palette = [(512, 512), (768, 512), (512, 768), (1024, 768), (768, 1024), (1400, 1000), (1000, 1400)]
        sizes = [palette[i % len(palette)] for i in range(n)]
    elif mode == "large":
        palette = [(1536, 1024), (1800, 1200), (2048, 1536)]
        sizes = [palette[i % len(palette)] for i in range(n)]
    else:
        raise ValueError(f"Unsupported image size mode: {mode}")

    images: List[Image.Image] = []
    for i, size in enumerate(sizes):
        color = (i * 37 % 255, i * 73 % 255, i * 17 % 255)
        images.append(Image.new("RGB", size=size, color=color))
    return images


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_scenario(
    fn: Callable[[], None],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> List[float]:
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


def _compute_stats(
    scenario: str,
    timings: Sequence[float],
    num_docs: int,
    batch_size: int,
    image_size_mode: str,
) -> ScenarioResult:
    ordered = sorted(timings)
    p50 = ordered[len(ordered) // 2]
    p95_idx = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    p95 = ordered[p95_idx]
    mean = statistics.mean(timings)
    std = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    throughput = num_docs / mean
    return ScenarioResult(
        scenario=scenario,
        num_docs=num_docs,
        image_size_mode=image_size_mode,
        batch_size=batch_size,
        mean_latency_s=mean,
        p50_latency_s=p50,
        p95_latency_s=p95,
        std_latency_s=std,
        throughput_docs_per_s=throughput,
    )


def _iter_chunks(xs: Sequence[Image.Image], batch_size: int) -> List[List[Image.Image]]:
    return [list(xs[i : i + batch_size]) for i in range(0, len(xs), batch_size)]


def _move_batch_to_device(
    batch: dict,
    device: torch.device,
    *,
    non_blocking: bool = False,
    pin_memory: bool = False,
) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            tensor = value
            if pin_memory and tensor.device.type == "cpu":
                tensor = tensor.pin_memory()
            moved[key] = tensor.to(device, non_blocking=non_blocking)
        else:
            moved[key] = value
    return moved


def _scenario_sequential_end_to_end(
    images: Sequence[Image.Image],
    processor: ColModernVBertProcessor,
    model: ColModernVBert,
    device: torch.device,
) -> None:
    with torch.no_grad():
        for image in images:
            batch = processor.process_images([image]).to(device)
            _ = model(**batch)


def _scenario_batched_end_to_end(
    images: Sequence[Image.Image],
    processor: ColModernVBertProcessor,
    model: ColModernVBert,
    device: torch.device,
    batch_size: int,
    *,
    non_blocking: bool = False,
    pin_memory: bool = False,
) -> None:
    with torch.no_grad():
        for chunk in _iter_chunks(images, batch_size):
            batch_cpu = processor.process_images(chunk)
            batch = _move_batch_to_device(batch_cpu, device, non_blocking=non_blocking, pin_memory=pin_memory)
            _ = model(**batch)


def _scenario_processor_only_sequential(images: Sequence[Image.Image], processor: ColModernVBertProcessor) -> None:
    for image in images:
        _ = processor.process_images([image])


def _scenario_processor_only_batched(
    images: Sequence[Image.Image],
    processor: ColModernVBertProcessor,
    batch_size: int,
) -> None:
    for chunk in _iter_chunks(images, batch_size):
        _ = processor.process_images(chunk)


def _scenario_processor_only_batched_threaded(
    images: Sequence[Image.Image],
    processor: ColModernVBertProcessor,
    batch_size: int,
    num_threads: int,
) -> None:
    chunks = _iter_chunks(images, batch_size)
    if num_threads <= 1:
        for chunk in chunks:
            _ = processor.process_images(chunk)
        return

    def _process(chunk: List[Image.Image]) -> None:
        _ = processor.process_images(chunk)

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        list(pool.map(_process, chunks))


def _scenario_model_only_from_preprocessed(
    images: Sequence[Image.Image],
    processor: ColModernVBertProcessor,
    model: ColModernVBert,
    device: torch.device,
    batch_size: int,
) -> None:
    preprocessed = [processor.process_images(chunk).to(device) for chunk in _iter_chunks(images, batch_size)]
    with torch.no_grad():
        for batch in preprocessed:
            _ = model(**batch)


def _scenario_model_only_cached_preprocessed(
    preprocessed: Sequence[dict],
    model: ColModernVBert,
) -> None:
    with torch.no_grad():
        for batch in preprocessed:
            _ = model(**batch)


def _extract_real_pixel_values(pixel_values: torch.Tensor) -> torch.Tensor:
    batch_size, num_images, _, _, _ = pixel_values.shape
    pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])
    nb_values_per_image = pixel_values.shape[1:].numel()
    real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
    if not any(real_images_inds):
        real_images_inds[0] = True
    return pixel_values[real_images_inds].contiguous()


def _scenario_vision_only_cached_preprocessed(
    preprocessed: Sequence[dict],
    model: ColModernVBert,
) -> None:
    core_model = model.model
    with torch.no_grad():
        for batch in preprocessed:
            pixel_values = _extract_real_pixel_values(batch["pixel_values"])
            image_hidden_states = core_model.vision_model(pixel_values=pixel_values).last_hidden_state
            _ = core_model.connector(image_hidden_states)


def _build_image_hidden_state_cache(
    preprocessed: Sequence[dict],
    model: ColModernVBert,
) -> List[torch.Tensor]:
    core_model = model.model
    cache: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in preprocessed:
            pixel_values = _extract_real_pixel_values(batch["pixel_values"])
            image_hidden_states = core_model.vision_model(pixel_values=pixel_values).last_hidden_state
            cache.append(core_model.connector(image_hidden_states))
    return cache


def _scenario_text_only_cached_preprocessed(
    preprocessed: Sequence[dict],
    image_hidden_states_cache: Sequence[torch.Tensor],
    model: ColModernVBert,
) -> None:
    core_model = model.model
    with torch.no_grad():
        for batch, image_hidden_states in zip(preprocessed, image_hidden_states_cache):
            input_ids = batch["input_ids"]
            inputs_embeds = core_model.text_model.get_input_embeddings()(input_ids).to(input_ids.device)
            inputs_embeds = core_model.inputs_merger(input_ids, inputs_embeds, image_hidden_states)
            _ = core_model.text_model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch.get("attention_mask"),
                position_ids=batch.get("position_ids"),
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )


def _scenario_split_vision_gpu_text_cpu_preprocessed(
    preprocessed: Sequence[dict],
    model: ColModernVBert,
    text_model_cpu: torch.nn.Module,
) -> None:
    core_model = model.model
    with torch.no_grad():
        for batch in preprocessed:
            pixel_values = _extract_real_pixel_values(batch["pixel_values"])
            pixel_values = pixel_values.to(next(core_model.vision_model.parameters()).device)
            image_hidden_states = core_model.vision_model(pixel_values=pixel_values).last_hidden_state
            image_hidden_states = core_model.connector(image_hidden_states).to("cpu")

            input_ids = batch["input_ids"].to("cpu")
            attention_mask = batch.get("attention_mask")
            attention_mask = attention_mask.to("cpu") if attention_mask is not None else None
            position_ids = batch.get("position_ids")
            position_ids = position_ids.to("cpu") if position_ids is not None else None

            inputs_embeds = text_model_cpu.get_input_embeddings()(input_ids)
            inputs_embeds = core_model.inputs_merger(input_ids, inputs_embeds, image_hidden_states)
            _ = text_model_cpu(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False,
            )


def _render_markdown(results: Sequence[ScenarioResult], args: argparse.Namespace, device_name: str) -> str:
    header = [
        "# ModernVBERT latency benchmark report",
        "",
        f"- model: `{args.model_name}`",
        f"- device: `{device_name}`",
        f"- num_docs: `{args.num_docs}`",
        f"- batch_sizes: `{args.batch_sizes}`",
        f"- image_size_modes: `{args.image_size_modes}`",
        f"- warmup: `{args.warmup}`",
        f"- repeats: `{args.repeats}`",
        "",
        "| scenario | image mode | batch | mean(s) | p50(s) | p95(s) | std(s) | docs/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [
        (
            f"| {r.scenario} | {r.image_size_mode} | {r.batch_size} | "
            f"{r.mean_latency_s:.4f} | {r.p50_latency_s:.4f} | {r.p95_latency_s:.4f} | "
            f"{r.std_latency_s:.4f} | {r.throughput_docs_per_s:.2f} |"
        )
        for r in results
    ]
    return "\n".join(header + rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ModernVBERT latency on GPU/CPU.")
    parser.add_argument("--model-name", default="ModernVBERT/colmodernvbert")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--num-docs", type=int, default=16)
    parser.add_argument("--batch-sizes", type=_parse_int_list, default=[1, 4, 8])
    parser.add_argument("--image-size-modes", type=lambda s: [x.strip() for x in s.split(",")], default=["uniform"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--scenarios",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=[
            "sequential_end_to_end",
            "batched_end_to_end",
            "processor_only_sequential",
            "processor_only_batched",
            "model_only_preprocessed",
            "vision_only_preprocessed",
            "text_only_preprocessed",
            "processor_only_batched_threaded",
            "split_vision_gpu_text_cpu_preprocessed",
        ],
        help="Comma-separated scenario names.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--processor-threads", type=int, default=1, help="Threads for threaded processor hypothesis.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin CPU tensors before H2D copy.")
    parser.add_argument("--non-blocking", action="store_true", help="Use non_blocking tensor transfer.")
    parser.add_argument("--output-dir", default="benchmark_reports")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available.")

    _seed_everything(args.seed)

    processor = _load_modernvbert_processor(args.model_name)
    model = ColModernVBert.from_pretrained(args.model_name).eval().to(device)

    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"

    print(
        f"Loaded model={args.model_name} on device={device_name}. "
        f"num_docs={args.num_docs}, modes={args.image_size_modes}, batch_sizes={args.batch_sizes}, "
        f"warmup={args.warmup}, repeats={args.repeats}, threads={args.processor_threads}, "
        f"pin_memory={args.pin_memory}, non_blocking={args.non_blocking}"
    )

    results: List[ScenarioResult] = []

    selected = set(args.scenarios)
    text_model_cpu = None
    if "split_vision_gpu_text_cpu_preprocessed" in selected:
        text_model_cpu = copy.deepcopy(model.model.text_model).to("cpu").eval()

    for mode in args.image_size_modes:
        images = _make_images(args.num_docs, mode)

        # Cache for model-only scenario to avoid recomputing processor cost in timing loop.
        model_only_cache: Dict[int, List[dict]] = {}
        vision_text_cache: Dict[int, Tuple[List[dict], List[torch.Tensor]]] = {}
        for batch_size in args.batch_sizes:
            if batch_size > args.num_docs:
                continue
            chunks = _iter_chunks(images, batch_size)
            model_only_cache[batch_size] = [processor.process_images(chunk).to(device) for chunk in chunks]
            image_hidden_states_cache = _build_image_hidden_state_cache(model_only_cache[batch_size], model)
            vision_text_cache[batch_size] = (model_only_cache[batch_size], image_hidden_states_cache)

        # Batch-size independent scenarios (run once per image mode)
        invariant: List[Tuple[str, int, Callable[[], None]]] = [
            (
                "sequential_end_to_end",
                1,
                lambda images=images: _scenario_sequential_end_to_end(images, processor, model, device),
            ),
            (
                "processor_only_sequential",
                1,
                lambda images=images: _scenario_processor_only_sequential(images, processor),
            ),
        ]

        for scenario_name, batch_size, scenario_fn in invariant:
            if scenario_name not in selected:
                continue
            print(f"Running [{mode}][batch={batch_size}] {scenario_name} ...")
            timings = _time_scenario(scenario_fn, device=device, warmup=args.warmup, repeats=args.repeats)
            results.append(
                _compute_stats(
                    scenario=scenario_name,
                    timings=timings,
                    num_docs=args.num_docs,
                    batch_size=batch_size,
                    image_size_mode=mode,
                )
            )
            print(
                f"[{mode}][batch={batch_size}] {scenario_name}: "
                f"mean={results[-1].mean_latency_s:.4f}s docs/s={results[-1].throughput_docs_per_s:.2f}"
            )

        for batch_size in args.batch_sizes:
            if batch_size > args.num_docs:
                continue

            variant: List[Tuple[str, Callable[[], None]]] = [
                (
                    "batched_end_to_end",
                    lambda images=images, b=batch_size: _scenario_batched_end_to_end(images, processor, model, device, b, non_blocking=args.non_blocking, pin_memory=args.pin_memory),
                ),
                (
                    "processor_only_batched",
                    lambda images=images, b=batch_size: _scenario_processor_only_batched(images, processor, b),
                ),
                (
                    "processor_only_batched_threaded",
                    lambda images=images, b=batch_size: _scenario_processor_only_batched_threaded(
                        images, processor, b, args.processor_threads
                    ),
                ),
                (
                    "model_only_preprocessed",
                    lambda cached=model_only_cache[batch_size]: _scenario_model_only_cached_preprocessed(cached, model),
                ),
                (
                    "vision_only_preprocessed",
                    lambda cached=model_only_cache[batch_size]: _scenario_vision_only_cached_preprocessed(cached, model),
                ),
                (
                    "text_only_preprocessed",
                    lambda cached=vision_text_cache[batch_size][0], ihs=vision_text_cache[batch_size][1]: _scenario_text_only_cached_preprocessed(
                        cached, ihs, model
                    ),
                ),
                (
                    "split_vision_gpu_text_cpu_preprocessed",
                    lambda cached=model_only_cache[batch_size], tm=text_model_cpu: _scenario_split_vision_gpu_text_cpu_preprocessed(
                        cached, model, tm
                    ),
                ),
            ]

            for scenario_name, scenario_fn in variant:
                if scenario_name not in selected:
                    continue
                print(f"Running [{mode}][batch={batch_size}] {scenario_name} ...")
                timings = _time_scenario(scenario_fn, device=device, warmup=args.warmup, repeats=args.repeats)
                results.append(
                    _compute_stats(
                        scenario=scenario_name,
                        timings=timings,
                        num_docs=args.num_docs,
                        batch_size=batch_size,
                        image_size_mode=mode,
                    )
                )
                print(
                    f"[{mode}][batch={batch_size}] {scenario_name}: "
                    f"mean={results[-1].mean_latency_s:.4f}s docs/s={results[-1].throughput_docs_per_s:.2f}"
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "modernvbert_latency_report.json"
    md_path = output_dir / "modernvbert_latency_report.md"

    payload: Dict[str, object] = {
        "config": {
            "model_name": args.model_name,
            "device": str(device),
            "device_name": device_name,
            "num_docs": args.num_docs,
            "batch_sizes": args.batch_sizes,
            "image_size_modes": args.image_size_modes,
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
