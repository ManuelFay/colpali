import time
from typing import List, Tuple

import pytest
import torch
from PIL import Image
from transformers.models.idefics3.image_processing_idefics3 import Idefics3ImageProcessor


@pytest.fixture(scope="module")
def mock_document_images() -> List[Image.Image]:
    sizes: List[Tuple[int, int]] = [
        (512, 512),
        (768, 512),
        (512, 768),
        (1024, 768),
        (768, 1024),
        (1400, 1000),
        (1000, 1400),
        (1280, 960),
    ]
    images = [Image.new("RGB", size, color=(i * 20 % 255, i * 40 % 255, i * 60 % 255)) for i, size in enumerate(sizes)]
    return images


def _timeit(fn, repeats: int = 1) -> float:
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed.append(time.perf_counter() - start)
    return min(elapsed)


def test_siglip_style_image_preprocessing_batch_vs_sequential_cpu(mock_document_images: List[Image.Image]):
    """Benchmark image preprocessing to compare per-image calls vs a single batched call on CPU."""
    image_processor = Idefics3ImageProcessor(
        do_resize=True,
        size={"longest_edge": 1024},
        do_image_splitting=True,
        max_image_size={"longest_edge": 512},
        do_pad=True,
    )

    def sequential():
        for image in mock_document_images:
            image_processor.preprocess([[image]], return_tensors="pt")

    def batched():
        image_processor.preprocess([[image] for image in mock_document_images], return_tensors="pt")

    sequential_s = _timeit(sequential)
    batched_s = _timeit(batched)

    sequential_out = [image_processor.preprocess([[image]], return_tensors="pt") for image in mock_document_images]
    batched_out = image_processor.preprocess([[image] for image in mock_document_images], return_tensors="pt")

    assert len(sequential_out) == len(mock_document_images)
    assert batched_out["pixel_values"].shape[0] == len(mock_document_images)
    assert sequential_s > 0
    assert batched_s > 0

    speedup = sequential_s / batched_s
    print(
        f"\nCPU image preprocessing benchmark: sequential={sequential_s:.4f}s, "
        f"batched={batched_s:.4f}s, speedup={speedup:.2f}x"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available in this environment")
def test_language_model_call_batching_cpu_vs_gpu_mock():
    """Benchmark sequential vs padded-batch LM-style forward passes (CPU and GPU when available)."""

    torch.manual_seed(0)
    hidden_size = 768
    seq_lens = [512, 640, 896, 1024, 384, 700, 256, 1200]
    batch = [torch.randn(seq, hidden_size) for seq in seq_lens]

    model = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_size, hidden_size),
    )

    def run(device: torch.device):
        local_model = model.to(device)
        local_batch = [x.to(device) for x in batch]

        def sequential():
            with torch.no_grad():
                for x in local_batch:
                    local_model(x)

        def batched():
            with torch.no_grad():
                padded = torch.nn.utils.rnn.pad_sequence(local_batch, batch_first=True)
                local_model(padded)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seq_s = _timeit(sequential)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_s = _timeit(batched)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        assert seq_s > 0
        assert batch_s > 0
        print(
            f"\n{device.type.upper()} LM benchmark: sequential={seq_s:.4f}s, "
            f"batched={batch_s:.4f}s, speedup={seq_s / batch_s:.2f}x"
        )

    run(torch.device("cpu"))
    run(torch.device("cuda"))
