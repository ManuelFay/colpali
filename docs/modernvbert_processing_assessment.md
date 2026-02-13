# ModernVBERT document-processing throughput assessment

## What I inspected

- `ColModernVBertProcessor.process_images` calls `self(...)` once for the full list of images, then delegates all heavy lifting to the Hugging Face `Idefics3Processor` stack.
- `ModernVBertModel.forward` flattens image tensors, removes all-zero padded image slots, runs the vision encoder once over the resulting image batch, projects image tokens, and merges with text embeddings before one language-model call.
- The wrapped processor path (`Idefics3Processor.__call__` -> `Idefics3ImageProcessor.preprocess`) performs nested Python loops during split/crop, conversion, normalization, and padding.

## Main performance findings (code-level)

1. **There is already macro-batching at the top-level API**, but not fully vectorized internals:
   - `process_images` is called on a list of images at once.
   - Internally, image preprocessing still loops over images (and over generated crops) in Python.

2. **The likely bottleneck is image preprocessing/cropping**, not just model forward:
   - `Idefics3ImageProcessor.preprocess` performs repeated per-image operations (split, resize, normalization, padding) with multiple list comprehensions and loops.
   - For large scanned documents with many crops, this can dominate wall-clock before GPU compute starts.

3. **Vision and language forward are already batched per model invocation**, but end-to-end pipelines often become effectively sequential when user code calls model/processor per-document.

4. **Ragged multimodal sequences reduce LM efficiency**:
   - Different image counts / token lengths per document produce uneven sequence sizes and high padding overhead if mixed naively.
   - Grouping by similar image-token budgets should improve throughput.

## Benchmark tests added

Added `tests/models/modernvbert/test_processing_throughput_benchmarks.py` with:

- CPU benchmark comparing image preprocessing done sequentially (one-image calls) vs one batched call using `Idefics3ImageProcessor` on mixed-size synthetic images.
- Mock LM benchmark comparing sequential per-sample forwards vs padded batched forwards.
- CUDA path included but skipped automatically when CUDA is unavailable.

## Improvement plan (proposal for approval)

### Phase 1: Low-risk pipeline changes

1. **Introduce two-stage batching in inference/indexing pipeline**
   - Stage A: preprocess/crop all images in DataLoader workers (CPU), emit tensors + metadata.
   - Stage B: batch all image crops for vision forward on GPU.
   - Stage C: pack/ bucket LM inputs by approximate token length and run large LM batches.

2. **Use DataLoader optimizations**
   - `num_workers > 0`, `pin_memory=True`, `persistent_workers=True`, tuned `prefetch_factor`.
   - Keep processor/image work in workers; only final tensor moves on main process.

3. **Asynchronous H2D and overlap**
   - Use non-blocking `.to(device, non_blocking=True)` from pinned buffers.
   - Optional CUDA streams to overlap transfer with compute.

### Phase 2: Better token/crop packing

4. **Bucket by image token budget**
   - Estimate token count from image size + splitting rules.
   - Build batches with similar token lengths to reduce padding waste in LM.

5. **Crop-level micro-batching**
   - Instead of document-by-document, create a global crop queue and process crops in large micro-batches.
   - Reassemble per-document outputs with index maps.

### Phase 3: Optional advanced optimizations

6. **`torch.compile` / fused kernels / mixed precision tuning** on vision and text components where stable.
7. **Prefetch next CPU batch while current GPU batch executes**.
8. **Profile-guided tuning** (PyTorch profiler + CUDA timeline) before/after each phase.

## Success metrics

- Throughput: documents/sec and crops/sec.
- GPU utilization and SM occupancy.
- End-to-end latency percentiles.
- CPU preprocessing share of total wall-clock.
- Padding ratio in LM batches.

## Reproducible GPU latency benchmark runner

Use `scripts/benchmarks/benchmark_modernvbert_latency.py` to generate before/after reports.

Example:

```bash
python scripts/benchmarks/benchmark_modernvbert_latency.py \
  --device cuda \
  --num-docs 64 \
  --batch-sizes 1,2,4,8,16 \
  --image-size-modes uniform,mixed,large \
  --warmup 2 \
  --repeats 8 \
  --output-dir benchmark_reports/before
```

The script produces:

- `modernvbert_latency_report.json`
- `modernvbert_latency_report.md`

The report includes multiple scenarios to isolate bottlenecks:

- `sequential_end_to_end`
- `batched_end_to_end`
- `processor_only_sequential`
- `processor_only_batched`
- `model_only_preprocessed`
