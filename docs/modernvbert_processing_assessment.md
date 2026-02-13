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


### Notes on runtime behavior

- The script defaults are intentionally **quick** (`num_docs=16`, `batch_sizes=1,4,8`, single `uniform` mode) so `python scripts/benchmarks/benchmark_modernvbert_latency.py` does not look stalled.
- For full profiling runs, increase to your target workload (example below).
- TensorFlow backend loading is disabled in the script (`TRANSFORMERS_NO_TF=1`) to avoid noisy CUDA factory registration logs when not needed.

Full run example:

```bash
python scripts/benchmarks/benchmark_modernvbert_latency.py \
  --device cuda \
  --num-docs 128 \
  --batch-sizes 1,2,4,8,16,32 \
  --image-size-modes uniform,mixed,large \
  --warmup 2 \
  --repeats 8 \
  --output-dir benchmark_reports/full_before
```


### Vision vs text (LLM) latency breakdown

The benchmark now supports two additional scenarios:

- `vision_only_preprocessed`: vision encoder + connector only (no text model)
- `text_only_preprocessed`: text model only, using cached image hidden states

Recommended command:

```bash
python scripts/benchmarks/benchmark_modernvbert_latency.py \
  --device cuda \
  --num-docs 64 \
  --batch-sizes 1,2,4,8,16 \
  --image-size-modes uniform,mixed \
  --warmup 2 \
  --repeats 8 \
  --scenarios model_only_preprocessed,vision_only_preprocessed,text_only_preprocessed,batched_end_to_end,processor_only_batched \
  --output-dir benchmark_reports/breakdown
```


### Memory-IO reduction hypotheses (CPU preprocess + split execution)

To test the hypothesis that memory transfers and CPU preprocessing are limiting throughput, run:

```bash
python scripts/benchmarks/benchmark_modernvbert_latency.py \
  --device cuda \
  --num-docs 64 \
  --batch-sizes 1,2,4,8,16 \
  --image-size-modes uniform,mixed \
  --warmup 2 \
  --repeats 8 \
  --processor-threads 8 \
  --pin-memory \
  --non-blocking \
  --scenarios batched_end_to_end,processor_only_batched,processor_only_batched_threaded,model_only_preprocessed,vision_only_preprocessed,text_only_preprocessed,split_vision_gpu_text_cpu_preprocessed \
  --output-dir benchmark_reports/memory_io_hypotheses
```

Interpretation guide:

- `processor_only_batched_threaded` vs `processor_only_batched`: CPU-side preprocessing parallelization benefit.
- `split_vision_gpu_text_cpu_preprocessed` vs `model_only_preprocessed`: whether moving text model to CPU helps overall throughput.
- `batched_end_to_end` with `--pin-memory --non-blocking`: host↔device transfer overhead reduction impact.

### Focused SigLIP-only embedding benchmark (sequential vs batched)

If you want to isolate pure SigLIP embedding throughput (without ModernVBERT text path), run:

```bash
python scripts/benchmarks/benchmark_siglip_embedding.py \
  --device cuda \
  --num-images 128 \
  --image-size 1024 \
  --batch-sizes 1,2,4,8,16,32 \
  --warmup 2 \
  --repeats 8 \
  --output-dir benchmark_reports/siglip_only
```

This writes:

- `siglip_embedding_benchmark.json`
- `siglip_embedding_benchmark.md`

### Interpreting the latest user-observed pattern

If your report shows the following shape:

- `vision_only_preprocessed` >> `text_only_preprocessed`
- `processor_only_batched_threaded` ≈ `processor_only_batched`
- `split_vision_gpu_text_cpu_preprocessed` much slower than `model_only_preprocessed`

then the bottleneck is not the text model. It means:

1. Vision + connector dominates model-side cost.
2. Threaded preprocessing is likely limited by Python-level orchestration, serialization, or non-vectorized internals (so simple threads do not scale).
3. Moving text to CPU adds PCIe transfer + CPU execution overhead, and is a net regression.

In this case, prioritize:

- reducing crop/token budget,
- true process-based DataLoader parallelism + overlap (prefetch/pinned memory/non-blocking H2D),
- bucketed batching by image-token budget to reduce padding.


### Component-level timing command (preprocessing vs connector vs inputs merger vs text forward)

To isolate your hypothesis around `inputs_merger`, run:

```bash
python scripts/benchmarks/benchmark_modernvbert_latency.py \
  --device cuda \
  --num-docs 64 \
  --batch-sizes 1,4,8,16 \
  --image-size-modes uniform,mixed \
  --warmup 2 \
  --repeats 8 \
  --scenarios processor_only_batched,vision_only_preprocessed,connector_only_preprocessed,text_only_preprocessed,inputs_merger_only_preprocessed,text_model_only_preprocessed,model_only_preprocessed \
  --output-dir benchmark_reports/component_breakdown
```

Interpretation:

- `processor_only_batched`: preprocessing cost.
- `connector_only_preprocessed`: projection-only cost after vision hidden states are produced.
- `inputs_merger_only_preprocessed`: merge cost only (without text transformer forward).
- `text_model_only_preprocessed`: text transformer forward over pre-merged embeddings.
- `text_only_preprocessed`: merge + text transformer combined.

### Consolidated findings from shared runs so far

Based on the timing outputs shared so far, the bottleneck pattern is now consistent across setups:

1. **LM/text is not the dominant cost in your workload**.
2. **Vision path + preprocessing dominate end-to-end latency**.
3. **Naive CPU threading of `process_images` has little to no impact**.
4. **Splitting vision on GPU and text on CPU is a regression** (transfer + CPU compute overhead).
5. **Pure SigLIP image embedding is fast in isolation**, which confirms overhead is mostly in multimodal/document pipeline expansion rather than raw SigLIP kernel speed.

#### L4 ModernVBERT breakdown (uniform mode, representative values from shared logs)

- `batched_end_to_end` ≈ **12.3–12.9s** for 16 docs (about **1.24–1.30 docs/s**).
- `processor_only_batched` ≈ **5.3–5.6s**.
- `model_only_preprocessed` ≈ **6.9–7.5s**.
- `vision_only_preprocessed` ≈ **6.2–6.7s**.
- `text_only_preprocessed` ≈ **0.69–0.75s**.
- `processor_only_batched_threaded` ≈ `processor_only_batched` (negligible delta).
- `split_vision_gpu_text_cpu_preprocessed` ≈ **16.2–19.2s** (worse than `model_only_preprocessed`).

Interpretation from those values:

- The model-side split is roughly **vision-dominated** (vision path is most of model-only time).
- Within the text side, the total text path is relatively small compared to vision and preprocessing.
- End-to-end optimizations should focus on **preprocessing + vision-token budget + batching quality**, not CPU-offloading the text model.

#### SigLIP-only benchmark (shared run)

For `google/siglip2-base-patch16-512` at 64 images, size 1024:

- Sequential (`batch=1`): **~2.03s**, **~31.6 images/s**.
- Batched (`batch=4`): **~1.92s**, **~33.3 images/s**.
- Batched (`batch=8`): **~2.00s**, **~32.0 images/s**.
- Batched (`batch=16`): **~2.05s**, **~31.2 images/s**.

This confirms raw SigLIP throughput on L4 is healthy; the larger ModernVBERT latency is caused by additional pipeline work (document preprocessing/splitting, connector + multimodal merge, and full multimodal forward behavior), not by an unexpectedly slow standalone SigLIP encoder.

#### Updated optimization priority (from evidence so far)

1. Keep building **component-level decomposition** (`connector_only`, `inputs_merger_only`, `text_model_only`) to verify your `inputs_merger` hypothesis quantitatively per batch/mode.
2. Reduce preprocessing burden with **process-based** parallel data loading/prefetch (threads alone are not enough).
3. Reduce vision token pressure by tuning resize/splitting/token budget and batching by similar token budgets.
4. Maintain text on GPU in normal path (CPU split has been consistently slower in your measurements).
