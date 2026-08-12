# Changelog

## Unreleased

### Removed the CUDA C++ backend

Triton is now the only backend. The `ak_kernels/cuda/` starters, the
`load_inline()` compilation utility, the `--backend cuda` flag, and the CUDA
export path in `export_hf.py` are gone, along with the `cuda` optional
dependency and the CUDA C++ playbooks in `program.md` / `kernelbench/program_kb.md`.

The pipeline is still parameterized by backend rather than hardcoded to Triton:
`extract.py` carries a `BACKENDS` registry and `export_hf.py` an `EXPORTERS`
map, so a backend can be added back by registering it and dropping starter
kernels into a subdirectory of `ak_kernels/`. Kernels only declare
`BACKEND = "..."` when they are not the default, so existing Triton kernels
need no change.

### Fixed

- `verify.py` crashed with a `TypeError` whenever any kernel in
  `orchestration_state.json` still had a null `speedup` -- which is its initial
  value -- taking down end-to-end verification.
- `_LinearWrapper` re-materialized `weight.t().contiguous()` on every forward,
  inside the timed region, biasing the end-to-end speedup against the optimized
  model. The transpose is now done once at wrap time.
- `softmax`, `cross_entropy`, `layernorm` and `rmsnorm` sized their Triton block
  as `next_power_of_2(row_width)`, so any row wider than a few thousand elements
  (a 50257-entry vocabulary rounds to 65536) could not compile. Each now
  dispatches to a chunked variant above `MAX_FUSED_BLOCK`.
- The roofline used the fp16 peak for every dtype, understating fp32 kernels'
  share of peak and misclassifying them as memory-bound. It now uses the peak
  for the dtype under test.
- `bench.py` exited 0 even when every correctness stage failed.
- `orchestrate.py` set a kernel's baseline from the first *kept* experiment,
  which reports the improved throughput -- pinning the reported speedup at 1.0x
  and losing the true starting point.
- A second replacement of the same module type in `verify.py` walked the
  already-patched tree and double-wrapped each module, while overwriting the
  saved original so `__exit__` restored a wrapper.
- `_LayerNormWrapper`/`_RMSNormWrapper` caught `TypeError` to probe call
  signatures, silently swallowing `TypeError`s raised inside the kernel itself.
  Signatures are now selected by inspection.
- The starter `matmul` and `fused_mlp` kernels left `tl.dot` at its default
  `allow_tf32=True`, running fp32 inputs at TF32 precision against a true-fp32
  reference and failing the fp32 tolerance.
- `fused_mlp`'s `gelu` path used the tanh approximation while the reference uses
  `F.gelu`'s exact erf form.
- `cross_entropy` ignored `ignore_index=-100`, reading out of bounds on ignored
  rows instead of excluding them from the mean as `F.cross_entropy` does.
- `reduce` returned shape `[1]` for a 1-D input where `x.sum(dim=-1)` returns a
  0-dim scalar.
- `extract.py`'s `scale_shape` scaled every dimension, including `head_dim` and
  `vocab`, generating `model_double` shapes the kernels legitimately reject.
  Only genuine workload dimensions are swept now.
- Extracted kernels now emit `EDGE_SIZES` derived from the model shape, instead
  of falling back to bench.py's generic built-in edge sizes.
- `_compare` accepted a kernel whose output dtype differed from the reference's,
  because both sides were cast to fp32 first.
- `_do_bench` passed `warmup`/`rep` to `triton.testing.do_bench`, which reads
  them as milliseconds, while the fallback path treated them as iteration counts
  and took a median where do_bench takes a mean.
- Several GPU entries (3090, 3080, A10, L4, 4090, 4080) carried sparsity-enabled
  peak TFLOPS, halving the reported `pct_peak_compute`.
- Row-wise kernels called `view()` without `contiguous()`, raising on
  non-contiguous inputs that `verify.py`'s wrappers can produce.
- `rotary_embedding` silently tiled `cos`/`sin` against layouts they do not
  align with; the broadcast requirement is now asserted.
- The timeout helper's Windows fallback raised `KeyboardInterrupt`, a
  `BaseException` that slipped past every `except Exception` handler and aborted
  the run instead of recording a stage failure. The harness is Unix-only now.

---

## v1.3.0 -- 2026-03-13

### AMD ROCm GPU Support (PR #3 by @andyluo7)

- Added AMD Instinct MI300X, MI325X, MI350X, MI355X to GPU database with correct peak FP16 TFLOPS, memory bandwidth, and L2 cache specs
- Added `gcnArchName`-based GPU detection for ROCm (device name is often empty on ROCm; `gcnArchName` like `gfx942` is always available)
- Guarded `clock_rate` access behind `hasattr` + `> 0` check (ROCm devices report `clock_rate=0`)
- Applied same fixes to `profile.py` fallback detector
- Tested on AMD Instinct MI300X (gfx942, ROCm 6.3) and MI350X (gfx950, ROCm 7.2)

### Bug Fixes

- **Fixed `verify.py` SyntaxError on Python 3.13+**: moved `global` declaration before variable usage in `main()` -- file would not even import on Python 3.13/3.14
- **Fixed CUDA flash_attention `sm_scale` parameter being ignored**: the `sm_scale` argument was accepted but the kernel hardcoded `rsqrtf(D)` instead of using it -- now correctly passes `sm_scale` through to the CUDA kernel
- **Fixed CUDA cross_entropy returning wrong dtype**: loss was cast back to input dtype instead of always returning `float32` (matching `F.cross_entropy` behavior)
- **Fixed Triton rotary_embedding broadcasting truncation**: `cos`/`sin` repeat used integer division which truncated when `n_rows` was not a multiple of `cos.shape[0]` -- now uses ceiling division and slices to exact size
- **Fixed Triton reduce output shape for non-last-dim reductions**: after permuting to move the reduce dim last, the output was reshaped using the original dim order instead of the permuted order

## v1.2.0 -- 2026-03-12

### Enhanced Profiler (Issue #1)

- Added `--export-trace` flag to export Chrome trace JSON for HTA/trace-blame analysis
- Added `--memory-snapshot` flag to capture CUDA memory snapshots for mosaic analysis
- Added `--torch-compile-log` flag to save torch.compile logs for tlparse analysis
- Added optional HTA (Holistic Trace Analysis) integration -- when installed, runs temporal and kernel breakdown analysis
- Added exported artifacts summary with suggested next steps for each tool
- Added `HolisticTraceAnalysis` as optional `profiling` dependency

### HuggingFace Kernels Export (Issue #2)

- Added `export_hf.py` -- exports optimized kernels to HuggingFace Kernels format
- Supports CUDA C++ kernels: auto-extracts CUDA source, parses function signatures, generates `build.toml` + `torch_binding.cpp` + `__init__.py`
- Supports Triton kernels: packages as a Python module with pyproject.toml
- Generates ready-to-upload project structure compatible with `kernels upload` CLI
- Added `kernels` and `huggingface-hub` as optional `hf-kernels` dependencies

## v1.1.0 -- 2026-03-12

### Native CUDA C++ Backend

- Added 9 CUDA C++ starter kernels with advanced GPU features:
  - **matmul** -- Tensor core GEMM via `wmma` API, 128x128 tiles, double-buffered shared memory
  - **softmax** -- Warp shuffle reductions, `half2` vectorized loads, grid-stride loop
  - **layernorm** -- Welford's single-pass algorithm, `float4` vectorized loads, warp shuffle stats
  - **rmsnorm** -- Warp shuffle cascade, `rsqrtf` fast inverse sqrt, `half2` vectorization
  - **flash_attention** -- Tiled online softmax, double-buffered shared memory, causal mask support
  - **fused_mlp** -- Fused SwiGLU (gate + up + SiLU + mul), shared memory tiling
  - **cross_entropy** -- Fused online log-sum-exp + NLL in single pass, warp reductions
  - **rotary_embedding** -- `__sincosf` intrinsic, `half2` read-modify-write
  - **reduce** -- Hierarchical warp shuffle + shared memory + grid-level atomic
- Added `kernels/cuda/_compile.py` -- shared compilation utility:
  - Hash-based caching (recompile only when source changes)
  - GPU architecture auto-detection via `torch.cuda.get_device_capability()`
  - Forward declaration extraction for cross-translation-unit linking
  - Thread-safe compilation with file locking
  - Detailed error diagnostics with source line numbers
- Added `--backend triton|cuda` flag to `extract.py`
- Added CUDA C++ optimization playbook to `program.md`
- Added `ninja` as optional dependency for faster compilation

### KernelBench Integration

- Added `kernelbench/bridge.py` -- problem loader supporting 3 sources:
  - HuggingFace datasets (`--source hf`)
  - Local KernelBench repo clone (`--source local`)
  - Individual Python files (`--source file`)
  - Automatic problem analysis (50+ operation patterns)
  - Starter `ModelNew` generation with CUDA/Triton templates
- Added `kernelbench/bench_kb.py` -- 4-stage evaluation pipeline:
  - Stage 1: Correctness (5 random input trials, atol/rtol=1e-2)
  - Stage 2: Stability (NaN/Inf detection)
  - Stage 3: Determinism (3 identical runs)
  - Stage 4: Performance (CUDA event timing, trimmed median)
  - Greppable output with `fast_p` at 7 thresholds
- Added `kernelbench/scorer.py` -- batch evaluation and metrics:
  - `fast_p` metric at thresholds: 1.0x, 1.1x, 1.25x, 1.5x, 2.0x, 3.0x, 5.0x
  - Incremental scoring with JSON persistence
  - Leaderboard-style reports with progress bars
- Added `kernelbench/program_kb.md` -- agent instructions for KernelBench mode:
  - Optimization playbook per difficulty level (L1-L4)
  - CUDA C++ and Triton strategy examples
  - Decision framework and anti-patterns

### Other

- Updated README with KernelBench section, dual-backend docs, and Discord link
- Added `datasets>=2.16.0` as optional `kernelbench` dependency

## v1.0.0 -- Initial Release

- Triton kernel optimization pipeline (profile, extract, bench, orchestrate, verify)
- 9 starter Triton kernels (matmul, softmax, layernorm, rmsnorm, flash_attention, fused_mlp, cross_entropy, rotary_embedding, reduce)
- 5-stage correctness harness + roofline analysis
- Amdahl's law orchestration for multi-kernel optimization
- Self-contained model definitions (GPT-2, LLaMA, BERT)
- TSV logging and experiment visualization
