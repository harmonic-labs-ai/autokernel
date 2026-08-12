# AutoKernel -- KernelBench Mode

You are an autonomous GPU kernel optimization agent competing on the KernelBench benchmark.
KernelBench provides 250+ PyTorch operations as `Model` classes. Your job: produce a `ModelNew`
class that is both **correct** (matches reference within atol=1e-2) and **fast** (speedup > 1.0x).

The key metric is **fast_p**: the fraction of problems where your solution is correct AND achieves
speedup >= p. Higher is better. The standard thresholds are 1.0x, 1.5x, 2.0x, 3.0x.

**Unlike one-shot LLM generation, you run 50-300 iterative experiments per problem.**
This is AutoKernel's advantage: systematic exploration beats guessing.

---

## Workflow Overview

```
bridge.py setup → kernel.py (ModelNew) → bench_kb.py → keep/revert → repeat
```

| Phase | What happens |
|-------|-------------|
| **Setup** | Load a KernelBench problem, generate starter kernel.py |
| **Optimize** | Edit ModelNew, benchmark, keep or revert -- iterative loop |
| **Score** | Run scorer.py for fast_p across multiple problems |

---

## Phase 1: Setup

### 1.1 Fetch problems

```bash
# Fetch all Level 1 problems from HuggingFace
uv run kernelbench/bridge.py fetch --source hf --level 1

# Or from a local KernelBench repo clone
uv run kernelbench/bridge.py fetch --source local --repo-path /path/to/KernelBench --level 1
```

### 1.2 List available problems

```bash
uv run kernelbench/bridge.py list --level 1
```

### 1.3 Set up a specific problem

```bash
uv run kernelbench/bridge.py setup --level 1 --problem 1 --source hf
```

This creates:
- `workspace/kb_active/reference.py` -- the original `Model` class (do not modify)
- `workspace/kb_active/metadata.json` -- problem analysis (operations, difficulty)
- `kernel.py` -- starter `ModelNew` (edit this)

### 1.4 Read the problem

Read kernel.py carefully. Understand:
- What `Model.forward()` does
- What the input shapes are (check `get_inputs()`)
- What operations are involved (check `metadata.json` analysis)
- Whether the model has learnable parameters

---

## Phase 2: Optimization Loop

**LOOP FOREVER. NEVER STOP. NEVER ASK THE HUMAN.**

### 2.1 Run baseline

```bash
uv run kernelbench/bench_kb.py > run.log 2>&1
```

Parse results:
```bash
grep "correctness\|speedup\|kernel_time_ms\|reference_time_ms\|fast_" run.log
```

### 2.2 Hypothesize

Think about what to try:
- Is this a compute-bound or memory-bound operation?
- Can I fuse multiple operations?
- Can I use a custom Triton kernel for the hot path?
- What precision should I use?

### 2.3 Edit kernel.py

Modify `ModelNew.forward()`. Common strategies:

**Strategy A: Custom Triton kernel**
```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        BLOCK_SIZE = 1024
        my_kernel[(triton.cdiv(n, BLOCK_SIZE),)](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
        return out
```

**Strategy B: Triton kernel**
```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, o_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(o_ptr + offs, x, mask=mask)

class ModelNew(nn.Module):
    def forward(self, x):
        out = torch.empty_like(x)
        N = x.numel()
        my_kernel[(N + 255) // 256](x, out, N, BLOCK=256)
        return out
```

**Strategy C: PyTorch optimization (no custom kernel)**
```python
class ModelNew(nn.Module):
    def forward(self, x):
        # Use torch.compile, memory format, or algorithmic improvements
        return torch._C._nn.gelu(x)  # faster internal path
```

### 2.4 Commit

```bash
git add kernel.py && git commit -m "kb exp N: <hypothesis>"
```

### 2.5 Run

```bash
uv run kernelbench/bench_kb.py > run.log 2>&1
```

### 2.6 Parse results

```bash
grep "correctness\|speedup\|kernel_time_ms\|fast_" run.log
```

### 2.7 Keep or revert

| Condition | Action |
|-----------|--------|
| correctness: FAIL | **REVERT**: `git reset --hard HEAD~1` |
| correctness: PASS, speedup improved | **KEEP** |
| correctness: PASS, speedup same or worse | **REVERT**: `git reset --hard HEAD~1` |

### 2.8 Repeat

Go back to 2.2. Each iteration should be one focused change.

---

## Optimization Playbook by Problem Level

### Level 1: Single Operations (easy-medium)

These are standalone ops: matmul, relu, conv2d, softmax, layernorm, etc.

**Strategy**: Replace the PyTorch op with a hand-written Triton kernel.

- **Elementwise ops** (relu, gelu, silu, sigmoid, tanh): Trivially parallelizable.
  One program per tile, `tl.load`/`tl.store` with a bounds mask.
  ```python
  @triton.jit
  def relu_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
      offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask = offs < N
      x = tl.load(in_ptr + offs, mask=mask, other=0.0)
      tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)
  ```

- **Reductions** (sum, mean, max, min): One program per output row, `tl.sum`/
  `tl.max` over a `BLOCK_SIZE` tile. If a row is wider than roughly 8192
  elements it will not fit in registers -- loop over the row in chunks with a
  `BLOCK_SIZE`-wide accumulator and reduce once at the end. See
  `ak_kernels/reduce.py` and the chunked variants in `ak_kernels/softmax.py`.

- **Matmul**: Tiled `tl.dot` with fp32 accumulators. See `ak_kernels/matmul.py`.
  Pass `allow_tf32=False` when the inputs are fp32 and the reference is not
  itself running TF32, or the comparison will fail on precision.

- **Convolutions**: Use `torch.nn.functional.conv2d` with optimal memory format
  (`torch.channels_last`), or write a custom im2col + GEMM kernel.

- **Normalization** (layernorm, batchnorm, rmsnorm): Welford's algorithm for
  single-pass stats, warp shuffle reductions, fused scale+bias epilogue.

### Level 2: Fusion Patterns (medium-hard)

These are 3-6 operations chained together: conv+bn+relu, linear+gelu+linear, etc.

**Strategy**: Fuse operations to eliminate intermediate memory traffic.

- Identify the operation chain in `Model.forward()`
- Write a single Triton kernel that does all operations without writing intermediates
- Keep the fused epilogue in fp32 registers; only cast on the final store
- Focus on tiling for the compute-heavy parts

### Level 3: Full Architectures (hard)

Complete models: MobileNet, VGG blocks, transformer layers.

**Strategy**: Identify the top bottleneck operation and optimize just that.

- Profile to find the hot path (usually matmul or attention)
- Replace just that one operation with a custom kernel
- Keep everything else as PyTorch for safety
- Fuse where possible (e.g., QKV projection + attention)

### Level 4: HuggingFace Models (very hard)

Pre-trained medium-sized models.

**Strategy**: Selective optimization.

- Use `torch.compile` as a baseline
- Replace specific nn.Module subclasses with optimized versions
- Focus on the operations that take the most time

---

## Triton Tips for KernelBench

### Data type handling

KernelBench problems use float32 by default. Load into fp32 accumulators for
anything involving a reduction, and cast back on store:

```python
x = tl.load(in_ptr + offs, mask=mask, other=0.0).to(tl.float32)
acc = tl.sum(x, axis=0)
tl.store(out_ptr + row, acc)   # store casts to the output tensor's dtype
```

For matmuls on fp32 inputs, `tl.dot` defaults to TF32 (10-bit mantissa). That is
usually fine for KernelBench's 1e-2 tolerance, but pass `allow_tf32=False` if a
tighter comparison fails.

### Block sizes

`BLOCK_SIZE` must be a power of two. `triton.next_power_of_2(n)` rounds up, but
do not feed it an unbounded row width: a 50257-element row rounds to 65536,
which will not fit in registers and fails to compile. Cap it and loop instead.

Sweep `num_warps` (2, 4, 8) and `num_stages` (2-4) as launch arguments -- they
often matter more than the tile shape.

### Common failure modes

- Forgetting `.contiguous()` before a `view()` on a transposed or sliced tensor.
- Masking loads but not stores (or vice versa) on ragged tails.
- Reading a masked-out lane's value: give `tl.load` an `other=` that is neutral
  for the reduction (`0.0` for sums, `-inf` for maxima).

---

## Decision Framework

### When to use Triton vs PyTorch

| Situation | Best approach |
|-----------|---------------|
| Simple elementwise op | Triton (trivial to write, removes a kernel launch) |
| Reduction | Triton (control the tiling and accumulator precision) |
| Matmul-like | Triton `tl.dot` (tensor cores via the compiler) |
| Conv2d | PyTorch with channels_last format (already optimized) |
| Multi-op fusion | Triton (single kernel, no intermediate memory) |
| Complex architecture | Selective: Triton for the hotspot, PyTorch for the rest |

### When to move on to the next problem

1. Speedup > 2.0x and stable across runs
2. 15+ experiments with no improvement (plateau)
3. The operation is already near hardware limits
4. Diminishing returns (spending too long on a small problem)

---

## Scoring

After optimizing individual problems, run the batch scorer:

```bash
# Score all Level 1 problems
uv run kernelbench/scorer.py --level 1

# Score specific problems
uv run kernelbench/scorer.py --level 1 --problems 1-20

# View aggregate results
uv run kernelbench/scorer.py --report
```

The scorer reports `fast_p` at thresholds: 1.0x, 1.1x, 1.25x, 1.5x, 2.0x, 3.0x, 5.0x.

**Target**: Achieve fast_1 > 0.80 (80% of problems correct and faster than PyTorch).
