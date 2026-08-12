"""
AutoKernel -- Softmax kernel.

Current kernel: Online Softmax (row-parallel)
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Each program instance handles one row of the input tensor.
Uses numerically stable approach: max -> subtract -> exp -> sum -> divide.

Rows up to MAX_FUSED_BLOCK elements are handled by a single-block kernel that
keeps the whole row in registers. Longer rows (vocab-sized logits, for example)
stream through a chunked kernel instead: a row of 50257 fp32 values would need
a 65536-wide block, which does not fit in registers and fails to compile.
"""

KERNEL_TYPE = "softmax"

import torch
import triton
import triton.language as tl

# Largest row the single-block kernel will take. Above this the chunked kernel
# runs instead, trading extra passes over memory for a block that fits.
MAX_FUSED_BLOCK = 8192


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    stride_input_row,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel softmax, one program per row, whole row in registers."""
    row_idx = tl.program_id(0)

    row_start_input = input_ptr + row_idx * stride_input_row
    row_start_output = output_ptr + row_idx * stride_output_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row
    row = tl.load(row_start_input + col_offsets, mask=mask, other=float("-inf")).to(tl.float32)

    # Numerically stable softmax: subtract max
    row_max = tl.max(row, axis=0)
    row = row - row_max

    # Exponentiate
    numerator = tl.exp(row)

    # Sum
    denominator = tl.sum(numerator, axis=0)

    # Divide
    result = numerator / denominator

    # Store
    tl.store(row_start_output + col_offsets, result, mask=mask)


@triton.jit
def softmax_kernel_chunked(
    input_ptr,
    output_ptr,
    n_cols,
    stride_input_row,
    stride_output_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel softmax for rows too long to hold in registers.

    Three streaming passes -- max, sum, normalize -- with the running max and
    sum held in BLOCK_SIZE-wide accumulators and reduced once at the end.
    Values are recomputed in the final pass rather than stashed in the output
    buffer, so no precision is lost to a low-precision round trip.
    """
    row_idx = tl.program_id(0)

    row_start_input = input_ptr + row_idx * stride_input_row
    row_start_output = output_ptr + row_idx * stride_output_row

    # Pass 1: row max.
    max_acc = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        chunk = tl.load(row_start_input + col_offsets, mask=mask,
                        other=float("-inf")).to(tl.float32)
        max_acc = tl.maximum(max_acc, chunk)
    row_max = tl.max(max_acc, axis=0)

    # Pass 2: sum of exp(x - max).
    sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        chunk = tl.load(row_start_input + col_offsets, mask=mask,
                        other=float("-inf")).to(tl.float32)
        sum_acc += tl.where(mask, tl.exp(chunk - row_max), 0.0)
    row_sum = tl.sum(sum_acc, axis=0)

    # Pass 3: recompute exp(x - max) and normalize.
    for start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        chunk = tl.load(row_start_input + col_offsets, mask=mask,
                        other=float("-inf")).to(tl.float32)
        result = tl.exp(chunk - row_max) / row_sum
        tl.store(row_start_output + col_offsets, result, mask=mask)


def kernel_fn(x: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.softmax_ref signature."""
    assert x.is_cuda

    # Flatten to 2D for row-parallel processing. contiguous() first: a view of
    # a transposed or sliced tensor raises otherwise.
    orig_shape = x.shape
    x = x.contiguous()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # Block size must be a power of 2 >= n_cols for the single-block kernel;
    # the chunked kernel caps it and loops instead.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)
    if BLOCK_SIZE <= MAX_FUSED_BLOCK:
        softmax_kernel[grid](
            x, output,
            n_cols,
            x.stride(0),
            output.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        softmax_kernel_chunked[grid](
            x, output,
            n_cols,
            x.stride(0),
            output.stride(0),
            BLOCK_SIZE=MAX_FUSED_BLOCK,
        )

    return output.view(orig_shape)
