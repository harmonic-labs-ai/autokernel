"""
AutoKernel -- Layer Normalization kernel.

Current kernel: LayerNorm (row-parallel)
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Each program instance handles one row of the input tensor.
Two-pass approach: compute mean, then variance, then normalize.

Rows up to MAX_FUSED_BLOCK elements keep the whole row in registers. Wider
rows stream through a chunked kernel: an 8192-element fp32 row needs 32KB of
registers per program, which spills catastrophically.
"""

KERNEL_TYPE = "layernorm"

import torch
import triton
import triton.language as tl

# Largest row the single-block kernel will take.
MAX_FUSED_BLOCK = 4096


@triton.jit
def layernorm_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel layer normalization. One program per row."""
    row_idx = tl.program_id(0)

    row_start_x = X_ptr + row_idx * stride_x_row
    row_start_y = Y_ptr + row_idx * stride_y_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load row into float32 for numerical stability
    x = tl.load(row_start_x + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Pass 1: compute mean
    mean = tl.sum(x, axis=0) / N

    # Pass 2: compute variance
    x_centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(x_centered * x_centered, axis=0) / N

    # Normalize
    inv_std = 1.0 / tl.sqrt(variance + eps)
    x_norm = x_centered * inv_std

    # Load weight and bias
    w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Apply affine transform
    y = x_norm * w + b

    # Store (cast back to input dtype via the store)
    tl.store(row_start_y + col_offsets, y, mask=mask)


@triton.jit
def layernorm_kernel_chunked(
    X_ptr,
    Y_ptr,
    W_ptr,
    B_ptr,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Layer normalization for rows too wide to hold in registers.

    Streams the row three times: sum, sum-of-squares, then normalize-and-store.
    Variance comes from E[x^2] - E[x]^2 computed in fp32.
    """
    row_idx = tl.program_id(0)

    row_start_x = X_ptr + row_idx * stride_x_row
    row_start_y = Y_ptr + row_idx * stride_y_row

    # Pass 1: mean.
    sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(row_start_x + col_offsets, mask=mask, other=0.0).to(tl.float32)
        sum_acc += x
    mean = tl.sum(sum_acc, axis=0) / N

    # Pass 2: variance around that mean.
    var_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(row_start_x + col_offsets, mask=mask, other=0.0).to(tl.float32)
        centered = tl.where(mask, x - mean, 0.0)
        var_acc += centered * centered
    variance = tl.sum(var_acc, axis=0) / N
    inv_std = 1.0 / tl.sqrt(variance + eps)

    # Pass 3: normalize, apply affine, store.
    for start in range(0, N, BLOCK_SIZE):
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(row_start_x + col_offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std * w + b
        tl.store(row_start_y + col_offsets, y, mask=mask)


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.layernorm_ref signature."""
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
    assert weight.shape[0] == n_cols
    assert bias.shape[0] == n_cols

    weight = weight.contiguous()
    bias = bias.contiguous()

    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)
    if BLOCK_SIZE <= MAX_FUSED_BLOCK:
        layernorm_kernel[grid](
            x, y,
            weight, bias,
            x.stride(0),
            y.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        layernorm_kernel_chunked[grid](
            x, y,
            weight, bias,
            x.stride(0),
            y.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=MAX_FUSED_BLOCK,
        )

    return y.view(orig_shape)
