"""
AutoKernel starter -- RMS Normalization
Basic Triton kernel. The agent improves this.

Rows up to MAX_FUSED_BLOCK elements keep the whole row in registers; wider
rows stream through a chunked kernel so the block size stays bounded.
"""

KERNEL_TYPE = "rmsnorm"

import torch
import triton
import triton.language as tl

# Largest row the single-block kernel will take.
MAX_FUSED_BLOCK = 4096


@triton.jit
def rmsnorm_kernel(
    X_ptr, W_ptr, OUT_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Row-parallel RMS normalization."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load row into float32 for numerical stability
    x = tl.load(X_ptr + row * stride_xm + offs * stride_xn, mask=mask, other=0.0).to(tl.float32)

    # Compute RMS
    sq_mean = tl.sum(x * x, axis=0) / N
    rms = tl.sqrt(sq_mean + eps)

    # Normalize
    x_norm = x / rms

    # Scale by weight
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w

    # Store (cast back to input dtype via the output tensor's dtype)
    tl.store(OUT_ptr + row * stride_om + offs * stride_on, out, mask=mask)


@triton.jit
def rmsnorm_kernel_chunked(
    X_ptr, W_ptr, OUT_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMS normalization for rows too wide to hold in registers."""
    row = tl.program_id(0)

    # Pass 1: mean of squares.
    sq_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X_ptr + row * stride_xm + offs * stride_xn,
                    mask=mask, other=0.0).to(tl.float32)
        sq_acc += x * x
    rms = tl.sqrt(tl.sum(sq_acc, axis=0) / N + eps)

    # Pass 2: normalize, scale, store.
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X_ptr + row * stride_xm + offs * stride_xn,
                    mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT_ptr + row * stride_om + offs * stride_on, x / rms * w, mask=mask)


def kernel_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.rmsnorm_ref signature."""
    assert x.is_cuda

    # Flatten to 2D so callers can pass [..., N] tensors, not just [M, N].
    orig_shape = x.shape
    x = x.contiguous()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim > 2:
        x = x.view(-1, x.shape[-1])

    M, N = x.shape
    assert weight.shape[0] == N
    weight = weight.contiguous()

    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)

    if BLOCK_SIZE <= MAX_FUSED_BLOCK:
        rmsnorm_kernel[(M,)](
            x, weight, out,
            M, N,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        rmsnorm_kernel_chunked[(M,)](
            x, weight, out,
            M, N,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            eps,
            BLOCK_SIZE=MAX_FUSED_BLOCK,
        )

    return out.view(orig_shape)
