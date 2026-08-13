"""
AutoKernel -- Rotary Position Embedding (RoPE), rotate_half convention.

Current kernel: RoPE application, split-half pairing
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

This is the convention HuggingFace LLaMA, Qwen2/2.5, Mistral and Gemma use.
The head dim is split in half and element i is rotated against i + head_dim//2:

  half = head_dim // 2
  x1 = x[..., :half]
  x2 = x[..., half:]
  out[..., :half] = x1 * cos[..., :half] - x2 * sin[..., :half]
  out[..., half:] = x2 * cos[..., half:] + x1 * sin[..., half:]

which is exactly `x * cos + rotate_half(x) * sin` written out per half.

Note the difference from ak_kernels/rotary_embedding.py, which pairs *adjacent*
elements (the GPT-J convention) and takes cos/sin of width head_dim // 2. Here
cos/sin span the full head_dim, with each angle duplicated across the halves --
HF builds them as cat([freqs, freqs], dim=-1). This kernel reads only the first
half of each row and relies on that duplication.
"""

KERNEL_TYPE = "rotary_embedding_half"

import torch
import triton
import triton.language as tl


@triton.jit
def rotary_embedding_half_kernel(
    X_ptr,
    COS_ptr,
    SIN_ptr,
    OUT_ptr,
    n_cos,
    stride_x_row,
    stride_cos_row,
    stride_sin_row,
    stride_out_row,
    half_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row; each row holds head_dim elements.

    Both halves are loaded and stored together, so the row is read once and
    written once regardless of head_dim.
    """
    row_idx = tl.program_id(0)
    # cos/sin are indexed by position and shared across heads, so they wrap.
    # Doing the wrap here means the caller never has to materialise a tiled
    # copy of them.
    cos_row = row_idx % n_cos

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < half_dim

    x_row = X_ptr + row_idx * stride_x_row
    x1 = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_row + half_dim + offs, mask=mask, other=0.0).to(tl.float32)

    # cos/sin are duplicated across the halves, so the first half suffices.
    cos = tl.load(COS_ptr + cos_row * stride_cos_row + offs, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(SIN_ptr + cos_row * stride_sin_row + offs, mask=mask, other=0.0).to(tl.float32)

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin

    out_row = OUT_ptr + row_idx * stride_out_row
    tl.store(out_row + offs, out1, mask=mask)
    tl.store(out_row + half_dim + offs, out2, mask=mask)


def kernel_fn(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Entry point called by bench.py. Matches reference.rotary_embedding_half_ref.

    Args:
        x: [..., head_dim] tensor to apply rotary embeddings to
        cos: [..., head_dim] precomputed cosines (halves duplicated)
        sin: [..., head_dim] precomputed sines (halves duplicated)

    Returns:
        Tensor of same shape as x with rotary embeddings applied.
    """
    assert x.is_cuda

    orig_shape = x.shape
    head_dim = x.shape[-1]
    half_dim = head_dim // 2

    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    assert cos.shape[-1] == head_dim, (
        f"rotate_half RoPE expects cos of width head_dim ({head_dim}), got "
        f"{cos.shape[-1]} -- the interleaved convention uses head_dim // 2, see "
        f"ak_kernels/rotary_embedding.py"
    )
    assert sin.shape[-1] == head_dim

    x_flat = x.contiguous().view(-1, head_dim)
    n_rows = x_flat.shape[0]

    # A broadcast dim of size 1 carries no rows, so drop leading singletons --
    # HF passes cos as [1, 1, seq_len, head_dim] against q of
    # [batch, heads, seq_len, head_dim], and those two 1s would otherwise look
    # like a layout mismatch.
    cos_s, sin_s = cos, sin
    while cos_s.dim() > 2 and cos_s.shape[0] == 1:
        cos_s = cos_s.squeeze(0)
    while sin_s.dim() > 2 and sin_s.shape[0] == 1:
        sin_s = sin_s.squeeze(0)

    # cos/sin are indexed by position, so they tile over the leading dims of x.
    # Only the [..., seq_len, head_dim] layout makes row r of x line up with row
    # r % n_cos of cos; assert it rather than silently rotating by wrong angles.
    cos_flat = cos_s.contiguous().view(-1, head_dim)
    sin_flat = sin_s.contiguous().view(-1, head_dim)
    assert cos_flat.shape == sin_flat.shape, "cos and sin must have the same shape"

    n_cos = cos_flat.shape[0]
    if n_cos < n_rows:
        assert n_rows % n_cos == 0, (
            f"cannot broadcast cos/sin with {n_cos} rows onto {n_rows} rows of x"
        )
        assert tuple(cos_s.shape[:-1]) == tuple(orig_shape[-1 - (cos_s.dim() - 1):-1]), (
            "cos/sin must align with the trailing dimensions of x before head_dim "
            f"(x shape {tuple(orig_shape)}, cos shape {tuple(cos.shape)})"
        )
        # The kernel wraps the row index itself, so no tiled copy is made here.
    else:
        assert n_cos == n_rows, f"cos/sin have {n_cos} rows but x has {n_rows}"

    out = torch.empty_like(x_flat)

    BLOCK_SIZE = triton.next_power_of_2(half_dim)

    rotary_embedding_half_kernel[(n_rows,)](
        x_flat,
        cos_flat,
        sin_flat,
        out,
        n_cos,
        x_flat.stride(0),
        cos_flat.stride(0),
        sin_flat.stride(0),
        out.stride(0),
        half_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.view(orig_shape)
