"""
AutoKernel -- Fused Cross Entropy kernel.

Current kernel: Fused log-softmax + NLL loss (row-parallel)
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Fuses log-softmax and negative log-likelihood loss into a single kernel pass
per row, avoiding materializing the full softmax output in global memory.
Each program handles one row (one sample in the batch).

Rows up to MAX_FUSED_BLOCK elements are handled by a single-block kernel. Real
vocabularies are larger than that (50257 logits would need a 65536-wide block,
which does not fit in registers), so longer rows stream through a chunked
kernel instead.
"""

KERNEL_TYPE = "cross_entropy"

import torch
import triton
import triton.language as tl

# Matches F.cross_entropy's default: targets equal to this contribute no loss
# and are excluded from the mean.
IGNORE_INDEX = -100

# Largest row the single-block kernel will take.
MAX_FUSED_BLOCK = 8192


@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    losses_ptr,
    weights_ptr,
    n_cols,
    stride_logits_row,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused cross-entropy: log_softmax + nll_loss per row.
    One program per row (batch element).
    """
    row_idx = tl.program_id(0)

    target = tl.load(targets_ptr + row_idx)
    if target == ignore_index:
        # Ignored rows contribute nothing and must not index the logits: a
        # negative sentinel would read out of bounds.
        tl.store(losses_ptr + row_idx, 0.0)
        tl.store(weights_ptr + row_idx, 0.0)
    else:
        row_start = logits_ptr + row_idx * stride_logits_row
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load logits row in float32
        logits = tl.load(row_start + col_offsets, mask=mask, other=float("-inf")).to(tl.float32)

        # Numerically stable log-softmax
        row_max = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - row_max)
        sum_exp = tl.sum(exp_logits, axis=0)
        log_sum_exp = tl.log(sum_exp)

        # We only need log_softmax at the target index:
        #   log_softmax[t] = logits[t] - max - log(sum(exp(logits - max)))
        target_logit = tl.load(row_start + target).to(tl.float32)
        log_softmax_target = (target_logit - row_max) - log_sum_exp

        # NLL loss = -log_softmax[target]
        tl.store(losses_ptr + row_idx, -log_softmax_target)
        tl.store(weights_ptr + row_idx, 1.0)


@triton.jit
def cross_entropy_kernel_chunked(
    logits_ptr,
    targets_ptr,
    losses_ptr,
    weights_ptr,
    n_cols,
    stride_logits_row,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    """Cross-entropy for vocabularies too large to hold in registers.

    Two streaming passes over the row (max, then sum-of-exp) with accumulators
    reduced once at the end.
    """
    row_idx = tl.program_id(0)

    target = tl.load(targets_ptr + row_idx)
    if target == ignore_index:
        tl.store(losses_ptr + row_idx, 0.0)
        tl.store(weights_ptr + row_idx, 0.0)
    else:
        row_start = logits_ptr + row_idx * stride_logits_row

        # Pass 1: row max.
        max_acc = tl.full((BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
        for start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            chunk = tl.load(row_start + col_offsets, mask=mask,
                            other=float("-inf")).to(tl.float32)
            max_acc = tl.maximum(max_acc, chunk)
        row_max = tl.max(max_acc, axis=0)

        # Pass 2: sum of exp(x - max).
        sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            chunk = tl.load(row_start + col_offsets, mask=mask,
                            other=float("-inf")).to(tl.float32)
            sum_acc += tl.where(mask, tl.exp(chunk - row_max), 0.0)
        log_sum_exp = tl.log(tl.sum(sum_acc, axis=0))

        target_logit = tl.load(row_start + target).to(tl.float32)
        log_softmax_target = (target_logit - row_max) - log_sum_exp

        tl.store(losses_ptr + row_idx, -log_softmax_target)
        tl.store(weights_ptr + row_idx, 1.0)


def kernel_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Entry point called by bench.py. Must match reference.cross_entropy_ref signature.

    Args:
        logits: [batch_size, vocab_size] raw logits (float16 or float32)
        targets: [batch_size] integer class indices (long)

    Returns:
        Scalar mean cross-entropy loss
    """
    assert logits.is_cuda and targets.is_cuda

    # Handle multi-dim: flatten to 2D. reshape() rather than view() so a
    # non-contiguous input does not raise.
    if logits.ndim > 2:
        logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)

    n_rows, n_cols = logits.shape
    assert targets.shape[0] == n_rows

    losses = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
    # Per-row weight: 0 for ignored rows, 1 otherwise. F.cross_entropy divides
    # by the number of contributing rows, not the batch size.
    weights = torch.empty(n_rows, device=logits.device, dtype=torch.float32)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)
    if BLOCK_SIZE <= MAX_FUSED_BLOCK:
        cross_entropy_kernel[grid](
            logits,
            targets,
            losses,
            weights,
            n_cols,
            logits.stride(0),
            IGNORE_INDEX,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        cross_entropy_kernel_chunked[grid](
            logits,
            targets,
            losses,
            weights,
            n_cols,
            logits.stride(0),
            IGNORE_INDEX,
            BLOCK_SIZE=MAX_FUSED_BLOCK,
        )

    total_weight = weights.sum()
    # All rows ignored: F.cross_entropy returns nan for this.
    return losses.sum() / total_weight
