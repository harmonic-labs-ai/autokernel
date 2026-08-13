#!/usr/bin/env python3
"""Validate that a profile report resolves to correct, model-specific shapes.

Prints the resolved shape table, then asserts the invariants that the shape
parser has actually got them right -- rather than silently falling back to
generic defaults, which is the failure this exists to catch.

Usage (7B olmOCR-2 language backbone, the default expectations):
    uv run tests/check_report.py

Usage (the tiny smoke-test model):
    uv run tests/check_report.py --preset tiny

Exits nonzero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract import (  # noqa: E402
    dedupe_kernels,
    infer_missing_fused_mlp_shapes,
    resolve_model_shape,
    shape_to_display,
)

# Architecture of the language backbone, per model. olmOCR-2 is Qwen2.5-VL-7B.
PRESETS = {
    "olmocr2": dict(hidden=3584, intermediate=18944, heads=28, kv_heads=4, vocab=152064),
    "tiny":    dict(hidden=896,  intermediate=4864,  heads=14, kv_heads=2, vocab=151936),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", default="workspace/profile_report.json")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="olmocr2")
    ap.add_argument("--seq", type=int, default=2048,
                    help="Sequence length the profile was run at.")
    args = ap.parse_args()

    arch = PRESETS[args.preset]
    hidden, inter = arch["hidden"], arch["intermediate"]
    heads, kv_heads, vocab = arch["heads"], arch["kv_heads"], arch["vocab"]
    head_dim = hidden // heads
    kv_dim = kv_heads * head_dim

    # The contraction dim of every GEMM in a Qwen2 block is one of these, and
    # the output dim is one of these. Anything else means the operand pair was
    # read wrong -- or that a generic default leaked through.
    valid_k = {hidden, inter}
    valid_n = {hidden, inter, vocab, kv_dim}

    with open(args.report) as f:
        report = json.load(f)
    rows = report["top_kernels"]
    sup = [r for r in rows if r.get("autokernel_supported")]

    resolved = []
    for r in sup:
        shape, source = resolve_model_shape(r)
        resolved.append(dict(
            rank=r.get("rank"), name=r.get("name", ""), op_type=r.get("op_type"),
            src=r.get("source", "?"), pct_total=r.get("pct_total", 0.0),
            gpu_time_ms=r.get("gpu_time_ms", 0.0),
            model_shape=shape, shape_source=source,
        ))
    infer_missing_fused_mlp_shapes(resolved)
    kept, dropped = dedupe_kernels(resolved)

    print(f"{len(rows)} rows, {len(sup)} supported -> {len(kept)} distinct "
          f"({len(dropped)} merged)\n")
    print(f"{'rk':<5}{'op_type':<18}{'src':<14}{'from':<10}{'shape':<46}name")
    print("-" * 120)
    for e in kept:
        print(f"{e['rank']:<5}{e['op_type']:<18}{e['src']:<14}{e['shape_source']:<10}"
              f"{shape_to_display(e['model_shape']):<46}{e['name'][:40]}")

    failures = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    # 1. Nothing may fall back to a generic default.
    for e in kept:
        check(e["shape_source"] in ("profiled", "inferred"),
              f"rank {e['rank']} ({e['op_type']}) fell back to a "
              f"'{e['shape_source']}' shape: {shape_to_display(e['model_shape'])}")

    # 2. Device-kernel rows carry no shapes and duplicate an aten row, so they
    #    must never survive as targets. ('?' means a pre-fix report.)
    stale = any(e["src"] == "?" for e in kept)
    if stale:
        print("\nNOTE: report has no 'source' field -- it predates the profiler fix, "
              "so the device-kernel exclusion check is skipped.")
    else:
        for e in kept:
            check(e["src"] == "aten_op",
                  f"rank {e['rank']} is a {e['src']} row but survived as a target")

    # 3. GEMM dims must belong to the architecture.
    for e in kept:
        if e["op_type"] != "matmul":
            continue
        s = e["model_shape"]
        check(s.get("K") in valid_k,
              f"rank {e['rank']} matmul K={s.get('K')} not in {sorted(valid_k)}")
        check(s.get("N") in valid_n,
              f"rank {e['rank']} matmul N={s.get('N')} not in {sorted(valid_n)}")
        check(s.get("M") == args.seq,
              f"rank {e['rank']} matmul M={s.get('M')}, expected {args.seq}")

    # 4. Attention axes must not be transposed -- this is the [B,S,H,D] vs
    #    [B,H,S,D] bug, which shows up as heads and seq_len swapped.
    for e in kept:
        if e["op_type"] != "flash_attention":
            continue
        s = e["model_shape"]
        check(s.get("heads") in (heads, kv_heads),
              f"rank {e['rank']} attention heads={s.get('heads')}, "
              f"expected {heads} or {kv_heads} -- axis order likely misread")
        check(s.get("seq_len") == args.seq,
              f"rank {e['rank']} attention seq_len={s.get('seq_len')}, "
              f"expected {args.seq} -- axis order likely misread")
        check(s.get("head_dim") == head_dim,
              f"rank {e['rank']} attention head_dim={s.get('head_dim')}, "
              f"expected {head_dim}")

    # 5. A recovered fused_mlp must match the real MLP geometry.
    for e in kept:
        if e["op_type"] != "fused_mlp":
            continue
        s = e["model_shape"]
        check(s.get("dim") == hidden and s.get("hidden") == inter,
              f"rank {e['rank']} fused_mlp dim/hidden = "
              f"{s.get('dim')}/{s.get('hidden')}, expected {hidden}/{inter}")

    # 6. Dedup must leave no two targets on the same shape.
    seen = {}
    for e in kept:
        key = (e["op_type"], tuple(sorted(e["model_shape"].items())))
        check(key not in seen,
              f"ranks {seen.get(key)} and {e['rank']} share a shape after dedup")
        seen[key] = e["rank"]

    # 7. RMSNorm and RoPE reach the plan at all. HF writes both as a chain of
    #    primitives, so a name-only classifier drops them -- and they are the
    #    largest memory-bound targets in a Qwen2 tower.
    by_type = {e["op_type"] for e in kept}
    check("rmsnorm" in by_type,
          "no rmsnorm target -- module-scope attribution did not fire")
    # HF LLaMA/Qwen2 rotate half the head dim; a kernel graded against the
    # interleaved oracle would compute a different rotation, so the plan must
    # name the split-half variant specifically.
    check("rotary_embedding_half" in by_type,
          "no rotary_embedding_half target -- the RoPE detector did not fire")
    check("rotary_embedding" not in by_type,
          "RoPE was tagged as the interleaved convention, but this model "
          "rotates half the head dim")

    # 8. ...and normalise over the hidden dim, rotating the real head geometry.
    for e in kept:
        s = e["model_shape"]
        if e["op_type"] == "rmsnorm":
            check(s.get("N") == hidden,
                  f"rank {e['rank']} rmsnorm N={s.get('N')}, expected {hidden}")
            check(s.get("M") == args.seq,
                  f"rank {e['rank']} rmsnorm M={s.get('M')}, expected {args.seq}")
        elif e["op_type"].startswith("rotary_embedding"):
            check(s.get("heads") in (heads, kv_heads),
                  f"rank {e['rank']} rope heads={s.get('heads')}, "
                  f"expected {heads} or {kv_heads}")
            check(s.get("seq_len") == args.seq,
                  f"rank {e['rank']} rope seq_len={s.get('seq_len')}, "
                  f"expected {args.seq}")
            check(s.get("head_dim") == head_dim,
                  f"rank {e['rank']} rope head_dim={s.get('head_dim')}, "
                  f"expected {head_dim}")

    # 9. The aten rows and the device kernels they launched are two independent
    #    measures of the same physical work, so their totals must agree. They
    #    diverge when device kernels leak into the aten population (inflating
    #    the total and creating phantom targets), or when folding a decomposed
    #    op into a composite double-counts or drops its members.
    aten_ms = sum(r["gpu_time_ms"] for r in rows if r.get("source") == "aten_op")
    dev_ms = sum(r["gpu_time_ms"] for r in rows if r.get("source") == "device_kernel")
    if dev_ms > 0:
        skew = abs(aten_ms - dev_ms) / dev_ms
        check(skew < 0.05,
              f"aten total {aten_ms:.1f} ms vs device total {dev_ms:.1f} ms "
              f"({skew:.1%} apart) -- work is being counted twice or lost")
        print(f"\naten {aten_ms:.2f} ms vs device {dev_ms:.2f} ms "
              f"({skew:.2%} apart) -- same work, counted once")

    # 10. Every composite must equal the sum of the parts it absorbed.
    for r in rows:
        parts = r.get("composed_of")
        if not parts:
            continue
        want = sum(p["gpu_time_ms"] for p in parts)
        got = r["gpu_time_ms"]
        check(abs(got - want) <= 0.01 * max(want, 1e-9),
              f"rank {r['rank']} {r['op_type']} is {got:.3f} ms but its "
              f"{len(parts)} parts sum to {want:.3f} ms")

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f_ in failures:
            print(f"  - {f_}")
        return 1

    print(f"\nPASS -- {len(kept)} targets, all shapes model-specific.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
