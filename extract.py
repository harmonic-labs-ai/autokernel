#!/usr/bin/env python3
"""
AutoKernel Kernel Extractor -- Generate baseline kernels from profiling results.

Usage:
    uv run extract.py                          # extract from workspace/profile_report.json
    uv run extract.py --top 5                  # extract only top-5 kernels
    uv run extract.py --kernel-type matmul     # extract only matmul kernels
    uv run extract.py --report path/to/report.json
    uv run extract.py --backend cuda           # use CUDA C++ starter kernels instead of Triton
    uv run extract.py --dtype bfloat16         # override the profiled dtype
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(SCRIPT_DIR, "workspace")
KERNELS_DIR = os.path.join(SCRIPT_DIR, "ak_kernels")
DEFAULT_REPORT_PATH = os.path.join(WORKSPACE_DIR, "profile_report.json")
OPTIMIZATION_PLAN_PATH = os.path.join(WORKSPACE_DIR, "optimization_plan.json")


# ---------------------------------------------------------------------------
# Shape key mappings per kernel type
# ---------------------------------------------------------------------------
# Each entry maps op_type -> list of (shape_key_aliases...) so we can parse
# various shape_info string formats from profile_report.json.

SHAPE_KEYS: Dict[str, List[str]] = {
    "matmul":            ["M", "N", "K"],
    "flash_attention":   ["B", "H", "N", "D"],
    "layernorm":         ["M", "N"],
    "softmax":           ["M", "N"],
    "cross_entropy":     ["batch", "vocab"],
    "fused_mlp":         ["M", "N", "K"],
    "rmsnorm":           ["M", "N"],
    "reduce":            ["M", "N"],
    "rotary_embedding":  ["B", "H", "N", "D"],
}

# Aliases: profile_report.json may use different key names than bench.py
# Map from alias -> canonical bench.py key, per op_type.
SHAPE_ALIAS_MAP: Dict[str, Dict[str, str]] = {
    "matmul": {},
    "flash_attention": {
        "B": "batch", "H": "heads", "N": "seq_len", "S": "seq_len", "D": "head_dim",
        "batch": "batch", "heads": "heads", "seq_len": "seq_len", "head_dim": "head_dim",
    },
    "layernorm": {
        "M": "batch", "N": "dim", "rows": "batch", "cols": "dim",
        "batch": "batch", "dim": "dim",
    },
    "softmax": {
        "M": "rows", "N": "cols", "rows": "rows", "cols": "cols",
    },
    "cross_entropy": {
        "batch": "batch", "vocab": "vocab",
    },
    "fused_mlp": {
        "M": "batch", "N": "hidden", "K": "dim",
        "batch": "batch", "dim": "dim", "hidden": "hidden",
    },
    "rmsnorm": {
        "M": "M", "N": "N",
    },
    "reduce": {
        "M": "M", "N": "N",
    },
    "rotary_embedding": {
        "B": "batch", "H": "heads", "N": "seq_len", "S": "seq_len", "D": "head_dim",
        "batch": "batch", "heads": "heads", "seq_len": "seq_len", "head_dim": "head_dim",
    },
}

# Default tolerances per op_type (matching bench.py structure, serialized for template)
TOLERANCES_MAP: Dict[str, Dict[str, Dict[str, float]]] = {
    "matmul": {
        "float16":  {"atol": 1e-2, "rtol": 1e-2},
        "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
        "float32":  {"atol": 1e-4, "rtol": 1e-4},
    },
    "flash_attention": {
        "float16":  {"atol": 1e-2, "rtol": 1e-2},
        "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
        "float32":  {"atol": 1e-4, "rtol": 1e-4},
    },
    "layernorm": {
        "float16":  {"atol": 1e-3, "rtol": 1e-3},
        "bfloat16": {"atol": 2e-3, "rtol": 2e-3},
        "float32":  {"atol": 1e-5, "rtol": 1e-5},
    },
    "softmax": {
        "float16":  {"atol": 1e-3, "rtol": 1e-3},
        "bfloat16": {"atol": 2e-3, "rtol": 2e-3},
        "float32":  {"atol": 1e-5, "rtol": 1e-5},
    },
    "cross_entropy": {
        "float16":  {"atol": 1e-2, "rtol": 1e-2},
        "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
        "float32":  {"atol": 1e-5, "rtol": 1e-5},
    },
    "fused_mlp": {
        "float16":  {"atol": 1e-2, "rtol": 1e-2},
        "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
        "float32":  {"atol": 1e-4, "rtol": 1e-4},
    },
    "rmsnorm": {
        "float16":  {"atol": 1e-2, "rtol": 1e-2},
        "bfloat16": {"atol": 1e-1, "rtol": 5e-2},
    },
    "reduce": {
        "float16":  {"atol": 1e-2, "rtol": 1e-2},
        "bfloat16": {"atol": 1e-1, "rtol": 5e-2},
    },
    "rotary_embedding": {
        "float16":  {"atol": 1e-3, "rtol": 1e-3},
        "bfloat16": {"atol": 2e-3, "rtol": 2e-3},
        "float32":  {"atol": 1e-5, "rtol": 1e-5},
    },
}

# Dtypes swept per op_type, mirroring bench.py's built-in test_dtypes.
# The profiled model's dtype is moved to the front of this list, since bench.py
# measures performance with test_dtypes[0].
TEST_DTYPES_MAP: Dict[str, List[str]] = {
    "matmul":           ["float16", "bfloat16", "float32"],
    "softmax":          ["float16", "bfloat16", "float32"],
    "layernorm":        ["float16", "bfloat16", "float32"],
    "flash_attention":  ["float16", "bfloat16"],
    "fused_mlp":        ["float16", "bfloat16", "float32"],
    "cross_entropy":    ["float16", "bfloat16", "float32"],
    "rotary_embedding": ["float16", "bfloat16", "float32"],
    "rmsnorm":          ["float16", "bfloat16"],
    "reduce":           ["float16", "bfloat16"],
}

# Canonical dtype names, so --dtype bf16 and --dtype bfloat16 agree.
DTYPE_ALIASES: Dict[str, str] = {
    "float16": "float16", "fp16": "float16", "half": "float16",
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float32": "float32", "fp32": "float32", "float": "float32",
}

# Fallback tolerances when a model's dtype has no entry in TOLERANCES_MAP.
DEFAULT_TOLERANCE_BY_DTYPE: Dict[str, Dict[str, float]] = {
    "float16":  {"atol": 1e-2, "rtol": 1e-2},
    "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
    "float32":  {"atol": 1e-4, "rtol": 1e-4},
}

# FLOPS formulas as source strings, per op_type
FLOPS_FN_SRC: Dict[str, str] = {
    "matmul":           'return 2 * s["M"] * s["N"] * s["K"]',
    "flash_attention":  'return 4 * s["batch"] * s["heads"] * (s["seq_len"] ** 2) * s["head_dim"]',
    "layernorm":        'return 8 * s["batch"] * s["dim"]',
    "softmax":          'return 5 * s["rows"] * s["cols"]',
    "cross_entropy":    'return 4 * s["batch"] * s["vocab"]',
    "fused_mlp":        'return 2 * s["batch"] * s["dim"] * s["hidden"] * 3',
    "rmsnorm":          'return 6 * s["M"] * s["N"]',
    "reduce":           'return s["M"] * s["N"]',
    "rotary_embedding": 'return 6 * s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"]',
}

# BYTES formulas as source strings, per op_type (dt_bytes is passed in)
BYTES_FN_SRC: Dict[str, str] = {
    "matmul":           'return (s["M"] * s["K"] + s["K"] * s["N"] + s["M"] * s["N"]) * dt_bytes',
    "flash_attention":  'return 4 * s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"] * dt_bytes',
    "layernorm":        'return (2 * s["batch"] * s["dim"] + 2 * s["dim"]) * dt_bytes',
    "softmax":          'return 2 * s["rows"] * s["cols"] * dt_bytes',
    "cross_entropy":    'return (s["batch"] * s["vocab"] + s["batch"]) * dt_bytes',
    "fused_mlp":        'return (s["batch"] * s["dim"] + s["hidden"] * s["dim"] * 3 + s["batch"] * s["dim"]) * dt_bytes',
    "rmsnorm":          'return (2 * s["M"] * s["N"] + s["N"]) * dt_bytes',
    "reduce":           'return (s["M"] * s["N"] + s["M"]) * dt_bytes',
    "rotary_embedding": 'return (s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"] * 2 + s["seq_len"] * s["head_dim"]) * dt_bytes',
}

# Speedup potential heuristic per op_type
SPEEDUP_ESTIMATES: Dict[str, str] = {
    "matmul":           "2-3x",
    "flash_attention":  "2-4x",
    "layernorm":        "1.5-3x",
    "softmax":          "1.5-3x",
    "cross_entropy":    "1.5-2x",
    "fused_mlp":        "2-3x",
    "rmsnorm":          "1.5-3x",
    "reduce":           "1.5-2x",
    "rotary_embedding": "1.5-2x",
}


# ---------------------------------------------------------------------------
# Shape parsing
# ---------------------------------------------------------------------------

# Attention entry points whose q/k/v are laid out [B, S, H, D] rather than the
# [B, H, S, D] that scaled_dot_product_attention and its backends use.
_ATTENTION_BSHD_OPS = {
    "_flash_attention_forward",
    "flash_attn_func",
    "flash_attn_varlen_func",
    "_flash_attention_backward",
}

# No production attention config has more heads than this; used to detect a
# misread [B, S, H, D] / [B, H, S, D] axis order.
_MAX_PLAUSIBLE_HEADS = 256


def _apply_alias_map(raw: Dict[str, int], op_type: str) -> Dict[str, int]:
    """Map raw shape keys onto the canonical bench.py keys for this op_type."""
    alias_map = SHAPE_ALIAS_MAP.get(op_type, {})
    if not alias_map:
        return raw
    return {alias_map.get(k, k): v for k, v in raw.items()}


def _canonical_op_name(name: str) -> str:
    """Reduce a profiler event name to a bare op name.

    "aten::addmm" -> "addmm", "aten::max_pool2d.default" -> "max_pool2d".
    Raw CUDA kernel names (e.g. "sm80_xmma_gemm_f16f16_...") pass through
    lowercased, which is harmless -- they carry no input shapes anyway.
    """
    n = str(name).strip()
    if "::" in n:
        n = n.split("::", 1)[1]
    if "." in n:
        n = n.split(".", 1)[0]
    return n.lower()


def _prod(dims: List[int]) -> int:
    total = 1
    for d in dims:
        total *= int(d)
    return total


def _gemm_shape_from_args(mats: List[List[int]]) -> Optional[Dict[str, int]]:
    """Derive {M, N, K} from the positional tensor args of a GEMM-family op.

    Covers aten::mm, addmm, bmm, baddbmm, matmul and linear uniformly by
    taking the last two >=2-D operands (which skips addmm's 1-D bias and
    linear's trailing bias) and reducing any leading/batch dims into M.

    linear stores its weight transposed as [N, K] rather than [K, N], so the
    contraction dim is matched against both axes to tell the two apart.
    """
    if len(mats) < 2:
        return None

    a, b = mats[-2], mats[-1]
    K = int(a[-1])
    M = _prod(a[:-1])

    if int(b[-2]) == K:
        N = int(b[-1])          # [K, N] -- mm / addmm / bmm / matmul
    elif int(b[-1]) == K:
        N = int(b[-2])          # [N, K] -- linear's transposed weight
    else:
        return None             # operands don't contract; not a GEMM we understand

    if M <= 0 or N <= 0 or K <= 0:
        return None
    return {"M": M, "N": N, "K": K}


def parse_shape_list(
    arg_shapes: Any,
    op_type: str,
    name: str = "",
) -> Optional[Dict[str, int]]:
    """Derive a canonical shape dict from torch profiler positional input shapes.

    torch.profiler records ``input_shapes`` as one entry per positional
    argument, in argument order, with scalars represented as empty lists --
    e.g. aten::addmm on a 4096x1536 @ 1536x1536 GEMM records
    ``[[1536], [4096, 1536], [1536, 1536]]``.

    Returns None if the shapes can't be interpreted for this op_type, so the
    caller can fall back to defaults.
    """
    if not isinstance(arg_shapes, (list, tuple)) or not arg_shapes:
        return None

    # Keep only real tensor operands, in argument order.
    mats: List[List[int]] = []
    for entry in arg_shapes:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            try:
                dims = [int(d) for d in entry]
            except (TypeError, ValueError):
                continue
            if all(d > 0 for d in dims):
                mats.append(dims)

    if not mats:
        return None

    op = _canonical_op_name(name)

    if op_type in ("matmul", "fused_mlp"):
        gemm = _gemm_shape_from_args(mats)
        return _apply_alias_map(gemm, op_type) if gemm else None

    if op_type in ("flash_attention", "rotary_embedding"):
        # Query is the first operand, but the axis order depends on the op:
        # the FlashAttention entry points take [B, S, H, D], while SDPA and
        # its fused backends take [B, H, S, D].
        q = mats[0]
        if len(q) != 4:
            return None
        if op in _ATTENTION_BSHD_OPS:
            B, S, H, D = q
        else:
            B, H, S, D = q
        # Guard the layout call: head counts are small and bounded, sequence
        # lengths are not, so an implausible head count means we guessed the
        # order wrong (e.g. reading [1, 2048, 28, 128] as 2048 heads of 28).
        if H > _MAX_PLAUSIBLE_HEADS and H > S:
            H, S = S, H
        return _apply_alias_map({"B": B, "H": H, "N": S, "D": D}, op_type)

    if op_type in ("layernorm", "rmsnorm", "softmax", "reduce", "cross_entropy"):
        # Row-wise ops: everything but the last dim collapses into the row count.
        x = mats[0]
        return _apply_alias_map({"M": _prod(x[:-1]), "N": int(x[-1])}, op_type)

    return None


def parse_shape_info(
    shape_info: Any,
    op_type: str,
    name: str = "",
) -> Optional[Dict[str, int]]:
    """
    Parse the ``shape_info`` field of a profile report entry into a shape dict.

    Handles the formats a report may carry:
      - a real list of positional arg shapes: [[4096, 1536], [1536, 1536]]
      - its string form, as written by str(evt.input_shapes)
      - hand-written key=value strings: "M=4096, N=4096, K=4096"

    Returns None if parsing fails.
    """
    # Structured list straight from the profiler.
    if isinstance(shape_info, (list, tuple)):
        return parse_shape_list(shape_info, op_type, name)

    if not shape_info or not isinstance(shape_info, str):
        return None

    # key=value form takes precedence -- it is already canonical.
    pairs = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\d+)", shape_info)
    if pairs:
        return _apply_alias_map({k: int(v) for k, v in pairs}, op_type)

    # Bracketed nested-list form, i.e. str() of the profiler's input_shapes.
    if "[" in shape_info:
        groups = re.findall(r"\[([0-9,\s]*)\]", shape_info)
        if groups:
            arg_shapes = [
                [int(tok) for tok in g.replace(" ", "").split(",") if tok]
                for g in groups
            ]
            return parse_shape_list(arg_shapes, op_type, name)

    return None


def shape_to_display(shape: Dict[str, int]) -> str:
    """Convert a shape dict to a display string like 'M=4096, N=4096, K=4096'."""
    return ", ".join(f"{k}={v}" for k, v in shape.items())


def scale_shape(shape: Dict[str, int], factor: float) -> Dict[str, int]:
    """
    Scale all shape dimensions by a factor, rounding to nearest integer.
    Ensures all values are at least 1.
    """
    return {k: max(1, int(round(v * factor))) for k, v in shape.items()}


def get_default_shape(op_type: str) -> Dict[str, int]:
    """
    Return a reasonable default shape for a given op_type when parsing fails.
    Based on the 'large' size from bench.py KERNEL_CONFIGS.
    """
    defaults: Dict[str, Dict[str, int]] = {
        "matmul":           {"M": 2048, "N": 2048, "K": 2048},
        "flash_attention":  {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 64},
        "layernorm":        {"batch": 4096, "dim": 2048},
        "softmax":          {"rows": 4096, "cols": 4096},
        "cross_entropy":    {"batch": 4096, "vocab": 32000},
        "fused_mlp":        {"batch": 2048, "dim": 2048, "hidden": 5504},
        "rmsnorm":          {"M": 4096, "N": 4096},
        "reduce":           {"M": 4096, "N": 4096},
        "rotary_embedding": {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 128},
    }
    return defaults.get(op_type, {"M": 2048, "N": 2048})


def resolve_model_shape(kernel_info: Dict[str, Any]) -> Tuple[Dict[str, int], str]:
    """Resolve one report entry to (shape, source).

    source is one of:
      "profiled" -- derived from shapes the profiler actually recorded
      "report"   -- taken from an explicit "shapes" dict in the report
      "default"  -- nothing usable in the report; generic fallback shape
    """
    op_type = kernel_info.get("op_type", "unknown")
    name = kernel_info.get("name", "")

    # Prefer the structured field; fall back to the stringified one.
    raw_shapes = kernel_info.get("input_shapes")
    if raw_shapes is None:
        raw_shapes = kernel_info.get("shape_info", kernel_info.get("shape", ""))

    shape = parse_shape_info(raw_shapes, op_type, name)
    if shape:
        return shape, "profiled"

    if isinstance(kernel_info.get("shapes"), dict) and kernel_info["shapes"]:
        return kernel_info["shapes"], "report"

    return get_default_shape(op_type), "default"


def infer_missing_fused_mlp_shapes(resolved: List[Dict[str, Any]]) -> int:
    """Recover fused_mlp shapes from the MLP's own projection pair, in place.

    The profiler classifies a bare activation (aten::silu) as fused_mlp, but
    its single operand [.., hidden] can't supply the (batch, dim, hidden)
    triple the fused kernel is benchmarked on. The two projections that
    bracket that activation are normally in the same report though --
    gate/up as (K=dim, N=hidden) and down as (K=hidden, N=dim) at the same M
    -- so the triple can be read off them instead of falling back to a
    generic default.

    Returns the number of entries filled in.
    """
    targets = [e for e in resolved
               if e["op_type"] == "fused_mlp" and e["shape_source"] != "profiled"]
    if not targets:
        return 0

    gemms = [e["model_shape"] for e in resolved
             if e["op_type"] == "matmul" and e["shape_source"] == "profiled"
             and {"M", "N", "K"} <= set(e["model_shape"])]

    best = None
    for a in gemms:
        for b in gemms:
            # a is gate/up (dim -> hidden), b is down (hidden -> dim).
            if a["K"] == b["N"] and a["N"] == b["K"] and a["M"] == b["M"]:
                cand = {"batch": a["M"], "dim": a["K"], "hidden": a["N"]}
                if best is None or cand["hidden"] > best["hidden"]:
                    best = cand
    if best is None:
        return 0

    for e in targets:
        e["model_shape"] = dict(best)
        e["shape_source"] = "inferred"
    return len(targets)


def _shape_key(op_type: str, shape: Dict[str, int]) -> Tuple:
    return (op_type, tuple(sorted((str(k), int(v)) for k, v in shape.items())))


def dedupe_kernels(
    resolved: List[Dict[str, Any]],
    time_tol: float = 0.02,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collapse report entries that describe the same physical kernel.

    Two things produce duplicates in a profile_report.json:

      1. The same op at the same shape appearing under more than one name.
      2. An ``aten::*`` op and the raw CUDA kernel it launched both being
         reported, since key_averages() covers CPU and CUDA activities alike.
         These have near-identical device time, but only the aten row carries
         input shapes -- so the kernel row lands on a default shape and looks
         like a distinct target.

    Case 1 is matched on (op_type, shape). Case 2 is matched on op_type plus
    device time, and only ever folds a default-shaped entry into a profiled
    one -- never the reverse, so a genuinely unparsed kernel is not silently
    absorbed by an unrelated neighbour.

    Merged entries accumulate pct_total so the plan's priority ordering still
    reflects the real share of GPU time. Returns (kept, dropped).
    """
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for entry in resolved:
        target = None

        for existing in kept:
            if existing["op_type"] != entry["op_type"]:
                continue

            if _shape_key(existing["op_type"], existing["model_shape"]) == \
                    _shape_key(entry["op_type"], entry["model_shape"]):
                target = existing
                break

            # aten op vs. its own CUDA kernel: same device time, but the
            # kernel row never had shapes to parse.
            if entry["shape_source"] == "default" and existing["shape_source"] == "profiled":
                a, b = entry["gpu_time_ms"], existing["gpu_time_ms"]
                if a > 0 and b > 0 and abs(a - b) <= time_tol * max(a, b):
                    target = existing
                    break

        if target is None:
            kept.append(entry)
            continue

        # Fold into the entry we are keeping. Device time is NOT summed: the
        # duplicate is the same work seen twice, not additional work.
        target["pct_total"] = round(target["pct_total"] + entry["pct_total"], 1)
        target.setdefault("merged_from", []).append({
            "rank": entry["rank"],
            "name": entry.get("name", ""),
            "gpu_time_ms": entry["gpu_time_ms"],
        })
        dropped.append(entry)

    return kept, dropped


# ---------------------------------------------------------------------------
# Kernel file generation
# ---------------------------------------------------------------------------

def normalize_dtype(dtype_str: Optional[str]) -> str:
    """Canonicalize a dtype name from the profile report. Defaults to float16."""
    if not dtype_str:
        return "float16"
    return DTYPE_ALIASES.get(str(dtype_str).lower().strip(), "float16")


def order_test_dtypes(op_type: str, model_dtype: str) -> List[str]:
    """Return the dtype sweep for an op with the model's dtype first.

    bench.py benchmarks with test_dtypes[0], so a bf16 model must not be
    measured in fp16.
    """
    candidates = list(TEST_DTYPES_MAP.get(op_type, ["float16", "bfloat16"]))
    return [model_dtype] + [d for d in candidates if d != model_dtype]


def read_starter_kernel(op_type: str, backend: str = "triton") -> Optional[str]:
    """Read the starter kernel file. Returns None if not found.

    For backend='triton': reads from ak_kernels/{op_type}.py
    For backend='cuda':   reads from ak_kernels/cuda/{op_type}.py
    """
    if backend == "cuda":
        path = os.path.join(KERNELS_DIR, "cuda", f"{op_type}.py")
    else:
        path = os.path.join(KERNELS_DIR, f"{op_type}.py")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_kernel_body(starter_code: str) -> str:
    """
    Extract the Triton kernel code from a starter file, stripping the
    original module docstring and KERNEL_TYPE declaration (which we replace
    in the template header).

    Returns everything from the first 'import' statement onward.
    """
    lines = starter_code.split("\n")

    # Find the first import line
    import_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_idx = i
            break

    if import_idx is not None:
        return "\n".join(lines[import_idx:])
    else:
        # Fallback: return everything after KERNEL_TYPE line
        for i, line in enumerate(lines):
            if line.strip().startswith("KERNEL_TYPE"):
                return "\n".join(lines[i + 1:])
        return starter_code


def generate_kernel_file(
    op_type: str,
    rank: int,
    pct_total: float,
    model_shape: Dict[str, int],
    model_name: str,
    gpu_time_ms: float,
    starter_code: str,
    backend: str = "triton",
    model_dtype: str = "float16",
    shape_source: str = "profiled",
) -> str:
    """Generate the complete kernel file content for extraction."""

    half_shape = scale_shape(model_shape, 0.5)
    double_shape = scale_shape(model_shape, 2.0)

    shape_display = shape_to_display(model_shape)
    half_display = shape_to_display(half_shape)
    double_display = shape_to_display(double_shape)

    test_dtypes = order_test_dtypes(op_type, model_dtype)

    tolerances = dict(TOLERANCES_MAP.get(op_type, DEFAULT_TOLERANCE_BY_DTYPE))
    # Every dtype in the sweep needs a tolerance entry.
    for dt in test_dtypes:
        if dt not in tolerances:
            tolerances[dt] = DEFAULT_TOLERANCE_BY_DTYPE.get(
                dt, {"atol": 1e-2, "rtol": 1e-2}
            )

    flops_fn_body = FLOPS_FN_SRC.get(op_type, 'return 0')
    bytes_fn_body = BYTES_FN_SRC.get(op_type, 'return 0')

    # Extract the kernel code body (imports + jit functions + kernel_fn)
    kernel_body = extract_kernel_body(starter_code)

    # Build the file
    lines = []

    # Header docstring
    lines.append('"""')
    lines.append(f"AutoKernel -- Extracted kernel from model profiling.")
    lines.append(f"Op type: {op_type}")
    lines.append(f"Rank: {rank} ({pct_total}% of GPU time)")
    lines.append(f"Model shape: {shape_display}")
    lines.append(f"Model dtype: {model_dtype}")
    if shape_source == "inferred":
        lines.append(f"")
        lines.append(f"NOTE: these sizes were not measured for this op directly -- they were")
        lines.append(f"derived from the surrounding projection GEMMs in the profile report.")
    elif shape_source != "profiled":
        lines.append(f"")
        lines.append(f"WARNING: shape source is '{shape_source}', not measured. The profile")
        lines.append(f"report carried no usable shapes for this op, so the sizes above are a")
        lines.append(f"generic fallback. Tuning against them optimizes a problem size the")
        lines.append(f"model may never run -- re-profile with record_shapes=True first.")
    lines.append(f"")
    lines.append(f"This kernel was extracted from profiling {model_name}.")
    lines.append(f"The agent optimizes this to maximize throughput at the model-specific shapes.")
    lines.append('"""')
    lines.append("")

    # KERNEL_TYPE and BACKEND
    lines.append(f'KERNEL_TYPE = "{op_type}"')
    if backend == "cuda":
        lines.append(f'BACKEND = "cuda"')
    lines.append("")

    # Model-specific shapes
    lines.append("# Model-specific shapes (the shapes that matter for THIS model)")
    lines.append(f"MODEL_SHAPES = {repr(model_shape)}")
    lines.append("")

    # Benchmark config
    lines.append("# Benchmark config -- bench.py loads these and overlays them onto its")
    lines.append("# built-in config for this kernel type, so the model's own shapes and")
    lines.append("# dtype are what get benchmarked.")
    lines.append("TEST_SIZES = [")
    lines.append(f'    ("model_primary", {repr(model_shape)}),')
    lines.append(f"    # Also test nearby sizes for robustness")
    lines.append(f'    ("model_half", {repr(half_shape)}),')
    lines.append(f'    ("model_double", {repr(double_shape)}),')
    lines.append("]")
    lines.append("")

    # Dtypes (model dtype first -- bench.py measures with TEST_DTYPES[0])
    lines.append("# Model dtype first: bench.py measures performance with TEST_DTYPES[0].")
    lines.append(f"TEST_DTYPES = {repr(test_dtypes)}")
    lines.append("")

    # Tolerances
    lines.append(f"TOLERANCES = {repr(tolerances)}")
    lines.append("")

    # FLOPS function
    lines.append("")
    lines.append("def FLOPS_FN(s):")
    lines.append(f"    {flops_fn_body}")
    lines.append("")

    # BYTES function
    lines.append("")
    lines.append("def BYTES_FN(s, dt_bytes):")
    lines.append(f"    {bytes_fn_body}")
    lines.append("")

    # Separator
    lines.append("")
    lines.append(f"# {'=' * 70}")
    backend_label = "CUDA C++" if backend == "cuda" else "Triton"
    backend_dir = f"ak_kernels/cuda/{op_type}.py" if backend == "cuda" else f"ak_kernels/{op_type}.py"
    lines.append(f"# {backend_label} kernel code (from {backend_dir})")
    lines.append(f"# {'=' * 70}")
    lines.append("")

    # Kernel body
    lines.append(kernel_body)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profile reading and validation
# ---------------------------------------------------------------------------

def load_profile_report(path: str) -> Optional[Dict[str, Any]]:
    """Load and validate the profile report JSON. Returns None on failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"ERROR: Failed to read profile report: {e}")
        return None


def get_supported_kernels(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract the list of supported (autokernel_supported=True) kernels from
    the profile report, sorted by rank.
    """
    kernels = report.get("top_kernels", report.get("kernels", report.get("bottleneck_kernels", [])))
    supported = []
    for k in kernels:
        if k.get("autokernel_supported", False):
            supported.append(k)

    # Sort by rank if available, otherwise by gpu_time_ms descending
    supported.sort(key=lambda x: x.get("rank", x.get("gpu_time_ms", 0)))
    # Ensure rank ordering (lower rank = higher priority)
    for i, k in enumerate(supported):
        if "rank" not in k:
            k["rank"] = i + 1

    return supported


# ---------------------------------------------------------------------------
# Optimization plan generation
# ---------------------------------------------------------------------------

def generate_optimization_plan(
    extracted: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the optimization_plan.json data structure."""
    kernels_to_optimize = []
    total_pct = 0.0

    for entry in extracted:
        total_pct += entry["pct_total"]
        kernels_to_optimize.append({
            "rank": entry["rank"],
            "file": entry["output_file"],
            "op_type": entry["op_type"],
            "model_shape": entry["model_shape"],
            # Downstream phases must not treat a fallback shape as measured.
            "shape_source": entry.get("shape_source", "profiled"),
            "merged_from": entry.get("merged_from", []),
            "gpu_time_ms": entry["gpu_time_ms"],
            "pct_total": entry["pct_total"],
            "estimated_speedup_potential": SPEEDUP_ESTIMATES.get(
                entry["op_type"], "1.5-2x"
            ),
        })

    defaulted = [k for k in kernels_to_optimize if k["shape_source"] == "default"]

    return {
        "kernels_to_optimize": kernels_to_optimize,
        "total_optimization_targets": len(kernels_to_optimize),
        "covered_gpu_time_pct": round(total_pct, 1),
        "kernels_with_default_shapes": len(defaulted),
    }


# ---------------------------------------------------------------------------
# Main extraction logic
# ---------------------------------------------------------------------------

def extract_kernels(
    report_path: str,
    top_n: Optional[int] = None,
    kernel_type_filter: Optional[str] = None,
    backend: str = "triton",
    dtype_override: Optional[str] = None,
) -> None:
    """Main extraction pipeline."""

    backend_label = "CUDA C++" if backend == "cuda" else "Triton"
    print(f"=== AutoKernel Kernel Extractor ({backend_label}) ===")
    print()

    # -- Load profile report --
    print(f"Reading profile from {report_path}...")
    report = load_profile_report(report_path)
    if report is None:
        print(f"ERROR: Profile report not found at {report_path}")
        print(f"       Run the profiler first: uv run profile_model.py")
        sys.exit(1)

    # -- Get model name --
    model_name = report.get("model_name", report.get("model", "unknown model"))

    # -- Get model dtype (bench.py measures with TEST_DTYPES[0]) --
    model_dtype = normalize_dtype(dtype_override or report.get("dtype"))
    print(f"Model dtype: {model_dtype}")

    # -- Get supported kernels --
    supported = get_supported_kernels(report)
    if not supported:
        print("ERROR: No supported kernels found in profile report.")
        print("       Ensure the profiler marks kernels with autokernel_supported=True.")
        sys.exit(1)

    # -- Apply filters --
    if kernel_type_filter:
        supported = [k for k in supported if k.get("op_type") == kernel_type_filter]
        if not supported:
            print(f"WARNING: No kernels of type '{kernel_type_filter}' found in profile report.")
            sys.exit(1)

    # NOTE: --top is applied after dedup, further down, so that N asks for N
    # distinct kernels rather than N report rows that may collapse into fewer.

    print(f"Found {len(supported)} supported kernels in the report.")
    print()

    # -- Ensure workspace directory exists --
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    # -- Resolve shapes up front, so duplicates can be collapsed before we
    #    write any files (two entries for one kernel would otherwise become
    #    two identical optimization targets).
    resolved = []
    for idx, kernel_info in enumerate(supported):
        model_shape, shape_source = resolve_model_shape(kernel_info)
        resolved.append({
            "rank": kernel_info.get("rank", idx + 1),
            "name": kernel_info.get("name", ""),
            "op_type": kernel_info.get("op_type", "unknown"),
            "pct_total": kernel_info.get("pct_total", kernel_info.get("pct_gpu_time", 0.0)),
            "gpu_time_ms": kernel_info.get(
                "gpu_time_ms", kernel_info.get("total_gpu_time_ms", 0.0)
            ),
            "model_shape": model_shape,
            "shape_source": shape_source,
        })

    n_inferred = infer_missing_fused_mlp_shapes(resolved)
    if n_inferred:
        print(f"  NOTE: recovered {n_inferred} fused_mlp shape(s) from the MLP "
              f"projection GEMMs rather than falling back to defaults.")
        print()

    resolved, duplicates = dedupe_kernels(resolved)

    for dup in duplicates:
        print(f"  NOTE: rank {dup['rank']} ({dup['op_type']}) duplicates an earlier "
              f"entry at the same shape -- merged, not extracted twice.")
    if duplicates:
        print()

    # Now that duplicates are gone, --top N means N distinct kernels.
    if top_n is not None:
        resolved = resolved[:top_n]

    print(f"Extracting {len(resolved)} distinct kernel(s).")
    print()

    defaulted = [e for e in resolved if e["shape_source"] == "default"]
    if defaulted:
        print(f"  WARNING: {len(defaulted)} kernel(s) had no parseable shape in the "
              f"report and fall back to generic defaults:")
        for e in defaulted:
            print(f"           rank {e['rank']} ({e['op_type']}) -> "
                  f"{shape_to_display(e['model_shape'])}")
        print(f"           These will be tuned at the wrong problem size. Check that "
              f"the profiler ran with record_shapes=True.")
        print()

    # -- Extract each kernel --
    print("Extracting kernels:")
    extracted = []
    skipped = 0

    for idx, entry in enumerate(resolved):
        rank = entry["rank"]
        op_type = entry["op_type"]
        pct_total = entry["pct_total"]
        gpu_time_ms = entry["gpu_time_ms"]
        model_shape = entry["model_shape"]
        shape_source = entry["shape_source"]

        # Read starter kernel
        starter_code = read_starter_kernel(op_type, backend=backend)
        if starter_code is None:
            starter_dir = "ak_kernels/cuda" if backend == "cuda" else "ak_kernels"
            print(f"  WARNING: No starter kernel found at {starter_dir}/{op_type}.py -- skipping.")
            skipped += 1
            continue

        # Generate output filename
        output_filename = f"kernel_{op_type}_{rank}.py"
        output_path = os.path.join(WORKSPACE_DIR, output_filename)
        # Relative path for display and plan
        output_relpath = f"workspace/{output_filename}"

        # Generate the customized kernel file
        kernel_content = generate_kernel_file(
            op_type=op_type,
            rank=rank,
            pct_total=pct_total,
            model_shape=model_shape,
            model_name=model_name,
            gpu_time_ms=gpu_time_ms,
            starter_code=starter_code,
            backend=backend,
            model_dtype=model_dtype,
            shape_source=shape_source,
        )

        # Write to workspace
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(kernel_content)

        # Print progress
        position = idx + 1
        total = len(resolved)
        shape_display = shape_to_display(model_shape)
        shape_note = "" if shape_source == "profiled" else f"  [{shape_source} shape]"
        print(f"  [{position}/{total}] {op_type} (rank {rank}, {pct_total}%) "
              f"-> {output_relpath}")
        print(f"        Model shape: {shape_display}{shape_note}")
        starter_dir = "ak_kernels/cuda" if backend == "cuda" else "ak_kernels"
        print(f"        Based on: {starter_dir}/{op_type}.py")
        print()

        extracted.append({
            "rank": rank,
            "op_type": op_type,
            "pct_total": pct_total,
            "gpu_time_ms": gpu_time_ms,
            "model_shape": model_shape,
            "shape_source": shape_source,
            "merged_from": entry.get("merged_from", []),
            "output_file": output_relpath,
        })

    if not extracted:
        print("ERROR: No kernels were successfully extracted.")
        if skipped > 0:
            print(f"       {skipped} kernel(s) skipped due to missing starter files.")
        sys.exit(1)

    # -- Generate optimization plan --
    plan = generate_optimization_plan(extracted)
    with open(OPTIMIZATION_PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=4)
    print(f"Optimization plan saved to workspace/optimization_plan.json")

    # -- Print next steps --
    print()
    top_kernel = extracted[0]
    top_file = top_kernel["output_file"]
    print("Next steps:")
    print(f"  1. Copy a kernel to kernel.py: cp {top_file} kernel.py")
    print(f"  2. Run benchmark: uv run bench.py")
    print(f"  3. Start optimizing (or let the agent do it via program.md)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoKernel Kernel Extractor -- Generate baseline kernels from profiling results.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=DEFAULT_REPORT_PATH,
        help="Path to profile_report.json (default: workspace/profile_report.json)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Extract only the top-N kernels by rank",
    )
    parser.add_argument(
        "--kernel-type",
        type=str,
        default=None,
        help="Extract only kernels of this type (e.g., matmul, flash_attention)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["triton", "cuda"],
        default="triton",
        help="Backend for starter kernels: 'triton' (default) or 'cuda' (native CUDA C++)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Override the dtype recorded in the profile report (e.g. bfloat16). "
             "This becomes TEST_DTYPES[0] in the generated kernels, which is what "
             "bench.py measures with.",
    )

    args = parser.parse_args()

    extract_kernels(
        report_path=args.report,
        top_n=args.top,
        kernel_type_filter=args.kernel_type,
        backend=args.backend,
        dtype_override=args.dtype,
    )


if __name__ == "__main__":
    main()
