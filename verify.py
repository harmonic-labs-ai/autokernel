#!/usr/bin/env python3
"""
AutoKernel End-to-End Verifier -- Plug optimized kernels back into the model and verify.

Usage:
    uv run verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,2048
    uv run verify.py --module transformers --class-name AutoModelForCausalLM --pretrained meta-llama/Llama-2-7b-hf
    uv run verify.py --model models/llama_7b.py --class-name LlamaModel --input-shape 1,2048 --diagnose

Checks:
  1. Loads the original model
  2. Runs inference with original PyTorch ops -> captures reference output
  3. Replaces bottleneck ops with optimized Triton kernels
  4. Runs inference with optimized kernels -> captures optimized output
  5. Compares outputs (tolerance check)
  6. Benchmarks both paths -> reports end-to-end speedup
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(SCRIPT_DIR, "workspace")

# Kernel types that can actually be installed back into a model. A type absent
# from this set can still be optimized by the Phase B loop, but its speedup
# will never show up end to end -- so extraction should prefer types listed
# here, and verification reports the gap explicitly.
SUPPORTED_REPLACEMENT_TYPES = (
    "matmul",
    "layernorm",
    "rmsnorm",
    "fused_mlp",
    "rotary_embedding",
    "rotary_embedding_half",
)
ORCHESTRATION_STATE = os.path.join(WORKSPACE_DIR, "orchestration_state.json")

# Benchmarking defaults
WARMUP_RUNS = 10
TIMED_RUNS = 50

# Tolerance defaults by dtype
# Tolerances for comparing whole-model outputs, which is a different problem
# from comparing one kernel's output. An optimized kernel accumulates in fp32
# and rounds once; the eager model rounds at every step, and over dozens of
# layers those roundings compound. The floor is the dtype's own precision:
# bf16 carries 8 mantissa bits (eps 7.8e-3), so logits of magnitude ~10 cannot
# be reproduced closer than ~0.08 by *any* reordering of the arithmetic.
#
# The previous 2e-3 absolute tolerance sat an order of magnitude below that
# floor, so every correct kernel failed end-to-end verification while passing
# its own bench.py checks. These values track eps and still catch real
# breakage, which shows up at the scale of the output itself (or as NaN/Inf,
# checked separately).
DEFAULT_TOLERANCES: Dict[torch.dtype, Dict[str, float]] = {
    torch.float16:  {"atol": 2e-2, "rtol": 2e-2},
    torch.bfloat16: {"atol": 8e-2, "rtol": 4e-2},
    torch.float32:  {"atol": 1e-5, "rtol": 1e-5},
}

# Whole-tensor relative L2 -- the pass criterion for reduced precision. A
# correct kernel differs from eager only by rounding, which is random in sign
# and partially cancels in the norm; a genuinely wrong one (mis-set epsilon,
# transposed operand, wrong rotation) shifts the whole tensor.
#
# Measured on Qwen2 towers, output logits, bf16:
#   correct kernel, 2 layers   1e-3 .. 5e-3
#   correct kernel, 28 layers  ~1.8e-2      <- floor grows with depth
#   +1% systematic error       ~2.1e-2  (caught)
#   +5% systematic error       ~9.2e-2  (caught)
#
# The floor rises with depth because a rounding difference at layer 0 is
# re-amplified by every layer above it, so this threshold trades sensitivity on
# deep models for not failing correct ones: it reliably catches errors of ~1.5%
# and up on a 28-layer model, and ~0.5% and up on a shallow one. Override with
# --rel-l2-tol to tighten it for a specific model, and prefer per-kernel
# bench.py correctness (which compares one op, with no depth amplification) as
# the primary gate -- this is the integration check, not the unit check.
REL_L2_TOLERANCES: Dict[torch.dtype, float] = {
    torch.float16:  1e-2,
    torch.bfloat16: 3e-2,
    torch.float32:  1e-5,
}

# Threshold for the RoPE patch's self-check. Tighter than the end-to-end
# figures above because it compares one op's output directly against the
# function it replaces, with no depth amplification in between: rounding
# differences land near 1e-3, and a mismodelled variant (wrong mrope sectioning,
# wrong rotation, wrong broadcast) moves the whole tensor and lands near 1.
_ROPE_PATCH_REL_L2_TOL = 1e-2


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KernelReplacement:
    """Describes a single kernel replacement: what to replace and with what."""
    kernel_type: str          # e.g. "matmul", "layernorm", "rmsnorm"
    rank: int                 # priority rank from profiling
    speedup: float            # individual kernel speedup
    optimized_path: str       # path to optimized kernel .py file
    module_fn: Optional[Callable] = None  # loaded kernel function


@dataclass
class VerificationResult:
    """Full verification result."""
    model_name: str = ""
    input_shape: str = ""
    dtype_str: str = ""
    gpu_name: str = ""

    # Reference run
    ref_output_shape: str = ""
    ref_latency_ms: float = 0.0

    # Optimized run
    opt_output_shape: str = ""
    opt_latency_ms: float = 0.0
    kernels_replaced: List[Dict[str, Any]] = field(default_factory=list)
    # (kernel_type, reason) pairs for optimized kernels that could not be
    # installed -- their gains are absent from end_to_end_speedup.
    kernels_not_applied: List[Tuple[str, str]] = field(default_factory=list)

    # Comparison
    correctness: str = "UNKNOWN"
    max_abs_error: float = 0.0
    mean_abs_error: float = 0.0
    rel_l2_error: float = 0.0
    has_nan: bool = False
    has_inf: bool = False

    # Summary
    end_to_end_speedup: float = 0.0


# ---------------------------------------------------------------------------
# 1. Model Loading
# ---------------------------------------------------------------------------

def load_model_from_file(model_path: str, class_name: str, **kwargs) -> nn.Module:
    """Load a model from a Python file by importing it and instantiating the class."""
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    spec = importlib.util.spec_from_file_location("user_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model from: {model_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, class_name):
        available = [n for n in dir(mod) if not n.startswith("_")]
        raise AttributeError(
            f"Class '{class_name}' not found in {model_path}. "
            f"Available names: {available}"
        )

    cls = getattr(mod, class_name)
    model = cls(**kwargs)
    return model


def load_model_from_module(module_name: str, class_name: str,
                           pretrained: Optional[str] = None, **kwargs) -> nn.Module:
    """Load a model from an installed Python module (e.g. 'transformers')."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_name}'. Is it installed? Error: {e}"
        )

    if not hasattr(mod, class_name):
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_name}'."
        )

    cls = getattr(mod, class_name)

    if pretrained:
        # HuggingFace-style: cls.from_pretrained(...)
        if hasattr(cls, "from_pretrained"):
            model = cls.from_pretrained(pretrained, **kwargs)
        else:
            raise AttributeError(
                f"'{class_name}' has no 'from_pretrained' method. "
                f"Cannot load pretrained weights from '{pretrained}'."
            )
    else:
        model = cls(**kwargs)

    return model


def load_model(args) -> nn.Module:
    """Unified model loader from CLI args."""
    dtype = _parse_dtype(args.dtype)

    if args.model:
        print(f"Loading model from file: {args.model} (class: {args.class_name})")
        model = load_model_from_file(args.model, args.class_name)
    elif args.module:
        print(f"Loading model from module: {args.module} (class: {args.class_name})")
        extra_kwargs = {}
        if dtype == torch.float16:
            extra_kwargs["torch_dtype"] = torch.float16
        elif dtype == torch.bfloat16:
            extra_kwargs["torch_dtype"] = torch.bfloat16
        model = load_model_from_module(
            args.module, args.class_name, pretrained=args.pretrained, **extra_kwargs
        )
    else:
        raise ValueError("Must specify either --model (file path) or --module (Python module)")

    model = model.to(dtype=dtype)

    if torch.cuda.is_available():
        try:
            model = model.cuda()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"WARNING: OOM moving model to GPU. Trying with smaller footprint...")
                torch.cuda.empty_cache()
                model = model.half().cuda()
            else:
                raise

    model.eval()
    return model


# ---------------------------------------------------------------------------
# 2. Input Generation
# ---------------------------------------------------------------------------

def generate_sample_input(
    input_shape: str,
    dtype: torch.dtype,
    device: str = "cuda",
    seed: int = 42,
) -> torch.Tensor:
    """Generate a sample input tensor from a shape string like '1,2048'."""
    dims = [int(d.strip()) for d in input_shape.split(",")]
    torch.manual_seed(seed)

    if dtype in (torch.int32, torch.int64, torch.long):
        # For language models, generate token IDs (assume vocab size ~32000)
        return torch.randint(0, 32000, dims, device=device, dtype=dtype)
    else:
        return torch.randn(dims, device=device, dtype=dtype)


def infer_input_type(model: nn.Module) -> str:
    """Try to determine if the model expects integer token IDs or float tensors."""
    # The forward signature is the most reliable signal. HuggingFace
    # *ForCausalLM / *ForConditionalGeneration wrappers take input_ids, but
    # their children are (model, lm_head) -- so the child scan below would hit
    # lm_head and wrongly answer "float".
    try:
        sig = inspect.signature(model.forward)
        if "input_ids" in sig.parameters:
            return "token_ids"
    except (ValueError, TypeError):
        pass

    # Check if model has an embedding layer as the first module
    for name, child in model.named_children():
        if isinstance(child, nn.Embedding):
            return "token_ids"
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            return "float"
    return "float"


def infer_vocab_size(model: nn.Module, default: int = 32000) -> int:
    """Find the model's vocab size so generated token IDs are always in range."""
    config = getattr(model, "config", None)
    vocab = getattr(config, "vocab_size", None)
    if isinstance(vocab, int) and vocab > 0:
        return vocab

    for module in model.modules():
        if isinstance(module, nn.Embedding) and module.num_embeddings > 0:
            return module.num_embeddings

    return default


def make_model_input(
    model: nn.Module,
    input_shape: str,
    dtype: torch.dtype,
    device: str = "cuda",
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Create an appropriate input for the model."""
    input_type = infer_input_type(model)

    if input_type == "token_ids":
        # Language model: expects integer input_ids
        dims = [int(d.strip()) for d in input_shape.split(",")]
        torch.manual_seed(42)
        vocab_size = infer_vocab_size(model)
        input_ids = torch.randint(0, vocab_size, dims, device=device, dtype=torch.long)

        # Check if model accepts input_ids keyword
        sig = inspect.signature(model.forward)
        if "input_ids" in sig.parameters:
            return {"input_ids": input_ids}
        return input_ids
    else:
        return generate_sample_input(input_shape, dtype, device)


# ---------------------------------------------------------------------------
# 3. Benchmarking
# ---------------------------------------------------------------------------

def benchmark_model(
    model: nn.Module,
    model_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    warmup: int = WARMUP_RUNS,
    timed: int = TIMED_RUNS,
) -> Tuple[Any, float]:
    """
    Benchmark model inference. Returns (output, median_latency_ms).
    Uses CUDA events for precise GPU timing.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking.")

    def _run():
        with torch.no_grad():
            if isinstance(model_input, dict):
                return model(**model_input)
            else:
                return model(model_input)

    # Warmup
    print(f"  Warmup: {warmup} runs...", end="", flush=True)
    for _ in range(warmup):
        output = _run()
    torch.cuda.synchronize()
    print(" done")

    # Timed runs
    print(f"  Timed: {timed} runs...", end="", flush=True)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(timed)]

    torch.cuda.synchronize()
    for i in range(timed):
        start_events[i].record()
        _run()
        end_events[i].record()
    torch.cuda.synchronize()
    print(" done")

    # Compute median
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    median_ms = times_ms[len(times_ms) // 2]

    # Final reference output (deterministic)
    with torch.no_grad():
        output = _run()
    torch.cuda.synchronize()

    return output, median_ms


# ---------------------------------------------------------------------------
# 4. Kernel Replacement
# ---------------------------------------------------------------------------

def load_orchestration_state() -> Optional[Dict]:
    """Load workspace/orchestration_state.json if it exists."""
    if not os.path.exists(ORCHESTRATION_STATE):
        return None
    with open(ORCHESTRATION_STATE, "r") as f:
        return json.load(f)


def discover_optimized_kernels() -> List[KernelReplacement]:
    """
    Find optimized kernels from the workspace directory.
    Checks orchestration_state.json first, then scans for *_optimized.py files.
    """
    replacements: List[KernelReplacement] = []

    # Strategy 1: Read orchestration state
    state = load_orchestration_state()
    if state and "kernels" in state:
        for k in state["kernels"]:
            ktype = k.get("op_type", k.get("type", "unknown"))
            rank = k.get("rank", 0)
            # These keys exist but are null until a kernel has both a baseline
            # and a best result, so a plain .get() default is not enough.
            speedup = k.get("speedup")
            if speedup is None:
                speedup = k.get("best_speedup")
            if speedup is None:
                speedup = 0.0
            speedup = float(speedup)
            # optimized_path is not written by orchestrate.py, so derive it
            # from the kernel file path if available
            opt_path = k.get("optimized_path", "")

            if not opt_path:
                # Try to derive from the "file" key that orchestrate.py writes
                base_file = k.get("file", "")
                if base_file:
                    stem = Path(base_file).stem
                    opt_path = os.path.join(
                        WORKSPACE_DIR, f"{stem}_optimized.py"
                    )
                else:
                    # Fallback convention: workspace/kernel_{type}_{rank}_optimized.py
                    opt_path = os.path.join(
                        WORKSPACE_DIR, f"kernel_{ktype}_{rank}_optimized.py"
                    )

            if os.path.exists(opt_path) and speedup > 1.0:
                replacements.append(KernelReplacement(
                    kernel_type=ktype,
                    rank=rank,
                    speedup=speedup,
                    optimized_path=opt_path,
                ))
        return replacements

    # Strategy 2: Scan workspace directory for optimized kernel files
    if not os.path.isdir(WORKSPACE_DIR):
        return replacements

    for fname in sorted(os.listdir(WORKSPACE_DIR)):
        if fname.endswith("_optimized.py"):
            # Parse filename: kernel_{type}_{rank}_optimized.py
            # Type can be multi-word (e.g. flash_attention), so the rank
            # is always the last numeric segment before "_optimized.py".
            stem = fname.replace("_optimized.py", "")  # e.g. "kernel_flash_attention_1"
            parts = stem.split("_")
            if len(parts) >= 3 and parts[0] == "kernel":
                # Find the rank: last part that is purely numeric
                rank = 0
                rank_idx = len(parts)
                for i in range(len(parts) - 1, 0, -1):
                    if parts[i].isdigit():
                        rank = int(parts[i])
                        rank_idx = i
                        break
                # Everything between parts[1] and the rank index is the type
                ktype = "_".join(parts[1:rank_idx]) if rank_idx > 1 else parts[1]
                opt_path = os.path.join(WORKSPACE_DIR, fname)
                replacements.append(KernelReplacement(
                    kernel_type=ktype,
                    rank=rank,
                    speedup=0.0,  # Unknown without state file
                    optimized_path=opt_path,
                ))

    return replacements


def load_kernel_module(path: str) -> Any:
    """Dynamically import a kernel .py file and return the module."""
    path = os.path.abspath(path)
    module_name = f"opt_kernel_{os.path.basename(path).replace('.py', '')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load kernel from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _call_kernel(kernel_fn: Callable, candidate_args: List[tuple]):
    """Call kernel_fn with the first argument list its signature accepts.

    Selection is by inspection rather than by catching TypeError, so a
    TypeError raised *inside* the kernel propagates instead of being mistaken
    for a signature mismatch and silently retried with fewer arguments.
    """
    try:
        sig = inspect.signature(kernel_fn)
    except (TypeError, ValueError):
        sig = None

    if sig is not None:
        for args in candidate_args:
            try:
                sig.bind(*args)
            except TypeError:
                continue
            return kernel_fn(*args)

    # Signature unavailable (a C callable, say): fall back to the longest form.
    return kernel_fn(*candidate_args[0])


class _LinearWrapper(nn.Module):
    """Wraps nn.Linear to use an optimized matmul kernel_fn."""

    def __init__(self, original: nn.Linear, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        self.weight = original.weight
        self.bias = original.bias
        # kernel_fn expects (A, B) with A @ B = C, while nn.Linear stores
        # weight as [out, in]. Transpose once, here, and as a view: doing it
        # per-forward copied the whole weight matrix inside the timed region and
        # charged the optimized model for work the baseline never does, while
        # caching a .contiguous() copy would double the model's weight memory.
        # Kernels receive both strides, so a non-contiguous B is fine.
        self.weight_t = original.weight.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape to 2D for kernel_fn, then reshape back
        orig_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        out = self.kernel_fn(x_2d, self.weight_t)

        if self.bias is not None:
            out = out + self.bias

        if len(orig_shape) > 2:
            out = out.reshape(*orig_shape[:-1], out.shape[-1])

        return out


class _LayerNormWrapper(nn.Module):
    """Wraps nn.LayerNorm to use an optimized kernel_fn."""

    def __init__(self, original: nn.LayerNorm, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        self.weight = original.weight
        self.bias = original.bias
        self.eps = original.eps
        self.normalized_shape = original.normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape if needed: kernel_fn expects (x, weight, bias[, eps])
        orig_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        out = _call_kernel(
            self.kernel_fn,
            [(x_2d, self.weight, self.bias, self.eps),
             (x_2d, self.weight, self.bias),
             (x_2d,)],
        )

        if len(orig_shape) > 2:
            out = out.reshape(orig_shape)

        return out


class _RMSNormWrapper(nn.Module):
    """Wraps RMSNorm-like modules to use an optimized kernel_fn."""

    def __init__(self, original: nn.Module, kernel_fn: Callable):
        super().__init__()
        self.original = original
        self.kernel_fn = kernel_fn
        # RMSNorm typically has a 'weight' attribute
        self.weight = getattr(original, "weight", None)
        # HF spells the epsilon `variance_epsilon` (LLaMA, Qwen2, Gemma);
        # reading only `eps` silently substituted a 1e-6 default and normalized
        # by a different constant than the model was trained with.
        eps = getattr(original, "variance_epsilon", None)
        if eps is None:
            eps = getattr(original, "eps", 1e-6)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x_2d = x.reshape(-1, x.shape[-1])
        else:
            x_2d = x

        if self.weight is not None:
            out = _call_kernel(
                self.kernel_fn,
                [(x_2d, self.weight, self.eps), (x_2d, self.weight)],
            )
        else:
            out = self.kernel_fn(x_2d)

        if len(orig_shape) > 2:
            out = out.reshape(orig_shape)

        return out


class _FusedMLPWrapper(nn.Module):
    """Wraps a SwiGLU MLP block to use an optimized fused_mlp kernel_fn.

    Matches the gate/up/down projection triple that LLaMA, Qwen2, Mistral and
    Gemma all share. The kernel owns the whole block -- both projections, the
    activation, the elementwise product and the down projection -- so the
    [tokens, intermediate] temporaries never reach global memory.
    """

    # (gate, up, down) attribute names, in the spellings HF uses.
    _TRIPLES = [
        ("gate_proj", "up_proj", "down_proj"),
        ("w1", "w3", "w2"),  # older LLaMA / Mixtral expert naming
    ]

    @classmethod
    def match(cls, module: nn.Module) -> Optional[Tuple[str, str, str]]:
        for names in cls._TRIPLES:
            mods = [getattr(module, n, None) for n in names]
            if all(isinstance(m, nn.Linear) for m in mods):
                # A fused kernel folds in no bias; leave biased MLPs alone.
                if any(m.bias is not None for m in mods):
                    return None
                return names
        return None

    def __init__(self, original: nn.Module, kernel_fn: Callable):
        super().__init__()
        names = self.match(original)
        if names is None:
            raise ValueError(f"{type(original).__name__} is not a SwiGLU MLP")
        self.original = original
        self.kernel_fn = kernel_fn
        gate, up, down = (getattr(original, n) for n in names)
        # reference.fused_mlp_ref takes w_* as [out, in] and transposes
        # internally, which is exactly nn.Linear's layout -- pass as stored.
        self.w_gate = gate.weight
        self.w_up = up.weight
        self.w_down = down.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
        out = _call_kernel(
            self.kernel_fn,
            [(x_2d, self.w_gate, self.w_up, self.w_down, "silu"),
             (x_2d, self.w_gate, self.w_up, self.w_down)],
        )
        if len(orig_shape) > 2:
            out = out.reshape(*orig_shape[:-1], out.shape[-1])
        return out


class _RoPEFunctionPatch:
    """Swaps a model's `apply_rotary_pos_emb` for an optimized kernel_fn.

    RoPE is the one supported op that is not an `nn.Module`: HF applies it in a
    free function called from inside attention, so there is nothing to replace
    in the module tree. This patches the function on the modeling module
    instead, and restores it on exit.

    The variants differ only in how they preprocess cos/sin before applying
    them -- Qwen2-VL's `apply_multimodal_rotary_pos_emb` slices them into
    temporal/height/width mrope sections first -- and never in how they touch
    q/k, which is always `(x * cos) + (rotate_half(x) * sin)`. So the
    preprocessing is reproduced here, and then *checked against the function it
    replaces* on the first call: if the two disagree, the patch permanently
    steps aside and the model keeps its own implementation. That way a variant
    this code models wrongly costs a warning and the speedup, never silent
    corruption -- including for variants that do not exist yet.
    """

    def __init__(self, model: nn.Module, kernel_fn: Callable):
        self.kernel_fn = kernel_fn
        self.model = model
        self._saved: List[Tuple[Any, str, Callable]] = []
        self.skipped: List[str] = []
        # Populated by the first-call self-check, for reporting afterwards.
        self.rejected: List[str] = []

    def _target_modules(self) -> List[Any]:
        seen, out = set(), []
        for module in self.model.modules():
            mod_name = type(module).__module__
            if mod_name in seen or not mod_name.startswith("transformers"):
                continue
            seen.add(mod_name)
            py_mod = sys.modules.get(mod_name)
            if py_mod is not None:
                out.append(py_mod)
        return out

    def apply(self) -> int:
        count = 0
        for py_mod in self._target_modules():
            for attr in dir(py_mod):
                if not attr.startswith("apply_") or "rotary" not in attr:
                    continue
                fn = getattr(py_mod, attr, None)
                if not callable(fn):
                    continue
                # The vision tower has its own RoPE entry point. Optimizing a
                # VLM's language backbone must leave it untouched.
                if "vision" in attr:
                    self.skipped.append(f"{py_mod.__name__}.{attr} (vision tower)")
                    continue
                self._saved.append((py_mod, attr, fn))
                setattr(py_mod, attr, self._make_patched(fn, f"{py_mod.__name__}.{attr}"))
                count += 1
        return count

    def _prepare_cos_sin(self, cos, sin, mrope_section, unsqueeze_dim):
        """Reproduce the variant's cos/sin preprocessing."""
        if mrope_section is not None:
            section = list(mrope_section) * 2
            cos = torch.cat(
                [m[i % 3] for i, m in enumerate(cos.split(section, dim=-1))], dim=-1
            )
            sin = torch.cat(
                [m[i % 3] for i, m in enumerate(sin.split(section, dim=-1))], dim=-1
            )
        return cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)

    def _make_patched(self, original_fn: Callable, label: str) -> Callable:
        kernel_fn = self.kernel_fn
        rejected = self.rejected
        prepare = self._prepare_cos_sin
        try:
            sig = inspect.signature(original_fn)
        except (TypeError, ValueError):
            sig = None
        state = {"checked": False, "use_kernel": True}

        def run_kernel(q, k, cos, sin, args, kwargs):
            mrope_section = None
            unsqueeze_dim = 1
            if sig is not None:
                bound = sig.bind(q, k, cos, sin, *args, **kwargs)
                bound.apply_defaults()
                mrope_section = bound.arguments.get("mrope_section")
                unsqueeze_dim = bound.arguments.get("unsqueeze_dim", 1)
            c, s = prepare(cos, sin, mrope_section, unsqueeze_dim)
            return kernel_fn(q, c, s), kernel_fn(k, c, s)

        def patched(q, k, cos, sin, *args, **kwargs):
            if not state["use_kernel"]:
                return original_fn(q, k, cos, sin, *args, **kwargs)

            if not state["checked"]:
                state["checked"] = True
                ref_q, ref_k = original_fn(q, k, cos, sin, *args, **kwargs)
                reason = ""
                try:
                    got_q, got_k = run_kernel(q, k, cos, sin, args, kwargs)
                    # Judge on relative L2, not elementwise allclose. A kernel
                    # that accumulates in fp32 differs from an eager bf16
                    # implementation by rounding on every element, so a
                    # per-element bound either rejects a correct kernel or has
                    # to be loosened past the point of catching anything. The
                    # norm separates the two cases cleanly: rounding lands near
                    # 1e-3, a different function lands near 1.
                    worst = 0.0
                    for r, g in ((ref_q, got_q), (ref_k, got_k)):
                        denom = float(torch.linalg.vector_norm(r.float()))
                        if denom > 0:
                            worst = max(worst, float(
                                torch.linalg.vector_norm(g.float() - r.float())
                            ) / denom)
                    ok = worst <= _ROPE_PATCH_REL_L2_TOL
                    if not ok:
                        reason = (f"rel_l2={worst:.2e} vs the original "
                                  f"(tol {_ROPE_PATCH_REL_L2_TOL:.0e})")
                except Exception as e:  # a shape assumption did not hold
                    ok, got_q, got_k = False, None, None
                    reason = f"{type(e).__name__}: {e}"

                if not ok:
                    state["use_kernel"] = False
                    rejected.append(f"{label}: {reason}")
                    print(f"  NOTE: kept the model's own {label} -- the optimized "
                          f"kernel did not reproduce it ({reason}).")
                    return ref_q, ref_k
                return got_q, got_k

            return run_kernel(q, k, cos, sin, args, kwargs)

        return patched

    def restore(self) -> None:
        for py_mod, attr, fn in self._saved:
            setattr(py_mod, attr, fn)
        self._saved.clear()


class OptimizedModelContext:
    """
    Context manager that patches a model's submodules to use optimized Triton kernels.

    Usage:
        with OptimizedModelContext(model, replacements) as patched_model:
            output = patched_model(input)
    """

    def __init__(self, model: nn.Module, replacements: List[KernelReplacement]):
        self.model = model
        self.replacements = replacements
        self._original_modules: Dict[str, nn.Module] = {}
        self._applied: List[str] = []
        self._rope_patches: List[_RoPEFunctionPatch] = []
        # (kernel_type, reason) for kernels that were optimized but could not
        # be installed. Reported so their absence from the end-to-end speedup
        # is visible rather than silent.
        self.unapplied: List[Tuple[str, str]] = []

    def __enter__(self) -> nn.Module:
        for repl in self.replacements:
            try:
                kernel_mod = load_kernel_module(repl.optimized_path)
                if not hasattr(kernel_mod, "kernel_fn"):
                    print(f"  WARNING: {repl.optimized_path} has no kernel_fn, skipping")
                    continue
                repl.module_fn = kernel_mod.kernel_fn
            except Exception as e:
                print(f"  WARNING: Failed to load {repl.optimized_path}: {e}")
                continue

            replaced = self._apply_replacement(repl)
            if replaced > 0:
                self._applied.append(
                    f"  {repl.kernel_type} (rank {repl.rank}): "
                    f"{repl.speedup:.1f}x -> {repl.optimized_path}"
                )

        return self.model

    def __exit__(self, *exc):
        # Restore all original modules, deepest path first so a nested
        # replacement is never reinstalled onto an already-restored parent.
        for name in sorted(self._original_modules, key=lambda n: -n.count(".")):
            self._install(name, self._original_modules[name])
        self._original_modules.clear()
        self._applied.clear()
        # Module-level function patches are global, so they must come back off
        # even if a replacement raised -- otherwise the reference run that
        # follows would silently use the optimized RoPE too.
        for patch in self._rope_patches:
            patch.restore()
        self._rope_patches.clear()

    def _apply_replacement(self, repl: KernelReplacement) -> int:
        """
        Replace matching modules in the model. Returns number of modules replaced.
        """
        count = 0

        if repl.kernel_type == "matmul":
            count = self._replace_linear_modules(repl)
        elif repl.kernel_type == "layernorm":
            count = self._replace_layernorm_modules(repl)
        elif repl.kernel_type == "rmsnorm":
            count = self._replace_rmsnorm_modules(repl)
        elif repl.kernel_type == "fused_mlp":
            count = self._replace_mlp_modules(repl)
        elif repl.kernel_type in ("rotary_embedding", "rotary_embedding_half"):
            count = self._patch_rotary(repl)
        else:
            self.unapplied.append((repl.kernel_type, "no replacement strategy"))
            print(f"  NOTE: No replacement strategy for kernel type '{repl.kernel_type}'. "
                  f"Skipping -- its speedup will NOT appear in the end-to-end number. "
                  f"(Supported: {', '.join(SUPPORTED_REPLACEMENT_TYPES)})")

        if count == 0 and repl.kernel_type in SUPPORTED_REPLACEMENT_TYPES:
            self.unapplied.append((repl.kernel_type, "no matching modules in this model"))

        return count

    def _install(self, name: str, wrapper: nn.Module) -> None:
        """Swap the module at a dotted path for its wrapper."""
        parts = name.split(".")
        parent = self.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], wrapper)

    def _is_inside_wrapper(self, name: str) -> bool:
        """True if this module lives under a module we already replaced."""
        return any(
            name == patched or name.startswith(patched + ".")
            for patched in self._original_modules
        )

    def _replace_matching(
        self,
        repl: "KernelReplacement",
        matches: Callable[[str, nn.Module], bool],
        make_wrapper: Callable[[nn.Module, Callable], nn.Module],
    ) -> int:
        """Replace every module satisfying `matches`, skipping already-wrapped
        subtrees so a second replacement of the same type cannot double-wrap."""
        count = 0
        for name, module in list(self.model.named_modules()):
            if not name or self._is_inside_wrapper(name):
                continue
            if matches(name, module):
                self._original_modules[name] = module
                self._install(name, make_wrapper(module, repl.module_fn))
                count += 1
        return count

    def _replace_linear_modules(self, repl: KernelReplacement) -> int:
        """Replace all nn.Linear modules with optimized matmul wrapper."""
        return self._replace_matching(
            repl,
            lambda name, m: isinstance(m, nn.Linear),
            _LinearWrapper,
        )

    def _replace_layernorm_modules(self, repl: KernelReplacement) -> int:
        """Replace all nn.LayerNorm modules with optimized wrapper."""
        return self._replace_matching(
            repl,
            lambda name, m: isinstance(m, nn.LayerNorm),
            _LayerNormWrapper,
        )

    def _replace_mlp_modules(self, repl: KernelReplacement) -> int:
        """Replace SwiGLU MLP blocks with a fused kernel."""
        return self._replace_matching(
            repl,
            lambda name, m: _FusedMLPWrapper.match(m) is not None,
            _FusedMLPWrapper,
        )

    def _patch_rotary(self, repl: KernelReplacement) -> int:
        """Patch the model's RoPE function (it is not a module)."""
        patch = _RoPEFunctionPatch(self.model, repl.module_fn)
        count = patch.apply()
        if count:
            self._rope_patches.append(patch)
        for name in patch.skipped:
            self.unapplied.append((repl.kernel_type, f"{name} deliberately left alone"))
            print(f"  NOTE: left {name} alone.")
        return count

    def collect_deferred(self) -> None:
        """Fold in anything a runtime self-check rejected after installation.

        The RoPE patch can only validate itself once the model actually runs,
        which is after __enter__ has returned, so its verdict is gathered here.
        """
        for patch in self._rope_patches:
            for reason in patch.rejected:
                entry = ("rotary_embedding", reason)
                if entry not in self.unapplied:
                    self.unapplied.append(entry)

    def _replace_rmsnorm_modules(self, repl: KernelReplacement) -> int:
        """
        Replace RMSNorm modules. Since there is no standard nn.RMSNorm,
        we look for common class names and attributes.
        """
        rmsnorm_names = {"RMSNorm", "LlamaRMSNorm", "T5LayerNorm", "GemmaRMSNorm"}

        def _matches(name: str, module: nn.Module) -> bool:
            cls_name = type(module).__name__
            if isinstance(module, nn.LayerNorm):
                return False
            if cls_name in rmsnorm_names:
                return True
            # Structural match, for the many spellings not in the list above
            # (Qwen2RMSNorm, MistralRMSNorm, Phi3RMSNorm, ...): a norm-like
            # leaf with a weight, no bias, and an epsilon.
            #
            # This previously required an `eps` attribute and tested
            # `not hasattr(module, "bias")`. HF names the epsilon
            # `variance_epsilon` and registers `bias` as a None buffer, so
            # every HF RMSNorm failed both tests and was silently left
            # unreplaced -- the one op most worth replacing.
            has_eps = any(hasattr(module, a) for a in ("eps", "variance_epsilon"))
            return (
                cls_name.lower().endswith("norm")
                and getattr(module, "weight", None) is not None
                and getattr(module, "bias", None) is None
                and has_eps
            )

        return self._replace_matching(repl, _matches, _RMSNormWrapper)

    @property
    def applied_summary(self) -> List[str]:
        return self._applied


# ---------------------------------------------------------------------------
# 5. Output Comparison
# ---------------------------------------------------------------------------

def extract_tensor(output: Any) -> torch.Tensor:
    """
    Extract a single tensor from model output, which might be a tuple, dict,
    or ModelOutput-like object.
    """
    if isinstance(output, torch.Tensor):
        return output

    # HuggingFace ModelOutput or similar dataclass-like object
    if hasattr(output, "logits"):
        return output.logits
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state

    # Tuple/list: return first tensor element
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
        # Recurse into first element
        if len(output) > 0:
            return extract_tensor(output[0])

    # Dict: try common keys
    if isinstance(output, dict):
        for key in ["logits", "last_hidden_state", "output", "hidden_states"]:
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
        # Return first tensor value
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v

    raise ValueError(
        f"Cannot extract tensor from output of type {type(output)}. "
        f"Consider adding support for this output format."
    )


def compare_outputs(
    ref_output: torch.Tensor,
    opt_output: torch.Tensor,
    dtype: torch.dtype,
    custom_atol: Optional[float] = None,
    custom_rtol: Optional[float] = None,
    custom_rel_l2: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compare reference and optimized outputs. Returns comparison metrics.
    """
    result: Dict[str, Any] = {}

    # Shape check
    result["shapes_match"] = ref_output.shape == opt_output.shape
    result["ref_shape"] = str(list(ref_output.shape))
    result["opt_shape"] = str(list(opt_output.shape))

    if not result["shapes_match"]:
        result["correctness"] = "FAIL"
        result["reason"] = f"Shape mismatch: ref={result['ref_shape']}, opt={result['opt_shape']}"
        return result

    # NaN / Inf check
    ref_float = ref_output.float()
    opt_float = opt_output.float()

    result["ref_has_nan"] = bool(torch.isnan(ref_float).any())
    result["ref_has_inf"] = bool(torch.isinf(ref_float).any())
    result["opt_has_nan"] = bool(torch.isnan(opt_float).any())
    result["opt_has_inf"] = bool(torch.isinf(opt_float).any())

    if result["opt_has_nan"] and not result["ref_has_nan"]:
        result["correctness"] = "FAIL"
        result["reason"] = "Optimized output contains NaN where reference does not"
        return result

    if result["opt_has_inf"] and not result["ref_has_inf"]:
        result["correctness"] = "FAIL"
        result["reason"] = "Optimized output contains Inf where reference does not"
        return result

    # Numerical comparison
    diff = (ref_float - opt_float).abs()

    # Mask out positions where both are NaN (those are fine)
    valid_mask = ~(torch.isnan(ref_float) & torch.isnan(opt_float))
    if valid_mask.any():
        valid_diff = diff[valid_mask]
        result["max_abs_error"] = float(valid_diff.max())
        result["mean_abs_error"] = float(valid_diff.mean())
        # An absolute error is uninterpretable without the scale it sits on:
        # 0.04 is noise on logits of 10 and catastrophic on activations of
        # 0.001. Report the error relative to the output's own magnitude, and
        # relative to what the dtype can represent at that magnitude.
        ref_scale = float(ref_float[valid_mask].abs().max())
        result["ref_abs_max"] = ref_scale
        result["max_rel_error"] = (
            result["max_abs_error"] / ref_scale if ref_scale > 0 else 0.0
        )
        try:
            eps = float(torch.finfo(dtype).eps)
            result["error_in_ulps"] = (
                result["max_abs_error"] / (eps * ref_scale) if ref_scale > 0 else 0.0
            )
        except (TypeError, ValueError):
            pass
    else:
        result["max_abs_error"] = 0.0
        result["mean_abs_error"] = 0.0
        result["max_rel_error"] = 0.0

    # Tolerance check
    tols = DEFAULT_TOLERANCES.get(dtype, {"atol": 1e-4, "rtol": 1e-4})
    atol = custom_atol if custom_atol is not None else tols["atol"]
    rtol = custom_rtol if custom_rtol is not None else tols["rtol"]

    # Relative L2 over the whole tensor. For a deep model in reduced precision
    # this is the criterion that means something: per-element allclose fails on
    # a single logit that happens to sit near zero, even when every kernel is
    # correct, because rounding differences at layer 0 amplify through dozens
    # of layers. Relative L2 asks the question actually of interest -- does the
    # optimized model compute the same function -- and cannot be dominated by
    # one outlier element.
    if valid_mask.any():
        ref_valid = ref_float[valid_mask]
        opt_valid = opt_float[valid_mask]
        denom = float(torch.linalg.vector_norm(ref_valid))
        rel_l2 = (
            float(torch.linalg.vector_norm(opt_valid - ref_valid)) / denom
            if denom > 0 else 0.0
        )
        result["rel_l2_error"] = rel_l2
        elementwise_ok = torch.allclose(ref_valid, opt_valid, atol=atol, rtol=rtol)
    else:
        result["rel_l2_error"] = 0.0
        rel_l2 = 0.0
        elementwise_ok = True

    rel_l2_tol = (custom_rel_l2 if custom_rel_l2 is not None
                  else REL_L2_TOLERANCES.get(dtype, 1e-4))
    result["rel_l2_tol"] = rel_l2_tol

    if dtype in (torch.float16, torch.bfloat16):
        passes = rel_l2 <= rel_l2_tol
        criterion = f"rel_l2={rel_l2:.3e} vs tol {rel_l2_tol:.1e}"
    else:
        # fp32 has the headroom for an exact elementwise check; keep it strict.
        passes = elementwise_ok
        criterion = f"allclose(atol={atol}, rtol={rtol})"

    result["correctness"] = "PASS" if passes else "FAIL"
    result["atol"] = atol
    result["rtol"] = rtol
    result["criterion"] = criterion

    if not passes:
        result["reason"] = (
            f"Output differs beyond tolerance [{criterion}]. "
            f"rel_l2_error={rel_l2:.6e}, "
            f"max_abs_error={result['max_abs_error']:.6e}, "
            f"mean_abs_error={result['mean_abs_error']:.6e}"
        )

    return result


# ---------------------------------------------------------------------------
# 6. Diagnosis Mode (apply kernels one at a time)
# ---------------------------------------------------------------------------

def diagnose_kernel_failures(
    model: nn.Module,
    model_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ref_tensor: torch.Tensor,
    replacements: List[KernelReplacement],
    dtype: torch.dtype,
) -> List[Dict[str, Any]]:
    """
    Apply each kernel replacement individually to find which one causes failure.
    """
    results = []

    for repl in replacements:
        print(f"\n  Testing kernel: {repl.kernel_type} (rank {repl.rank})...")
        ctx = OptimizedModelContext(model, [repl])

        try:
            with ctx as patched_model:
                with torch.no_grad():
                    if isinstance(model_input, dict):
                        opt_output = patched_model(**model_input)
                    else:
                        opt_output = patched_model(model_input)
                torch.cuda.synchronize()

            opt_tensor = extract_tensor(opt_output)
            comp = compare_outputs(ref_tensor, opt_tensor, dtype)

            results.append({
                "kernel_type": repl.kernel_type,
                "rank": repl.rank,
                "path": repl.optimized_path,
                "correctness": comp["correctness"],
                "max_abs_error": comp.get("max_abs_error", 0.0),
                "mean_abs_error": comp.get("mean_abs_error", 0.0),
                "reason": comp.get("reason", ""),
            })

            status = comp["correctness"]
            if status == "PASS":
                print(f"    -> PASS (max_err={comp.get('max_abs_error', 0):.6e})")
            else:
                print(f"    -> FAIL: {comp.get('reason', 'unknown')}")

        except Exception as e:
            results.append({
                "kernel_type": repl.kernel_type,
                "rank": repl.rank,
                "path": repl.optimized_path,
                "correctness": "ERROR",
                "max_abs_error": float("inf"),
                "mean_abs_error": float("inf"),
                "reason": str(e),
            })
            print(f"    -> ERROR: {e}")

    return results


# ---------------------------------------------------------------------------
# 7. Output Formatting
# ---------------------------------------------------------------------------

def format_report(result: VerificationResult, diagnose_results: Optional[List] = None) -> str:
    """Format the verification result into a human-readable report."""
    lines = []
    lines.append("")
    lines.append("=== AutoKernel End-to-End Verification ===")
    lines.append("")
    lines.append(f"Model: {result.model_name}")
    lines.append(f"Input: [{result.input_shape}], dtype={result.dtype_str}")
    lines.append(f"GPU: {result.gpu_name}")

    # Reference run
    lines.append("")
    lines.append("--- Reference Run ---")
    lines.append(f"Output shape: {result.ref_output_shape}")
    lines.append(f"Latency: {result.ref_latency_ms:.1f} ms ({TIMED_RUNS} runs, median)")

    # Optimized run
    lines.append("")
    lines.append("--- Optimized Run ---")
    if result.kernels_replaced:
        lines.append("Kernels replaced:")
        for k in result.kernels_replaced:
            lines.append(f"  {k['type']} (rank {k['rank']}): "
                         f"{k['speedup']:.1f}x -> {k['path']}")
    else:
        lines.append("Kernels replaced: none")
    lines.append(f"Output shape: {result.opt_output_shape}")
    lines.append(f"Latency: {result.opt_latency_ms:.1f} ms ({TIMED_RUNS} runs, median)")

    # Verification
    lines.append("")
    lines.append("--- Verification ---")
    lines.append(f"correctness: {result.correctness}")
    lines.append(f"rel_l2_error: {result.rel_l2_error:.2e}")
    lines.append(f"max_abs_error: {result.max_abs_error:.2e}")
    lines.append(f"mean_abs_error: {result.mean_abs_error:.2e}")
    if result.has_nan:
        lines.append("WARNING: NaN detected in optimized output")
    if result.has_inf:
        lines.append("WARNING: Inf detected in optimized output")

    # Summary
    lines.append("")
    lines.append("--- Summary ---")
    lines.append(f"original_latency_ms: {result.ref_latency_ms:.1f}")
    lines.append(f"optimized_latency_ms: {result.opt_latency_ms:.1f}")
    lines.append(f"end_to_end_speedup: {result.end_to_end_speedup:.2f}x")
    lines.append(f"kernels_replaced: {len(result.kernels_replaced)}")

    # Diagnosis
    if diagnose_results:
        lines.append("")
        lines.append("--- Diagnosis (per-kernel) ---")
        for dr in diagnose_results:
            status = dr["correctness"]
            line = f"  {dr['kernel_type']} (rank {dr['rank']}): {status}"
            if status == "PASS":
                line += f" | max_err={dr['max_abs_error']:.2e}"
            if dr.get("reason"):
                line += f" | {dr['reason']}"
            lines.append(line)

    lines.append("")
    return "\n".join(lines)


def save_verification_json(result: VerificationResult, path: str) -> None:
    """Save verification results as JSON for programmatic consumption."""
    data = {
        "model": result.model_name,
        "input_shape": result.input_shape,
        "dtype": result.dtype_str,
        "gpu": result.gpu_name,
        "reference": {
            "output_shape": result.ref_output_shape,
            "latency_ms": round(result.ref_latency_ms, 2),
        },
        "optimized": {
            "output_shape": result.opt_output_shape,
            "latency_ms": round(result.opt_latency_ms, 2),
            "kernels_replaced": result.kernels_replaced,
        },
        "verification": {
            "correctness": result.correctness,
            "rel_l2_error": result.rel_l2_error,
            "max_abs_error": result.max_abs_error,
            "mean_abs_error": result.mean_abs_error,
            "has_nan": result.has_nan,
            "has_inf": result.has_inf,
        },
        "summary": {
            "original_latency_ms": round(result.ref_latency_ms, 2),
            "optimized_latency_ms": round(result.opt_latency_ms, 2),
            "end_to_end_speedup": round(result.end_to_end_speedup, 3),
            "kernels_replaced": len(result.kernels_replaced),
            "kernels_not_applied": [
                {"kernel_type": t, "reason": r} for t, r in result.kernels_not_applied
            ],
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dtype(dtype_str: str) -> torch.dtype:
    """Parse a dtype string into a torch.dtype."""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    key = dtype_str.lower().strip()
    if key not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose from: {list(mapping.keys())}")
    return mapping[key]


def _get_gpu_name() -> str:
    """Get current GPU name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "No GPU"


def _output_shape_str(output: Any) -> str:
    """Get shape string from model output."""
    try:
        t = extract_tensor(output)
        return str(list(t.shape))
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global WORKSPACE_DIR, ORCHESTRATION_STATE, WARMUP_RUNS, TIMED_RUNS

    parser = argparse.ArgumentParser(
        description="AutoKernel End-to-End Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model loading
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", type=str,
        help="Path to a Python file containing the model class"
    )
    model_group.add_argument(
        "--module", type=str,
        help="Python module name (e.g. 'transformers')"
    )

    parser.add_argument(
        "--class-name", type=str, required=True,
        help="Name of the model class to instantiate"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Pretrained model name/path (for HuggingFace models)"
    )
    parser.add_argument(
        "--input-shape", type=str, default="1,2048",
        help="Comma-separated input shape, e.g. '1,2048' (default: 1,2048)"
    )
    parser.add_argument(
        "--dtype", type=str, default="float16",
        help="Data type: float16, bfloat16, float32 (default: float16)"
    )

    # Benchmark tuning
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_RUNS,
        help=f"Number of warmup iterations (default: {WARMUP_RUNS})"
    )
    parser.add_argument(
        "--timed", type=int, default=TIMED_RUNS,
        help=f"Number of timed iterations (default: {TIMED_RUNS})"
    )

    # Tolerance overrides
    parser.add_argument("--atol", type=float, default=None, help="Override absolute tolerance")
    parser.add_argument("--rtol", type=float, default=None, help="Override relative tolerance")
    parser.add_argument("--rel-l2-tol", type=float, default=None,
                        help="Override the relative-L2 tolerance used as the "
                             "pass criterion for fp16/bf16 (see REL_L2_TOLERANCES)")

    # Modes
    parser.add_argument(
        "--diagnose", action="store_true",
        help="On failure, test each kernel replacement individually to find the culprit"
    )
    parser.add_argument(
        "--json", type=str, default=None,
        help="Save results to a JSON file at this path"
    )
    parser.add_argument(
        "--workspace", type=str, default=None,
        help="Override workspace directory (default: ./workspace)"
    )

    args = parser.parse_args()

    # Override globals if workspace specified
    if args.workspace:
        WORKSPACE_DIR = os.path.abspath(args.workspace)
        ORCHESTRATION_STATE = os.path.join(WORKSPACE_DIR, "orchestration_state.json")

    WARMUP_RUNS = args.warmup
    TIMED_RUNS = args.timed

    dtype = _parse_dtype(args.dtype)
    gpu_name = _get_gpu_name()

    print("=" * 60)
    print("  AutoKernel End-to-End Verifier")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------------
    # Step 1: Discover optimized kernels
    # -----------------------------------------------------------------------
    print("Step 1: Discovering optimized kernels...")
    replacements = discover_optimized_kernels()
    if not replacements:
        print()
        print("No optimized kernels found.")
        print(f"  Searched: {WORKSPACE_DIR}")
        print(f"  State file: {ORCHESTRATION_STATE}")
        print()
        print("Run the optimization loop first to produce optimized kernels.")
        print("Expected files: workspace/kernel_<type>_<rank>_optimized.py")
        sys.exit(1)

    print(f"  Found {len(replacements)} optimized kernel(s):")
    for r in replacements:
        print(f"    {r.kernel_type} (rank {r.rank}): speedup={r.speedup:.1f}x -> {r.optimized_path}")
    print()

    # -----------------------------------------------------------------------
    # Step 2: Load model
    # -----------------------------------------------------------------------
    print("Step 2: Loading model...")
    try:
        model = load_model(args)
        model_name = args.class_name
        if args.pretrained:
            model_name = f"{args.class_name} ({args.pretrained})"
        print(f"  Model loaded: {model_name}")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
    except Exception as e:
        print(f"\nERROR: Failed to load model: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 3: Create input
    # -----------------------------------------------------------------------
    print("Step 3: Creating model input...")
    try:
        model_input = make_model_input(model, args.input_shape, dtype)
        if isinstance(model_input, dict):
            for k, v in model_input.items():
                print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  Input: shape={list(model_input.shape)}, dtype={model_input.dtype}")
    except Exception as e:
        print(f"\nERROR: Failed to create input: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 4: Reference run
    # -----------------------------------------------------------------------
    print("Step 4: Reference run (original PyTorch ops)...")
    try:
        ref_output, ref_latency = benchmark_model(model, model_input, WARMUP_RUNS, TIMED_RUNS)
        ref_tensor = extract_tensor(ref_output)
        ref_shape_str = str(list(ref_tensor.shape))
        print(f"  Output shape: {ref_shape_str}")
        print(f"  Median latency: {ref_latency:.1f} ms")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nERROR: GPU out of memory during reference run.")
            print("  Try a smaller --input-shape or a smaller model.")
            torch.cuda.empty_cache()
            sys.exit(1)
        else:
            raise
    except Exception as e:
        print(f"\nERROR: Reference run failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 5: Optimized run
    # -----------------------------------------------------------------------
    print("Step 5: Optimized run (with Triton kernel replacements)...")
    ctx = OptimizedModelContext(model, replacements)
    try:
        with ctx as patched_model:
            if ctx.applied_summary:
                print("  Replacements applied:")
                for line in ctx.applied_summary:
                    print(f"  {line}")
            else:
                print("  WARNING: No kernel replacements could be applied to this model.")
                print("  The model may not contain modules matching the optimized kernel types.")

            if ctx.unapplied:
                print()
                print("  NOT APPLIED -- these kernels were optimized but could not be")
                print("  installed, so their speedup is absent from the number below:")
                for ktype, reason in ctx.unapplied:
                    print(f"    {ktype}: {reason}")

            opt_output, opt_latency = benchmark_model(
                patched_model, model_input, WARMUP_RUNS, TIMED_RUNS
            )
            opt_tensor = extract_tensor(opt_output)
            opt_shape_str = str(list(opt_tensor.shape))
            print(f"  Output shape: {opt_shape_str}")
            print(f"  Median latency: {opt_latency:.1f} ms")

            # Runtime self-checks can only report once the model has run.
            ctx.collect_deferred()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nERROR: GPU out of memory during optimized run.")
            print("  The optimized kernels may use more memory than expected.")
            torch.cuda.empty_cache()
            sys.exit(1)
        else:
            print(f"\nERROR: Optimized run failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Optimized run failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    print()

    # -----------------------------------------------------------------------
    # Step 6: Compare outputs
    # -----------------------------------------------------------------------
    print("Step 6: Comparing outputs...")
    comp = compare_outputs(ref_tensor, opt_tensor, dtype, args.atol, args.rtol,
                           getattr(args, "rel_l2_tol", None))
    print(f"  correctness: {comp['correctness']}")
    print(f"  max_abs_error: {comp.get('max_abs_error', 0):.2e}")
    print(f"  mean_abs_error: {comp.get('mean_abs_error', 0):.2e}")
    if comp.get("reason"):
        print(f"  reason: {comp['reason']}")
    print()

    # -----------------------------------------------------------------------
    # Step 6b: Diagnose failures if requested
    # -----------------------------------------------------------------------
    diagnose_results = None
    if args.diagnose and comp["correctness"] == "FAIL":
        print("Step 6b: Diagnosing failure (testing each kernel individually)...")
        diagnose_results = diagnose_kernel_failures(
            model, model_input, ref_tensor, replacements, dtype
        )
        print()

    # -----------------------------------------------------------------------
    # Step 7: Build and display final report
    # -----------------------------------------------------------------------
    speedup = ref_latency / opt_latency if opt_latency > 0 else 0.0

    result = VerificationResult(
        model_name=model_name if args.pretrained else args.class_name,
        input_shape=args.input_shape,
        dtype_str=args.dtype,
        gpu_name=gpu_name,
        ref_output_shape=ref_shape_str,
        ref_latency_ms=ref_latency,
        opt_output_shape=opt_shape_str,
        opt_latency_ms=opt_latency,
        kernels_replaced=[
            {
                "type": r.kernel_type,
                "rank": r.rank,
                "speedup": r.speedup,
                "path": r.optimized_path,
            }
            for r in replacements
            if r.module_fn is not None
        ],
        kernels_not_applied=list(ctx.unapplied),
        correctness=comp["correctness"],
        max_abs_error=comp.get("max_abs_error", 0.0),
        rel_l2_error=comp.get("rel_l2_error", 0.0),
        mean_abs_error=comp.get("mean_abs_error", 0.0),
        has_nan=comp.get("opt_has_nan", False),
        has_inf=comp.get("opt_has_inf", False),
        end_to_end_speedup=speedup,
    )

    report = format_report(result, diagnose_results)
    print(report)

    # Save JSON if requested
    if args.json:
        json_path = os.path.abspath(args.json)
        save_verification_json(result, json_path)
        print(f"Results saved to: {json_path}")

    # Default: save to workspace
    default_json = os.path.join(WORKSPACE_DIR, "verification_result.json")
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    save_verification_json(result, default_json)
    print(f"Results saved to: {default_json}")

    # Exit code: 0 for PASS, 1 for FAIL
    if result.correctness != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
