"""Tiny stand-ins for olmOCR-2's language backbone, for pipeline smoke tests.

olmOCR-2 is a Qwen2.5-VL derivative, so its language tower is Qwen2
architecture: GQA attention, SwiGLU MLP, RMSNorm, and a very wide lm_head.
These models keep that op mix and the real head/GQA/vocab structure but
shrink the hidden size, so a profiling run finishes in seconds on random
weights with nothing to download.

The point is to exercise the *real* torch.profiler output format -- which is
what the shape parser consumes -- not to measure anything meaningful.

Two variants, because the attention backends record their q/k/v with
different axis orders and the parser has to handle both:

    TinyQwenFA2   -> aten::_flash_attention_forward, laid out [B, S, H, D]
    TinyQwenSDPA  -> aten::scaled_dot_product_attention, laid out [B, H, S, D]

Usage:
    uv run profile_model.py --model tests/tiny_qwen.py \
        --class-name TinyQwenFA2 --input-shape 1,2048 --dtype bfloat16
"""

from transformers import Qwen2Config, Qwen2ForCausalLM


def _tiny_config(attn_implementation: str) -> Qwen2Config:
    return Qwen2Config(
        # Shrunk so the model is a few hundred MB of random weights.
        hidden_size=896,
        intermediate_size=4864,
        num_hidden_layers=2,
        # Head count, GQA ratio and head_dim are kept realistic: they decide
        # the attention shape the parser has to read back.
        num_attention_heads=14,
        num_key_value_heads=2,
        # Kept at the real Qwen2.5 vocab so the lm_head GEMM is exercised at
        # its true width, which is where the M=2048,N=152064 row comes from.
        vocab_size=151936,
        max_position_embeddings=4096,
        attn_implementation=attn_implementation,
    )


class TinyQwenFA2(Qwen2ForCausalLM):
    """Flash-Attention-2 backend -> aten::_flash_attention_forward ([B, S, H, D])."""

    def __init__(self) -> None:
        super().__init__(_tiny_config("flash_attention_2"))


class TinyQwenSDPA(Qwen2ForCausalLM):
    """SDPA backend -> aten::scaled_dot_product_attention ([B, H, S, D])."""

    def __init__(self) -> None:
        super().__init__(_tiny_config("sdpa"))
