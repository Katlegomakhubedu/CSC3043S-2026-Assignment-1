"""Checks on the Task 2 question script.

The one thing here that is easy to get quietly wrong is Q7's denominator.
`generate(max_new_tokens=N)` does not always produce N tokens: the cached path
decodes into a fixed-size buffer of `context_length` and stops on reaching it
(§4.2 permits this), while the uncached path re-slices a sliding window and has
no such limit. Dividing both by N therefore compares two different amounts of
work. `tokens_actually_generated` computes the real count; these tests check it
against generation with a byte-level tokenizer, whose decode/encode round-trip
is exact, so the token count can be recovered from the output string.
"""
import pytest
import torch

from src.model import TransformerLM, TransformerConfig
from src.tokenizer import BPETokenizer
from src.generate import generate
from scripts.run_task2_questions import (tokens_actually_generated,
                                          parameter_breakdown, RELU_D_FF)


def make_byte_tokenizer():
    vocab = {0: b"<|endoftext|>"}
    for b in range(256):
        vocab[1 + b] = bytes([b])
    return BPETokenizer(vocab, [], special_tokens=["<|endoftext|>"])


def make_model(context_length):
    torch.manual_seed(0)
    cfg = TransformerConfig(vocab_size=257, context_length=context_length,
                            n_layers=2, d_model=16, n_heads=4, d_ff=32)
    model = TransformerLM(cfg)
    model.eval()
    return model


@pytest.mark.parametrize("context_length", [32, 64])
@pytest.mark.parametrize("requested", [8, 32, 128])
@pytest.mark.parametrize("use_cache", [True, False])
def test_token_count_formula_matches_real_generation(context_length, requested, use_cache):
    model = make_model(context_length)
    tokenizer = make_byte_tokenizer()
    prompt = "abcdef"
    n_prompt = len(tokenizer.encode(prompt))

    out = generate(model, tokenizer, prompt, max_new_tokens=requested,
                   temperature=0.0, use_cache=use_cache)
    assert "<|endoftext|>" not in out, "early stop would invalidate this comparison"

    actual = len(tokenizer.encode(out)) - n_prompt
    predicted = tokens_actually_generated(n_prompt, requested, context_length, use_cache)
    assert actual == predicted


def test_uncached_path_is_not_capped_by_context_length():
    # The uncached path slides a window, so it keeps generating past the context.
    assert tokens_actually_generated(4, 256, 256, use_cache=False) == 256


def test_cached_path_stops_at_context_length():
    # 4 prompt tokens fill 4 cache slots; the prefill's own logits give one
    # token without consuming a slot, so 256 - 4 + 1 = 253.
    assert tokens_actually_generated(4, 256, 256, use_cache=True) == 253
    # Below the cap, nothing is truncated.
    assert tokens_actually_generated(4, 128, 256, use_cache=True) == 128


def test_q5_relu_variant_is_parameter_matched_at_the_assignment_dims():
    """§7.2's claim, on the real §4.1 dimensions: 1344 SwiGLU vs 2048 ReLU
    should land within 2% on total parameters."""
    base = dict(vocab_size=4000, context_length=256, n_layers=4,
                d_model=512, n_heads=8)
    swiglu = TransformerLM(TransformerConfig(**base, d_ff=1344, ffn_type="swiglu"))
    relu = TransformerLM(TransformerConfig(**base, d_ff=RELU_D_FF, ffn_type="relu"))

    swiglu_total = parameter_breakdown(swiglu)["total"]
    relu_total = parameter_breakdown(relu)["total"]
    assert abs(relu_total - swiglu_total) / swiglu_total < 0.02


def test_parameter_breakdown_sums_to_total():
    model = make_model(32)
    counts = parameter_breakdown(model)
    assert counts["embedding"] + counts["lm_head"] + counts["non_embedding"] == counts["total"]
    assert counts["total"] == sum(p.numel() for p in model.parameters())
