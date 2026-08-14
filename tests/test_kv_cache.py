"""
KV-cache correctness, per assignment §4.2's required check:
"with a fixed seed and greedy decoding, cached and uncached generation must
produce identical token sequences for a 100-token continuation, and the
logits must agree to within floating-point tolerance."

Also covers the specific bugs this cache implementation had:
 - no causal mask applied during multi-token prefill
 - stale self.cache_len leaking into the non-cached path's RoPE positions
 - the decode loop re-feeding the last prompt token (duplicating it in the
   cache under the wrong position) instead of reusing the prefill's own
   last-position logits
 - the cache being hardcoded to batch size 1
"""
import torch
import pytest

from model import TransformerLM, TransformerConfig
from tokenizer import BPETokenizer
from generate import generate


# A byte-level vocab (<|endoftext|> + 256 byte values) has exactly 257 IDs;
# the model's vocab_size must match so every ID the model can emit is decodable.
BYTE_VOCAB_SIZE = 257


def make_model(context_length=64, seed=0, vocab_size=BYTE_VOCAB_SIZE):
    torch.manual_seed(seed)
    cfg = TransformerConfig(vocab_size=vocab_size, context_length=context_length,
                             n_layers=2, d_model=16, n_heads=4, d_ff=32)
    model = TransformerLM(cfg)
    model.eval()
    return model


def make_byte_tokenizer():
    vocab = {0: b"<|endoftext|>"}
    for b in range(256):
        vocab[1 + b] = bytes([b])
    return BPETokenizer(vocab, [], special_tokens=["<|endoftext|>"])


# ---------------------------------------------------------------------------
# Low-level: cached prefill vs. an ordinary uncached forward pass
# ---------------------------------------------------------------------------

def test_prefill_logits_match_uncached_forward():
    model = make_model()
    x = torch.randint(0, BYTE_VOCAB_SIZE, (1, 20))

    with torch.no_grad():
        logits_uncached = model(x, use_cache=False)
        for layer in model.layers:
            layer.attn.reset_cache()
        logits_cached = model(x, use_cache=True)

    max_abs_diff = (logits_uncached - logits_cached).abs().max().item()
    assert max_abs_diff < 1e-4, f"prefill vs uncached logits diverge: max abs diff {max_abs_diff}"


def test_incremental_decode_matches_full_recompute_at_every_step():
    """Feed one token at a time through the cache and check each step's
    logits against a from-scratch forward pass over the whole prefix so far."""
    model = make_model()
    full_seq = torch.randint(0, BYTE_VOCAB_SIZE, (1, 15))

    for layer in model.layers:
        layer.attn.reset_cache()

    with torch.no_grad():
        # Prefill on the first token.
        cached_logits = model(full_seq[:, :1], use_cache=True)
        uncached_logits = model(full_seq[:, :1], use_cache=False)
        assert (cached_logits - uncached_logits).abs().max().item() < 1e-4

        for t in range(1, full_seq.shape[1]):
            cached_logits = model(full_seq[:, t:t + 1], use_cache=True)
            uncached_logits = model(full_seq[:, :t + 1], use_cache=False)
            diff = (cached_logits[:, -1, :] - uncached_logits[:, -1, :]).abs().max().item()
            assert diff < 1e-4, f"step {t}: cached vs uncached logits diverge (max abs diff {diff})"


def test_cache_supports_batch_greater_than_one():
    # The cache used to hardcode batch=1 when allocating its buffers.
    model = make_model()
    x = torch.randint(0, BYTE_VOCAB_SIZE, (4, 10))
    with torch.no_grad():
        logits_uncached = model(x, use_cache=False)
        for layer in model.layers:
            layer.attn.reset_cache()
        logits_cached = model(x, use_cache=True)
    assert logits_cached.shape == (4, 10, BYTE_VOCAB_SIZE)
    assert (logits_uncached - logits_cached).abs().max().item() < 1e-4


# ---------------------------------------------------------------------------
# End-to-end: generate() with and without the cache
# ---------------------------------------------------------------------------

def test_generate_greedy_cached_equals_uncached():
    model = make_model(context_length=128)
    tokenizer = make_byte_tokenizer()

    out_cached = generate(model, tokenizer, "The quick brown fox", max_new_tokens=100,
                           temperature=0.0, use_cache=True)
    out_uncached = generate(model, tokenizer, "The quick brown fox", max_new_tokens=100,
                             temperature=0.0, use_cache=False)
    assert out_cached == out_uncached


def test_generate_sampling_is_reproducible_with_a_fixed_seed():
    model = make_model(context_length=64)
    tokenizer = make_byte_tokenizer()
    a = generate(model, tokenizer, "once upon a time", max_new_tokens=30,
                 temperature=1.0, top_k=20, top_p=0.9, seed=123, use_cache=True)
    b = generate(model, tokenizer, "once upon a time", max_new_tokens=30,
                 temperature=1.0, top_k=20, top_p=0.9, seed=123, use_cache=True)
    assert a == b


def test_generate_stops_cleanly_at_context_length_instead_of_crashing():
    context_length = 24
    model = make_model(context_length=context_length)
    tokenizer = make_byte_tokenizer()
    prompt = "x" * (context_length - 2)
    out = generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.0, use_cache=True)
    # Must return (not raise) even though max_new_tokens would overflow the cache.
    assert len(out) >= len(prompt)


def test_generate_rejects_empty_prompt():
    model = make_model()
    tokenizer = make_byte_tokenizer()
    with pytest.raises(ValueError):
        generate(model, tokenizer, "", max_new_tokens=5, use_cache=True)
