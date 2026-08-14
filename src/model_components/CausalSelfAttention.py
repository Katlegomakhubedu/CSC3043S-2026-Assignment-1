import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .RMSNorm import RMSNorm
from .RoPE import RotaryPositionalEmbedding

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Scaled dot-product attention.
    params:
        q:    (..., n_queries, d_k)
        k:    (..., n_keys, d_k)
        v:    (..., n_keys, d_v)
        mask: optional boolean (n_queries, n_keys); True = attend, False = do not attend
    returns:
        out:  (..., n_queries, d_v)
    """
    d_k = q.size(-1)

    # Step 1: scores e_ij = q_i . k_j / sqrt(d_k)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)        # (..., n_queries, n_keys)

    # Step 2: block the disallowed positions with -inf so softmax gives them zero weight
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # Step 3: normalise over the KEY dimension, then take the weighted average of values
    attn = F.softmax(scores, dim=-1)                         # (..., n_queries, n_keys)
    return attn @ v
class CausalSelfAttention(nn.Module):
    """
    Causal multi-head self-attention with QK norm and RoPE.
    params:
        d_model:     model dimension
        n_heads:     number of attention heads
        rope:        a shared RotaryPositionalEmbedding module
        use_qk_norm: whether to RMSNorm the queries and keys before attention

    KV cache: buffers are allocated lazily on the first call that passes
    use_cache=True, sized to the batch of THAT call. All subsequent cached
    calls (the rest of one generate() run) must use the same batch size;
    call reset_cache() (or just let a fresh forward with use_cache=False
    fall through) before starting a new sequence / a different batch size.
    """

    def __init__(self, d_model, n_heads, context_length, rope, use_rope=True, use_qk_norm=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.context_length = context_length
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = rope
        self.q_norm = RMSNorm(self.d_head) if use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.d_head) if use_qk_norm else nn.Identity()

        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0

    def reset_cache(self):
        """Drop any cached K/V so the next use_cache=True call starts a fresh prefill."""
        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0

    def forward(self, x, use_cache=False):
        """
        params:
            x: (batch, seq_len, d_model)
        returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        # Positions come from the cache only on the cached path. On the non-cached
        # path every call is a fresh full-sequence pass starting at position 0 -
        # using self.cache_len here too would leak whatever cache state a *previous*
        # use_cache=True call on this module left behind into this call's RoPE angles.
        if use_cache:
            positions = torch.arange(self.cache_len, self.cache_len + seq_len, device=x.device)
        else:
            positions = torch.arange(seq_len, device=x.device)

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rope:
            q, k = self.rope(q, positions), self.rope(k, positions)

        if use_cache:
            # Allocate the cache on the first call of a generation run, sized to
            # this call's batch (see class docstring: batch size must stay fixed
            # for the lifetime of the cache).
            if self.k_cache is None:
                self.k_cache = torch.zeros(batch, self.n_heads, self.context_length, self.d_head, device=x.device, dtype=k.dtype)
                self.v_cache = torch.zeros_like(self.k_cache)
                self.cache_len = 0

            if self.cache_len + seq_len > self.context_length:
                raise RuntimeError(
                    f"KV cache overflow: {self.cache_len} cached + {seq_len} new tokens "
                    f"exceeds context_length={self.context_length}. Stop generation before "
                    f"this point (see generate()'s context-length guard)."
                )

            # Store current K, V
            self.k_cache[:, :, self.cache_len:self.cache_len + seq_len] = k
            self.v_cache[:, :, self.cache_len:self.cache_len + seq_len] = v
            k_full = self.k_cache[:, :, :self.cache_len + seq_len]
            v_full = self.v_cache[:, :, :self.cache_len + seq_len]
            self.cache_len += seq_len
        else:
            k_full, v_full = k, v
            # Deliberately NOT touching self.cache_len here. It used to be reset
            # to 0 on every non-cached call, which corrupts an in-progress cache:
            # e.g. one unrelated use_cache=False forward pass (a training step, an
            # eval batch, ...) interleaved with a generate() run would zero
            # cache_len, causing the next cached call to overwrite the cache from
            # position 0 instead of appending, and to use the wrong RoPE
            # positions. Cache state is only ever touched by the cached path and
            # by reset_cache().

        # A causal mask is needed whenever more than one query position is being
        # scored at once - that is true both for an ordinary uncached forward pass
        # over a sequence AND for a cached *prefill* step (several prompt tokens
        # attending to each other for the first time). It is NOT needed for a
        # single-token cached decode step (seq_len == 1): that one query is free
        # to attend to everything already in the cache. Gating this on `use_cache`
        # (i.e. skipping the mask whenever caching is on) silently attends to
        # future prompt tokens during prefill - that was the bug here.
        is_causal = seq_len > 1
        out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)
