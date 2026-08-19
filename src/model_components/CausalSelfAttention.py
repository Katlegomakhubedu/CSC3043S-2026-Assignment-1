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

    The KV cache is allocated lazily on the first use_cache=True call and sized
    to that call's batch, so the rest of a generate() run must keep the same
    batch size. Call reset_cache() before a new sequence or batch size.
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
        # Positions come from the cache only on the cached path: an uncached
        # call is a fresh pass from position 0, and reading cache_len here would
        # leak a previous cached call's state into this call's RoPE angles.
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
            # Allocated on the first call of a generation run, sized to this
            # call's batch (which must then stay fixed - see the class docstring).
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

            # Append this step's K and V, then attend over everything cached.
            self.k_cache[:, :, self.cache_len:self.cache_len + seq_len] = k
            self.v_cache[:, :, self.cache_len:self.cache_len + seq_len] = v
            k_full = self.k_cache[:, :, :self.cache_len + seq_len]
            v_full = self.v_cache[:, :, :self.cache_len + seq_len]
            self.cache_len += seq_len
        else:
            k_full, v_full = k, v

        is_causal = seq_len > 1
        out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)
