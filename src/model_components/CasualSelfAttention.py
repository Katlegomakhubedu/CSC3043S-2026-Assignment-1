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
    """

    def __init__(self, d_model, n_heads, rope, use_qk_norm=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads          # d_q = d_k = d_v = d_model / n_heads

        # One full-width projection each; the heads are carved out by reshaping.
        # No bias terms, following modern LLMs.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = rope
        self.q_norm = RMSNorm(self.d_head) if use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.d_head) if use_qk_norm else nn.Identity()

    def forward(self, x):
        """
        params:
            x: (batch, seq_len, d_model)
        returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device)

        # Step 1: project, then split into heads -> (batch, n_heads, seq_len, d_head)
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Step 2: QK norm, then RoPE on queries and keys only (never on values)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q, positions), self.rope(k, positions)

        # Step 3: causal mask - query i attends to key j only if j <= i
        mask = positions[None, :] <= positions[:, None]              # (seq_len, seq_len)

        out = scaled_dot_product_attention(q, k, v, mask)            # (b, n_heads, seq_len, d_head)

        # Step 4: concatenate the heads and project back to the residual stream
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)