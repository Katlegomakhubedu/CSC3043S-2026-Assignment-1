import torch
import torch.nn as nn
from .CasualSelfAttention import CausalSelfAttention
from .RMSNorm import RMSNorm
from .SwiGLU import SwiGLU

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block: attention sub-layer then feed-forward sub-layer.
    params:
        d_model, n_heads, d_ff: sizes
        rope:        shared RotaryPositionalEmbedding module
        use_qk_norm: passed through to the attention module
    """

    def __init__(self, d_model, n_heads, d_ff, rope, use_qk_norm=True):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rope, use_qk_norm)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        """
        params:
            x: (batch, seq_len, d_model)
        returns:
            (batch, seq_len, d_model)
        """
        # Sub-layer 1: normalise, attend, add the residual.
        x = x + self.attn(self.attn_norm(x))
        # Sub-layer 2: normalise, feed-forward, add the residual.
        x = x + self.ffn(self.ffn_norm(x))
        return x