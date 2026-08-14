import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    Position-wise SwiGLU feed-forward network (no bias terms).
    params:
        d_model: model dimension (input and output size)
        d_ff:    inner hidden dimension, canonically about (8/3) * d_model
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)      # gate branch
        self.w3 = nn.Linear(d_model, d_ff, bias=False)      # value branch
        self.w2 = nn.Linear(d_ff, d_model, bias=False)      # projection back to d_model

    def forward(self, x):
        """
        params:
            x: (..., d_model)
        returns:
            tensor of shape (..., d_model)
        """
        # SiLU(W1 x) is the gate; it multiplies W3 x element-wise; W2 projects back down.
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def compute_d_ff(d_model, multiple_of=64):
    """Round (8/3) * d_model to the NEAREST multiple of `multiple_of`.

    (Not ceiling: rounding up gives 1408 for d_model=512, but the
    assignment's reference value is 1344 = round(1365.33 / 64) * 64,
    i.e. nearest, not up.)
    """
    d_ff = 8 * d_model / 3
    return multiple_of * round(d_ff / multiple_of)


def relu_ffn_hidden_dim(d_ff_swiglu):
    """
    Hidden dim for a 2-matrix ReLU FFN that matches a 3-matrix SwiGLU FFN's
    parameter count at the same d_model.

    SwiGLU has 3 (d_model x d_ff) matrices -> 3 * d_model * d_ff params.
    A plain ReLU FFN has 2 -> 2 * d_model * d_ff_relu params.
    Equal params requires d_ff_relu = 1.5 * d_ff_swiglu exactly.
    """
    return round(1.5 * d_ff_swiglu)


class ReLUFFN(nn.Module):
    """Plain (non-gated) ReLU feed-forward. Use `relu_ffn_hidden_dim` to size
    `d_ff` so this has the same parameter count as a SwiGLU FFN, for the
    §7.2 ablation."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))