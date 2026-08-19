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
        # SiLU(W1 x) gates W3 x element-wise; W2 projects back to d_model.
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def compute_d_ff(d_model, multiple_of=64):
    """Round (8/3) * d_model to the nearest multiple of `multiple_of`.

    Nearest, not up: rounding up gives 1408 at d_model=512, where §4.1's
    reference value is 1344.
    """
    d_ff = 8 * d_model / 3
    return multiple_of * round(d_ff / multiple_of)


def relu_ffn_hidden_dim(d_ff_swiglu):
    """
    Hidden dim for a 2-matrix ReLU FFN whose parameter count matches a 3-matrix
    SwiGLU FFN at the same d_model: 3 * d_model * d_ff == 2 * d_model * d_ff_relu
    requires d_ff_relu = 1.5 * d_ff_swiglu.
    """
    return round(1.5 * d_ff_swiglu)


class ReLUFFN(nn.Module):
    """Plain (non-gated) ReLU feed-forward, FFN(x) = W2 ReLU(W1 x)."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))