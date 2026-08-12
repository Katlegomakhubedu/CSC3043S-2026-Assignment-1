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
    """Round (8/3) * d_model up to the nearest multiple of `multiple_of`."""
    d_ff = int(8 * d_model / 3)
    return multiple_of * ((d_ff + multiple_of - 1) // multiple_of)