import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root mean square layer normalisation.
    params:
        d:   size of the dimension to normalise (the last dimension of the input)
        eps: constant added inside the square root for numerical stability
    """

    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d))     # learned gain g, initialised to 1

    def forward(self, x):
        """
        params:
            x: (..., d)
        returns:
            tensor of the same shape as x
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)                     # upcast to avoid overflow when squaring

        # Step 1: root mean square over the LAST dimension, keeping the dimension for broadcasting
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)   # (..., 1)

        # Step 2: divide by the RMS and apply the learned gain
        out = (x / rms) * self.gain.to(torch.float32)                      # (..., d)

        return out.to(in_dtype)                     # cast back to the original dtype