import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary position embeddings (RoPE). Has no learnable parameters.
    params:
        d_head:      dimension of the query/key vectors (must be even)
        max_seq_len: longest sequence we will ever need positions for
        theta:       the base Theta in theta_k = Theta^(-2k/d)
    """

    def __init__(self, d_head, max_seq_len, theta=10000.0):
        super().__init__()
        assert d_head % 2 == 0, "RoPE rotates pairs of dimensions, so d_head must be even"

        k = torch.arange(0, d_head // 2, dtype=torch.float32)        # (d_head/2,)
        inv_freq = theta ** (-2.0 * k / d_head)                      # theta_k
        positions = torch.arange(max_seq_len, dtype=torch.float32)   # (max_seq_len,)
        angles = torch.outer(positions, inv_freq)                    # (max_seq_len, d_head/2)

        # Buffers, not parameters: these are fixed, and not saved into checkpoints.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, positions):
        """
        params:
            x:         (..., seq_len, d_head)
            positions: (seq_len,) absolute position of each element along the sequence axis
        returns:
            rotated tensor of the same shape as x
        """
        # Step 1: look up the precomputed angles for these positions
        cos = self.cos[positions]                   # (seq_len, d_head/2)
        sin = self.sin[positions]                   # (seq_len, d_head/2)

        # Step 2: split the last dimension into the even and odd members of each pair
        x_even, x_odd = x[..., 0::2], x[..., 1::2]  # each (..., seq_len, d_head/2)

        # Step 3: apply the 2D rotation to every pair
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # Step 4: interleave the pairs back into the original layout
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)