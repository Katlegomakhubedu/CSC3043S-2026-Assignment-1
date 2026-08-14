import numpy as np
import torch
from typing import Optional

class DataLoader:
    """
    Memory-mapped data loader for pre-encoded uint16 token arrays.
    Supports both training (random sampling) and validation (fixed indices).
    """
    def __init__(self, data_path: str, batch_size: int, seq_len: int, seed: int = 42):
        self.data = np.load(data_path, mmap_mode='r')
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_tokens = len(self.data)
        self.rng = np.random.default_rng(seed)

    def get_batch(self, fixed_indices: Optional[np.ndarray] = None):
        """
        Returns (inputs, targets) as torch tensors of shape (batch_size, seq_len).

        If fixed_indices is provided (shape: (batch_size,)), use those start positions
        instead of random sampling. This is used for fixed validation batches.
        """
        # Need seq_len + 1 tokens per sample so both inputs and targets (targets
        # shifted one position right) come out as seq_len long. Slicing seq_len
        # tokens and then using [1:] for targets (the old code) gave inputs one
        # token longer than targets - shapes that don't line up in cross_entropy.
        if fixed_indices is None:
            starts = self.rng.integers(0, self.num_tokens - self.seq_len - 1, size=self.batch_size)
        else:
            starts = fixed_indices

        # Create a batch by slicing from each start position
        batch = np.stack([self.data[s:s + self.seq_len + 1] for s in starts])
        inputs = torch.from_numpy(batch[:, :-1].astype(np.int64))
        targets = torch.from_numpy(batch[:, 1:].astype(np.int64))   # shifted by one
        return inputs, targets