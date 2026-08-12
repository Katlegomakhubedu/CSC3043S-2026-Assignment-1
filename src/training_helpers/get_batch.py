import torch
import numpy as np

def get_batch(token_ids, batch_size, context_length, device, rng):
    """
    Sample a batch of (input, target) sequences from a flat array of token IDs.
    params:
        token_ids:      1-D numpy array of token IDs
        batch_size:     number of sequences in the batch
        context_length: length of each sequence
        device:         torch device to place the tensors on
        rng:            a numpy random Generator
    returns:
        x: (batch_size, context_length) inputs
        y: (batch_size, context_length) targets, i.e. x shifted one position left
    """
    # Step 1: pick random start indices, leaving room for the target shifted one to the right
    max_start = len(token_ids) - context_length - 1
    starts = rng.integers(0, max_start, size=batch_size)

    # Step 2: slice out the inputs and the targets (inputs shifted by one)
    x = np.stack([token_ids[s:s + context_length] for s in starts])
    y = np.stack([token_ids[s + 1:s + 1 + context_length] for s in starts])

    # Step 3: convert to int64 tensors on the target device
    x = torch.from_numpy(x.astype(np.int64)).to(device)
    y = torch.from_numpy(y.astype(np.int64)).to(device)
    return x, y