import torch
import torch.nn.functional as F
import numpy as np
from training_helpers.get_batch import get_batch

@torch.no_grad()
def estimate_loss(model, token_ids, batch_size, context_length, device, n_batches=10, seed=1234):
    """Average loss over several batches, with the model in eval mode."""
    model.eval()
    rng = np.random.default_rng(seed)     # fixed seed => comparable across evaluations
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(token_ids, batch_size, context_length, device, rng)
        logits = model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    model.train()
    return total / n_batches