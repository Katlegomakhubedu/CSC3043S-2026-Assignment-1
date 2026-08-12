import time
import math
import torch    
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from training_helpers.get_batch import get_batch
from training_helpers.estimate_loss import estimate_loss
from training_helpers.manage_checkpoint import save_checkpoint
from .seed import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

def train(model, train_ids, val_ids, num_steps, batch_size, learning_rate=1e-8,
          eval_every=50, checkpoint_path="checkpoints/model.pt", seed=0):
    """
    Train the model with plain SGD.
    returns:
        history: dict with 'step', 'train_loss' and 'val_loss' lists
    """
    context_length = model.config.context_length
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    history = {"step": [], "train_loss": [], "val_loss": []}

    model.train()
    start = time.time()
    for step in range(1, num_steps + 1):
        # Step 1: sample a batch of inputs and targets
        x, y = get_batch(train_ids, batch_size, context_length, device, rng)

        # Step 2: forward pass -> logits of shape (batch, seq_len, vocab_size)
        logits = model(x)

        # Step 3: cross-entropy loss, flattening batch and sequence together
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        # Step 4: backward pass and parameter update
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % eval_every == 0 or step == 1:
            val_loss = estimate_loss(model, val_ids, batch_size, context_length, device,
                                     n_batches=5)
            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss)
            print(f"step {step:5d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} "
                  f"| perplexity {math.exp(val_loss):7.1f} | {time.time() - start:5.0f}s")

    save_checkpoint(model, optimizer, num_steps, checkpoint_path)
    print(f"\nSaved checkpoint to {checkpoint_path}")
    return history