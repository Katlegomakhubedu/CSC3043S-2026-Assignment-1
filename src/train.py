import os
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from .training_helpers.get_batch import get_batch
from .training_helpers.estimate_loss import estimate_loss
from .training_helpers.manage_checkpoint import save_checkpoint, load_checkpoint
from .seed import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

def train(model, train_ids, val_ids, num_steps, batch_size, learning_rate=3e-3,
          warmup_steps=200, weight_decay=0.1, grad_clip=1.0,
          eval_every=50, save_every=1000, checkpoint_dir="checkpoints",
          log_dir="logs", run_name="baseline", seed=0, resume_from=None,
          use_amp=True):
    """
    Train with AdamW, warmup+cosine, gradient clipping, BF16 AMP,
    checkpointing, and CSV logging.
    """
    # Determinism
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    context_length = model.config.context_length

    # ---- AdamW with param groups ----
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if p.ndim >= 2 and "norm" not in n],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.ndim < 2 or "norm" in n],
         "weight_decay": 0.0},
    ]
    
    # Sanity check: parameter groups partition the model exactly
    total_params = sum(p.numel() for p in model.parameters())
    group1_params = sum(p.numel() for p in param_groups[0]['params'])
    group2_params = sum(p.numel() for p in param_groups[1]['params'])
    assert group1_params + group2_params == total_params, "Parameter groups do not partition the model!"

    optimizer = AdamW(param_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    # ---- Warmup + Cosine scheduler ----
    warmup = LinearLR(optimizer, start_factor=1e-4, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=num_steps - warmup_steps, eta_min=0.1 * learning_rate)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    # ---- Resume from checkpoint ----
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step = load_checkpoint(resume_from, model, optimizer, scheduler, map_location=device)
        print(f"Resumed from step {start_step}")

    # ---- Fixed validation batches (reproducible) ----
    val_rng = np.random.default_rng(seed + 1)
    val_batch_indices = val_rng.integers(0, len(val_ids) - context_length,
                                         size=5 * batch_size).reshape(5, batch_size)

    # ---- Logging ----
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{run_name}.csv")
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("step,wall_time,tokens,lr,train_loss,val_loss,grad_norm\n")
    
    grad_log_file = os.path.join(log_dir, f"{run_name}_grad_norm.csv")
    if not os.path.exists(grad_log_file):
        with open(grad_log_file, 'w') as f:
            f.write("step,grad_norm\n")

    os.makedirs(checkpoint_dir, exist_ok=True)

    run_config = {
        'num_steps': num_steps,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'warmup_steps': warmup_steps,
        'weight_decay': weight_decay,
        'grad_clip': grad_clip,
        'eval_every': eval_every,
        'save_every': save_every,
        'seed': seed,
        'use_amp': use_amp,
    }

    # ---- Training loop ----
    model.train()
    rng = np.random.default_rng(seed)
    history = {"step": [], "train_loss": [], "val_loss": []}
    total_tokens = 0
    start_time = time.time()
    amp_enabled = use_amp and device.type == "cuda"

    for step in range(start_step + 1, num_steps + 1):
        x, y = get_batch(train_ids, batch_size, context_length, device, rng)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_tokens += x.numel()

        with open(grad_log_file, 'a') as f:
            f.write(f"{step},{grad_norm:.6f}\n")

        if step % eval_every == 0 or step == 1 or step == num_steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i in range(5):
                    vx, vy = get_batch(val_ids, batch_size, context_length, device,
                                       fixed_indices=val_batch_indices[i])
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                        vlogits = model(vx)
                        vloss = F.cross_entropy(vlogits.reshape(-1, logits.size(-1)), vy.reshape(-1))
                    val_losses.append(vloss.item())
            val_loss = np.mean(val_losses)
            model.train()

            lr = scheduler.get_last_lr()[0]
            with open(log_file, 'a') as f:
                f.write(f"{step},{time.time()-start_time:.1f},{total_tokens},{lr:.6f},"
                        f"{loss.item():.4f},{val_loss:.4f},{grad_norm:.4f}\n")

            history["step"].append(step)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss)

            print(f"step {step:5d} | train {loss.item():.4f} | val {val_loss:.4f} "
                  f"| lr {lr:.2e} | grad {grad_norm:.4f}")

        if step % save_every == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, step,
                             os.path.join(checkpoint_dir, f"{run_name}_step{step}.pt"),
                             config=run_config)

    save_checkpoint(model, optimizer, scheduler, num_steps,
                     os.path.join(checkpoint_dir, f"{run_name}_final.pt"),
                     config=run_config)

    print(f"Checkpoints saved to {checkpoint_dir}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--valid_data', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, default=4000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=1344)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--run_name', type=str, default='baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', dest='use_amp', action='store_false')
    args = parser.parse_args()

    # Manually map argparse keys to train() parameter names
    train_kwargs = {
        'num_steps': args.steps,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'warmup_steps': args.warmup_steps,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'eval_every': args.eval_every,
        'save_every': args.save_every,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'run_name': args.run_name,
        'seed': args.seed,
        'resume_from': args.resume_from,
        'use_amp': args.use_amp,
    }

    from model import TransformerLM, TransformerConfig
    config = TransformerConfig(
        vocab_size=args.vocab_size, context_length=args.context_length,
        n_layers=args.n_layers, d_model=args.d_model, n_heads=args.n_heads,
        d_ff=args.d_ff, use_qk_norm=True, use_rmsnorm=True, use_rope=True
    )
    model = TransformerLM(config).to(device)

    train_ids = np.load(args.train_data, mmap_mode='r')
    val_ids = np.load(args.valid_data, mmap_mode='r')

    train(model, train_ids, val_ids, **train_kwargs)