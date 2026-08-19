"""Training loop for the §5 deliverable: AdamW, warmup + cosine, gradient
clipping, bf16 autocast, checkpoint/resume, and CSV logging.

Two CSVs are written per run, and every figure in the report comes from them:

  logs/<run>.csv        one row per evaluation - step, wall_time, tokens, lr,
                        train_loss, val_loss, grad_norm (§5.5's required set)
  logs/<run>_steps.csv  one row per optimiser step - grad_norm and step_time.
                        The pre-clipping gradient norm is logged every step
                        because that is the diagnostic that moves first when a
                        run is about to diverge (§5.3), and step_time is what
                        Q8's fp32-vs-bf16 comparison is measured from.

Rows are buffered in memory and flushed at each evaluation, so no file I/O
happens inside the timed part of a step.
"""
import os
import math
import time
import random
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from .training_helpers.get_batch import get_batch
from .training_helpers.manage_checkpoint import save_checkpoint, load_checkpoint

# Kept for callers that import a device from here. `train()` deliberately uses
# the device the model is already on rather than this global, so a caller can
# train two models on two devices in one process.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_VAL_BATCHES = 5   # §5.5: validation on a fixed set of batches, same every run


def build_param_groups(model, weight_decay):
    """§5.1: decay the weight matrices only.

    RMSNorm gains and biases go in a group with weight_decay=0 - decaying a
    normalisation gain towards zero is not regularisation, it is a bug.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and "norm" not in name else no_decay).append(p)

    counted = sum(p.numel() for p in decay) + sum(p.numel() for p in no_decay)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert counted == total, "parameter groups do not partition the model"

    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def build_scheduler(optimizer, num_steps, warmup_steps, learning_rate):
    """§5.2: linear warmup into cosine decay to 0.1 x lr, over this run's length.

    The cosine period is `num_steps - warmup_steps`, so decay finishes exactly
    at the last step. A shortened sweep run therefore has a correspondingly
    shortened schedule - otherwise the sweep compares learning rates at
    different points on their schedules, which is not a comparison at all.
    """
    if warmup_steps >= num_steps:
        raise ValueError(
            f"warmup_steps={warmup_steps} must be < num_steps={num_steps}; the "
            f"cosine phase would otherwise have zero or negative length. Shorten "
            f"the warmup along with the run.")

    eta_min = 0.1 * learning_rate
    cosine = CosineAnnealingLR(optimizer, T_max=num_steps - warmup_steps, eta_min=eta_min)
    if warmup_steps == 0:
        # Q9's no-warmup arm. LinearLR with total_iters=0 is not a no-op, so the
        # warmup phase is dropped entirely rather than configured away.
        return cosine
    warmup = LinearLR(optimizer, start_factor=1e-4, total_iters=warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def resolve_amp(requested, run_device):
    """§5.4: bf16 autocast on CUDA, fp32 everywhere else.

    Returns (enabled, reason). The reason is recorded in the run config so a
    timing number can never be read as bf16 when it silently ran in fp32.
    """
    if not requested:
        return False, "disabled by use_amp=False"
    if run_device.type != "cuda":
        return False, f"no bf16 on device '{run_device.type}' - fell back to fp32"
    if not torch.cuda.is_bf16_supported():
        return False, "CUDA device does not support bf16 - fell back to fp32"
    return True, "bf16 autocast"


def train(model, train_ids, val_ids, num_steps, batch_size, learning_rate=3e-3,
          warmup_steps=200, weight_decay=0.1, grad_clip=1.0,
          eval_every=50, save_every=1000, checkpoint_dir="checkpoints",
          log_dir="logs", run_name="baseline", seed=0, resume_from=None,
          use_amp=True, stop_on_divergence=True, divergence_factor=1.5):
    """Train `model` and return a history dict.

    params:
        model:        a TransformerLM, already on the target device
        train_ids:    1-D uint16 token array, memory-mapped (§5.5)
        val_ids:      same, for the fixed validation batches
        num_steps:    optimiser steps; also the length of the cosine schedule
        warmup_steps: 0 for no warmup (Q9's second arm)
        resume_from:  checkpoint path to restart from, mid-run
        use_amp:      request bf16 autocast; ignored off CUDA (§5.4)
        stop_on_divergence: stop as soon as the run has clearly diverged
                      (§7.1 - there is nothing to learn from the remaining
                      steps and the GPU budget is not free)
        divergence_factor: validation loss above this multiple of the loss at
                      step 1 counts as diverged, alongside any non-finite loss
    returns:
        dict with the evaluation history, the per-step series, and the resolved
        run config - see the keys assembled at the end.
    """
    # §5.5: determinism from the single seed, set here rather than at import so
    # importing this module never mutates a caller's RNG state.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    run_device = next(model.parameters()).device
    context_length = model.config.context_length

    optimizer = AdamW(build_param_groups(model, weight_decay),
                      lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    scheduler = build_scheduler(optimizer, num_steps, warmup_steps, learning_rate)
    amp_enabled, amp_reason = resolve_amp(use_amp, run_device)

    # ---- Resume ----
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step, prior_config = load_checkpoint(resume_from, model, optimizer,
                                                   scheduler, map_location=run_device)
        print(f"Resumed from step {start_step}")
        # The scheduler state that just came back was built for the run that
        # wrote the checkpoint. Resuming with a different --steps therefore
        # continues on that run's cosine period, not the one just constructed,
        # and §5.2's "cosine finishes at the last step" quietly stops holding.
        prior_steps = (prior_config or {}).get('num_steps')
        if prior_steps is not None and prior_steps != num_steps:
            print(f"  ! WARNING: checkpoint was written by a {prior_steps}-step run "
                  f"but this one is {num_steps} steps. The restored schedule is the "
                  f"{prior_steps}-step one - resume with --steps {prior_steps} to "
                  f"continue the same schedule.")

    # ---- Fixed validation batches (§5.5: same seed every run, so curves from
    # different runs are measured against the same data) ----
    val_rng = np.random.default_rng(seed + 1)
    val_batch_indices = val_rng.integers(
        0, len(val_ids) - context_length - 1,
        size=N_VAL_BATCHES * batch_size).reshape(N_VAL_BATCHES, batch_size)

    run_config = {
        'num_steps': num_steps, 'batch_size': batch_size,
        'learning_rate': learning_rate, 'warmup_steps': warmup_steps,
        'weight_decay': weight_decay, 'grad_clip': grad_clip,
        'eval_every': eval_every, 'save_every': save_every, 'seed': seed,
        'use_amp': use_amp, 'amp_enabled': amp_enabled, 'amp_reason': amp_reason,
        'device': str(run_device), 'context_length': context_length,
    }

    # ---- Logging ----
    # Truncated on a fresh run and appended to only when resuming. Appending
    # unconditionally (the previous behaviour) left re-runs of the same
    # run_name interleaved in one file, so anything reading the last row back
    # got a number from whichever earlier run happened to have written it.
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    eval_log = os.path.join(log_dir, f"{run_name}.csv")
    step_log = os.path.join(log_dir, f"{run_name}_steps.csv")
    if start_step == 0:
        with open(eval_log, "w") as f:
            f.write("step,wall_time,tokens,lr,train_loss,val_loss,grad_norm\n")
        with open(step_log, "w") as f:
            f.write("step,grad_norm,step_time\n")

    print(f"[{run_name}] {num_steps} steps on {run_device} | {amp_reason} | "
          f"lr {learning_rate:.2e}, warmup {warmup_steps}, batch {batch_size}")

    # ---- Training loop ----
    model.train()
    history = {"step": [], "train_loss": [], "val_loss": [], "lr": [],
               "grad_norm": [], "wall_time": []}
    steps = {"step": [], "grad_norm": [], "step_time": []}
    pending = []                # per-step rows, flushed at each evaluation
    total_tokens = 0
    start_time = time.perf_counter()
    diverged, diverged_at, diverged_reason = False, None, None
    baseline_val_loss = None    # the step-1 loss the divergence check compares to
    last_step = start_step

    def sync():
        if run_device.type == "cuda":
            torch.cuda.synchronize()

    for step in range(start_step + 1, num_steps + 1):
        # The batch is a function of (seed, step), not of a generator advanced
        # once per step. A single long-lived generator is rebuilt from `seed` on
        # resume while the loop restarts mid-run, so a run resumed at step 26
        # would replay the batches from steps 1-25 and diverge from the
        # uninterrupted trajectory §5.6 requires it to match. Seeding per step
        # makes step N's batch the same wherever the run was interrupted.
        step_rng = np.random.default_rng([seed, step])
        x, y = get_batch(train_ids, batch_size, context_length, run_device, step_rng)

        # Timed region: forward, backward, clip, step. The batch is prepared
        # above and logs are flushed below, so neither lands in the measurement.
        # CUDA kernels queue asynchronously, so the step is not over until the
        # device says so - without this sync Q8 would time the enqueue rather
        # than the work, and bf16 and fp32 would come out indistinguishable.
        sync()
        t0 = time.perf_counter()

        with torch.autocast(device_type=run_device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # §5.3: returns the norm *before* clipping, which is what gets logged.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        sync()
        step_time = time.perf_counter() - t0

        total_tokens += x.numel()
        last_step = step
        grad_norm = float(grad_norm)
        pending.append((step, grad_norm, step_time))
        steps["step"].append(step)
        steps["grad_norm"].append(grad_norm)
        steps["step_time"].append(step_time)

        if step % eval_every == 0 or step == 1 or step == num_steps:
            val_loss = evaluate_fixed_batches(model, val_ids, val_batch_indices,
                                              batch_size, context_length,
                                              run_device, amp_enabled)
            lr = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - start_time
            train_loss = loss.item()

            with open(eval_log, "a") as f:
                f.write(f"{step},{elapsed:.1f},{total_tokens},{lr:.8f},"
                        f"{train_loss:.4f},{val_loss:.4f},{grad_norm:.4f}\n")
            with open(step_log, "a") as f:
                for s, g, t in pending:
                    f.write(f"{s},{g:.6f},{t:.6f}\n")
            pending.clear()

            history["step"].append(step)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(lr)
            history["grad_norm"].append(grad_norm)
            history["wall_time"].append(elapsed)

            print(f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| lr {lr:.2e} | grad {grad_norm:.4f} | {step_time*1e3:.1f} ms/step")

            # §7.1: kill a diverged run as soon as it is visibly diverged. The
            # sweep is *required* to contain one, so this is a normal outcome
            # to record rather than an error - `diverged` goes in the history
            # and Q10 reads it back.
            if baseline_val_loss is None:
                baseline_val_loss = val_loss
            if not math.isfinite(train_loss) or not math.isfinite(val_loss):
                diverged, diverged_reason = True, "loss became NaN or infinite"
            elif val_loss > divergence_factor * baseline_val_loss:
                diverged, diverged_reason = True, (
                    f"validation loss {val_loss:.4f} exceeded "
                    f"{divergence_factor}x its step-1 value {baseline_val_loss:.4f}")
            if diverged:
                diverged_at = step
                print(f"  ! DIVERGED at step {step}: {diverged_reason}")
                if stop_on_divergence:
                    print(f"  ! stopping early - {num_steps - step} steps not run")
                    break

        if save_every and step % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step,
                            os.path.join(checkpoint_dir, f"{run_name}_step{step}.pt"),
                            config=run_config)

    if pending:
        with open(step_log, "a") as f:
            for s, g, t in pending:
                f.write(f"{s},{g:.6f},{t:.6f}\n")

    run_config['completed_steps'] = last_step
    run_config['diverged'] = diverged
    final_ckpt = os.path.join(checkpoint_dir, f"{run_name}_final.pt")
    save_checkpoint(model, optimizer, scheduler, last_step, final_ckpt, config=run_config)
    print(f"[{run_name}] {'DIVERGED' if diverged else 'done'} at step {last_step} "
          f"- checkpoint {final_ckpt}")

    history["diverged"] = diverged
    history["diverged_at"] = diverged_at
    history["diverged_reason"] = diverged_reason
    history["completed_steps"] = last_step
    history["total_seconds"] = time.perf_counter() - start_time
    history["steps"] = steps
    history["config"] = run_config
    history["eval_log"] = eval_log
    history["step_log"] = step_log
    history["final_checkpoint"] = final_ckpt
    return history


@torch.no_grad()
def evaluate_fixed_batches(model, val_ids, val_batch_indices, batch_size,
                           context_length, run_device, amp_enabled):
    """Mean loss over the fixed validation batches, model restored to train mode."""
    model.eval()
    losses = []
    for indices in val_batch_indices:
        vx, vy = get_batch(val_ids, batch_size, context_length, run_device,
                           fixed_indices=indices)
        with torch.autocast(device_type=run_device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled):
            vlogits = model(vx)
            vloss = F.cross_entropy(vlogits.reshape(-1, vlogits.size(-1)), vy.reshape(-1))
        losses.append(vloss.item())
    model.train()
    return float(np.mean(losses))


def build_parser():
    """§5.5: every hyperparameter settable from the command line."""
    p = argparse.ArgumentParser(description="Train the Transformer LM (section 5).")
    p.add_argument('--train_data', type=str, required=True)
    p.add_argument('--valid_data', type=str, required=True)
    # architecture
    p.add_argument('--vocab_size', type=int, default=4000)
    p.add_argument('--context_length', type=int, default=256)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--d_ff', type=int, default=1344)
    p.add_argument('--rope_theta', type=float, default=10000.0)
    # §7.2 ablation switches - the architecture was not reachable from the CLI
    # without these, and the ablations are launched as command lines.
    p.add_argument('--ffn_type', choices=['swiglu', 'relu'], default='swiglu')
    p.add_argument('--no_rmsnorm', dest='use_rmsnorm', action='store_false')
    p.add_argument('--no_rope', dest='use_rope', action='store_false')
    p.add_argument('--no_qk_norm', dest='use_qk_norm', action='store_false')
    p.set_defaults(use_rmsnorm=True, use_rope=True, use_qk_norm=True)
    # optimisation
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--lr', type=float, default=3e-3)
    p.add_argument('--warmup_steps', type=int, default=200)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    # bookkeeping
    p.add_argument('--eval_every', type=int, default=50)
    p.add_argument('--save_every', type=int, default=1000)
    p.add_argument('--log_dir', type=str, default='logs')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    p.add_argument('--run_name', type=str, default='baseline')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', dest='resume_from', type=str, default=None,
                   help="Checkpoint to restart from, mid-run.")
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--no_amp', dest='use_amp', action='store_false',
                   help="Force fp32. bf16 autocast is the default on CUDA.")
    p.add_argument('--no_stop_on_divergence', dest='stop_on_divergence',
                   action='store_false',
                   help="Run a diverged job to completion instead of killing it "
                        "at the first sign (section 7.1 says to kill it).")
    p.add_argument('--divergence_factor', type=float, default=1.5)
    p.set_defaults(use_amp=True, stop_on_divergence=True)
    return p


def main(argv=None):
    from .model import TransformerLM, TransformerConfig

    args = build_parser().parse_args(argv)
    run_device = torch.device(args.device) if args.device else device

    config = TransformerConfig(
        vocab_size=args.vocab_size, context_length=args.context_length,
        n_layers=args.n_layers, d_model=args.d_model, n_heads=args.n_heads,
        d_ff=args.d_ff, rope_theta=args.rope_theta, ffn_type=args.ffn_type,
        use_qk_norm=args.use_qk_norm, use_rmsnorm=args.use_rmsnorm,
        use_rope=args.use_rope,
    )
    model = TransformerLM(config).to(run_device)

    # §5.5: memory-mapped, never read into RAM.
    train_ids = np.load(args.train_data, mmap_mode='r')
    val_ids = np.load(args.valid_data, mmap_mode='r')

    return train(model, train_ids, val_ids,
                 num_steps=args.steps, batch_size=args.batch_size,
                 learning_rate=args.lr, warmup_steps=args.warmup_steps,
                 weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                 eval_every=args.eval_every, save_every=args.save_every,
                 checkpoint_dir=args.checkpoint_dir, log_dir=args.log_dir,
                 run_name=args.run_name, seed=args.seed,
                 resume_from=args.resume_from, use_amp=args.use_amp,
                 stop_on_divergence=args.stop_on_divergence,
                 divergence_factor=args.divergence_factor)


if __name__ == "__main__":
    main()
