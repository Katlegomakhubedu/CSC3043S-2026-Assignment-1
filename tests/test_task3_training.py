"""Task 3 (§5): the training loop, and the §5.6 checks to run before spending
GPU hours.

§5.6 lists five things to confirm on CPU first - loss at initialisation near
ln(vocab_size), overfitting a single batch, a resumed run reproducing the
uninterrupted trajectory, and a short run that neither NaNs nor stalls. Those
are exactly the properties that are expensive to discover on a queued GPU job,
so they are tests here rather than a checklist someone remembers to work
through.

The rest cover the parts of §5.1-5.3 that are quietly wrong-able: which
parameters weight decay reaches, whether the cosine period matches the run, and
whether the logged gradient norm is the pre-clipping one.
"""
import csv
import math
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.model import TransformerLM, TransformerConfig
from src.train import (train, build_param_groups, build_scheduler, resolve_amp,
                       evaluate_fixed_batches)

CONTEXT = 8
VOCAB = 32


def make_model(vocab_size=VOCAB, context_length=CONTEXT, seed=0, **overrides):
    """A small model, identically initialised for a given seed."""
    cfg = dict(vocab_size=vocab_size, context_length=context_length, n_layers=2,
               d_model=32, n_heads=4, d_ff=64)
    cfg.update(overrides)
    torch.manual_seed(seed)
    return TransformerLM(TransformerConfig(**cfg))


def single_batch_data(context_length=CONTEXT, seed=0):
    """A corpus with exactly one window in it, so every step sees the same batch.

    get_batch draws starts from [0, len - context_length - 1); at this length
    that interval contains only 0.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, VOCAB, size=context_length + 2, dtype=np.uint16)


def random_corpus(n=2000, seed=1):
    return np.random.default_rng(seed).integers(0, VOCAB, size=n, dtype=np.uint16)


def run(tmp_path, model, train_ids, val_ids, **kwargs):
    """train() with the bookkeeping pointed at a temp dir."""
    kwargs.setdefault("batch_size", 4)
    kwargs.setdefault("save_every", 0)
    kwargs.setdefault("eval_every", 5)
    kwargs.setdefault("seed", 0)
    return train(model, train_ids, val_ids,
                 log_dir=str(tmp_path / "logs"),
                 checkpoint_dir=str(tmp_path / "ckpt"), **kwargs)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# §5.6 - confirm before spending GPU hours
# ---------------------------------------------------------------------------

def test_loss_at_initialisation_is_near_ln_vocab_size():
    """§5.6: a freshly initialised model is uniform over the vocabulary, so its
    loss is ln(vocab_size). A loss far from this means the initialisation or
    the LM head is wrong, and no amount of tuning will fix it."""
    model = make_model(vocab_size=4000)
    x = torch.randint(0, 4000, (2, CONTEXT))
    y = torch.randint(0, 4000, (2, CONTEXT))

    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 4000), y.reshape(-1))

    assert loss.item() == pytest.approx(math.log(4000), abs=0.3)


def test_model_overfits_a_single_batch(tmp_path):
    """§5.6: with one batch repeated, the model must reach near-zero loss. If it
    cannot memorise 8 tokens the bug is in the model or the loss, not the
    hyperparameters."""
    data = single_batch_data()
    model = make_model()
    history = run(tmp_path, model, data, data, num_steps=400, batch_size=2,
                  learning_rate=1e-2, warmup_steps=20, eval_every=100,
                  run_name="overfit")

    assert history["train_loss"][-1] < 0.05, (
        f"did not memorise a single batch: final train loss "
        f"{history['train_loss'][-1]:.4f}")


def test_resume_reproduces_the_uninterrupted_trajectory(tmp_path):
    """§5.6: a 20-step run resumed from a step-10 checkpoint must land on the
    same losses as one that ran straight through.

    This is the check that caught the batch sampler: a generator advanced once
    per step gets rebuilt from `seed` on resume while the loop restarts at step
    11, so the resumed run replayed steps 1-10's batches. Seeding per step from
    (seed, step) makes the batch a function of the step number instead.

    The checkpoint has to be one taken *during* the 20-step run. A completed
    10-step run carries a 10-step cosine period, so resuming it into a 20-step
    run grafts on a schedule built for a different length - which is what
    train() now warns about rather than the trajectory mismatch it looks like.
    """
    train_ids, val_ids = random_corpus(), random_corpus(seed=2)

    # save_every=10 leaves a step-10 checkpoint behind, exactly as a job killed
    # at step 10 would have.
    uninterrupted = run(tmp_path, make_model(), train_ids, val_ids, num_steps=20,
                        learning_rate=1e-3, warmup_steps=4, eval_every=5,
                        save_every=10, run_name="straight")

    resumed = run(tmp_path, make_model(), train_ids, val_ids, num_steps=20,
                  learning_rate=1e-3, warmup_steps=4, eval_every=5,
                  run_name="secondhalf",
                  resume_from=str(tmp_path / "ckpt" / "straight_step10.pt"))

    assert resumed["step"] == [15, 20], "resumed run should start after step 10"
    for step, got in zip(resumed["step"], resumed["val_loss"]):
        expected = uninterrupted["val_loss"][uninterrupted["step"].index(step)]
        assert got == pytest.approx(expected, rel=1e-4), (
            f"step {step}: resumed val loss {got:.6f} != uninterrupted {expected:.6f}")


def test_short_run_is_finite_and_the_loss_goes_down(tmp_path):
    """§5.6: 40 steps with no NaNs, and a loss that moves in the right direction."""
    train_ids, val_ids = random_corpus(), random_corpus(seed=2)
    history = run(tmp_path, make_model(), train_ids, val_ids, num_steps=40,
                  learning_rate=3e-3, warmup_steps=8, eval_every=10,
                  run_name="short")

    assert all(np.isfinite(history["train_loss"])), history["train_loss"]
    assert all(np.isfinite(history["val_loss"])), history["val_loss"]
    assert all(np.isfinite(history["steps"]["grad_norm"]))
    assert history["val_loss"][-1] < history["val_loss"][0]


# ---------------------------------------------------------------------------
# §5.1 - optimiser and parameter groups
# ---------------------------------------------------------------------------

def test_weight_decay_reaches_only_the_weight_matrices():
    """§5.1: decay the ndim>=2 matrices; RMSNorm gains and biases must sit in a
    group with weight_decay=0. Decaying a normalisation gain towards zero is a
    bug, not regularisation."""
    model = make_model()
    decay_group, no_decay_group = build_param_groups(model, weight_decay=0.1)

    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    decayed = {id(p) for p in decay_group["params"]}
    for name, p in model.named_parameters():
        if "norm" in name or p.ndim < 2:
            assert id(p) not in decayed, f"{name} (ndim={p.ndim}) must not be decayed"
        else:
            assert id(p) in decayed, f"{name} is a weight matrix and should be decayed"

    # Every RMSNorm gain in the model is actually reached by that rule.
    gains = [n for n, _ in model.named_parameters() if n.endswith("gain")]
    assert gains, "expected RMSNorm gains in the model"


def test_parameter_groups_partition_the_model():
    model = make_model()
    groups = build_param_groups(model, weight_decay=0.1)
    counted = sum(p.numel() for g in groups for p in g["params"])
    assert counted == sum(p.numel() for p in model.parameters())


def test_optimiser_is_adamw_with_the_prescribed_betas_and_eps(tmp_path):
    """§5.1 fixes beta=(0.9, 0.95) and eps=1e-8; AdamW's decoupled decay is not
    the same optimiser as Adam(weight_decay=)."""
    model = make_model()
    opt = AdamW(build_param_groups(model, 0.1), lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    assert isinstance(opt, AdamW)
    for group in opt.param_groups:
        assert group["betas"] == (0.9, 0.95)
        assert group["eps"] == 1e-8


# ---------------------------------------------------------------------------
# §5.2 - learning-rate schedule
# ---------------------------------------------------------------------------

def lr_trace(num_steps, warmup_steps, lr):
    """The learning rate actually seen at each step of a run of this length."""
    model = make_model()
    opt = AdamW(build_param_groups(model, 0.1), lr=lr)
    sched = build_scheduler(opt, num_steps, warmup_steps, lr)
    trace = []
    for _ in range(num_steps):
        trace.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    return trace


def test_warmup_rises_then_cosine_decays_to_a_tenth_of_lr_max():
    """§5.2: the cosine period is the run's length, so decay finishes exactly at
    the last step, at 0.1 x lr_max."""
    lr, num_steps, warmup = 1e-3, 100, 20
    trace = lr_trace(num_steps, warmup, lr)

    assert trace[0] < 0.1 * lr, "warmup should start near zero"
    assert trace[warmup] == pytest.approx(lr, rel=1e-6), "peak at the end of warmup"
    assert trace[:warmup] == sorted(trace[:warmup]), "warmup should be monotonic"
    assert trace[warmup:] == sorted(trace[warmup:], reverse=True), "cosine should decay"
    assert trace[-1] == pytest.approx(0.1 * lr, rel=0.02), "decay ends at 0.1 x lr_max"


def test_no_warmup_arm_starts_at_lr_max():
    """Q9's second arm. LinearLR(total_iters=0) is not a no-op, so the warmup
    phase has to be dropped rather than configured away."""
    lr = 1e-3
    trace = lr_trace(100, 0, lr)
    assert trace[0] == pytest.approx(lr, rel=1e-6)
    assert trace == sorted(trace, reverse=True)
    assert trace[-1] == pytest.approx(0.1 * lr, rel=0.02)


def test_shortening_the_run_shortens_the_schedule():
    """§5.2: a shortened sweep run must get a correspondingly shortened schedule,
    otherwise the sweep compares learning rates at different points on their
    schedules."""
    for num_steps in (50, 200):
        trace = lr_trace(num_steps, 10, 1e-3)
        assert len(trace) == num_steps
        assert trace[-1] == pytest.approx(1e-4, rel=0.05)


def test_warmup_longer_than_the_run_is_rejected():
    """A 100-step run with the default 200-step warmup gives CosineAnnealingLR a
    negative period. It used to build that scheduler anyway and produce a
    nonsense schedule; now it fails loudly."""
    model = make_model()
    opt = AdamW(build_param_groups(model, 0.1), lr=1e-3)
    with pytest.raises(ValueError, match="must be <"):
        build_scheduler(opt, num_steps=100, warmup_steps=200, learning_rate=1e-3)


# ---------------------------------------------------------------------------
# §5.3 / §5.5 - gradient norm and logging
# ---------------------------------------------------------------------------

def test_logged_gradient_norm_is_the_pre_clipping_one(tmp_path):
    """§5.3: clip_grad_norm_ returns the norm *before* clipping, and that is the
    diagnostic worth logging. Logging the post-clip value would just print the
    clip threshold back at every step where it mattered."""
    train_ids, val_ids = random_corpus(), random_corpus(seed=2)
    history = run(tmp_path, make_model(), train_ids, val_ids, num_steps=30,
                  learning_rate=1e-2, warmup_steps=1, grad_clip=1e-4,
                  eval_every=10, run_name="clip")

    norms = history["steps"]["grad_norm"]
    assert max(norms) > 1e-4, (
        "with a clip of 1e-4 the logged norms should still exceed it; they look "
        "like post-clipping values")


def test_logs_have_the_columns_section_5_5_requires(tmp_path):
    train_ids, val_ids = random_corpus(), random_corpus(seed=2)
    history = run(tmp_path, make_model(), train_ids, val_ids, num_steps=10,
                  learning_rate=1e-3, warmup_steps=2, eval_every=5,
                  run_name="cols")

    required = {"step", "wall_time", "tokens", "lr", "train_loss", "val_loss",
                "grad_norm"}
    rows = read_csv(history["eval_log"])
    assert required <= set(rows[0].keys())
    assert [int(r["step"]) for r in rows] == [1, 5, 10]
    # Tokens processed must accumulate, not restate the batch size.
    assert [int(r["tokens"]) for r in rows] == sorted(int(r["tokens"]) for r in rows)


def test_step_log_has_one_row_per_step(tmp_path):
    """§5.3 asks for the gradient norm every step, not every evaluation."""
    train_ids, val_ids = random_corpus(), random_corpus(seed=2)
    history = run(tmp_path, make_model(), train_ids, val_ids, num_steps=13,
                  learning_rate=1e-3, warmup_steps=2, eval_every=5,
                  run_name="perstep")

    rows = read_csv(history["step_log"])
    assert [int(r["step"]) for r in rows] == list(range(1, 14))
    assert all(float(r["step_time"]) > 0 for r in rows)


def test_a_fresh_run_replaces_its_log_rather_than_appending(tmp_path):
    """Re-running a name used to interleave both runs in one file, so anything
    reading the last row back got whichever run happened to write it."""
    train_ids, val_ids = random_corpus(), random_corpus(seed=2)
    for _ in range(2):
        history = run(tmp_path, make_model(), train_ids, val_ids, num_steps=10,
                      learning_rate=1e-3, warmup_steps=2, eval_every=5,
                      run_name="rerun")

    assert [int(r["step"]) for r in read_csv(history["eval_log"])] == [1, 5, 10]
    assert len(read_csv(history["step_log"])) == 10


def test_resuming_appends_to_the_existing_log(tmp_path):
    """The other half of that rule: a resumed run must not truncate the history
    of the steps it is continuing from."""
    train_ids, val_ids = random_corpus(), random_corpus(seed=2)
    first = run(tmp_path, make_model(), train_ids, val_ids, num_steps=20,
                learning_rate=1e-3, warmup_steps=2, eval_every=5, save_every=10,
                run_name="cont")
    assert [int(r["step"]) for r in read_csv(first["eval_log"])] == [1, 5, 10, 15, 20]

    history = run(tmp_path, make_model(), train_ids, val_ids, num_steps=20,
                  learning_rate=1e-3, warmup_steps=2, eval_every=5, run_name="cont",
                  resume_from=str(tmp_path / "ckpt" / "cont_step10.pt"))

    steps = [int(r["step"]) for r in read_csv(history["eval_log"])]
    assert steps[:5] == [1, 5, 10, 15, 20], "resume must not truncate the earlier rows"
    assert steps[5:] == [15, 20], "the resumed steps are appended"


def test_resuming_with_a_different_run_length_warns(tmp_path, capsys):
    """The scheduler state restored from a checkpoint was built for that run's
    length. Continuing it under a different --steps silently breaks §5.2's
    "cosine finishes at the last step", so it has to say so."""
    train_ids, val_ids = random_corpus(), random_corpus(seed=2)
    run(tmp_path, make_model(), train_ids, val_ids, num_steps=10, learning_rate=1e-3,
        warmup_steps=2, eval_every=5, save_every=10, run_name="tenstep")

    run(tmp_path, make_model(), train_ids, val_ids, num_steps=20, learning_rate=1e-3,
        warmup_steps=2, eval_every=5, run_name="grafted",
        resume_from=str(tmp_path / "ckpt" / "tenstep_step10.pt"))

    out = capsys.readouterr().out
    assert "WARNING" in out and "10-step run" in out


# ---------------------------------------------------------------------------
# §5.4 - mixed precision
# ---------------------------------------------------------------------------

def test_amp_falls_back_to_fp32_off_cuda():
    """§5.4: bf16 is CUDA-only here, and the fallback must be recorded so a
    timing number can never be read as bf16 when it silently ran fp32."""
    enabled, reason = resolve_amp(True, torch.device("cpu"))
    assert enabled is False
    assert "fp32" in reason

    enabled, reason = resolve_amp(False, torch.device("cpu"))
    assert enabled is False


def test_validation_uses_the_same_fixed_batches_every_call():
    """§5.5: validation on a fixed set of batches, so curves from different runs
    are comparable."""
    val_ids = random_corpus()
    model = make_model()
    model.eval()
    indices = np.random.default_rng(1).integers(
        0, len(val_ids) - CONTEXT - 1, size=(3, 2))

    first = evaluate_fixed_batches(model, val_ids, indices, 2, CONTEXT,
                                   torch.device("cpu"), amp_enabled=False)
    second = evaluate_fixed_batches(model, val_ids, indices, 2, CONTEXT,
                                    torch.device("cpu"), amp_enabled=False)
    assert first == pytest.approx(second)


def test_evaluate_restores_train_mode():
    model = make_model()
    model.train()
    evaluate_fixed_batches(model, random_corpus(), np.zeros((2, 2), dtype=int), 2,
                           CONTEXT, torch.device("cpu"), amp_enabled=False)
    assert model.training, "evaluation must leave the model back in train mode"


# ---------------------------------------------------------------------------
# Q8/Q9 script helpers
# ---------------------------------------------------------------------------

def test_step_time_stats_discards_the_opening_steps():
    from scripts.run_task3_questions import step_time_stats

    # Step 1 carries one-off allocator and kernel-load cost; it must not drag
    # the reported mean.
    steps = {1: (0.0, 1.0), 2: (0.0, 0.010), 3: (0.0, 0.012), 4: (0.0, 0.011)}
    stats = step_time_stats(steps, discard=1)
    assert stats["n_steps_measured"] == 3
    assert stats["n_steps_discarded"] == 1
    assert stats["mean_ms"] == pytest.approx(11.0, rel=1e-6)

    # Discarding everything would leave nothing to report, so it falls back.
    assert step_time_stats(steps, discard=10)["n_steps_measured"] == 4


def test_autocast_keeps_the_delicate_operations_in_fp32():
    """Q8's second half, measured rather than asserted: autocast runs the big
    matrix multiplies in bf16 but keeps the softmax, the RMSNorm reciprocal
    square root and the final cross-entropy in fp32."""
    from scripts.run_task3_questions import probe_autocast_dtypes

    probe = probe_autocast_dtypes(torch.device("cpu"))
    ops = probe["ops"]
    assert probe["device"] == "cpu"

    assert ops["matmul (attention scores, FFN, LM head)"] == "bfloat16"
    assert ops["linear"] == "bfloat16"
    for op in ("softmax", "cross_entropy", "log_softmax",
               "rsqrt (the RMSNorm reciprocal square root)",
               "RMSNorm module output"):
        assert ops[op] == "float32", f"{op} should stay in fp32 under autocast"


def test_q9_arms_start_from_identical_weights():
    """The comparison Q9 makes is only about warmup, so both arms must be built
    from the same initialisation. Reusing one model object across runs - the
    thing this replaced - had the second arm inherit the first arm's training."""
    from scripts.run_task3_questions import build_model

    device = torch.device("cpu")
    a = build_model(device, vocab_size=VOCAB, context_length=CONTEXT, n_layers=1,
                    d_model=32, n_heads=4, d_ff=64)
    b = build_model(device, vocab_size=VOCAB, context_length=CONTEXT, n_layers=1,
                    d_model=32, n_heads=4, d_ff=64)

    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), f"{name} differs between two fresh models"
