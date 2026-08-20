"""Task 3 end to end: training, answering Q8-Q9 (§5.7).

  * Q8 - mean step time for the standard configuration in fp32 and in bf16
         autocast, plus measured evidence of which operations autocast keeps
         in fp32.
  * Q9 - the same learning rate with and without warmup, all else fixed:
         both validation curves on one figure, and the pre-clipping gradient
         norm at step 1, step 50 and the final step for each run.

Three things this script is careful about, because each of them silently
invalidates the answer:

  * Every run gets a **freshly initialised model**, built under the same seed.
    Reusing one model object across runs means the second arm starts from
    weights the first arm already trained, so Q9 would be comparing "warmup"
    against "no warmup, but 200 steps of training in hand".
  * Step time is measured **inside the training loop** (src/train.py), around
    the forward/backward/step only, with a CUDA sync on both sides. Dividing
    total wall-clock by the step count instead folds in validation passes and
    checkpoint writes.
  * bf16 needs CUDA (§5.4). On CPU both arms run fp32 and the Q8 comparison is
    vacuous - the script says so rather than reporting two equal numbers as
    though they were a result.

Every number printed also lands in logs/task3_results.json, and both figures
are drawn from the per-run CSVs in logs/ rather than from memory (§5.5).

Examples
--------
  python scripts/run_task3_questions.py
  python scripts/run_task3_questions.py --questions 9 --lr 1e-3
  python scripts/run_task3_questions.py --q8_steps 300 --q9_steps 1000
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import json
import platform
import statistics

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")   # headless: this script only saves a PNG
import matplotlib.pyplot as plt

from src.model import TransformerLM, TransformerConfig
from src.model_components.RMSNorm import RMSNorm
from src.train import train, resolve_amp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_CONFIG = dict(vocab_size=4000, context_length=256, n_layers=4, d_model=512,
                   n_heads=8, d_ff=1344, rope_theta=10000.0, use_qk_norm=True)

# §7's standard run configuration, held fixed across both arms of each question.
STANDARD_RUN = dict(batch_size=32, weight_decay=0.1, grad_clip=1.0)

INIT_SEED = 0        # model initialisation; identical for every arm
RUN_SEED = 42        # batch order and validation batches


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def describe_machine(device):
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def build_model(device, vocab_size, **overrides):
    """A fresh model, initialised identically every call.

    The seed is set here rather than by the caller because the initialisation
    draws from the global torch RNG: two models built without reseeding differ,
    and an A/B comparison of training runs would then be confounded by their
    starting weights.
    """
    cfg = dict(BASE_CONFIG)
    cfg["vocab_size"] = vocab_size
    cfg.update(overrides)
    torch.manual_seed(INIT_SEED)
    return TransformerLM(TransformerConfig(**cfg)).to(device)


def resolve_data(args):
    """Token arrays for training, memory-mapped, plus a note on where they came from.

    Preference order: whatever --train_data/--valid_data name, then the Task 1
    encoded corpus, then a disjoint split of the encoded validation set (useful
    while the train split is still encoding), then random tokens.

    The random fallback exists so the plumbing can be exercised without the
    corpus, but Q9's curves mean nothing on uniform random tokens - there is no
    structure to learn, so both arms sit flat at ln(vocab_size). The caller
    records `synthetic` so a figure made this way cannot be mistaken for a
    result.
    """
    def load(path):
        return np.load(path, mmap_mode="r")

    if args.train_data and args.valid_data:
        return load(args.train_data), load(args.valid_data), {
            "source": "explicit", "train": args.train_data, "valid": args.valid_data,
            "synthetic": False}

    train_npy = os.path.join(REPO_ROOT, "train_encoded.npy")
    # §2 reserves the last 2,000 validation documents as a test set to be
    # touched exactly once, so anything that guides a modelling decision scores
    # the split array rather than the whole validation file.
    valid_npy = os.path.join(REPO_ROOT, "valid_split_encoded.npy")
    if not os.path.exists(valid_npy):
        valid_npy = os.path.join(REPO_ROOT, "valid_encoded.npy")
    if os.path.exists(train_npy) and os.path.exists(valid_npy):
        return load(train_npy), load(valid_npy), {
            "source": "task 1 encoded corpus", "train": "train_encoded.npy",
            "valid": os.path.basename(valid_npy), "synthetic": False}

    if os.path.exists(valid_npy):
        # Only the validation split has been encoded so far. Split it in two
        # disjoint halves rather than training and validating on the same
        # tokens, which would make the validation curve meaningless.
        data = load(valid_npy)
        cut = int(0.9 * len(data))
        print(f"  ! train_encoded.npy not found - splitting {os.path.basename(valid_npy)} "
              f"{cut:,}/{len(data)-cut:,} into disjoint train/val halves.")
        return data[:cut], data[cut:], {
            "source": f"{os.path.basename(valid_npy)} split 90/10 (train split not yet encoded)",
            "n_train_tokens": cut, "n_val_tokens": len(data) - cut, "synthetic": False}

    print("  ! No encoded corpus found - falling back to RANDOM tokens. "
          "Q9's curves will be flat and are not a result.")
    rng = np.random.default_rng(0)
    vocab = BASE_CONFIG["vocab_size"]
    return (rng.integers(0, vocab, size=400_000, dtype=np.uint16),
            rng.integers(0, vocab, size=40_000, dtype=np.uint16),
            {"source": "random tokens (no corpus available)", "synthetic": True})


def read_step_log(log_dir, run_name):
    """The per-step CSV a run wrote: step -> (grad_norm, step_time).

    Read back from disk rather than taken from the returned history, so every
    number quoted in the report demonstrably comes from a committed log (§5.5).
    """
    path = os.path.join(log_dir, f"{run_name}_steps.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {int(r["step"]): (float(r["grad_norm"]), float(r["step_time"])) for r in rows}


def read_eval_log(log_dir, run_name):
    """The per-evaluation CSV: parallel lists of step and val_loss."""
    path = os.path.join(log_dir, f"{run_name}.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return ([int(r["step"]) for r in rows],
            [float(r["val_loss"]) for r in rows],
            [float(r["train_loss"]) for r in rows])


def step_time_stats(steps, discard):
    """Mean/median step time in ms, ignoring the first `discard` steps.

    The opening steps carry one-off costs - allocator growth, cuDNN autotuning,
    lazy kernel loading - that are not part of the steady-state step time Q8
    asks for. The discarded count is reported alongside the mean so the number
    can be read for what it is.
    """
    ordered = [steps[k][1] for k in sorted(steps)]
    kept = ordered[discard:] or ordered
    return {
        "mean_ms": 1e3 * statistics.fmean(kept),
        "median_ms": 1e3 * statistics.median(kept),
        "stdev_ms": 1e3 * statistics.stdev(kept) if len(kept) > 1 else 0.0,
        "n_steps_measured": len(kept),
        "n_steps_discarded": len(ordered) - len(kept),
    }


# ---------------------------------------------------------------------------
# Q8
# ---------------------------------------------------------------------------

def probe_autocast_dtypes(device):
    """Which operations autocast keeps in fp32, measured rather than asserted.

    Runs one representative op of each kind under bf16 autocast and records the
    dtype that comes back. This is the evidence for Q8's second half: autocast
    runs the big matrix multiplies in bf16 but keeps reductions and anything
    numerically delicate in fp32, which is what makes the softmax, the RMSNorm
    reciprocal square root and the final cross-entropy safe.

    Uses its own autocast context (not the training loop's), so the probe still
    answers the question on a CPU-only machine even though §5.4 has training
    fall back to fp32 there. The cast policy is per-backend, so the device the
    probe ran on is recorded with the answer.
    """
    torch.manual_seed(0)
    x = torch.randn(2, 8, 64, device=device)
    w = torch.randn(64, 64, device=device)
    scores = torch.randn(2, 4, 8, 8, device=device)
    q = torch.randn(2, 4, 8, 16, device=device)
    logits = torch.randn(16, 4000, device=device)
    targets = torch.randint(0, 4000, (16,), device=device)
    norm = RMSNorm(64).to(device)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        probes = {
            "matmul (attention scores, FFN, LM head)": (x @ w).dtype,
            "linear": F.linear(x, w).dtype,
            "scaled_dot_product_attention": F.scaled_dot_product_attention(q, q, q).dtype,
            "softmax": F.softmax(scores, dim=-1).dtype,
            "silu (SwiGLU activation)": F.silu(x).dtype,
            "rsqrt (the RMSNorm reciprocal square root)": torch.rsqrt(x.abs() + 1e-5).dtype,
            "sum (reduction)": x.sum().dtype,
            "RMSNorm module output": norm(x).dtype,
            "cross_entropy": F.cross_entropy(logits, targets).dtype,
            "log_softmax": F.log_softmax(logits, dim=-1).dtype,
        }

    probed = {name: str(dtype).replace("torch.", "") for name, dtype in probes.items()}
    return {"device": device.type, "ops": probed}


def run_q8(args, device, train_ids, val_ids, vocab_size, results):
    """Mean step time for the standard configuration, fp32 versus bf16 autocast."""
    print("\n" + "=" * 70)
    print("Q8: mean step time, fp32 vs bf16 autocast")
    print("=" * 70)

    bf16_possible, bf16_reason = resolve_amp(True, device)
    if not bf16_possible:
        print(f"  ! {bf16_reason}. Both arms will run fp32, so the comparison "
              f"below is not a Q8 answer - re-run on a CUDA machine.")

    arms = {}
    for label, use_amp in (("bf16", True), ("fp32", False)):
        run_name = f"q8_{label}"
        # Fresh model per arm: precision is being compared, not training
        # progress, so both arms must start from the same weights.
        model = build_model(device, vocab_size)
        train(model, train_ids, val_ids,
              num_steps=args.q8_steps, warmup_steps=min(args.warmup, args.q8_steps - 1),
              learning_rate=args.lr, eval_every=args.q8_steps,   # eval only at the end
              save_every=0, use_amp=use_amp, run_name=run_name,
              log_dir=args.log_dir, checkpoint_dir=args.checkpoint_dir,
              seed=RUN_SEED, **STANDARD_RUN)
        stats = step_time_stats(read_step_log(args.log_dir, run_name), args.q8_discard)
        stats["amp_enabled"] = use_amp and bf16_possible
        arms[label] = stats
        del model

    speedup = arms["fp32"]["mean_ms"] / arms["bf16"]["mean_ms"]
    dtypes = probe_autocast_dtypes(device)

    print(f"\n[Q8] Mean step time over {arms['bf16']['n_steps_measured']} steps "
          f"(first {args.q8_discard} discarded), batch {STANDARD_RUN['batch_size']} "
          f"x {BASE_CONFIG['context_length']} tokens:")
    print(f"  {'precision':<10} {'mean ms':>10} {'median ms':>10} {'stdev ms':>10}")
    for label in ("fp32", "bf16"):
        s = arms[label]
        print(f"  {label:<10} {s['mean_ms']:>10.2f} {s['median_ms']:>10.2f} {s['stdev_ms']:>10.2f}")
    print(f"  bf16 speedup: {speedup:.2f}x")

    print(f"\n[Q8] Dtypes under bf16 autocast, measured on {dtypes['device']}:")
    for name, dtype in dtypes["ops"].items():
        print(f"  {dtype:<9} {name}")

    results["q8"] = {
        "steps_per_arm": args.q8_steps,
        "steps_discarded": args.q8_discard,
        "batch_size": STANDARD_RUN["batch_size"],
        "context_length": BASE_CONFIG["context_length"],
        "fp32": arms["fp32"],
        "bf16": arms["bf16"],
        "bf16_speedup": speedup,
        "bf16_actually_enabled": bf16_possible,
        "comparison_valid": bf16_possible,
        "note": (bf16_reason if not bf16_possible else
                 "bf16 autocast active; both arms otherwise identical."),
        "autocast_dtypes": dtypes,
    }
    return results["q8"]


# ---------------------------------------------------------------------------
# Q9
# ---------------------------------------------------------------------------

def run_q9(args, device, train_ids, val_ids, vocab_size, results):
    """The same learning rate with and without warmup, all else fixed."""
    print("\n" + "=" * 70)
    print(f"Q9: warmup vs no warmup at lr={args.lr:.1e}")
    print("=" * 70)

    arms = {
        "with_warmup": args.warmup,
        "no_warmup": 0,
    }
    curves, norms = {}, {}
    for label, warmup_steps in arms.items():
        run_name = f"q9_{label}"
        # Fresh model per arm, identical initialisation - the whole comparison
        # rests on warmup being the only difference between the two runs.
        model = build_model(device, vocab_size)
        train(model, train_ids, val_ids,
              num_steps=args.q9_steps, warmup_steps=warmup_steps,
              learning_rate=args.lr, eval_every=args.eval_every,
              save_every=0, use_amp=not args.no_amp, run_name=run_name,
              log_dir=args.log_dir, checkpoint_dir=args.checkpoint_dir,
              seed=RUN_SEED, **STANDARD_RUN)

        steps_csv = read_step_log(args.log_dir, run_name)
        curves[label] = read_eval_log(args.log_dir, run_name)
        final_step = max(steps_csv)
        norms[label] = {
            "step_1": steps_csv[1][0],
            "step_50": steps_csv[50][0] if 50 in steps_csv else None,
            f"step_{final_step}": steps_csv[final_step][0],
            "max": max(g for g, _ in steps_csv.values()),
        }
        del model

    png = plot_q9(args, curves, norms)

    final_step = args.q9_steps
    print(f"\n[Q9] Pre-clipping gradient norm:")
    print(f"  {'run':<16} {'step 1':>10} {'step 50':>10} {'step ' + str(final_step):>12} {'max':>10}")
    for label in arms:
        n = norms[label]
        s50 = f"{n['step_50']:.4f}" if n["step_50"] is not None else "-"
        print(f"  {label:<16} {n['step_1']:>10.4f} {s50:>10} "
              f"{n[f'step_{final_step}']:>12.4f} {n['max']:>10.4f}")

    print(f"\n[Q9] Final validation loss:")
    for label in arms:
        print(f"  {label:<16} {curves[label][1][-1]:.4f}")
    print(f"\n[Q9] Figure written to {png}")

    results["q9"] = {
        "learning_rate": args.lr,
        "steps": args.q9_steps,
        "warmup_steps": args.warmup,
        "grad_norms": norms,
        "final_val_loss": {label: curves[label][1][-1] for label in arms},
        "val_curves": {label: {"step": curves[label][0], "val_loss": curves[label][1]}
                       for label in arms},
        "figure": os.path.basename(png),
    }
    return results["q9"]


def plot_q9(args, curves, norms):
    """Validation curves plus the gradient norms the table quotes, on one figure."""
    fig, (ax_loss, ax_grad) = plt.subplots(1, 2, figsize=(12, 4.5))
    styles = {"with_warmup": ("With warmup", "tab:blue", "-"),
              "no_warmup": ("No warmup", "tab:red", "--")}

    for label, (steps, val_loss, _) in curves.items():
        name, colour, dash = styles[label]
        ax_loss.plot(steps, val_loss, dash, color=colour, label=name)
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Validation loss")
    ax_loss.set_title(f"Validation loss, lr={args.lr:.1e}")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # The gradient norm is the diagnostic that moves first (§5.3), so it goes
    # on the same figure as the curve it explains.
    for label in curves:
        name, colour, dash = styles[label]
        steps_csv = read_step_log(args.log_dir, f"q9_{label}")
        xs = sorted(steps_csv)
        ax_grad.plot(xs, [steps_csv[s][0] for s in xs], dash, color=colour,
                     label=name, linewidth=0.9)
    ax_grad.axhline(STANDARD_RUN["grad_clip"], color="grey", linestyle=":",
                    label=f"clip at {STANDARD_RUN['grad_clip']}")
    ax_grad.set_yscale("log")
    ax_grad.set_xlabel("Step")
    ax_grad.set_ylabel("Pre-clipping gradient norm")
    ax_grad.set_title("Gradient norm before clipping")
    ax_grad.legend()
    ax_grad.grid(True, alpha=0.3)

    fig.tight_layout()
    png = os.path.join(args.out_dir, "task3_q9_warmup.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def save_results(results, log_dir):
    path = os.path.join(log_dir, "task3_results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prior = {}
    if os.path.exists(path):
        # Running one question (--questions 9) must not discard the other.
        with open(path, encoding="utf-8") as f:
            prior = json.load(f)
    prior.update(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prior, f, indent=2, default=str)
    return path


def build_parser():
    p = argparse.ArgumentParser(
        description="Task 3: training. Answers Q8-Q9 (section 5.7).")
    p.add_argument("--questions", nargs="+", type=int, default=[8, 9], choices=[8, 9])
    p.add_argument("--train_data", default=None,
                   help="Encoded uint16 .npy. Auto-discovered if omitted.")
    p.add_argument("--valid_data", default=None)
    p.add_argument("--lr", type=float, default=3e-3,
                   help="Q9's learning rate - set this to the Q10 sweep winner.")
    p.add_argument("--warmup", type=int, default=200,
                   help="Warmup steps for the 'with warmup' arm (section 5.2).")
    p.add_argument("--q8_steps", type=int, default=200,
                   help="Timed steps per precision arm; a few hundred is enough.")
    p.add_argument("--q8_discard", type=int, default=20,
                   help="Opening steps excluded from the Q8 mean (warm-up cost).")
    p.add_argument("--q9_steps", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--no_amp", action="store_true",
                   help="Force fp32 for Q9 (Q8 always runs both precisions).")
    p.add_argument("--device", default=None, help="cuda / cpu. Defaults to cuda if available.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny model and step counts, into logs/smoke/. Exercises "
                        "every step of both questions in about a minute, before "
                        "any GPU time is spent on the real thing (section 5.6).")
    p.add_argument("--log_dir", default=os.path.join(REPO_ROOT, "logs"))
    p.add_argument("--checkpoint_dir", default=os.path.join(REPO_ROOT, "checkpoints"))
    p.add_argument("--out_dir", default=REPO_ROOT)
    return p


def apply_smoke_settings(args):
    """Shrink everything so the plumbing can be checked without a GPU.

    Kept separate from the real defaults, and pointed at logs/smoke/, so a smoke
    run can never leave numbers behind that look like an answer.
    """
    BASE_CONFIG.update(context_length=64, n_layers=2, d_model=128, n_heads=4, d_ff=256)
    STANDARD_RUN.update(batch_size=8)
    args.q8_steps, args.q8_discard = 10, 2
    args.q9_steps, args.warmup, args.eval_every = 40, 8, 10
    args.log_dir = os.path.join(args.log_dir, "smoke")
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, "smoke")
    args.out_dir = os.path.join(args.out_dir, "logs", "smoke")
    print("SMOKE RUN - tiny model, output in logs/smoke/. Not a Q8/Q9 answer.\n")


def warn_if_expensive(args, device):
    """A CPU run of the real configuration is hours long, and Q8 is vacuous on it.

    Worth saying before the first step rather than after someone watches a
    12-second step tick over for an afternoon.
    """
    if device.type == "cuda" or args.smoke:
        return
    total = (2 * args.q8_steps if 8 in args.questions else 0) + \
            (2 * args.q9_steps if 9 in args.questions else 0)
    print(f"  ! Running {total} steps of the section 4.1 model on CPU. Expect "
          f"roughly 10s/step at batch {STANDARD_RUN['batch_size']}, i.e. "
          f"~{total * 10 / 3600:.1f} hours, and Q8's fp32-vs-bf16 comparison "
          f"cannot be made without CUDA. Use --smoke to check the pipeline, and "
          f"run the real thing on a GPU.\n")


def main(argv=None):
    args = build_parser().parse_args(argv)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        apply_smoke_settings(args)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    train_ids, val_ids, data_info = resolve_data(args)
    vocab_size = BASE_CONFIG["vocab_size"]

    print(f"Device: {device}")
    print(f"Data:   {data_info['source']} "
          f"({len(train_ids):,} train / {len(val_ids):,} val tokens)")
    warn_if_expensive(args, device)

    results = {
        "machine": describe_machine(device),
        "config": vars(args),
        "model_config": {**BASE_CONFIG, "vocab_size": vocab_size},
        "standard_run": STANDARD_RUN,
        "data": data_info,
    }

    if 8 in args.questions:
        run_q8(args, device, train_ids, val_ids, vocab_size, results)
    if 9 in args.questions:
        run_q9(args, device, train_ids, val_ids, vocab_size, results)

    path = save_results(results, args.log_dir)
    print("\n" + "=" * 70)
    print("TASK 3 COMPLETE")
    print("=" * 70)
    print(f"All numbers written to {path}")


if __name__ == "__main__":
    main()
