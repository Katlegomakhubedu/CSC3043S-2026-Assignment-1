"""Task 4 end to end: experiments, answering Q10-Q17 (§7.5).

  * Q10 - learning-rate sweep, including a divergent run; the best learning
          rate found here is the baseline for everything below.
  * Q11 - no-RMSNorm ablation, at the best learning rate and at a reduced one.
  * Q12 - NoPE against the RoPE baseline.
  * Q13 - parameter-matched ReLU FFN against SwiGLU.
  * Q14 - the ablation gaps side by side with the Q10 learning-rate gap.
  * Q15 - perplexity, BPC and parameter count for the two vocabulary sizes.
  * Q16 - GPU-hours by phase.
  * Q17 - position-wise validation loss for NoPE and RoPE.

§7's rule is that when you vary one thing you hold everything else fixed -
seed, tokens processed, and validation batches included. That is what most of
this file is for:

  * One `Experiment` record per run, all sharing STANDARD_RUN. An experiment
    names only what it changes, so nothing can drift between arms by accident.
  * Models are built from a fixed seed, so two arms differ in the thing under
    test and not in their initialisation.
  * Runs are **cached**. Each finished run writes logs/<name>_run.json; asking
    for it again reuses it rather than spending the GPU hours twice. `--force`
    retrains. This is what makes Q14 (which reads six runs) cheap to re-run.
  * Data is the memory-mapped encoded corpus from Task 1. Re-tokenizing the
    2.2GB training text per question - what this script used to do - reads the
    whole corpus into RAM three times over and violates §5.5.

Every number printed also lands in logs/task4_results.json.

Examples
--------
  python scripts/run_task4_questions.py --plan            # what would run, and roughly how long
  python scripts/run_task4_questions.py --questions 10    # the sweep
  python scripts/run_task4_questions.py --questions 11 12 13 14
  python scripts/run_task4_questions.py --smoke           # whole pipeline, tiny, ~minutes
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import json
import math
import platform
import time
from dataclasses import dataclass, field

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")   # headless: this script only saves PNGs
import matplotlib.pyplot as plt

from src.model import TransformerLM, TransformerConfig
from src.train import train, resolve_amp
from src.evaluate import evaluate, evaluate_by_position, chars_from_meta
from src.data import load_token_parts
from src.training_helpers.manage_checkpoint import load_checkpoint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# §4.1's model. Only the ablations change any of this, and each names what it
# changes rather than restating the whole config.
BASE_CONFIG = dict(vocab_size=4000, context_length=256, n_layers=4, d_model=512,
                   n_heads=8, d_ff=1344, rope_theta=10000.0, use_qk_norm=True,
                   use_rmsnorm=True, use_rope=True, ffn_type="swiglu")

# §7's standard run configuration. Every run inherits this; the sweep overrides
# only `num_steps`, and §7.4's final model only `num_steps` again.
STANDARD_RUN = dict(batch_size=32, num_steps=5000, warmup_steps=200,
                    weight_decay=0.1, grad_clip=1.0, eval_every=50,
                    save_every=0, seed=42)

SWEEP_LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]   # §7.1's suggested grid
RELU_D_FF = 2048                             # §7.2: matches SwiGLU's 1344 to within 2%
INIT_SEED = 0                                # model init, identical for every arm


# ---------------------------------------------------------------------------
# experiment records
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """One training run: a name, what it changes, and nothing else.

    `config_overrides` are TransformerConfig fields; `run_overrides` are train()
    arguments. Everything unnamed comes from BASE_CONFIG and STANDARD_RUN, so an
    ablation cannot silently differ from its baseline in a second way.
    """
    name: str
    phase: str                                   # for Q16's GPU-hours table
    lr: float
    config_overrides: dict = field(default_factory=dict)
    run_overrides: dict = field(default_factory=dict)
    note: str = ""

    def model_config(self, vocab_size=None):
        cfg = dict(BASE_CONFIG)
        cfg.update(self.config_overrides)
        if vocab_size is not None:
            cfg["vocab_size"] = vocab_size
        return cfg

    def run_kwargs(self):
        kwargs = dict(STANDARD_RUN)
        kwargs.update(self.run_overrides)
        kwargs["learning_rate"] = self.lr
        return kwargs


def build_model(config, device):
    """A model built from a fixed seed, so arms differ only by design."""
    torch.manual_seed(INIT_SEED)
    return TransformerLM(TransformerConfig(**config)).to(device)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def corpus_paths(data_dir, vocab_size, primary_vocab_size):
    """Where Task 1 §3.5 puts the encoded corpus for a given vocabulary size.

    Task 1 writes the training split as two arrays, because the corpus ships as
    two files, and suffixes the second tokenizer's outputs (`valid_vocab1000.npy`).
    """
    suffix = "" if vocab_size == primary_vocab_size else f"_vocab{vocab_size}"

    # A combined train.npy wins if scripts/combine_train_parts.py has been run;
    # otherwise the parts are read as one sequence. Both are the same tokens in
    # the same order, so which one is present changes nothing but bookkeeping.
    combined = os.path.join(data_dir, f"train{suffix}.npy")
    parts = [os.path.join(data_dir, f"train_part{i}{suffix}.npy") for i in (1, 2)]
    if os.path.exists(combined):
        train = [combined]
    else:
        train = [p for p in parts if os.path.exists(p)] or parts[:1]

    return {
        "train": train,
        # §2 reserves the last 2,000 validation documents as a test set to be
        # touched exactly once. Model selection therefore scores the *split*
        # validation array; the full valid.npy still contains those documents.
        "valid": os.path.join(data_dir, f"valid_split{suffix}.npy"),
        "suffix": suffix,
    }


def load_corpus(args, vocab_size):
    """Memory-mapped encoded arrays for a vocabulary size, plus char counts.

    Nothing here re-tokenizes: the encoded corpus from Task 1 is the input, and
    the two training parts are addressed as one sequence rather than copied
    into a single array (§5.5).
    """
    if args.smoke:
        return synthetic_corpus(vocab_size)

    paths = corpus_paths(args.data_dir, vocab_size, BASE_CONFIG["vocab_size"])

    if not os.path.exists(paths["valid"]):
        if paths["suffix"]:
            build = (f"  python scripts/make_splits.py "
                     f"--vocab vocab{vocab_size}_vocab.pkl "
                     f"--merges vocab{vocab_size}_merges.pkl "
                     f"--suffix {paths['suffix']}")
        else:
            build = ("  python scripts/make_splits.py "
                     "--vocab vocab.pkl --merges merges.pkl")
        raise SystemExit(
            f"{os.path.basename(paths['valid'])} is missing. Section 2's "
            f"validation/test split has to exist before any model selection "
            f"happens, or the reserved test documents get scored at every "
            f"evaluation. Build it with:\n" + build)

    missing = [p for p in paths["train"] if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "Task 4 trains on the encoded corpus from Task 1 section 3.5, and "
            "these are missing:\n  " +
            "\n  ".join(os.path.basename(p) for p in missing) +
            f"\n\nRun Task 1 phase 2 for vocab_size={vocab_size}, or point "
            f"--data_dir at wherever they already are. --smoke exercises this "
            f"script on synthetic data without them.")

    train_chars = sum(chars_from_meta(p)[0] or 0 for p in paths["train"])
    valid_chars, convention = chars_from_meta(paths["valid"])
    train_ids = load_token_parts(paths["train"])

    return {
        "train": train_ids,
        "valid": np.load(paths["valid"], mmap_mode="r"),
        "valid_path": paths["valid"],
        "valid_chars": valid_chars,
        "train_chars": train_chars,
        "char_convention": convention,
        "vocab_size": vocab_size,
        "source": (f"{len(paths['train'])} train part(s) + "
                   f"{os.path.basename(paths['valid'])}"),
    }


def synthetic_corpus(vocab_size):
    """Random tokens, for --smoke only. There is nothing here to learn."""
    rng = np.random.default_rng(0)
    return {
        "train": rng.integers(0, vocab_size, size=200_000, dtype=np.uint16),
        "valid": rng.integers(0, vocab_size, size=40_000, dtype=np.uint16),
        "valid_path": None,
        "valid_chars": 40_000 * 4,     # a plausible chars-per-token, for plumbing only
        "train_chars": 200_000 * 4,
        "char_convention": "synthetic",
        "vocab_size": vocab_size,
        "source": "synthetic (--smoke)",
    }


# ---------------------------------------------------------------------------
# the run cache
# ---------------------------------------------------------------------------

def run_meta_path(args, name):
    return os.path.join(args.log_dir, f"{name}_run.json")


def load_run(args, name):
    """A finished run's record, or None."""
    path = run_meta_path(args, name)
    if args.force or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ensure_run(args, exp, corpus, device):
    """Train `exp` unless it has already been run, and return its record.

    These are 5,000-step GPU jobs; Q14 alone reads six of them. Re-running the
    analysis must not re-run the training, so a finished run is reused unless
    --force says otherwise.
    """
    cached = load_run(args, exp.name)
    if cached is not None:
        status = "DIVERGED" if cached.get("diverged") else "ok"
        print(f"  [cached] {exp.name}: final val {cached['final_val_loss']:.4f} ({status})")
        return cached

    config = exp.model_config(corpus["vocab_size"])
    kwargs = exp.run_kwargs()
    print(f"\n--- {exp.name} ({exp.phase}) lr={exp.lr:.1e} "
          f"{exp.note or ''}".rstrip() + " ---")

    model = build_model(config, device)
    started = time.time()
    history = train(model, corpus["train"], corpus["valid"],
                    run_name=exp.name, log_dir=args.log_dir,
                    checkpoint_dir=args.checkpoint_dir,
                    use_amp=not args.no_amp, **kwargs)
    wall_seconds = time.time() - started

    record = {
        "name": exp.name,
        "phase": exp.phase,
        "note": exp.note,
        "learning_rate": exp.lr,
        "model_config": config,
        "run_config": {k: v for k, v in kwargs.items()},
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "diverged": history["diverged"],
        "diverged_at": history["diverged_at"],
        "diverged_reason": history["diverged_reason"],
        "completed_steps": history["completed_steps"],
        "final_val_loss": history["val_loss"][-1],
        "final_train_loss": history["train_loss"][-1],
        "wall_seconds": wall_seconds,
        "device": str(device),
        "amp_enabled": history["config"]["amp_enabled"],
        "checkpoint": history["final_checkpoint"],
        "eval_log": history["eval_log"],
        "data_source": corpus["source"],
    }
    with open(run_meta_path(args, exp.name), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def read_curve(record):
    """(steps, val_loss) from a run's evaluation CSV (§5.5: plots come from logs)."""
    with open(record["eval_log"], newline="") as f:
        rows = list(csv.DictReader(f))
    return [int(r["step"]) for r in rows], [float(r["val_loss"]) for r in rows]


def load_trained_model(record, device):
    """Rebuild a run's model from its recorded config and restore its weights."""
    model = TransformerLM(TransformerConfig(**record["model_config"])).to(device)
    load_checkpoint(record["checkpoint"], model, map_location=device,
                    restore_rng=False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# experiment definitions
# ---------------------------------------------------------------------------

def sweep_experiments(args):
    """§7.1: at least five peak learning rates over at least two orders of
    magnitude, at a reduced step count with the cosine period matched to it."""
    return [Experiment(name=f"lr_sweep_{lr:.0e}", phase="lr_sweep", lr=lr,
                       run_overrides={"num_steps": args.sweep_steps,
                                      "warmup_steps": min(STANDARD_RUN["warmup_steps"],
                                                          max(1, args.sweep_steps // 10))},
                       note=f"sweep at {args.sweep_steps} steps")
            for lr in args.sweep_lrs]


def baseline_experiment(best_lr):
    """§7.1: the standard run at the best learning rate. Everything else is
    compared against this one run."""
    return Experiment(name="baseline", phase="baseline", lr=best_lr,
                      note="standard run, the comparison point for sections 7.2-7.4")


def ablation_experiments(best_lr, reduced_lr):
    """§7.2's three ablations, each one standard run against the baseline."""
    return [
        Experiment(name="ablation_no_rmsnorm", phase="ablations", lr=best_lr,
                   config_overrides={"use_rmsnorm": False},
                   note="RMSNorm removed from blocks and final norm"),
        # §7.2 asks for a reduced learning rate as well, because this ablation
        # is the one that typically will not train at the baseline's lr.
        Experiment(name="ablation_no_rmsnorm_lowlr", phase="ablations", lr=reduced_lr,
                   config_overrides={"use_rmsnorm": False},
                   note="same ablation at a reduced learning rate (section 7.2)"),
        Experiment(name="ablation_nope", phase="ablations", lr=best_lr,
                   config_overrides={"use_rope": False},
                   note="RoPE removed entirely"),
        Experiment(name="ablation_relu", phase="ablations", lr=best_lr,
                   config_overrides={"ffn_type": "relu", "d_ff": RELU_D_FF},
                   note=f"ReLU FFN at d_ff={RELU_D_FF}, parameter-matched"),
    ]


def reduced_lr(args, best_lr):
    """Section 7.2's lower learning rate for the no-RMSNorm run.

    Relative to the best learning rate rather than absolute: a fixed value ends
    up *above* the baseline whenever the sweep picks something smaller, which
    would make "the same ablation at a reduced learning rate" untrue.
    """
    if args.reduced_lr is not None:
        return args.reduced_lr
    return best_lr / args.reduced_lr_factor


def resolve_best_lr(args, results):
    """The best learning rate: from Q10 if it has been run, else --best_lr."""
    q10 = results.get("q10") or (load_run(args, "_q10_summary") or {})
    if args.best_lr is not None:
        return args.best_lr, "--best_lr"
    if q10.get("best_lr") is not None:
        return q10["best_lr"], "Q10 sweep"
    raise SystemExit(
        "No best learning rate available. Run the sweep first:\n"
        "  python scripts/run_task4_questions.py --questions 10\n"
        "or pass one explicitly with --best_lr.")


# ---------------------------------------------------------------------------
# Q10 - learning-rate sweep
# ---------------------------------------------------------------------------

def run_q10(args, corpus, device, results):
    print("\n" + "=" * 70)
    print("Q10: learning-rate sweep")
    print("=" * 70)

    records = [ensure_run(args, exp, corpus, device)
               for exp in sweep_experiments(args)]

    converged = [r for r in records if not r["diverged"]]
    diverged = [r for r in records if r["diverged"]]
    if not converged:
        raise SystemExit("every sweep run diverged - lower the grid and re-run")

    ranked = sorted(converged, key=lambda r: r["final_val_loss"])
    best, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
    lowest_diverging = min((r["learning_rate"] for r in diverged), default=None)

    png = plot_q10(args, records)

    print(f"\n[Q10] {len(records)} runs at {args.sweep_steps} steps:")
    print(f"  {'lr':>10} {'final val':>11} {'steps':>7}  status")
    for r in sorted(records, key=lambda r: r["learning_rate"]):
        status = f"DIVERGED at {r['diverged_at']}" if r["diverged"] else "ok"
        print(f"  {r['learning_rate']:>10.0e} {r['final_val_loss']:>11.4f} "
              f"{r['completed_steps']:>7}  {status}")
    print(f"\n  best learning rate:      {best['learning_rate']:.0e} "
          f"(val {best['final_val_loss']:.4f})")
    if second:
        print(f"  second best:             {second['learning_rate']:.0e} "
              f"(val {second['final_val_loss']:.4f}, "
              f"+{second['final_val_loss'] - best['final_val_loss']:.4f})")
    print(f"  lowest diverging lr:     "
          f"{f'{lowest_diverging:.0e}' if lowest_diverging else 'none diverged'}")
    if lowest_diverging:
        print(f"  usable band:             {best['learning_rate']:.0e} to "
              f"{lowest_diverging:.0e} - a factor of "
              f"{lowest_diverging / best['learning_rate']:.1f}")
    print(f"  figure: {png}")

    results["q10"] = {
        "sweep_steps": args.sweep_steps,
        "runs": [{k: r[k] for k in ("name", "learning_rate", "final_val_loss",
                                    "diverged", "diverged_at", "completed_steps")}
                 for r in records],
        "best_lr": best["learning_rate"],
        "best_val_loss": best["final_val_loss"],
        "second_best_lr": second["learning_rate"] if second else None,
        "second_best_val_loss": second["final_val_loss"] if second else None,
        "lr_gap": (second["final_val_loss"] - best["final_val_loss"]) if second else None,
        "lowest_diverging_lr": lowest_diverging,
        "margin_to_divergence": (lowest_diverging / best["learning_rate"]
                                 if lowest_diverging else None),
        "figure": os.path.basename(png),
    }
    return results["q10"]


def plot_q10(args, records):
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(records)))
    for colour, r in zip(colours, sorted(records, key=lambda r: r["learning_rate"])):
        steps, val = read_curve(r)
        label = f"{r['learning_rate']:.0e}" + (" (diverged)" if r["diverged"] else "")
        ax.plot(steps, val, "--" if r["diverged"] else "-", color=colour,
                label=label, linewidth=1.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation loss")
    ax.set_title(f"Learning-rate sweep ({args.sweep_steps} steps)")

    # A diverged run reaches a loss many times the others', which flattens every
    # converged curve into one indistinguishable band. Scale the axis to the
    # runs that trained and let the divergent one leave the top of the plot -
    # it is still on the figure, and running off the axis reads as divergence.
    converged = [read_curve(r)[1] for r in records if not r["diverged"]]
    if converged and any(r["diverged"] for r in records):
        low = min(min(v) for v in converged)
        high = max(max(v) for v in converged)
        margin = 0.15 * (high - low) or 0.5
        ax.set_ylim(low - margin, high + margin)
        ax.annotate("divergent run continues off-scale", xy=(0.02, 0.95),
                    xycoords="axes fraction", fontsize=8, color="grey")

    ax.legend(title="peak lr", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(args.out_dir, "task4_q10_lr_sweep.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# Q11-Q13 - ablations
# ---------------------------------------------------------------------------

def ablation_figure(args, baseline, arms, title, filename, ylabel="Validation loss"):
    """One ablation against the baseline, from the run logs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    steps, val = read_curve(baseline)
    ax.plot(steps, val, "-", color="tab:blue", label="baseline", linewidth=1.8)
    for colour, r in zip(("tab:red", "tab:orange", "tab:green"), arms):
        s, v = read_curve(r)
        label = r["note"] or r["name"]
        if r["diverged"]:
            label += " (diverged)"
        ax.plot(s, v, "--", color=colour, label=label, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(args.out_dir, filename)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def run_ablation(args, corpus, device, results, question, names, title, filename):
    """Shared body of Q11-Q13: train the baseline and the named arms, plot, report."""
    best_lr, lr_source = resolve_best_lr(args, results)
    baseline = ensure_run(args, baseline_experiment(best_lr), corpus, device)

    by_name = {e.name: e for e in ablation_experiments(best_lr, reduced_lr(args, best_lr))}
    arms = [ensure_run(args, by_name[n], corpus, device) for n in names]

    png = ablation_figure(args, baseline, arms, title, filename)

    print(f"\n[{question.upper()}] against the baseline "
          f"(lr {best_lr:.0e}, from {lr_source}):")
    print(f"  {'run':<32} {'final val':>11} {'gap':>9}")
    print(f"  {'baseline':<32} {baseline['final_val_loss']:>11.4f} {'-':>9}")
    entries = []
    for r in arms:
        gap = r["final_val_loss"] - baseline["final_val_loss"]
        flag = "  DIVERGED" if r["diverged"] else ""
        print(f"  {r['name']:<32} {r['final_val_loss']:>11.4f} {gap:>+9.4f}{flag}")
        entries.append({"name": r["name"], "note": r["note"],
                        "learning_rate": r["learning_rate"],
                        "final_val_loss": r["final_val_loss"], "gap": gap,
                        "diverged": r["diverged"]})
    print(f"  figure: {png}")

    results[question] = {
        "best_lr": best_lr,
        "baseline_val_loss": baseline["final_val_loss"],
        "arms": entries,
        "figure": os.path.basename(png),
    }
    return results[question]


def run_q11(args, corpus, device, results):
    print("\n" + "=" * 70)
    print("Q11: RMSNorm ablation")
    print("=" * 70)
    return run_ablation(
        args, corpus, device, results, "q11",
        ["ablation_no_rmsnorm", "ablation_no_rmsnorm_lowlr"],
        "No RMSNorm against the baseline", "task4_q11_rmsnorm.png")


def run_q12(args, corpus, device, results):
    print("\n" + "=" * 70)
    print("Q12: positional encoding ablation (NoPE)")
    print("=" * 70)
    return run_ablation(
        args, corpus, device, results, "q12", ["ablation_nope"],
        "NoPE against the RoPE baseline", "task4_q12_nope.png")


def run_q13(args, corpus, device, results):
    print("\n" + "=" * 70)
    print("Q13: SwiGLU against a parameter-matched ReLU FFN")
    print("=" * 70)
    out = run_ablation(
        args, corpus, device, results, "q13", ["ablation_relu"],
        f"SwiGLU against ReLU (d_ff={RELU_D_FF})", "task4_q13_swiglu_relu.png")

    # §7.2 claims the two are matched to within 2%; state the actual counts so
    # the comparison can be read as parameter-matched rather than assumed to be.
    baseline = load_run(args, "baseline")
    relu = load_run(args, "ablation_relu")
    if baseline and relu:
        delta = abs(relu["n_parameters"] - baseline["n_parameters"]) / baseline["n_parameters"]
        print(f"  parameters: SwiGLU {baseline['n_parameters']:,} vs "
              f"ReLU {relu['n_parameters']:,} ({delta:.2%} apart)")
        out["parameter_delta_fraction"] = delta
        out["swiglu_parameters"] = baseline["n_parameters"]
        out["relu_parameters"] = relu["n_parameters"]
    return out


# ---------------------------------------------------------------------------
# Q14 - the gaps side by side
# ---------------------------------------------------------------------------

def run_q14(args, corpus, device, results):
    """§7.5: the ablation gaps against the Q10 learning-rate gap, one table."""
    print("\n" + "=" * 70)
    print("Q14: which design choice mattered most")
    print("=" * 70)

    q10 = results.get("q10")
    if q10 is None:
        raise SystemExit("Q14 reads Q10's learning-rate gap - run --questions 10 first.")

    baseline = load_run(args, "baseline")
    if baseline is None:
        raise SystemExit("Q14 needs the baseline run - run --questions 11 (or 12, 13) first.")

    rows = []
    if q10.get("lr_gap") is not None:
        rows.append({
            "choice": f"learning rate ({q10['best_lr']:.0e} vs "
                      f"{q10['second_best_lr']:.0e})",
            "gap": q10["lr_gap"],
            "note": "best vs second-best in the sweep"})

    for name, label in (("ablation_no_rmsnorm", "RMSNorm"),
                        ("ablation_nope", "RoPE (vs NoPE)"),
                        ("ablation_relu", "SwiGLU (vs ReLU)")):
        record = load_run(args, name)
        if record is None:
            print(f"  (skipping {label} - {name} has not been run)")
            continue
        rows.append({
            "choice": label,
            "gap": record["final_val_loss"] - baseline["final_val_loss"],
            "note": ("ablation diverged" if record["diverged"]
                     else "ablation minus baseline")})

    rows.sort(key=lambda r: abs(r["gap"]), reverse=True)

    print(f"\n  {'design choice':<38} {'val loss gap':>13}  note")
    for r in rows:
        print(f"  {r['choice']:<38} {r['gap']:>+13.4f}  {r['note']}")
    if rows:
        print(f"\n  Largest effect at this scale: {rows[0]['choice']} "
              f"({rows[0]['gap']:+.4f}).")

    results["q14"] = {"baseline_val_loss": baseline["final_val_loss"], "rows": rows}
    return results["q14"]


# ---------------------------------------------------------------------------
# Q15 - vocabulary size
# ---------------------------------------------------------------------------

def run_q15(args, corpus, device, results):
    """§7.3: one standard run per tokenizer, compared on BPC rather than
    perplexity - the two vocabularies are not counting the same events."""
    print("\n" + "=" * 70)
    print(f"Q15: vocabulary size {BASE_CONFIG['vocab_size']} against {args.second_vocab_size}")
    print("=" * 70)

    if args.second_vocab_size == BASE_CONFIG["vocab_size"]:
        # §7.3 is a comparison against a *clearly different* vocabulary size.
        # Comparing a tokenizer with itself would print two identical rows and
        # a conclusion about which is better, which is worse than not answering.
        print(f"  ! --second_vocab_size equals the primary vocabulary "
              f"({args.second_vocab_size}); section 7.3 needs a clearly different "
              f"one. Skipping Q15.")
        results["q15"] = {"skipped": "second_vocab_size equals the primary vocabulary"}
        return results["q15"]

    best_lr, _ = resolve_best_lr(args, results)
    entries = []

    for vocab_size in (BASE_CONFIG["vocab_size"], args.second_vocab_size):
        primary = vocab_size == BASE_CONFIG["vocab_size"]
        vocab_corpus = corpus if primary else load_corpus(args, vocab_size)
        exp = (baseline_experiment(best_lr) if primary else
               Experiment(name=f"vocab_{vocab_size}", phase="vocab_study", lr=best_lr,
                          note=f"second tokenizer, vocab {vocab_size} (section 7.3)"))
        record = ensure_run(args, exp, vocab_corpus, device)

        model = load_trained_model(record, device)
        metrics = evaluate(model, vocab_corpus["valid"], args.eval_batch_size,
                           record["model_config"]["context_length"], device,
                           total_chars=vocab_corpus["valid_chars"],
                           amp_enabled=resolve_amp(not args.no_amp, device)[0],
                           max_windows=args.eval_windows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        entries.append({
            "vocab_size": vocab_size, "run": record["name"],
            "n_parameters": record["n_parameters"],
            "loss": metrics["loss"], "perplexity": metrics["perplexity"],
            "bpc": metrics.get("bpc"), "n_tokens": metrics["n_tokens"],
            "chars_per_token": metrics.get("chars_per_token"),
            "char_convention": vocab_corpus["char_convention"],
        })

    print(f"\n[Q15] held-out validation, characters counted "
          f"{entries[0]['char_convention']}:")
    print(f"  {'vocab':>7} {'params':>12} {'loss':>8} {'perplexity':>12} {'BPC':>8} {'chars/tok':>10}")
    for e in entries:
        bpc = f"{e['bpc']:.4f}" if e["bpc"] is not None else "n/a"
        cpt = f"{e['chars_per_token']:.3f}" if e["chars_per_token"] else "n/a"
        print(f"  {e['vocab_size']:>7} {e['n_parameters']:>12,} {e['loss']:>8.4f} "
              f"{e['perplexity']:>12.2f} {bpc:>8} {cpt:>10}")

    if all(e["bpc"] is not None for e in entries):
        better = min(entries, key=lambda e: e["bpc"])
        print(f"\n  Lower BPC: vocab {better['vocab_size']} "
              f"({better['bpc']:.4f}). Perplexity is not comparable across these "
              f"two - they are not counting the same events - so BPC decides it.")
        results_better = better["vocab_size"]
    else:
        results_better = None

    results["q15"] = {"models": entries, "better_on_bpc": results_better}
    return results["q15"]


# ---------------------------------------------------------------------------
# Q16 - GPU hours
# ---------------------------------------------------------------------------

def run_q16(args, corpus, device, results):
    """§7.5: hours by phase, from the per-run records rather than a tally kept
    by hand."""
    print("\n" + "=" * 70)
    print("Q16: GPU-hours by phase")
    print("=" * 70)

    phases = {}
    for filename in sorted(os.listdir(args.log_dir)):
        if not filename.endswith("_run.json"):
            continue
        with open(os.path.join(args.log_dir, filename), encoding="utf-8") as f:
            record = json.load(f)
        phase = phases.setdefault(record.get("phase", "other"),
                                  {"seconds": 0.0, "runs": 0, "steps": 0})
        phase["seconds"] += record.get("wall_seconds", 0.0)
        phase["runs"] += 1
        phase["steps"] += record.get("completed_steps", 0)

    measured = sum(p["seconds"] for p in phases.values())
    # Development and debugging is not in the logs by construction: it is the
    # time spent before any run was worth keeping. It has to be supplied.
    dev_hours = args.dev_hours

    print(f"\n  {'phase':<16} {'runs':>5} {'steps':>9} {'hours':>8}")
    for name, p in sorted(phases.items(), key=lambda kv: -kv[1]["seconds"]):
        print(f"  {name:<16} {p['runs']:>5} {p['steps']:>9,} {p['seconds']/3600:>8.2f}")
    print(f"  {'development':<16} {'-':>5} {'-':>9} {dev_hours:>8.2f}  (--dev_hours)")
    print(f"  {'TOTAL':<16} {'':>5} {'':>9} {measured/3600 + dev_hours:>8.2f}")

    if device.type != "cuda":
        print("\n  ! These are wall-clock hours on "
              f"{device.type}, not GPU-hours. Re-run on the GPU for the reported figure.")

    results["q16"] = {
        "phases": {k: {**v, "hours": v["seconds"] / 3600} for k, v in phases.items()},
        "development_hours": dev_hours,
        "measured_hours": measured / 3600,
        "total_hours": measured / 3600 + dev_hours,
        "device": str(device),
        "is_gpu": device.type == "cuda",
    }
    return results["q16"]


# ---------------------------------------------------------------------------
# Q17 - position-wise loss
# ---------------------------------------------------------------------------

def run_q17(args, corpus, device, results):
    """§7.5: mean validation loss by position, bucketed, for NoPE and RoPE.

    The aggregate curve averages over the whole window, so it cannot show
    *where* a model without positional information loses out. This can.
    """
    print("\n" + "=" * 70)
    print("Q17: position-wise validation loss")
    print("=" * 70)

    best_lr, _ = resolve_best_lr(args, results)
    ensure_run(args, baseline_experiment(best_lr), corpus, device)
    by_name = {e.name: e for e in ablation_experiments(best_lr, reduced_lr(args, best_lr))}
    ensure_run(args, by_name["ablation_nope"], corpus, device)

    amp = resolve_amp(not args.no_amp, device)[0]
    series = {}
    for label, name in (("RoPE (baseline)", "baseline"), ("NoPE", "ablation_nope")):
        record = load_run(args, name)
        model = load_trained_model(record, device)
        series[label] = evaluate_by_position(
            model, corpus["valid"], args.eval_batch_size,
            record["model_config"]["context_length"], device,
            n_buckets=args.position_buckets, amp_enabled=amp,
            max_windows=args.eval_windows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    png = plot_q17(args, series)

    print(f"\n[Q17] mean validation loss by position "
          f"({series['NoPE']['n_windows']:,} windows):")
    header = "  " + f"{'positions':<14}" + "".join(f"{k:>18}" for k in series)
    print(header)
    for i in range(args.position_buckets):
        span = series["NoPE"]["buckets"][i]
        label = f"{span['start']}-{span['end']}"
        row = "  " + f"{label:<14}"
        row += "".join(f"{s['buckets'][i]['loss']:>18.4f}" for s in series.values())
        print(row)

    first = {k: s["buckets"][0]["loss"] for k, s in series.items()}
    last = {k: s["buckets"][-1]["loss"] for k, s in series.items()}
    print("\n  Improvement from the first bucket to the last:")
    for k in series:
        print(f"    {k:<18} {first[k] - last[k]:+.4f}")

    results["q17"] = {
        "buckets": {k: s["buckets"] for k, s in series.items()},
        "per_position": {k: s["per_position"] for k, s in series.items()},
        "n_windows": series["NoPE"]["n_windows"],
        "figure": os.path.basename(png),
    }
    return results["q17"]


def plot_q17(args, series):
    fig, ax = plt.subplots(figsize=(8, 5))
    for colour, (label, s) in zip(("tab:blue", "tab:red"), series.items()):
        centres = [(b["start"] + b["end"]) / 2 for b in s["buckets"]]
        ax.plot(centres, [b["loss"] for b in s["buckets"]], "-o", color=colour,
                label=label, markersize=4)
    ax.set_xlabel("Position in the context window")
    ax.set_ylabel("Mean validation loss")
    ax.set_title("Validation loss by position")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    png = os.path.join(args.out_dir, "task4_q17_position.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# §7.4 - the final model
# ---------------------------------------------------------------------------

def run_final_model(args, corpus, device, results):
    """§7.4: the best configuration for roughly four times the standard step
    count, cosine period set to the full run. Task 5 evaluates this one."""
    print("\n" + "=" * 70)
    print("Section 7.4: final model")
    print("=" * 70)

    best_lr, _ = resolve_best_lr(args, results)
    steps = args.final_steps or 4 * STANDARD_RUN["num_steps"]
    exp = Experiment(name="final_model", phase="final_model", lr=best_lr,
                     run_overrides={"num_steps": steps,
                                    "save_every": args.final_save_every},
                     note=f"section 7.4: {steps} steps, cosine period matched")
    record = ensure_run(args, exp, corpus, device)

    print(f"\n[§7.4] final model: {record['completed_steps']} steps, "
          f"val {record['final_val_loss']:.4f}")
    print(f"  checkpoint: {record['checkpoint']}  (Task 5 evaluates this)")
    results["final_model"] = record
    return record


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def save_results(results, log_dir):
    path = os.path.join(log_dir, "task4_results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prior = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            prior = json.load(f)
    prior.update(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prior, f, indent=2, default=str)
    return path


def describe_machine(device):
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "device": str(device)}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def print_plan(args, results):
    """What a full run would train, and roughly what it costs."""
    try:
        best_lr, source = resolve_best_lr(args, results)
    except SystemExit:
        best_lr, source = float("nan"), "not yet known (run Q10 first)"

    planned = sweep_experiments(args) + [baseline_experiment(best_lr)] + \
        ablation_experiments(best_lr, reduced_lr(args, best_lr))
    total_steps = sum(e.run_kwargs()["num_steps"] for e in planned)

    print(f"Best learning rate: {source}")
    print(f"\n  {'run':<32} {'phase':<12} {'steps':>7} {'lr':>9}  cached")
    for e in planned:
        cached = "yes" if load_run(args, e.name) else "no"
        lr = "TBD" if math.isnan(e.lr) else f"{e.lr:.0e}"
        print(f"  {e.name:<32} {e.phase:<12} {e.run_kwargs()['num_steps']:>7} "
              f"{lr:>9}  {cached}")
    print(f"\n  {total_steps:,} steps total for Q10-Q14 and Q17 "
          f"(plus section 7.4's final model at "
          f"{args.final_steps or 4 * STANDARD_RUN['num_steps']:,}).")
    print("  At ~0.1 s/step on an A100 that is roughly "
          f"{total_steps * 0.1 / 3600:.1f} GPU-hours; measure your own step time "
          "with Task 3's Q8 and scale.")


def apply_smoke_settings(args):
    """Tiny everything, into logs/smoke/, so the pipeline can be checked first."""
    BASE_CONFIG.update(context_length=64, n_layers=2, d_model=128, n_heads=4, d_ff=128)
    STANDARD_RUN.update(batch_size=8, num_steps=30, warmup_steps=5, eval_every=10,
                        save_every=0)
    args.sweep_steps = 20
    args.sweep_lrs = [1e-4, 1e-2, 1.0]     # 1.0 is there to make something diverge
    args.eval_windows = 20
    args.final_steps = 40
    args.position_buckets = 4
    args.second_vocab_size = BASE_CONFIG["vocab_size"]   # no second corpus in smoke
    args.log_dir = os.path.join(args.log_dir, "smoke")
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, "smoke")
    args.out_dir = os.path.join(REPO_ROOT, "logs", "smoke")
    print("SMOKE RUN - tiny model on synthetic data, into logs/smoke/. "
          "Not a Q10-Q17 answer.\n")


def build_parser():
    p = argparse.ArgumentParser(
        description="Task 4: experiments. Answers Q10-Q17 (section 7.5).")
    p.add_argument("--questions", nargs="+", type=int,
                   default=[10, 11, 12, 13, 14, 15, 16, 17],
                   choices=[10, 11, 12, 13, 14, 15, 16, 17])
    p.add_argument("--final_model", action="store_true",
                   help="Also train section 7.4's final model (4x the standard steps).")
    p.add_argument("--plan", action="store_true",
                   help="Print what would be trained and roughly what it costs, "
                        "then stop.")
    p.add_argument("--force", action="store_true",
                   help="Retrain runs that are already cached.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny model on synthetic data, into logs/smoke/.")
    # §7.1
    p.add_argument("--sweep_lrs", nargs="+", type=float, default=SWEEP_LRS)
    p.add_argument("--sweep_steps", type=int, default=1000,
                   help="Reduced step count for the sweep; the cosine period "
                        "follows it (section 7.1).")
    p.add_argument("--best_lr", type=float, default=None,
                   help="Skip Q10 and use this learning rate as the baseline's.")
    p.add_argument("--reduced_lr", type=float, default=None,
                   help="The lower learning rate for section 7.2's no-RMSNorm "
                        "second run. Defaults to best_lr / --reduced_lr_factor.")
    p.add_argument("--reduced_lr_factor", type=float, default=3.0,
                   help="How far below the best learning rate the reduced run "
                        "sits when --reduced_lr is not given.")
    p.add_argument("--final_save_every", type=int, default=2000,
                   help="Checkpoint interval for section 7.4's long run, which "
                        "is the only one worth being able to resume.")
    p.add_argument("--final_steps", type=int, default=None,
                   help="Section 7.4's run length. Defaults to 4x the standard run.")
    # evaluation
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--eval_windows", type=int, default=None,
                   help="Cap on validation windows scored, for a quicker estimate.")
    p.add_argument("--position_buckets", type=int, default=8)
    p.add_argument("--second_vocab_size", type=int, default=1000,
                   help="Section 7.3's comparison tokenizer.")
    p.add_argument("--dev_hours", type=float, default=0.0,
                   help="Development and debugging hours for Q16; not in the logs.")
    # plumbing
    p.add_argument("--data_dir", default=REPO_ROOT)
    p.add_argument("--log_dir", default=os.path.join(REPO_ROOT, "logs"))
    p.add_argument("--checkpoint_dir", default=os.path.join(REPO_ROOT, "checkpoints"))
    p.add_argument("--out_dir", default=REPO_ROOT)
    p.add_argument("--device", default=None)
    p.add_argument("--no_amp", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        apply_smoke_settings(args)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    existing = os.path.join(args.log_dir, "task4_results.json")
    if os.path.exists(existing):
        with open(existing, encoding="utf-8") as f:
            results.update(json.load(f))

    if args.plan:
        print_plan(args, results)
        return

    corpus = load_corpus(args, BASE_CONFIG["vocab_size"])
    print(f"Device: {device}")
    print(f"Data:   {corpus['source']} "
          f"({len(corpus['train']):,} train / {len(corpus['valid']):,} val tokens)")

    results["machine"] = describe_machine(device)
    results["base_config"] = dict(BASE_CONFIG)
    results["standard_run"] = dict(STANDARD_RUN)

    handlers = {10: run_q10, 11: run_q11, 12: run_q12, 13: run_q13,
                14: run_q14, 15: run_q15, 16: run_q16, 17: run_q17}
    for q in sorted(args.questions):
        handlers[q](args, corpus, device, results)

    if args.final_model:
        run_final_model(args, corpus, device, results)

    path = save_results(results, args.log_dir)
    print("\n" + "=" * 70)
    print("TASK 4 COMPLETE")
    print("=" * 70)
    print(f"All numbers written to {path}")


if __name__ == "__main__":
    main()