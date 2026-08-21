"""Regenerate every figure in the report from the files in logs/.

Section 5.5: "Every plot in your report must be generated from these logs by a
script in your repo." The run_task*_questions.py scripts each plot their own
figures as a side effect of running, which means reproducing a figure means
re-running (or at least re-loading) the experiment. This script is the other
half: it reads only logs/, so the figures can be rebuilt - restyled, relabelled,
rescaled - without touching a checkpoint or a GPU.

    python scripts/make_plots.py                 # all figures -> figures/
    python scripts/make_plots.py --only q10 q17

Section 10.2 asks for axes labelled with units, a legend on every figure, and
the same axis limits when curves are meant to be compared. The three ablation
figures (Q11-Q13) are therefore drawn on one shared y-axis range computed
across all of them, so the panels can be read against each other directly.
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# One size for every single-panel figure, so they line up when the report puts
# two of them side by side.
PANEL = (5.2, 3.4)
WIDE = (7.4, 3.0)

STEP_LABEL = "Optimiser step"
LOSS_LABEL = "Validation loss (nats / token)"


# ---------------------------------------------------------------------------
# log readers
# ---------------------------------------------------------------------------

def read_eval_log(log_dir, name):
    """logs/<name>.csv - one row per evaluation (section 5.5)."""
    path = os.path.join(log_dir, name + ".csv")
    steps, val, train, grad = [], [], [], []
    seen = set()
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            step = int(row["step"])
            # q8_*.csv is appended to by both arms of the timing run, so the
            # same steps appear twice; keep the first occurrence of each.
            if step in seen:
                continue
            seen.add(step)
            steps.append(step)
            val.append(float(row["val_loss"]))
            train.append(float(row["train_loss"]))
            grad.append(float(row["grad_norm"]))
    return {"step": np.array(steps), "val": np.array(val),
            "train": np.array(train), "grad": np.array(grad)}


def read_step_log(log_dir, name):
    """logs/<name>_steps.csv - per-step pre-clipping gradient norm and step time."""
    path = os.path.join(log_dir, name + "_steps.csv")
    steps, grad = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            steps.append(int(row["step"]))
            grad.append(float(row["grad_norm"]))
    return np.array(steps), np.array(grad)


def read_json(log_dir, name):
    with open(os.path.join(log_dir, name), encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, out_dir, filename, dpi):
    fig.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("  wrote " + path)
    return path


def clamp_to_converged(ax, curves, diverged_present, pad=0.12):
    """Scale y to the runs that trained.

    A diverged run reaches a loss two orders of magnitude above the others, and
    on a shared axis that flattens every converged curve into one band. Scaling
    to the converged runs lets the divergent one leave the top of the plot,
    which is itself a readable statement of what happened.
    """
    if not curves:
        return
    low = min(float(np.min(c)) for c in curves)
    high = max(float(np.max(c)) for c in curves)
    margin = pad * (high - low) or 0.5
    ax.set_ylim(low - margin, high + margin)
    if diverged_present:
        ax.annotate("divergent run continues off-scale", xy=(0.03, 0.93),
                    xycoords="axes fraction", fontsize=7, color="0.35")


# ---------------------------------------------------------------------------
# Q3 - compression ratio against vocabulary size
# ---------------------------------------------------------------------------

def fig_q3(args):
    rows = []
    with open(os.path.join(args.log_dir, "vocab_study.csv"), newline="",
              encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((int(row["vocab_size"]), float(row["bytes_per_token"]),
                         int(row["embed_lm_head_params"])))
    rows.sort()
    vocab = np.array([r[0] for r in rows])
    bpt = np.array([r[1] for r in rows])
    params = np.array([r[2] for r in rows]) / 1e6

    fig, ax = plt.subplots(figsize=PANEL)
    ax.plot(vocab, bpt, "o-", color="tab:blue", label="compression")
    ax.set_xscale("log")
    ax.set_xticks(vocab)
    ax.set_xticklabels([str(v) for v in vocab])
    ax.set_xlabel("Vocabulary size (tokens)")
    ax.set_ylabel("Compression (bytes / token)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, alpha=0.3)

    # The second axis is the reason the curve alone does not decide the
    # question: past ~4k the compression gain is flat while the embedding and
    # LM head keep growing linearly.
    ax2 = ax.twinx()
    ax2.plot(vocab, params, "s--", color="tab:red", label="embedding + LM head")
    ax2.set_ylabel("Embedding + LM head (M parameters)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=7,
              loc="upper left")
    ax.set_title("BPE compression and parameter cost vs vocabulary size",
                 fontsize=10)
    return save(fig, args.out_dir, "fig_q3_compression.png", args.dpi)


# ---------------------------------------------------------------------------
# Q7 - generation throughput with and without the KV cache
# ---------------------------------------------------------------------------

def fig_q7(args):
    q7 = read_json(args.log_dir, "task2_results.json")["q7"]
    rows = q7["rows"]
    x = [r["max_new_tokens"] for r in rows]
    cached = [r["cached"]["tokens_per_second"] for r in rows]
    uncached = [r["uncached"]["tokens_per_second"] for r in rows]

    fig, ax = plt.subplots(figsize=PANEL)
    ax.plot(x, cached, "o-", color="tab:blue", label="with KV cache")
    ax.plot(x, uncached, "s--", color="tab:red", label="without KV cache")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel("Tokens generated (max_new_tokens)")
    ax.set_ylabel("Throughput (tokens / second)")
    ax.set_ylim(0, max(cached + uncached) * 1.25)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Generation throughput (" + q7["device"] + ", batch 1)",
                 fontsize=10)
    return save(fig, args.out_dir, "fig_q7_throughput.png", args.dpi)


# ---------------------------------------------------------------------------
# Q9 - warmup against no warmup
# ---------------------------------------------------------------------------

def fig_q9(args):
    with_w = read_eval_log(args.log_dir, "q9_with_warmup")
    no_w = read_eval_log(args.log_dir, "q9_no_warmup")
    gw_step, gw = read_step_log(args.log_dir, "q9_with_warmup")
    gn_step, gn = read_step_log(args.log_dir, "q9_no_warmup")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=WIDE)

    ax1.plot(with_w["step"], with_w["val"], "-", color="tab:blue",
             label="200-step warmup")
    ax1.plot(no_w["step"], no_w["val"], "--", color="tab:red", label="no warmup")
    ax1.set_xlabel(STEP_LABEL)
    ax1.set_ylabel(LOSS_LABEL)
    ax1.set_ylim(1.5, 5.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.set_title("Validation loss", fontsize=10)

    # Per-step, not per-evaluation: the whole effect of warmup is in the first
    # ~50 steps and an every-50-steps log samples straight past it.
    ax2.plot(gw_step, gw, "-", color="tab:blue", linewidth=0.8,
             label="200-step warmup")
    ax2.plot(gn_step, gn, "-", color="tab:red", linewidth=0.8, label="no warmup")
    ax2.set_yscale("log")
    ax2.set_xlabel(STEP_LABEL)
    ax2.set_ylabel("Gradient norm (pre-clipping, L2)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title("Pre-clipping gradient norm", fontsize=10)

    fig.suptitle("Effect of warmup at lr_max = 3e-3 (2,000 steps)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(args.out_dir, "fig_q9_warmup.png")
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    print("  wrote " + path)
    return path


# ---------------------------------------------------------------------------
# Q10 - learning-rate sweep
# ---------------------------------------------------------------------------

def fig_q10(args):
    results = read_json(args.log_dir, "task4_results.json")["q10"]
    runs = sorted(results["runs"], key=lambda r: r["learning_rate"])
    colours = plt.cm.viridis(np.linspace(0, 0.88, len(runs)))

    fig, ax = plt.subplots(figsize=PANEL)
    converged = []
    for colour, run in zip(colours, runs):
        curve = read_eval_log(args.log_dir, run["name"])
        label = "%.0e" % run["learning_rate"]
        if run["diverged"]:
            label += " (diverged)"
        else:
            converged.append(curve["val"])
        ax.plot(curve["step"], curve["val"], "--" if run["diverged"] else "-",
                color=colour, label=label, linewidth=1.5)

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel(LOSS_LABEL)
    clamp_to_converged(ax, converged, any(r["diverged"] for r in runs))
    ax.grid(True, alpha=0.3)
    ax.legend(title="peak learning rate", fontsize=7, title_fontsize=7)
    ax.set_title("Learning-rate sweep (%d steps, cosine period matched)"
                 % results["sweep_steps"], fontsize=10)
    return save(fig, args.out_dir, "fig_q10_lr_sweep.png", args.dpi)


# ---------------------------------------------------------------------------
# Q11-Q13 - ablations, on one shared y-axis
# ---------------------------------------------------------------------------

ABLATION_FIGURES = {
    "q11": ("fig_q11_rmsnorm.png", "RMSNorm ablation",
            [("ablation_no_rmsnorm", "no RMSNorm, lr 3e-3", "tab:red", "--"),
             ("ablation_no_rmsnorm_lowlr", "no RMSNorm, lr 1e-3", "tab:orange",
              "-.")]),
    "q12": ("fig_q12_nope.png", "Positional encoding ablation",
            [("ablation_nope", "NoPE (RoPE removed)", "tab:red", "--")]),
    "q13": ("fig_q13_ffn.png", "Feed-forward ablation",
            [("ablation_relu", "ReLU FFN, d_ff = 2048", "tab:green", "--")]),
}


ABLATION_YLIM_FROM_STEP = 500


def shared_ablation_ylim(args):
    """One y-range across Q11-Q13 so the three panels are directly comparable.

    Section 10.2: "use the same axis limits when comparing curves". Two things
    are deliberately outside the range. The diverged arm, for the reason in
    clamp_to_converged. And the first few hundred steps: every run starts at
    ln(4000) = 8.29 and the whole ablation gap is under 0.15 nats, so a range
    that includes step 1 compresses all three curves into one line. The range
    is taken from step 500 on, and the early descent runs off the top.
    """
    names = ["baseline", "ablation_no_rmsnorm_lowlr", "ablation_nope",
             "ablation_relu"]
    tails = []
    for name in names:
        curve = read_eval_log(args.log_dir, name)
        tails.append(curve["val"][curve["step"] >= ABLATION_YLIM_FROM_STEP])
    low = min(float(np.min(t)) for t in tails)
    high = max(float(np.max(t)) for t in tails)
    margin = 0.08 * (high - low)
    return low - margin, high + margin


def fig_ablation(args, key, ylim):
    filename, title, arms = ABLATION_FIGURES[key]
    records = {r["name"]: r for r in
               read_json(args.log_dir, "task4_results.json")[key]["arms"]}

    fig, ax = plt.subplots(figsize=PANEL)
    base = read_eval_log(args.log_dir, "baseline")
    ax.plot(base["step"], base["val"], "-", color="tab:blue",
            label="baseline (RMSNorm + RoPE + SwiGLU)", linewidth=1.8)

    any_diverged = False
    for name, label, colour, style in arms:
        curve = read_eval_log(args.log_dir, name)
        if records.get(name, {}).get("diverged"):
            label += " (diverged)"
            any_diverged = True
        ax.plot(curve["step"], curve["val"], style, color=colour, label=label,
                linewidth=1.5)

    ax.set_xlabel(STEP_LABEL)
    ax.set_ylabel(LOSS_LABEL)
    ax.set_xlim(0, 5000)
    ax.set_ylim(*ylim)
    note = "y cropped from step %d; early descent off-scale" % ABLATION_YLIM_FROM_STEP
    if any_diverged:
        note = "divergent run and early descent continue off-scale"
    ax.annotate(note, xy=(0.03, 0.05), xycoords="axes fraction", fontsize=7,
                color="0.35")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(title + " (5,000 steps, lr 3e-3 unless stated)", fontsize=10)
    return save(fig, args.out_dir, filename, args.dpi)


# ---------------------------------------------------------------------------
# Q17 - validation loss against position in the context window
# ---------------------------------------------------------------------------

def fig_q17(args):
    q17 = read_json(args.log_dir, "task4_results.json")["q17"]
    fig, ax = plt.subplots(figsize=PANEL)
    styles = {"RoPE (baseline)": ("tab:blue", "-", "o"),
              "NoPE": ("tab:red", "--", "s")}

    for label, buckets in q17["buckets"].items():
        colour, style, marker = styles.get(label, ("tab:grey", ":", "^"))
        centres = [(b["start"] + b["end"]) / 2 for b in buckets]
        losses = [b["loss"] for b in buckets]
        ax.plot(centres, losses, style, marker=marker, color=colour,
                label=label, linewidth=1.5, markersize=4)

    ax.set_xlabel("Position within the 256-token context window")
    ax.set_ylabel("Mean validation loss (nats / token)")
    ax.set_xticks([b["start"] for b in next(iter(q17["buckets"].values()))])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Position-wise loss (%s windows, 32-token buckets)"
                 % format(q17["n_windows"], ","), fontsize=10)
    return save(fig, args.out_dir, "fig_q17_position.png", args.dpi)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--out_dir", default="figures")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--only", nargs="*", default=None,
                        help="subset of q3 q7 q9 q10 q11 q12 q13 q17")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    wanted = set(args.only) if args.only else None

    def want(key):
        return wanted is None or key in wanted

    print("reading " + args.log_dir + "/ -> writing " + args.out_dir + "/")
    if want("q3"):
        fig_q3(args)
    if want("q7"):
        fig_q7(args)
    if want("q9"):
        fig_q9(args)
    if want("q10"):
        fig_q10(args)
    if any(want(k) for k in ("q11", "q12", "q13")):
        ylim = shared_ablation_ylim(args)
        for key in ("q11", "q12", "q13"):
            if want(key):
                fig_ablation(args, key, ylim)
    if want("q17"):
        fig_q17(args)


if __name__ == "__main__":
    main()
