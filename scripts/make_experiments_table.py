"""Build EXPERIMENTS.md (deliverable 10.3) from the per-run JSON in logs/.

Every factual column is read straight out of logs/<name>_run.json, so the table
cannot drift from the runs it describes. Re-run after any new experiment:

    python scripts/make_experiments_table.py

The "what I learned" column is left as TODO on purpose -- that is the analysis
and it has to be written by hand.
"""

import glob
import json
import os

PHASE_ORDER = ["baseline", "lr_sweep", "ablations", "vocab_study", "final_model"]
PHASE_TITLES = {
    "baseline": "Baseline",
    "lr_sweep": "Task 4 Q10 - learning-rate sweep",
    "ablations": "Task 4 Q11-Q13 - architecture ablations",
    "vocab_study": "Task 4 - vocabulary-size study",
    "final_model": "Task 5 - final model",
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_summary(run):
    """One-line configuration string: only the knobs that vary between runs."""
    mc = run.get("model_config", {})
    rc = run.get("run_config", {})
    parts = [
        f"vocab={mc.get('vocab_size')}",
        f"lr={run.get('learning_rate'):g}",
        f"bs={rc.get('batch_size')}",
        f"warmup={rc.get('warmup_steps')}",
    ]
    if not mc.get("use_rmsnorm", True):
        parts.append("**no RMSNorm**")
    if not mc.get("use_rope", True):
        parts.append("**no RoPE**")
    if mc.get("ffn_type") != "swiglu":
        parts.append(f"**ffn={mc.get('ffn_type')}**")
    parts.append(f"d_ff={mc.get('d_ff')}")
    return ", ".join(parts)


def steps_cell(run):
    done = run.get("completed_steps")
    asked = run.get("run_config", {}).get("num_steps")
    if run.get("diverged"):
        return f"{done} / {asked} (**stopped**)"
    return str(done)


def loss_cell(run):
    loss = run.get("final_val_loss")
    if loss is None:
        return "-"
    if run.get("diverged"):
        return f"{loss:.4f} (diverged)"
    return f"{loss:.4f}"


def hms(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def main():
    runs = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "logs", "*_run.json"))):
        with open(path, encoding="utf-8") as fh:
            run = json.load(fh)
        run["_source"] = os.path.relpath(path, REPO_ROOT).replace("\\", "/")
        runs.append(run)

    total_seconds = sum(r.get("wall_seconds", 0.0) for r in runs)

    lines = [
        "# Experiment log",
        "",
        "Every training run launched for this assignment, including the one that "
        "diverged. Generated from `logs/*_run.json` by "
        "`scripts/make_experiments_table.py` -- every number is traceable to the "
        "JSON file named in the last column.",
        "",
        f"**{len(runs)} runs, {total_seconds / 3600:.2f} GPU-hours total** "
        "(single CUDA device, AMP enabled).",
        "",
        "Fixed across every run unless the configuration column says otherwise: "
        "context_length=256, n_layers=4, d_model=512, n_heads=8, SwiGLU FFN, "
        "RoPE (theta=10000), QK-norm, AdamW with weight_decay=0.1, grad_clip=1.0, "
        "cosine schedule, seed=42.",
        "",
    ]

    seen = set()
    for phase in PHASE_ORDER + sorted({r.get("phase") for r in runs} - set(PHASE_ORDER)):
        if phase in seen:
            continue
        seen.add(phase)
        phase_runs = [r for r in runs if r.get("phase") == phase]
        if not phase_runs:
            continue
        phase_runs.sort(key=lambda r: r.get("learning_rate", 0.0))

        lines += [
            f"## {PHASE_TITLES.get(phase, phase)}",
            "",
            "| Run | Configuration | Steps | Final val loss | Params | Wall-clock | GPU-hours | What I learned | Log |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for run in phase_runs:
            lines.append(
                "| `{name}` | {cfg} | {steps} | {loss} | {params:,} | {wall} | {gpuh:.2f} | TODO | `{src}` |".format(
                    name=run.get("name"),
                    cfg=config_summary(run),
                    steps=steps_cell(run),
                    loss=loss_cell(run),
                    params=run.get("n_parameters", 0),
                    wall=hms(run.get("wall_seconds", 0.0)),
                    gpuh=run.get("wall_seconds", 0.0) / 3600.0,
                    src=run["_source"],
                )
            )
        lines.append("")

    failed = [r for r in runs if r.get("diverged")]
    if failed:
        lines += ["## Runs that failed", ""]
        for run in failed:
            lines.append(
                f"- `{run.get('name')}` stopped at step {run.get('diverged_at')}: "
                f"{run.get('diverged_reason')}. Cost: {hms(run.get('wall_seconds', 0.0))} "
                f"({run.get('wall_seconds', 0.0) / 3600:.2f} GPU-hours)."
            )
        lines.append("")

    out_path = os.path.join(REPO_ROOT, "EXPERIMENTS.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out_path} ({len(runs)} runs, {total_seconds / 3600:.2f} GPU-hours)")


if __name__ == "__main__":
    main()
