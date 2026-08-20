"""Task 5 end to end: the final model, answering Q18-Q20 (§8.1).

  * Q18 - loss, perplexity and BPC on validation and on test.
  * Q19 - 256 tokens from "Once upon a time," under three decoding settings,
          at least one top-p and one top-k, labelled, with a recommendation.
  * Q20 - one decoding setting that degenerates, and one systematic failure
          mode that is not a decoding artefact, each evidenced by real output.

Three rules this script exists to enforce:

  * **The test set is touched once.** §2 reserves the last 2,000 validation
    documents for exactly one evaluation, and §8 says "once only". Every touch
    is appended to logs/test_set_ledger.json, and a second one has to be asked
    for explicitly. The ledger is the evidence that the rule was kept.
  * **Nothing runs on an untrained model.** The previous version of this script
    had the checkpoint load commented out, so every number and every sample
    came from random initialisation - noise presented as a result. Loading is
    now required, and `--allow_untrained` exists only for plumbing checks and
    stamps every output it produces.
  * **Q20's evidence is generated, not written.** A failure mode is found by
    generating and measuring, and the samples are quoted from what came back.
    Inventing an illustrative "model output" would be fabricating evidence.

Every number and every sample lands in logs/task5_results.json, and the raw
samples in task5_samples.txt.

Examples
--------
  python scripts/run_task5_questions.py --questions 19 20
  python scripts/run_task5_questions.py --questions 18        # touches the test set
  python scripts/run_task5_questions.py --smoke               # plumbing, untrained
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import collections
import json
import platform
import re
import time
from datetime import datetime, timezone

import numpy as np
import torch

from src.model import TransformerLM, TransformerConfig
from src.tokenizer import BPETokenizer
from src.evaluate import evaluate, chars_from_meta
from src.generate import generate
from src.train import resolve_amp
from src.training_helpers.manage_checkpoint import load_checkpoint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
END_OF_TEXT = "<|endoftext|>"
PROMPT = "Once upon a time,"          # §8.1 Q19 fixes this prompt

# §8.1 Q19: three settings, at least one top-p and one top-k. Greedy is the
# fourth because it is the reference the others are judged against - it shows
# what the model does with no sampling at all, which is what makes the
# repetition in Q20 legible as a decoding artefact rather than a model defect.
DECODING_SETTINGS = [
    {"label": "greedy (temperature=0)", "temperature": 0.0},
    {"label": "temperature=0.8, top-k=40", "temperature": 0.8, "top_k": 40},
    {"label": "temperature=0.9, top-p=0.95", "temperature": 0.9, "top_p": 0.95},
]

# §8.1 Q20 asks for a setting that visibly degenerates. Low temperature with no
# truncation concentrates the distribution until the model re-enters a loop.
DEGENERATE_SETTING = {"label": "temperature=0.15 (no truncation)", "temperature": 0.15}


# ---------------------------------------------------------------------------
# model and data
# ---------------------------------------------------------------------------

def load_tokenizer(args):
    vocab = os.path.join(args.data_dir, args.vocab)
    merges = os.path.join(args.data_dir, args.merges)
    for path in (vocab, merges):
        if not os.path.exists(path):
            raise SystemExit(
                f"missing {os.path.basename(path)} - Task 1 has to have produced "
                f"the tokenizer before the final model can be read or generated from")
    return BPETokenizer.from_files(vocab, merges, [END_OF_TEXT])


def load_final_model(args, device):
    """§7.4's final model, from the run record Task 4 wrote.

    Refuses to proceed on random weights: an untrained model still produces
    numbers and still generates text, and neither means anything. The old
    version of this script had the load commented out and did exactly that.
    """
    record_path = os.path.join(args.log_dir, f"{args.run_name}_run.json")
    if os.path.exists(record_path):
        with open(record_path, encoding="utf-8") as f:
            record = json.load(f)
        config, checkpoint = record["model_config"], record["checkpoint"]
    else:
        config, checkpoint, record = None, args.checkpoint, None

    if config is None or not checkpoint or not os.path.exists(checkpoint):
        if not args.allow_untrained:
            raise SystemExit(
                f"No trained final model found.\n"
                f"  looked for: {record_path}\n"
                f"              {checkpoint or '(no --checkpoint given)'}\n\n"
                f"Train it first (section 7.4):\n"
                f"  python scripts/run_task4_questions.py --final_model\n\n"
                f"Or pass --checkpoint explicitly. --allow_untrained runs on random "
                f"weights for plumbing checks only; its output is not a result.")
        print("  ! --allow_untrained: running on RANDOM weights. Nothing below is a result.")
        config = dict(vocab_size=args.vocab_size, context_length=args.context_length,
                      n_layers=4, d_model=512, n_heads=8, d_ff=1344)
        model = TransformerLM(TransformerConfig(**config)).to(device)
        model.eval()
        return model, config, None

    model = TransformerLM(TransformerConfig(**config)).to(device)
    step, _ = load_checkpoint(checkpoint, model, map_location=device, restore_rng=False)
    model.eval()
    print(f"Loaded {os.path.basename(checkpoint)} (step {step}), "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    return model, config, record


def load_split(args, name):
    """One encoded split plus its character count (§6 needs C, not just N)."""
    npy = os.path.join(args.data_dir, f"{name}{args.split_suffix}.npy")
    if not os.path.exists(npy):
        raise SystemExit(
            f"missing {os.path.basename(npy)}. Build section 2's splits with:\n"
            f"  python scripts/make_splits.py --vocab vocab.pkl --merges merges.pkl")
    chars, convention = chars_from_meta(npy)
    return np.load(npy, mmap_mode="r"), chars, convention, npy


# ---------------------------------------------------------------------------
# the test-set ledger
# ---------------------------------------------------------------------------

def ledger_path(args):
    return os.path.join(args.log_dir, "test_set_ledger.json")


def read_ledger(args):
    path = ledger_path(args)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def record_test_touch(args, entry):
    """Append one test-set evaluation to the ledger.

    §2 says the test set is touched exactly once. A rule with no record of
    whether it was kept is not much of a rule, so every touch is logged with
    what was evaluated and when - and the report can cite the ledger.
    """
    entries = read_ledger(args)
    entries.append(entry)
    with open(ledger_path(args), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    return entries


def guard_test_set(args):
    """Stop a second test-set evaluation unless it is asked for explicitly."""
    prior = read_ledger(args)
    if not prior:
        return
    print(f"\n  ! The test set has already been evaluated {len(prior)} time(s):")
    for entry in prior:
        print(f"      {entry['when']}  {entry['run_name']}  "
              f"loss {entry['loss']:.4f}")
    if not args.touch_test_again:
        raise SystemExit(
            "Section 2 reserves the test set for exactly one evaluation, and the "
            "ledger says it has been used. The reported numbers are in "
            "logs/task5_results.json and the ledger above.\n"
            "If a re-run is genuinely intended (a retrained final model, say), "
            "pass --touch_test_again; it is recorded as a further touch.")
    print("  ! --touch_test_again given; this touch is being recorded too.")


# ---------------------------------------------------------------------------
# Q18
# ---------------------------------------------------------------------------

def run_q18(args, model, config, record, tokenizer, device, results):
    """§8.1: loss, perplexity and BPC on validation and on test."""
    print("\n" + "=" * 70)
    print("Q18: final model on validation and test")
    print("=" * 70)

    guard_test_set(args)
    amp = resolve_amp(not args.no_amp, device)[0]
    rows = {}

    for split in ("valid_split", "test_split"):
        data, chars, convention, npy = load_split(args, split)
        metrics = evaluate(model, data, args.eval_batch_size,
                           config["context_length"], device, total_chars=chars,
                           amp_enabled=amp, max_windows=args.eval_windows)
        metrics["convention"] = convention
        metrics["source"] = os.path.basename(npy)
        rows[split] = metrics

    print(f"\n[Q18] characters counted {rows['valid_split']['convention']}, "
          f"whole non-overlapping windows of {config['context_length']}:")
    print(f"  {'split':<14} {'tokens':>11} {'loss':>9} {'perplexity':>12} {'BPC':>8}")
    for split, label in (("valid_split", "validation"), ("test_split", "test")):
        m = rows[split]
        print(f"  {label:<14} {m['n_tokens']:>11,} {m['loss']:>9.4f} "
              f"{m['perplexity']:>12.2f} {m['bpc']:>8.4f}")

    gap = rows["test_split"]["loss"] - rows["valid_split"]["loss"]
    print(f"\n  test - validation loss: {gap:+.4f}")

    if record is not None:
        entry = {
            "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "run_name": args.run_name,
            "checkpoint": record["checkpoint"],
            "loss": rows["test_split"]["loss"],
            "perplexity": rows["test_split"]["perplexity"],
            "bpc": rows["test_split"]["bpc"],
            "n_tokens": rows["test_split"]["n_tokens"],
        }
        entries = record_test_touch(args, entry)
        print(f"  test-set touches recorded: {len(entries)} "
              f"(logs/{os.path.basename(ledger_path(args))})")
    else:
        print("  (untrained model - not recorded as a test-set touch)")

    results["q18"] = {"validation": rows["valid_split"], "test": rows["test_split"],
                      "test_minus_validation_loss": gap,
                      "trained": record is not None}
    return results["q18"]


# ---------------------------------------------------------------------------
# Q19 / Q20 - generation
# ---------------------------------------------------------------------------

def console_safe(text):
    """Generated text, made printable on whatever the terminal can encode.

    A model can emit any byte sequence its tokenizer can decode, and Windows
    consoles default to cp1252 - printing a sample raw kills the run partway
    through Q19. task5_samples.txt is written as UTF-8 regardless, so nothing is lost
    from the record; only the console view is degraded.
    """
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def sample_with(model, tokenizer, setting, args, max_new_tokens=None):
    """One generation under one labelled setting."""
    kwargs = {k: v for k, v in setting.items() if k != "label"}
    started = time.perf_counter()
    text = generate(model, tokenizer, PROMPT,
                    max_new_tokens=max_new_tokens or args.max_new_tokens,
                    seed=args.seed, **kwargs)
    return {
        "label": setting["label"],
        "settings": kwargs,
        "prompt": PROMPT,
        "text": text,
        "continuation": text[len(PROMPT):],
        "n_tokens": len(tokenizer.encode(text)),
        "seconds": time.perf_counter() - started,
    }


def repetition_metrics(text, tokenizer):
    """How repetitive a sample is, so Q20's claim is measured rather than asserted.

    Two numbers, because they catch different failures: the distinct-token ratio
    falls when a small set of tokens dominates, and the longest repeated line
    catches a loop that cycles through a longer phrase.
    """
    ids = tokenizer.encode(text)
    distinct_ratio = len(set(ids)) / len(ids) if ids else 0.0

    words = re.findall(r"\w+", text.lower())
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    counts = collections.Counter(trigrams)
    top_trigram, top_count = (counts.most_common(1)[0] if counts else ((), 0))

    return {
        "distinct_token_ratio": distinct_ratio,
        "n_tokens": len(ids),
        "most_repeated_trigram": " ".join(top_trigram),
        "most_repeated_trigram_count": top_count,
        "repeated_trigram_fraction": (top_count / len(trigrams)) if trigrams else 0.0,
    }


def first_lines(text, n=3, width=100):
    """At most `n` lines of a sample - §8.1 Q20 allows three lines of evidence."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [line[:width] for line in lines[:n]]


def run_q19(args, model, config, record, tokenizer, device, results):
    """§8.1: 256 tokens under three settings, labelled, with a recommendation."""
    print("\n" + "=" * 70)
    print(f"Q19: generation from {PROMPT!r}")
    print("=" * 70)

    samples = []
    for setting in DECODING_SETTINGS:
        sample = sample_with(model, tokenizer, setting, args)
        sample.update(repetition_metrics(sample["text"], tokenizer))
        samples.append(sample)

        print(f"\n--- {sample['label']} "
              f"({sample['n_tokens']} tokens, distinct ratio "
              f"{sample['distinct_token_ratio']:.2f}) ---")
        print(console_safe(sample["text"][:args.print_chars]))
        if len(sample["text"]) > args.print_chars:
            print(f"  ... [{len(sample['text']) - args.print_chars} more characters]")

    # The recommendation is derived, not asserted: among the sampled settings,
    # prefer the most varied output. Greedy is excluded - it cannot vary, and
    # shipping it means every user gets the same story.
    sampled = [s for s in samples if s["settings"].get("temperature", 0) > 0]
    recommended = max(sampled, key=lambda s: s["distinct_token_ratio"]) if sampled else None
    if recommended:
        print(f"\n[Q19] Would ship: {recommended['label']} - highest distinct-token "
              f"ratio ({recommended['distinct_token_ratio']:.2f}) among the sampling "
              f"settings, i.e. the least repetitive without going incoherent; greedy "
              f"is excluded because it returns the same story to every user.")

    results["q19"] = {
        "prompt": PROMPT,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "samples": samples,
        "recommended": recommended["label"] if recommended else None,
        "trained": record is not None,
    }
    return results["q19"]


def run_q20(args, model, config, record, tokenizer, device, results):
    """§8.1: a degenerate decoding setting, and a failure mode that is not one.

    Both are evidenced from generated text. The failure mode is looked for
    empirically - across several prompts, under the recommended setting, so it
    cannot be blamed on the sampler.
    """
    print("\n" + "=" * 70)
    print("Q20: degenerate decoding, and a failure mode that is not decoding")
    print("=" * 70)

    degenerate = sample_with(model, tokenizer, DEGENERATE_SETTING, args)
    degenerate.update(repetition_metrics(degenerate["text"], tokenizer))

    healthy = sample_with(model, tokenizer, DECODING_SETTINGS[-1], args)
    healthy.update(repetition_metrics(healthy["text"], tokenizer))

    print(f"\n--- degenerate: {degenerate['label']} ---")
    for line in first_lines(degenerate["continuation"]):
        print(console_safe(f"  | {line}"))
    print(f"  distinct-token ratio {degenerate['distinct_token_ratio']:.2f} "
          f"against {healthy['distinct_token_ratio']:.2f} for "
          f"{healthy['label']}; most repeated trigram "
          + console_safe(f"{degenerate['most_repeated_trigram']!r}")
          + f" x{degenerate['most_repeated_trigram_count']}")

    # §8.1 wants a failure mode that is *not* a decoding artefact, so it is
    # looked for under a healthy sampler and across several prompts: anything
    # that survives that is a property of the model, not of the sampling.
    probes = []
    for prompt in args.failure_prompts:
        kwargs = {k: v for k, v in DECODING_SETTINGS[-1].items() if k != "label"}
        text = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens,
                        seed=args.seed, **kwargs)
        probes.append({"prompt": prompt, "text": text,
                       "continuation": text[len(prompt):],
                       "lines": first_lines(text[len(prompt):]),
                       **repetition_metrics(text, tokenizer)})

    print(f"\n--- candidate failure mode, under {DECODING_SETTINGS[-1]['label']} ---")
    for probe in probes:
        print(f"\n  prompt: {probe['prompt']!r}")
        for line in probe["lines"][:2]:
            print(f"  | {line}")

    mean_distinct = float(np.mean([p["distinct_token_ratio"] for p in probes]))
    print(f"\n  mean distinct-token ratio across {len(probes)} prompts: "
          f"{mean_distinct:.2f} (a healthy sampler, so repetition here would be "
          f"the model rather than the decoding)")
    print("\n  Read the samples above and name the failure mode in the report - "
          "what the model does wrong that changing the sampler does not fix.")

    results["q20"] = {
        "degenerate": {"sample": degenerate,
                       "comparison": {"label": healthy["label"],
                                      "distinct_token_ratio": healthy["distinct_token_ratio"]},
                       "evidence_lines": first_lines(degenerate["continuation"])},
        "failure_mode_probes": probes,
        "mean_distinct_ratio_across_prompts": mean_distinct,
        "setting_used_for_probes": DECODING_SETTINGS[-1]["label"],
        "trained": record is not None,
    }
    return results["q20"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def write_samples(results, path, trained):
    """The generated text, in full, next to the settings that produced it."""
    blocks = [f"Prompt: {PROMPT!r}",
              f"Trained model: {trained}",
              f"Written: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
              ""]
    for question in ("q19", "q20"):
        block = results.get(question)
        if not block:
            continue
        samples = block.get("samples") or []
        if question == "q20":
            samples = [block["degenerate"]["sample"]] + block["failure_mode_probes"]
        for sample in samples:
            blocks.append("=" * 70)
            blocks.append(f"[{question.upper()}] {sample.get('label', sample.get('prompt'))}")
            blocks.append(f"settings: {sample.get('settings', DECODING_SETTINGS[-1])}")
            blocks.append("=" * 70)
            blocks.append(sample["text"])
            blocks.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    return path


def save_results(results, log_dir):
    path = os.path.join(log_dir, "task5_results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prior = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            prior = json.load(f)
    prior.update(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prior, f, indent=2, default=str)
    return path


def build_parser():
    p = argparse.ArgumentParser(
        description="Task 5: final model and generation. Answers Q18-Q20 (section 8.1).")
    p.add_argument("--questions", nargs="+", type=int, default=[18, 19, 20],
                   choices=[18, 19, 20])
    p.add_argument("--run_name", default="final_model",
                   help="Task 4 run whose checkpoint is the final model (section 7.4).")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit checkpoint path, if there is no run record.")
    p.add_argument("--allow_untrained", action="store_true",
                   help="Run on random weights for a plumbing check. The output "
                        "is not a result and is stamped as such.")
    p.add_argument("--touch_test_again", action="store_true",
                   help="Evaluate the test set even though the ledger says it has "
                        "been used. Section 2 says once.")
    # generation
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Section 8.1 Q19 asks for 256.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_chars", type=int, default=1200)
    p.add_argument("--failure_prompts", nargs="+",
                   default=["Once upon a time,",
                            "The little girl found a key that",
                            "Tom and Sara wanted to build"],
                   help="Prompts probed for a non-decoding failure mode.")
    # evaluation
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--eval_windows", type=int, default=None)
    p.add_argument("--vocab_size", type=int, default=4000)
    p.add_argument("--context_length", type=int, default=256)
    # plumbing
    p.add_argument("--split_suffix", default="",
                   help="Task 1 naming suffix for the splits, e.g. '_vocab1000'.")
    p.add_argument("--vocab", default="vocab.pkl")
    p.add_argument("--merges", default="merges.pkl")
    p.add_argument("--data_dir", default=REPO_ROOT)
    p.add_argument("--log_dir", default=os.path.join(REPO_ROOT, "logs"))
    p.add_argument("--out_dir", default=REPO_ROOT)
    p.add_argument("--device", default=None)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Short generations on an untrained model, into logs/smoke/.")
    return p


def apply_smoke_settings(args):
    args.allow_untrained = True
    args.max_new_tokens = 32
    args.eval_windows = 20
    args.print_chars = 300
    args.failure_prompts = args.failure_prompts[:2]
    args.log_dir = os.path.join(args.log_dir, "smoke")
    args.out_dir = os.path.join(REPO_ROOT, "logs", "smoke")
    print("SMOKE RUN - untrained model, short samples, into logs/smoke/. "
          "Not a Q18-Q20 answer.\n")


def main(argv=None):
    args = build_parser().parse_args(argv)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke:
        apply_smoke_settings(args)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = load_tokenizer(args)
    model, config, record = load_final_model(args, device)
    print(f"Device: {device}")

    results = {
        "machine": {"platform": platform.platform(), "torch": torch.__version__,
                    "device": str(device)},
        "model_config": config,
        "final_model_run": record["name"] if record else None,
        "trained": record is not None,
    }

    handlers = {18: run_q18, 19: run_q19, 20: run_q20}
    for q in sorted(args.questions):
        handlers[q](args, model, config, record, tokenizer, device, results)

    path = save_results(results, args.log_dir)
    samples_path = write_samples(results,
                                 os.path.join(args.out_dir, "task5_samples.txt"),
                                 record is not None)
    print("\n" + "=" * 70)
    print("TASK 5 COMPLETE")
    print("=" * 70)
    print(f"Numbers  -> {path}")
    print(f"Samples  -> {samples_path}")


if __name__ == "__main__":
    main()
