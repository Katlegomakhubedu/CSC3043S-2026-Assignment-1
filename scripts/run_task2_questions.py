"""Task 2 end to end: model and inference, answering Q5-Q7 (§4.4).

  * Q5 - parameter counts split into embedding / LM head / non-embedding, for
         the SwiGLU model and the §7.2 parameter-matched ReLU variant.
  * Q6 - evidence the KV cache is correct: cached and uncached greedy
         generations identical over 100 tokens (§4.2), logits agreeing to
         floating-point tolerance.
  * Q7 - generation throughput with and without the cache, 16 to 256 tokens.

None of the three needs trained weights or GPU time: parameter counts follow
from the architecture, cache correctness is an identity that holds for any
weights, and throughput follows the shape of the computation rather than its
values. The results file records that, so the report can state it.

Every number printed also lands in logs/task2_results.json.

Examples
--------
  python scripts/run_task2_questions.py
  python scripts/run_task2_questions.py --questions 7 --repeats 5
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import pickle
import platform
import time

import torch

import matplotlib
matplotlib.use("Agg")   # headless: this script only saves a PNG
import matplotlib.pyplot as plt

from src.model import TransformerLM, TransformerConfig
from src.tokenizer import BPETokenizer
from src.generate import generate

SPECIAL_TOKENS = ["<|endoftext|>"]

# §4.1 base model. vocab_size is overridden by the tokenizer actually loaded, so
# the model can never emit an ID the tokenizer cannot decode.
BASE_CONFIG = dict(vocab_size=4000, context_length=256, n_layers=4, d_model=512,
                   n_heads=8, d_ff=1344, rope_theta=10000.0, use_qk_norm=True)

# §7.2's ReLU variant: two matrices at 2048 against SwiGLU's three at 1344,
# which matches the parameter count to within 2%. d_ff is the only difference.
RELU_D_FF = 2048


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def describe_machine():
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def load_tokenizer(repo_root):
    """The Task 1 tokenizer if it has been built, else a byte-level fallback.

    The fallback lets Q6 and Q7 run before Task 1 is finished. Which one was
    used is recorded, since the vocabulary size moves the Q7 numbers (the LM
    head is the widest matrix in a decode step).
    """
    vocab_path = os.path.join(repo_root, "vocab.pkl")
    merges_path = os.path.join(repo_root, "merges.pkl")
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        return BPETokenizer(vocab, merges, SPECIAL_TOKENS), "task1 (vocab.pkl/merges.pkl)"

    print("  ! vocab.pkl/merges.pkl not found - falling back to a byte-level tokenizer.")
    vocab = {0: SPECIAL_TOKENS[0].encode("utf-8")}
    for b in range(256):
        vocab[1 + b] = bytes([b])
    return BPETokenizer(vocab, [], SPECIAL_TOKENS), "byte-level fallback"


def build_model(tokenizer, device, **overrides):
    cfg = dict(BASE_CONFIG)
    cfg["vocab_size"] = len(tokenizer.vocab)
    cfg.update(overrides)
    torch.manual_seed(0)         # same initialisation every run
    model = TransformerLM(TransformerConfig(**cfg)).to(device)
    model.eval()
    return model


def parameter_breakdown(model):
    total = model.num_parameters()
    embedding = model.token_embeddings.weight.numel()
    lm_head = model.lm_head.weight.numel()
    return {
        "embedding": embedding,
        "lm_head": lm_head,
        "non_embedding": total - embedding - lm_head,
        "total": total,
    }


def tokens_actually_generated(n_prompt_tokens, max_new_tokens, context_length, use_cache):
    """How many tokens `generate` really produces, which is not `max_new_tokens`.

    The cached path stops on filling its fixed-size context_length buffer (§4.2);
    the uncached path slides a window and has no such limit. At
    max_new_tokens=256 with a 4-token prompt that is 253 tokens against 256, so
    dividing both by max_new_tokens would compare different amounts of work and
    understate cached throughput. tests/test_task2_questions.py checks this
    count against real generation.

    Assumes generation is not cut short by <|endoftext|>; the caller checks that.
    """
    if not use_cache:
        return max_new_tokens
    # Prefill fills one slot per prompt token, and its last-position logits give
    # one more token without consuming a slot - hence the +1.
    prompt_in_cache = min(n_prompt_tokens, context_length)
    return min(max_new_tokens, context_length - prompt_in_cache + 1)


def save_results(results, out_dir):
    path = os.path.join(out_dir, "logs", "task2_results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prior = {}
    if os.path.exists(path):
        # Running one question (--questions 7) must not discard the others.
        with open(path, encoding="utf-8") as f:
            prior = json.load(f)
    prior.update(results)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prior, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Q5: parameter counts
# ---------------------------------------------------------------------------

def run_q5(tokenizer, device):
    """§4.4 Q5: the parameter split, and the SwiGLU/ReLU parameter match."""
    print("\n" + "=" * 70)
    print("Q5: parameter count")
    print("=" * 70)

    swiglu = build_model(tokenizer, device, ffn_type="swiglu", d_ff=BASE_CONFIG["d_ff"])
    relu = build_model(tokenizer, device, ffn_type="relu", d_ff=RELU_D_FF)

    swiglu_params = parameter_breakdown(swiglu)
    relu_params = parameter_breakdown(relu)

    difference = relu_params["total"] - swiglu_params["total"]
    relative = abs(difference) / swiglu_params["total"]

    print(f"\n{'':<16} | {'SwiGLU (d_ff=1344)':>20} | {'ReLU (d_ff=2048)':>20}")
    print("-" * 64)
    for key, label in (("embedding", "Embedding"), ("lm_head", "LM head"),
                        ("non_embedding", "Non-embedding"), ("total", "Total")):
        if key == "total":
            print("-" * 64)
        print(f"{label:<16} | {swiglu_params[key]:>20,} | {relu_params[key]:>20,}")

    print(f"\nDifference in total parameters: {difference:+,} ({relative:.2%})")
    print(f"Within the 2% section 7.2 promises: {relative < 0.02}")
    print("\nNote: SwiGLU uses three d_model x d_ff matrices, ReLU two, which is")
    print("why the two variants use different d_ff to reach the same size.")

    # §4.1's "roughly 17M excluding the embedding and LM head" is really the
    # total *including* them at vocab_size=4000, so report the measured split.
    print(f"\nFor the report: non-embedding is {swiglu_params['non_embedding'] / 1e6:.2f}M, "
          f"total {swiglu_params['total'] / 1e6:.2f}M at vocab_size="
          f"{len(tokenizer.vocab)}.")
    print("(Section 4.1's \"roughly 17M excluding embedding and LM head\" matches")
    print(" the total INCLUDING them, not the non-embedding count.)")

    return {
        "swiglu": {"d_ff": BASE_CONFIG["d_ff"], **swiglu_params},
        "relu": {"d_ff": RELU_D_FF, **relu_params},
        "difference_total": difference,
        "relative_difference": relative,
        "within_2_percent": bool(relative < 0.02),
        "vocab_size": len(tokenizer.vocab),
    }


# ---------------------------------------------------------------------------
# Q6: KV-cache correctness
# ---------------------------------------------------------------------------

def greedy_logits_uncached(model, token_ids, n_steps, device):
    """Greedy-decode `n_steps` tokens the uncached way, recording each step's logits."""
    ids = list(token_ids)
    context_length = model.config.context_length
    logits_per_step = []
    with torch.no_grad():
        for _ in range(n_steps):
            window = torch.tensor([ids[-context_length:]], dtype=torch.long, device=device)
            logits = model(window, use_cache=False)[0, -1, :].float()
            logits_per_step.append(logits)
            ids.append(int(logits.argmax()))
    return ids, logits_per_step


def greedy_logits_cached(model, token_ids, forced_ids, device):
    """Replay `forced_ids` through the cached path, recording each step's logits.

    Driven by the sequence the uncached path produced, so both sides see the
    same inputs at every step and any difference is the cache's doing rather
    than a diverging sequence.
    """
    for layer in model.layers:
        layer.attn.reset_cache()

    logits_per_step = []
    with torch.no_grad():
        # Prefill the whole prompt; its last-position logits are step 1's prediction.
        prompt = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
        logits = model(prompt, use_cache=True)[0, -1, :].float()
        logits_per_step.append(logits)

        # Then one token at a time, exactly as generate() decodes.
        for token in forced_ids[:-1]:
            step = torch.tensor([[token]], dtype=torch.long, device=device)
            logits = model(step, use_cache=True)[0, -1, :].float()
            logits_per_step.append(logits)
    return logits_per_step


def run_q6(tokenizer, device, prompt, n_tokens=100):
    """§4.4 Q6 / §4.2's required check, at the 100 tokens §4.2 specifies."""
    print("\n" + "=" * 70)
    print(f"Q6: KV-cache correctness ({n_tokens}-token greedy continuation)")
    print("=" * 70)

    model = build_model(tokenizer, device)

    # -- 1. End to end: do the two paths produce the same text? --------------
    out_cached = generate(model, tokenizer, prompt, max_new_tokens=n_tokens,
                          temperature=0.0, use_cache=True)
    out_uncached = generate(model, tokenizer, prompt, max_new_tokens=n_tokens,
                            temperature=0.0, use_cache=False)
    generations_identical = out_cached == out_uncached

    # -- 2. Step by step: how far apart are the logits? ----------------------
    prompt_ids = tokenizer.encode(prompt)
    all_ids, uncached_logits = greedy_logits_uncached(model, prompt_ids, n_tokens, device)
    generated_ids = all_ids[len(prompt_ids):]
    cached_logits = greedy_logits_cached(model, prompt_ids, generated_ids, device)

    per_step = [(u - c).abs().max().item() for u, c in zip(uncached_logits, cached_logits)]
    max_abs_diff = max(per_step)
    argmax_agreement = all(int(u.argmax()) == int(c.argmax())
                           for u, c in zip(uncached_logits, cached_logits))

    print(f"\n  Greedy generations identical      : {generations_identical}")
    print(f"  Steps compared                    : {len(per_step)}")
    print(f"  Argmax identical at every step    : {argmax_agreement}")
    print(f"  Max absolute logit difference     : {max_abs_diff:.3e}")
    print(f"  Mean absolute logit difference    : {sum(per_step) / len(per_step):.3e}")
    print(f"  Worst step                        : {per_step.index(max_abs_diff)}")

    verdict = generations_identical and argmax_agreement and max_abs_diff < 1e-4
    print(f"\n  KV cache correct: {verdict}")
    if not verdict:
        print("  ! Cached and uncached paths disagree - do not report Q6 as passing.")

    return {
        "prompt": prompt,
        "n_tokens_requested": n_tokens,
        "steps_compared": len(per_step),
        "generations_identical": bool(generations_identical),
        "argmax_identical_every_step": bool(argmax_agreement),
        "max_abs_logit_difference": max_abs_diff,
        "mean_abs_logit_difference": sum(per_step) / len(per_step),
        "tolerance": 1e-4,
        "passed": bool(verdict),
    }


# ---------------------------------------------------------------------------
# Q7: generation throughput
# ---------------------------------------------------------------------------

def run_q7(tokenizer, device, prompt, out_dir, token_counts, repeats):
    """§4.4 Q7: throughput against tokens generated, with and without the cache."""
    print("\n" + "=" * 70)
    print(f"Q7: generation throughput ({repeats} timed repeat(s) per point)")
    print("=" * 70)

    model = build_model(tokenizer, device)
    context_length = model.config.context_length
    n_prompt_tokens = len(tokenizer.encode(prompt))

    # Warm up both paths, so lazy allocation and kernel selection do not land
    # entirely on the 16-token point.
    for use_cache in (True, False):
        generate(model, tokenizer, prompt, max_new_tokens=8,
                 temperature=0.0, use_cache=use_cache)

    rows = []
    for count in token_counts:
        row = {"max_new_tokens": count}
        for use_cache in (True, False):
            key = "cached" if use_cache else "uncached"
            n_tokens = tokens_actually_generated(n_prompt_tokens, count,
                                                 context_length, use_cache)
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                out = generate(model, tokenizer, prompt, max_new_tokens=count,
                               temperature=0.0, use_cache=use_cache)
                times.append(time.perf_counter() - start)

            # An early <|endoftext|> would invalidate the token count above.
            if out.count("<|endoftext|>") > prompt.count("<|endoftext|>"):
                raise RuntimeError(
                    f"generation stopped early on <|endoftext|> at count={count}, "
                    f"use_cache={use_cache}; the throughput denominator would be "
                    f"wrong. Choose a different --prompt.")

            mean_time = sum(times) / len(times)
            row[key] = {
                "tokens_generated": n_tokens,
                "seconds_mean": mean_time,
                "seconds_trials": times,
                "tokens_per_second": n_tokens / mean_time,
            }
        row["speedup"] = row["cached"]["tokens_per_second"] / row["uncached"]["tokens_per_second"]
        rows.append(row)
        print(f"  {count:>4} requested | cached {row['cached']['tokens_generated']:>3} tok "
              f"@ {row['cached']['tokens_per_second']:7.1f} tok/s | "
              f"uncached {row['uncached']['tokens_generated']:>3} tok "
              f"@ {row['uncached']['tokens_per_second']:7.1f} tok/s | "
              f"{row['speedup']:.2f}x")

    # -- figure --------------------------------------------------------------
    x = [r["max_new_tokens"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [r["cached"]["tokens_per_second"] for r in rows], "o-",
            color="tab:blue", label="With KV cache")
    ax.plot(x, [r["uncached"]["tokens_per_second"] for r in rows], "s--",
            color="tab:red", label="Without KV cache")
    ax.set_xlabel("Tokens generated (max_new_tokens)")
    ax.set_ylabel("Throughput (tokens / second)")
    ax.set_title(f"Generation throughput with and without the KV cache\n"
                 f"({device}, {model.config.n_layers} layers, d_model="
                 f"{model.config.d_model}, context_length={context_length})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    png = os.path.join(out_dir, "task2_q7_throughput.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)

    final = rows[-1]
    print(f"\n  Figure saved to {png}")
    print(f"  Speedup at {final['max_new_tokens']} tokens: {final['speedup']:.2f}x")

    return {
        "prompt": prompt,
        "n_prompt_tokens": n_prompt_tokens,
        "context_length": context_length,
        "device": str(device),
        "repeats": repeats,
        "rows": rows,
        "speedup_at_max": final["speedup"],
        "max_new_tokens_at_max": final["max_new_tokens"],
        "figure": os.path.basename(png),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description="Task 2: model and inference. Answers Q5-Q7 (section 4.4).")
    parser.add_argument("--questions", nargs="+", type=int, default=[5, 6, 7],
                        choices=[5, 6, 7])
    parser.add_argument("--prompt", default="Once upon a time",
                        help="Prompt for the Q6 and Q7 generations.")
    parser.add_argument("--q6_tokens", type=int, default=100,
                        help="Continuation length for Q6 (section 4.2 specifies 100).")
    parser.add_argument("--token_counts", nargs="+", type=int,
                        default=[16, 32, 64, 128, 256],
                        help="Q7 x-axis: tokens generated (section 4.4 asks for 16 to 256).")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Timed repeats per Q7 point; the mean is reported.")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu. Defaults to cuda when available.")
    parser.add_argument("--out_dir", default=repo_root)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)

    tokenizer, tokenizer_source = load_tokenizer(repo_root)
    print(f"Device: {device}")
    print(f"Tokenizer: {tokenizer_source}, vocab_size={len(tokenizer.vocab)}")
    print("Model: freshly initialised - Q5-Q7 do not depend on trained weights.")

    results = {
        "machine": describe_machine(),
        "config": vars(args),
        "device": str(device),
        "tokenizer_source": tokenizer_source,
        "vocab_size": len(tokenizer.vocab),
        "model_config": {**BASE_CONFIG, "vocab_size": len(tokenizer.vocab)},
        "trained_weights": False,
    }

    if 5 in args.questions:
        results["q5"] = run_q5(tokenizer, device)
    if 6 in args.questions:
        results["q6"] = run_q6(tokenizer, device, args.prompt, args.q6_tokens)
    if 7 in args.questions:
        results["q7"] = run_q7(tokenizer, device, args.prompt, args.out_dir,
                               args.token_counts, args.repeats)

    path = save_results(results, args.out_dir)
    print("\n" + "=" * 70)
    print("TASK 2 COMPLETE")
    print("=" * 70)
    print(f"All numbers written to {path}")


if __name__ == "__main__":
    main()