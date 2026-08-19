"""Compression ratio vs vocabulary size (§3.4, Q3).

Trains BPE once to the largest size of interest and reads the compression ratio
for every smaller size off the same run, since each merge reduces the corpus
token count by exactly the merged pair's count.

Also reports each size's embedding/LM-head parameter cost, which §3.4 asks the
choice to be justified against alongside the curve.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv

import matplotlib
matplotlib.use("Agg")   # headless: this script only saves a PNG
import matplotlib.pyplot as plt

from src.tokenizer import (get_word_freq_from_files, get_word_freq_parallel,
                            init_vocab, train_bpe_incremental)


def train_bpe_with_counts(input_path, max_vocab_size, special_tokens, workers=1):
    """Train to `max_vocab_size`, tracking the corpus token count after every merge.

    Uses the same incremental trainer as `train_bpe`, so the curve reflects the
    real algorithm rather than a reimplementation.

    Returns (vocab, merges, token_counts, initial_vocab_size, word_freq).
    """
    input_paths = [input_path] if isinstance(input_path, str) else list(input_path)
    if workers and workers > 1:
        word_freq = get_word_freq_parallel(input_paths, special_tokens, workers)
    else:
        word_freq = get_word_freq_from_files(input_paths, special_tokens)

    vocab = init_vocab(special_tokens)
    initial_vocab_size = len(vocab)
    num_merges = max_vocab_size - initial_vocab_size

    merges, token_counts = train_bpe_incremental(
        word_freq, vocab, num_merges, track_token_counts=True)

    return vocab, merges, token_counts, initial_vocab_size, word_freq


def corpus_size_from_word_freq(word_freq):
    """(characters, UTF-8 bytes) of the pre-tokenized text, from the frequency table.

    The GPT-2 regex tiles a document completely, so the corpus is the
    concatenation of its pre-tokens and its size reads straight off `word_freq`
    - no second pass over a 2.2GB file. This counts exactly the text the token
    counts cover: document text with the <|endoftext|> delimiters excluded.
    """
    n_bytes = 0
    n_chars = 0
    for word, freq in word_freq.items():
        n_bytes += len(word) * freq                       # one element per byte
        n_chars += len(b"".join(word).decode("utf-8", errors="replace")) * freq
    return n_chars, n_bytes


def compute_compression_metrics(word_freq, token_counts, initial_vocab_size,
                                target_sizes, d_model=512):
    """Bytes/characters per token at each target vocabulary size.

    `token_counts[i]` is the corpus token count after `i` merges, so the count
    for vocabulary size V is `token_counts[V - initial_vocab_size]`.
    """
    n_chars, n_bytes = corpus_size_from_word_freq(word_freq)

    metrics = []
    for target in target_sizes:
        index = target - initial_vocab_size
        if not 0 <= index < len(token_counts):
            print(f"  ! skipping vocab_size={target}: outside the trained range "
                  f"({initial_vocab_size}..{initial_vocab_size + len(token_counts) - 1})")
            continue

        token_count = token_counts[index]
        metrics.append({
            "vocab_size": target,
            "tokens": token_count,
            "bytes_per_token": n_bytes / token_count,
            "chars_per_token": n_chars / token_count,
            # Embedding + untied LM head: the part of the budget that scales
            # with the vocabulary (§4.1).
            "embed_lm_head_params": 2 * target * d_model,
        })
    return metrics, n_chars, n_bytes


def plot_compression(metrics, output_file="compression_ratio.png"):
    sizes = [m["vocab_size"] for m in metrics]
    bpt = [m["bytes_per_token"] for m in metrics]
    params = [m["embed_lm_head_params"] / 1e6 for m in metrics]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel("Vocabulary size (tokens)")
    ax1.set_xscale("log")
    ax1.set_ylabel("Compression ratio (bytes / token)", color="tab:blue")
    line1, = ax1.plot(sizes, bpt, "o-", color="tab:blue", label="Compression (bytes/token)")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([str(s) for s in sizes])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Embedding + LM head (millions of parameters)", color="tab:red")
    line2, = ax2.plot(sizes, params, "s--", color="tab:red",
                      label="Embedding + LM head params (M)")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title("Compression ratio and parameter cost vs vocabulary size")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(handles=[line1, line2], loc="center right")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def write_csv(metrics, output_file):
    """Write the Q3 table, so the report's numbers are traceable to a file."""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def main():
    import argparse
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Compression-ratio vs vocab-size study (§3.4).")
    parser.add_argument("--input", nargs="+",
                        default=[os.path.join(repo_root, "data", "TinyStoriesV2-GPT4-valid.txt")],
                        help="Corpus to run the study on (§3.4 allows the validation file "
                             "or a subsample of the training file).")
    parser.add_argument("--target_sizes", type=int, nargs="+",
                        default=[1000, 2000, 4000, 8000, 16000])
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out_dir", default=repo_root)
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]
    max_size = max(args.target_sizes)

    print(f"Training BPE once to vocab_size={max_size} on "
          f"{', '.join(os.path.basename(p) for p in args.input)} ...")
    _, _, token_counts, initial_size, word_freq = train_bpe_with_counts(
        args.input, max_size, special_tokens, workers=args.workers)

    metrics, n_chars, n_bytes = compute_compression_metrics(
        word_freq, token_counts, initial_size, args.target_sizes, d_model=args.d_model)

    png = os.path.join(args.out_dir, "compression_ratio.png")
    csv_path = os.path.join(args.out_dir, "logs", "vocab_study.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    plot_compression(metrics, png)
    write_csv(metrics, csv_path)

    print(f"\nCorpus (delimiters excluded): {n_chars:,} chars, {n_bytes:,} UTF-8 bytes")
    print("\n{:<12} | {:>14} | {:>12} | {:>12} | {:>14}".format(
        "Vocab size", "Tokens", "Bytes/token", "Chars/token", "Embed+head"))
    print("-" * 76)
    for m in metrics:
        print("{:<12} | {:>14,} | {:>12.3f} | {:>12.3f} | {:>13,}".format(
            m["vocab_size"], m["tokens"], m["bytes_per_token"],
            m["chars_per_token"], m["embed_lm_head_params"]))

    print(f"\nSaved {png}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
