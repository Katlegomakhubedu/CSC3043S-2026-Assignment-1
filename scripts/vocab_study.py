import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")  # headless: this script only needs to save a PNG, and
                        # plt.show() with no display (Colab CPU runtime, CI, ...)
                        # blocks forever instead of erroring.
import matplotlib.pyplot as plt
import os
from src.tokenizer import read_txt, get_word_freq_from_files, init_vocab, train_bpe_incremental

def train_bpe_with_counts(input_path, max_vocab_size, special_tokens):
    """
    Trains BPE up to `max_vocab_size`, using the same incremental merge-counting
    trainer as `train_bpe` (src/tokenizer.py) so this vocab-size study reflects
    the actual training algorithm, not a separate (slower) reimplementation.
    `input_path` may be a single path or a list of paths.
    Returns: (vocab, merges, token_counts, initial_vocab_size)
    """
    input_paths = [input_path] if isinstance(input_path, str) else list(input_path)
    word_freq = get_word_freq_from_files(input_paths, special_tokens)

    vocab = init_vocab(special_tokens)
    initial_vocab_size = len(vocab)
    num_merges = max_vocab_size - initial_vocab_size

    merges, token_counts = train_bpe_incremental(word_freq, vocab, num_merges, track_token_counts=True)

    return vocab, merges, token_counts, initial_vocab_size

def compute_compression_metrics(input_path, token_counts, initial_vocab_size, target_sizes):
    """
    Computes bytes/token and characters/token for target vocabulary sizes.
    `input_path` may be a single path or a list of paths (same corpus used to train).
    """
    input_paths = [input_path] if isinstance(input_path, str) else list(input_path)
    byte_count = sum(os.path.getsize(p) for p in input_paths)
    char_count = sum(len(read_txt(p)) for p in input_paths)

    # Build a map: vocab_size -> total_token_count
    size_to_tokens = {initial_vocab_size + i: count for i, count in enumerate(token_counts)}

    metrics = []
    for target in target_sizes:
        # Find the recorded token count for this target size
        if target in size_to_tokens:
            token_count = size_to_tokens[target]
        else:
            # Just pick the closest available size if exact match not found
            available = [s for s in size_to_tokens if s <= target]
            if available:
                closest = max(available)
                token_count = size_to_tokens[closest]
            else:
                continue

        metrics.append({
            "size": target,
            "bytes_per_token": byte_count / token_count,
            "chars_per_token": char_count / token_count
        })
    return metrics

def plot_compression(metrics, output_file="compression_ratio.png"):
    """
    Plots compression ratio (dual y-axes) and saves the figure.
    """
    sizes = [m["size"] for m in metrics]
    bpt = [m["bytes_per_token"] for m in metrics]
    cpt = [m["chars_per_token"] for m in metrics]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    ax1.set_xlabel("Vocabulary Size")
    ax1.set_xscale('log')
    ax1.set_ylabel("Bytes per Token", color='tab:blue')
    ax1.plot(sizes, bpt, 'o-', color='tab:blue', label="Bytes / Token")
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Characters per Token", color='tab:orange')
    ax2.plot(sizes, cpt, 's--', color='tab:orange', label="Characters / Token")
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title("Compression Ratio vs. Vocabulary Size")
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    fig.tight_layout()
    
    plt.savefig(output_file)
    plt.show()

def main():
    import argparse
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Compression-ratio vs vocab-size study (§3.4).")
    parser.add_argument("--input", default=os.path.join(repo_root, "data", "TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--target_sizes", type=int, nargs="+", default=[1000, 2000, 4000, 8000, 16000])
    args = parser.parse_args()

    INPUT_PATH = args.input
    SPECIAL_TOKENS = ["<|endoftext|>"]
    TARGET_SIZES = args.target_sizes
    MAX_SIZE = max(TARGET_SIZES)

    #Train BPE once
    print(f"Training BPE up to vocab_size = {MAX_SIZE} ...")
    _, _, token_counts, initial_size = train_bpe_with_counts(INPUT_PATH, MAX_SIZE, SPECIAL_TOKENS)

    # Compute metrics for all target sizes
    metrics = compute_compression_metrics(INPUT_PATH, token_counts, initial_size, TARGET_SIZES)

    plot_compression(metrics)

    # Print table to console
    print("\n{:<16} | {:<12} | {:<12}".format("Vocabulary size", "Bytes/token", "Chars/token"))
    print("-" * 48)
    for m in metrics:
        print(f"{m['size']:<16} | {m['bytes_per_token']:<12.3f} | {m['chars_per_token']:<12.3f}")

if __name__ == "__main__":
    main()
