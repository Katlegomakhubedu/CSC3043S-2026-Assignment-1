import matplotlib.pyplot as plt
import os
from collections import Counter
from src.tokenizer import read_txt, split_text, get_pretokens, count_pretokens, merge_pair

def train_bpe_with_counts(input_path, max_vocab_size, special_tokens):
    """
    Trains BPE up to `max_vocab_size`.
    Returns: (vocab, merges, token_counts, initial_vocab_size)
    """
    # --- Setup ---
    raw_text = read_txt(input_path)
    documents = split_text(raw_text, special_tokens)
    
    # Pre-tokenize and get initial frequencies
    pretokens = get_pretokens(documents)
    word_freq = count_pretokens(pretokens)

    # Initialize vocabulary: Special tokens + 256 byte values
    vocab = {}
    for i, st in enumerate(special_tokens):
        vocab[i] = st.encode("utf-8")
    
    offset = len(special_tokens)
    for b in range(256):
        vocab[offset + b] = bytes([b])
    
    initial_vocab_size = len(vocab)
    
    # Count total tokens in the corpus initially
    total_tokens = sum(freq * len(word) for word, freq in word_freq.items())
    token_counts = [total_tokens]  # record count at initial size
    
    merges = []
    num_merges = max_vocab_size - initial_vocab_size

    # --- Iterative merge loop ---
    for _ in range(num_merges):
        # Count adjacent pairs across all words
        pair_counts = Counter()
        for word, freq in word_freq.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pair_counts[pair] += freq

        if not pair_counts:
            break

        # Tie-breaking: lexicographically greatest pair among max-count pairs
        best_count = max(pair_counts.values())
        best_pair = max(p for p, c in pair_counts.items() if c == best_count)

        # Apply the merge to all words
        new_word_freq = {}
        for word, freq in word_freq.items():
            merged = merge_pair(word, best_pair)
            new_word_freq[merged] = new_word_freq.get(merged, 0) + freq
        word_freq = new_word_freq

        # Update token count
        total_tokens -= best_count
        token_counts.append(total_tokens)

        # Record the merge
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

    return vocab, merges, token_counts, initial_vocab_size

def compute_compression_metrics(input_path, token_counts, initial_vocab_size, target_sizes):
    """
    Computes bytes/token and characters/token for target vocabulary sizes.
    """
    byte_count = os.path.getsize(input_path)
    char_count = len(read_txt(input_path))

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
    # Configuration
    INPUT_PATH = r"C:\Users\katle\OneDrive - University of Cape Town\Final Year\CS3043S\data\TinyStoriesV2-GPT4-valid.txt"
    SPECIAL_TOKENS = ["<|endoftext|>"]
    TARGET_SIZES = [1000, 2000, 4000, 8000, 16000]
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
