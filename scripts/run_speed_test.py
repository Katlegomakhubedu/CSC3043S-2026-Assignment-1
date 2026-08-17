"""
This script performs a speed comparison between the naive BPE trainer and the
optimized, incremental BPE trainer to gather results for Task 1 (Q2).

It does the following:
1.  Reads a 10MB slice from a given input file.
2.  Times the "naive" BPE trainer (which recounts on every merge).
3.  Times the "incremental" BPE trainer (which updates counts efficiently).
4.  Calculates and prints the speedup factor.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import argparse
from src.tokenizer import (
    train_bpe_incremental, init_vocab, get_pretokens, split_text_by_specials,
    get_word_freq_from_files
)
# The naive implementation is in the test file, so we import it from there.
from tests.test_tokenizer import naive_train_bpe

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Run BPE trainer speed comparison for Task 1, Q2.")
    parser.add_argument("--input", default=os.path.join(repo_root, "data", "TinyStoriesV2-GPT4-train.txt"),
                        help="Path to the training .txt file to slice.")
    parser.add_argument("--slice_mb", type=int, default=10,
                        help="The size of the data slice in megabytes.")
    parser.add_argument("--vocab_size", type=int, default=1000,
                        help="The vocabulary size for the speed test.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        print("Please download the dataset and place it in the 'data' directory.")
        return

    # --- Prepare Data Slice ---
    print(f"--- Preparing {args.slice_mb}MB data slice ---")
    slice_size_bytes = args.slice_mb * 1024 * 1024
    with open(args.input, "r", encoding="utf-8") as f:
        data_slice = f.read(slice_size_bytes)
    
    special_tokens = ["<|endoftext|>"]
    documents = split_text_by_specials(data_slice, special_tokens)
    pretokens = get_pretokens(documents)
    word_freq = get_word_freq_from_files([args.input], special_tokens)


    num_merges = args.vocab_size - len(init_vocab(special_tokens))

    # --- Q2: Time the naive implementation ---
    print("\n--- Running Naive BPE Trainer ---")
    vocab_naive = init_vocab(special_tokens)
    start_time_naive = time.time()
    naive_train_bpe(word_freq, vocab_naive, num_merges)
    end_time_naive = time.time()
    time_naive = end_time_naive - start_time_naive
    print(f"[Q2 Result] Naive trainer time: {time_naive:.2f} seconds")

    # --- Q2: Time the incremental implementation ---
    print("\n--- Running Incremental BPE Trainer ---")
    vocab_incremental = init_vocab(special_tokens)
    start_time_incremental = time.time()
    train_bpe_incremental(word_freq, vocab_incremental, num_merges)
    end_time_incremental = time.time()
    time_incremental = end_time_incremental - start_time_incremental
    print(f"[Q2 Result] Incremental trainer time: {time_incremental:.2f} seconds")

    # --- Q2: Report Speedup ---
    print("\n--- Speed Comparison ---")
    if time_incremental > 0:
        speedup = time_naive / time_incremental
        print(f"[Q2 Result] Speedup: {speedup:.2f}x")
    else:
        print("[Q2 Result] Could not calculate speedup (incremental time was near zero).")

if __name__ == "__main__":
    main()
