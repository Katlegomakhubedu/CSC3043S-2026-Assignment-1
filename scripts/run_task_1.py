"""
This script performs a full BPE training run on a large corpus to gather
results for Task 1 (Q1 and Q4) of the assignment.

It does the following:
1.  Times the BPE training process on the full training file (for Q1).
2.  Saves the resulting vocabulary and merges files.
3.  Analyzes the trained tokenizer to find the longest token (for Q4).
4.  Prints the first 5 and last 5 learned merges (for Q4).
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import pickle
import argparse
from src.tokenizer import train_bpe

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Run full BPE training and analysis for Task 1.")
    parser.add_argument("--input", default=os.path.join(repo_root, "data", "TinyStoriesV2-GPT4-valid.txt"),
                        help="Path to the full training .txt file.")
    parser.add_argument("--vocab_size", type=int, default=4000,
                        help="The vocabulary size to train.")
    parser.add_argument("--vocab_out", default=os.path.join(repo_root, "vocab.pkl"),
                        help="Path to save the trained vocabulary.")
    parser.add_argument("--merges_out", default=os.path.join(repo_root, "merges.pkl"),
                        help="Path to save the learned merges.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        print("Please download the dataset and place it in the 'data' directory.")
        return

    # --- Q1: Time the training process ---
    print(f"--- Starting BPE training (vocab_size={args.vocab_size}) on {os.path.basename(args.input)} ---")
    start_time = time.time()
    vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens=["<|endoftext|>"])
    end_time = time.time()
    training_time = end_time - start_time
    print(f"--- BPE training finished ---")
    print(f"\n[Q1 Result] Wall-clock time for training: {training_time:.2f} seconds")

    # Save the results
    with open(args.vocab_out, "wb") as f:
        pickle.dump(vocab, f)
    with open(args.merges_out, "wb") as f:
        pickle.dump(merges, f)
    print(f"Saved vocab to '{args.vocab_out}' and merges to '{args.merges_out}'")

    # --- Q4: Analyze the vocabulary and merges ---
    print("\n--- Analyzing results for Q4 ---")
    
    # Find the longest token
    if vocab:
        longest_token = max(vocab.values(), key=len)
        print(f"[Q4 Result] Longest token in vocabulary ({len(longest_token)} bytes): {longest_token}")
    
    # Print first and last 5 merges
    if merges:
        print("\n[Q4 Result] First 5 merges:")
        for i, pair in enumerate(merges[:5]):
            # repr() is used to make whitespace and control characters visible
            print(f"  {i+1}: {repr(pair[0])} + {repr(pair[1])}")

        print("\n[Q4 Result] Last 5 merges:")
        for i, pair in enumerate(merges[-5:]):
            # repr() is used to make whitespace and control characters visible
            print(f"  {len(merges) - 5 + i + 1}: {repr(pair[0])} + {repr(pair[1])}")

    print("\n--- For Q1 (Encoding Time) ---")
    print("To get the encoding time, run the following command:")
    print(f"python scripts/encode_corpus.py --input {args.input} --output encoded_train.npy --vocab {args.vocab_out} --merges {args.merges_out}")
    print("You will need to time this command yourself.")

if __name__ == "__main__":
    main()
