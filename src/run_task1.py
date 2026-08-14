import os
import sys
import time
import pickle
import argparse
from collections import Counter

# src/*.py import each other with bare names (e.g. `from tokenizer import ...`),
# which needs src/ itself on sys.path; scripts/*.py import via `from src.x import
# ...`, which needs the repo root on sys.path instead. This script uses both
# styles, so both directories need to be on sys.path.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _REPO_ROOT)

# Import from your existing modules
from tokenizer import train_bpe, read_txt, split_text, get_pretokens, count_pretokens, merge_pair
from scripts.vocab_study import train_bpe_with_counts, compute_compression_metrics, plot_compression
from scripts.encode_corpus import encode_corpus
from tokenizer import BPETokenizer

# ============================================================
# CONFIGURATION
# ============================================================
REPO_ROOT = _REPO_ROOT
DATA_DIR = os.path.join(REPO_ROOT, "data")

# TinyStories' train split ships as two part files that together form one
# corpus (see tokenizer.get_word_freq_from_files) - list both here.
DEFAULT_TRAIN_FILES = [
    os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train-part1.txt"),
    os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train-part2.txt"),
]
DEFAULT_VALID_FILE = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")

parser = argparse.ArgumentParser(description="Run Task 1 (BPE tokenizer) end to end.")
parser.add_argument("--train_files", nargs="+", default=DEFAULT_TRAIN_FILES,
                     help="One or more files making up the training corpus.")
parser.add_argument("--valid_file", default=DEFAULT_VALID_FILE)
parser.add_argument("--vocab_size", type=int, default=4000)
parser.add_argument("--out_dir", default=REPO_ROOT)
args = parser.parse_args()

TRAIN_FILES = args.train_files
VALID_FILE = args.valid_file
SPECIAL_TOKENS = ["<|endoftext|>"]
CHOSEN_VOCAB_SIZE = args.vocab_size

for f in TRAIN_FILES + [VALID_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Expected data file not found: {f}")

# ============================================================
# 1. VOCABULARY STUDY (Q3, Q4 - reuse vocab_study.py logic)
# ============================================================
print("\n[STEP 1] Running vocabulary study (Q3, Q4)...")
target_sizes = [1000, 2000, 4000, 8000, 16000]
max_size = max(target_sizes)

# Reuse train_bpe_with_counts from vocab_study.py
vocab, merges, token_counts, initial_size = train_bpe_with_counts(TRAIN_FILES, max_size, SPECIAL_TOKENS)

# Reuse compute_compression_metrics and plot_compression
metrics = compute_compression_metrics(TRAIN_FILES, token_counts, initial_size, target_sizes)
plot_compression(metrics, output_file=os.path.join(args.out_dir, "compression_ratio.png"))
print("  -> Q3 Plot saved: compression_ratio.png")

# ============================================================
# 2. TRAIN MAIN TOKENIZER (Q1, Q4) - reuse train_bpe (incremental)
# ============================================================
print(f"\n[STEP 2] Training chosen tokenizer (vocab={CHOSEN_VOCAB_SIZE})...")
start_time = time.time()
vocab_chosen, merges_chosen = train_bpe(TRAIN_FILES, CHOSEN_VOCAB_SIZE, SPECIAL_TOKENS)
train_time = time.time() - start_time

# Save tokenizer
with open(os.path.join(args.out_dir, "vocab.pkl"), "wb") as f: pickle.dump(vocab_chosen, f)
with open(os.path.join(args.out_dir, "merges.pkl"), "wb") as f: pickle.dump(merges_chosen, f)
print(f"  -> Saved vocab.pkl, merges.pkl")

# ============================================================
# 3. ENCODE FULL CORPUS (Q1) - reuse encode_corpus.py
# ============================================================
print("\n[STEP 3] Encoding full corpus (reusing encode_corpus.py)...")
enc_start = time.time()

tokenizer_obj = BPETokenizer(vocab_chosen, merges_chosen, SPECIAL_TOKENS)

for train_file in TRAIN_FILES:
    out_name = os.path.splitext(os.path.basename(train_file))[0] + ".npy"
    encode_corpus(train_file, os.path.join(args.out_dir, out_name), tokenizer_obj)
encode_corpus(VALID_FILE, os.path.join(args.out_dir, "valid_encoded.npy"), tokenizer_obj)
enc_time = time.time() - enc_start

# ============================================================
# 4. Q2 SPEEDUP TEST (10 MB slice)
# ============================================================
print("\n[STEP 4] Measuring Q2 speedup on 10 MB slice...")
slice_path = os.path.join(args.out_dir, "temp_10mb.txt")
with open(TRAIN_FILES[0], 'r', encoding='utf-8') as f:
    with open(slice_path, 'w', encoding='utf-8') as out:
        out.write(f.read(10 * 1024 * 1024))


def naive_train(path, vs, st):
    """Tutorial-style baseline: recount every pair from scratch after every
    merge. This is the O(merges * corpus_size) implementation §3.2 asks you
    to replace - kept here only as the "before" side of the Q2 comparison."""
    raw = read_txt(path)
    docs = split_text(raw, st)
    p_tokens = get_pretokens(docs)
    wf = count_pretokens(p_tokens)
    v = {}
    for i, tok in enumerate(st): v[i] = tok.encode("utf-8")
    off = len(st)
    for b in range(256): v[off+b] = bytes([b])
    m = []
    for _ in range(vs - len(v)):
        pair_counts = Counter()
        for word, freq in wf.items():
            for i in range(len(word)-1):
                pair_counts[(word[i], word[i+1])] += freq
        if not pair_counts: break
        best_count = max(pair_counts.values())
        best_pair = max(p for p, c in pair_counts.items() if c == best_count)
        new_wf = {}
        for word, freq in wf.items():
            merged = merge_pair(word, best_pair)
            new_wf[merged] = new_wf.get(merged, 0) + freq
        wf = new_wf
        m.append(best_pair)
        v[len(v)] = best_pair[0] + best_pair[1]
    return v, m

# Optimised (incremental) - the real train_bpe implementation.
start = time.time()
train_bpe(slice_path, 1000, SPECIAL_TOKENS)
opt_time = time.time() - start

# Naive (recount every merge) - the tutorial baseline.
start = time.time()
naive_train(slice_path, 1000, SPECIAL_TOKENS)
naive_time = time.time() - start
os.remove(slice_path)
speedup = naive_time / opt_time

# ============================================================
# PRINT ANSWERS FOR Q1-Q4
# ============================================================
print("\n" + "="*60)
print("ANSWERS TO TASK 1 (Q1-Q4)")
print("="*60)

print("\n[Q1] Wall-clock times:")
print(f"  BPE Training (vocab={CHOSEN_VOCAB_SIZE}): {train_time:.1f}s ({train_time/60:.1f} min)")
print(f"  Encoding full corpus: {enc_time:.1f}s ({enc_time/60:.1f} min)")
print("  Machine: <<< fill in the machine you actually ran this on >>>")

print("\n[Q2] Speedup on 10 MB slice:")
print("  +----------------+----------+")
print("  | Implementation  | Time (s) |")
print("  +----------------+----------+")
print(f"  | Tutorial-1      | {naive_time:6.2f}  |")
print(f"  | Optimised (Inc) | {opt_time:6.2f}  |")
print("  +----------------+----------+")
print(f"  | Speedup         | {speedup:6.1f}x  |")
print("  +----------------+----------+")

print("\n[Q3] Compression ratio plot saved as 'compression_ratio.png'")
print(f"  Vocabulary size choice: {CHOSEN_VOCAB_SIZE}")
print("  Reason: <<< justify from the compression_ratio.png curve + parameter-count tradeoff >>>")

print("\n[Q4] Token analysis:")
longest = max(vocab_chosen.values(), key=len)
print(f"  Longest token: {longest}")
print(f"  First 5 merges: {merges_chosen[:5]}")
print(f"  Last 5 merges:  {merges_chosen[-5:] if len(merges_chosen) >= 5 else merges_chosen}")
print("  Explanation: <<< does the merge progression make sense? >>>")

print("\n" + "="*60)
print("Task 1 complete. You now have:")
print("   - compression_ratio.png (Q3)")
print("   - vocab.pkl, merges.pkl (Q4)")
print("   - <train-part>.npy x N, valid_encoded.npy (ready for GPU training)")
print("   - Answers Q1-Q4 printed above (fill in the <<< >>> placeholders).")
print("="*60)
