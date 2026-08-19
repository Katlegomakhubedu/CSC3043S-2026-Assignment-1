"""Task 1 end to end: train the byte-level BPE tokenizer and answer Q1-Q4 (§3).

This is the single entry point for Task 1. It replaces three earlier scripts
(src/run_task1.py, scripts/run_task_1.py, scripts/run_speed_test.py) that
overlapped, disagreed, and two of which could not run at all.

The corpus is pre-tokenized ONCE and everything else is derived from that single
pass:

  * the compression curve for Q3   - every merge reduces the corpus token count
                                     by exactly the merged pair's count, so the
                                     whole curve comes off one training run;
  * the primary tokenizer          - the first k merges of a longer run *are*
    and the §3.5 second tokenizer    the merges a shorter run would learn, so
                                     both are truncations of one merge list;
  * the Q1 timings                 - pre-tokenization and merging timed apart,
                                     since only their sum is the answer to Q1.

Run a smoke test first - `--limit_mb 50` exercises every step in a couple of
minutes - before committing to the full corpus.

Examples
--------
  # quick end-to-end check on a 50MB slice
  python scripts/run_task1.py --limit_mb 50 --out_dir /tmp/task1_smoke

  # the real thing, pre-tokenizing in parallel
  python scripts/run_task1.py --workers 8
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json
import multiprocessing
import pickle
import platform
import shutil
import tempfile
import time

from src.tokenizer import (BPETokenizer, init_vocab, train_bpe_incremental,
                            train_bpe_naive, derive_vocab_and_merges,
                            get_word_freq_from_files, get_word_freq_parallel)
from scripts.vocab_study import (compute_compression_metrics, plot_compression,
                                  write_csv, corpus_size_from_word_freq)
from scripts.encode_corpus import encode_corpus

SPECIAL_TOKENS = ["<|endoftext|>"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def describe_machine():
    """Q1 asks which machine the timings were taken on."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def make_slice(source_files, out_path, n_bytes):
    """Copy the first `n_bytes` of the corpus into `out_path`, cut at a document
    boundary so the slice is a whole number of documents."""
    written = 0
    with open(out_path, "w", encoding="utf-8", newline="") as out:
        for path in source_files:
            with open(path, "r", encoding="utf-8", newline="") as f:
                while written < n_bytes:
                    chunk = f.read(min(1 << 20, n_bytes - written))
                    if not chunk:
                        break
                    out.write(chunk)
                    written += len(chunk)
            if written >= n_bytes:
                break
        # Terminate on a document boundary.
        out.write(SPECIAL_TOKENS[0])
    return out_path


def save_tokenizer(vocab, merges, out_dir, prefix):
    vocab_path = os.path.join(out_dir, f"{prefix}vocab.pkl")
    merges_path = os.path.join(out_dir, f"{prefix}merges.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    return vocab_path, merges_path


def save_results(results, out_dir):
    """Every number the report quotes must be traceable to a file in the repo."""
    path = os.path.join(out_dir, "logs", "task1_results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def describe_merges(vocab, merges, n=5):
    """Q4: longest token, plus the first and last few merges."""
    longest = max(vocab.values(), key=len)
    return {
        "longest_token": repr(longest),
        "longest_token_length_bytes": len(longest),
        "first_merges": [f"{left!r} + {right!r}" for left, right in merges[:n]],
        "last_merges": [f"{left!r} + {right!r}" for left, right in merges[-n:]],
    }


# ---------------------------------------------------------------------------
# Q3 / §3.4: the vocabulary-size study, which is what *decides* the size
# ---------------------------------------------------------------------------

def run_study_and_choose(args, results, word_freq, base_size, corpus_bytes, t_pretok):
    """Build the compression curve, then commit to a vocabulary size.

    §3.4 is the step that chooses the vocabulary size, so the curve has to exist
    before any size is committed to - this runs before the Q1 timing rather than
    after it. One run to the largest size of interest produces every point on
    the curve, because each merge reduces the corpus token count by exactly the
    merged pair's count.

    Returns (merges_max, merges_primary, vocab_primary), or (None, None, None)
    if no `--vocab_size` was given, meaning the caller should stop and let the
    curve inform the choice.
    """
    max_study = max(args.study_sizes + [args.second_vocab_size]
                    + ([args.vocab_size] if args.vocab_size else []))
    print(f"\n[2/6] Training merges to {max_study} for the Q3 curve ...")
    vocab_max = init_vocab(SPECIAL_TOKENS)
    t0 = time.time()
    merges_max, token_counts = train_bpe_incremental(
        word_freq, vocab_max, max_study - base_size, track_token_counts=True)
    print(f"      {len(merges_max):,} merges in {(time.time() - t0) / 60:.1f} min")

    # Save the full merge list so that choosing a different vocabulary size later
    # costs nothing: any size <= max_study is a truncation of this list, and
    # re-running with --from_merges skips the pre-tokenization pass entirely.
    merges_max_path = os.path.join(args.out_dir, f"merges_upto{max_study}.pkl")
    with open(merges_max_path, "wb") as f:
        pickle.dump(merges_max, f)

    metrics, _, _ = compute_compression_metrics(
        word_freq, token_counts, base_size, args.study_sizes, d_model=args.d_model)
    png = os.path.join(args.out_dir, "compression_ratio.png")
    csv_path = os.path.join(args.out_dir, "logs", "vocab_study.csv")
    plot_compression(metrics, png)
    write_csv(metrics, csv_path)
    results["q3"] = {"metrics": metrics, "figure": os.path.basename(png),
                     "csv": os.path.relpath(csv_path, args.out_dir),
                     "merges_file": os.path.basename(merges_max_path)}

    print(f"\n[3/6] Compression ratio vs vocabulary size (Q3)")
    print(f"\n{'Vocab':<8} | {'Bytes/token':>12} | {'Chars/token':>12} | {'Embed+head':>12}")
    print("-" * 54)
    for m in metrics:
        print(f"{m['vocab_size']:<8} | {m['bytes_per_token']:>12.3f} | "
              f"{m['chars_per_token']:>12.3f} | {m['embed_lm_head_params']:>12,}")

    if args.vocab_size is None:
        results_path = save_results(results, args.out_dir)
        print("\n" + "=" * 66)
        print("STUDY COMPLETE - no --vocab_size given, stopping here.")
        print("=" * 66)
        print("Pick a size from the curve above, weighing the compression gain")
        print("against the parameter cost of the embedding and LM head, then")
        print("re-run. The merge list is saved, so this does NOT repeat the")
        print("pre-tokenization pass:")
        print(f"\n  python scripts/run_task1.py --vocab_size <N> \\")
        print(f"      --from_merges {merges_max_path}\n")
        print(f"Study numbers written to {results_path}")
        return None, None, None

    # -- Q1: time a merge run at the chosen size ----------------------------
    print(f"\n[4/6] Timing a merge run at vocab_size={args.vocab_size} (Q1) ...")
    vocab_primary = init_vocab(SPECIAL_TOKENS)
    t0 = time.time()
    merges_primary, _ = train_bpe_incremental(
        word_freq, vocab_primary, args.vocab_size - base_size)
    t_merge_chosen = time.time() - t0
    print(f"      {len(merges_primary):,} merges in {t_merge_chosen / 60:.1f} min")

    results["q1"] = {
        "vocab_size": args.vocab_size,
        "corpus_bytes": corpus_bytes,
        "workers": args.workers,
        "pretokenize_seconds": round(t_pretok, 1),
        "merge_seconds": round(t_merge_chosen, 1),
        "train_total_seconds": round(t_pretok + t_merge_chosen, 1),
    }

    # The tokenizer derived from the longer run must equal the one trained
    # directly here; assert it rather than trusting it.
    assert derive_vocab_and_merges(merges_max, args.vocab_size, SPECIAL_TOKENS)[1] \
        == merges_primary, "truncated merge list disagrees with the direct run"

    return merges_max, merges_primary, vocab_primary


# ---------------------------------------------------------------------------
# Q2: tutorial-style vs incremental merge counting on a fixed slice
# ---------------------------------------------------------------------------

def run_q2(slice_path, vocab_size, results):
    """§3.2's claim, measured: incremental merge counting vs full recounting.

    Both sides are given the *same* pre-token frequency table, so the comparison
    isolates the merge loop - which is the thing §3.2 asks you to change.
    Pre-tokenization is reported separately, and the end-to-end row adds it back
    to both sides so the user-visible speedup is not overstated.
    """
    print(f"\n[Q2] Tutorial vs optimised merge counting "
          f"(vocab_size={vocab_size}, {os.path.getsize(slice_path) / 1e6:.1f}MB slice)")

    t0 = time.time()
    word_freq = get_word_freq_from_files([slice_path], SPECIAL_TOKENS)
    t_pretok = time.time() - t0

    num_merges = vocab_size - len(init_vocab(SPECIAL_TOKENS))

    vocab_opt = init_vocab(SPECIAL_TOKENS)
    t0 = time.time()
    merges_opt, _ = train_bpe_incremental(word_freq, vocab_opt, num_merges)
    t_opt = time.time() - t0

    vocab_naive = init_vocab(SPECIAL_TOKENS)
    t0 = time.time()
    merges_naive = train_bpe_naive(word_freq, vocab_naive, num_merges)
    t_naive = time.time() - t0

    # The optimisation is only worth reporting if it computed the same answer.
    assert merges_opt == merges_naive, "incremental and naive trainers disagree"

    results["q2"] = {
        "slice_bytes": os.path.getsize(slice_path),
        "vocab_size": vocab_size,
        "pretokenize_seconds": round(t_pretok, 2),
        "merge_seconds_tutorial": round(t_naive, 2),
        "merge_seconds_optimised": round(t_opt, 2),
        "merge_speedup": round(t_naive / t_opt, 1) if t_opt else None,
        "end_to_end_seconds_tutorial": round(t_pretok + t_naive, 2),
        "end_to_end_seconds_optimised": round(t_pretok + t_opt, 2),
        "end_to_end_speedup": round((t_pretok + t_naive) / (t_pretok + t_opt), 1) if t_opt else None,
        "merges_identical": True,
    }

    print(f"  pre-tokenization (shared) : {t_pretok:8.2f} s")
    print(f"  merge loop, tutorial      : {t_naive:8.2f} s")
    print(f"  merge loop, optimised     : {t_opt:8.2f} s   -> {t_naive / t_opt:.1f}x")
    print(f"  end to end, tutorial      : {t_pretok + t_naive:8.2f} s")
    print(f"  end to end, optimised     : {t_pretok + t_opt:8.2f} s   "
          f"-> {(t_pretok + t_naive) / (t_pretok + t_opt):.1f}x")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(repo_root, "data")

    parser = argparse.ArgumentParser(
        description="Task 1: train the BPE tokenizer, encode the corpus, answer Q1-Q4.")
    parser.add_argument("--train_files", nargs="+", default=[
        os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt"),
        os.path.join(data_dir, "TinyStoriesV2-GPT4-train-part2.txt"),
    ], help="One or more files that together make up the training corpus.")
    parser.add_argument("--valid_file", default=os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Primary vocabulary size (section 4.1 reference: 4000). "
                             "Omit to stop after the section 3.4 study, so the "
                             "compression curve can inform the choice.")
    parser.add_argument("--from_merges", default=None,
                        help="Reuse the merge list saved by an earlier run "
                             "(merges_upto<N>.pkl) instead of pre-tokenizing and "
                             "training again. Any vocab_size <= N is a truncation "
                             "of it, so changing your mind about the size is free.")
    parser.add_argument("--second_vocab_size", type=int, default=1000,
                        help="Second tokenizer for the §7.3 vocabulary study.")
    parser.add_argument("--study_sizes", type=int, nargs="+",
                        default=[1000, 2000, 4000, 8000, 16000],
                        help="Vocabulary sizes for the Q3 compression curve.")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel pre-tokenization workers (§3.2, recommended).")
    parser.add_argument("--q2_slice_mb", type=int, default=10)
    parser.add_argument("--limit_mb", type=int, default=0,
                        help="Smoke test: use only the first N MB of the training corpus. "
                             "0 uses the full corpus.")
    parser.add_argument("--skip_encode", action="store_true",
                        help="Skip corpus encoding (the slowest step).")
    parser.add_argument("--skip_q2", action="store_true")
    parser.add_argument("--out_dir", default=repo_root)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)

    missing = [p for p in args.train_files + [args.valid_file] if not os.path.exists(p)]
    if missing:
        parser.error("missing data file(s):\n  " + "\n  ".join(missing))

    results = {}
    prior_path = os.path.join(args.out_dir, "logs", "task1_results.json")
    if args.from_merges and os.path.exists(prior_path):
        # A --from_merges run only derives and encodes; it must not clobber the
        # Q1 timings and Q3 curve produced by the run that did the training.
        with open(prior_path, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Carrying forward earlier results from {prior_path}")
    results["machine"] = describe_machine()
    results["config"] = vars(args)
    tmp_dir = tempfile.mkdtemp(prefix="task1_")

    try:
        train_files = args.train_files
        if args.limit_mb:
            print(f"[smoke test] using only the first {args.limit_mb}MB of the training corpus")
            train_files = [make_slice(train_files,
                                       os.path.join(tmp_dir, "train_slice.txt"),
                                       args.limit_mb * 1_000_000)]
            results["config"]["limited_to_bytes"] = args.limit_mb * 1_000_000

        corpus_bytes = sum(os.path.getsize(p) for p in train_files)
        print(f"\nTraining corpus: {corpus_bytes / 1e9:.2f} GB across "
              f"{len(train_files)} file(s)")

        base_size = len(init_vocab(SPECIAL_TOKENS))

        if args.from_merges:
            # Reusing an earlier run's merge list: the pre-tokenization pass and
            # the study have already been paid for, so go straight to deriving
            # tokenizers and encoding.
            if args.vocab_size is None:
                parser.error("--from_merges requires --vocab_size "
                             "(the point of reusing merges is to commit to a size)")
            print(f"\n[1/3] Loading merges from {os.path.basename(args.from_merges)} ...")
            with open(args.from_merges, "rb") as f:
                merges_max = pickle.load(f)
            print(f"      {len(merges_max):,} merges, covering vocab_size up to "
                  f"{base_size + len(merges_max):,} - pre-tokenization skipped")
            results["reused_merges"] = os.path.basename(args.from_merges)
            vocab_primary, merges_primary = derive_vocab_and_merges(
                merges_max, args.vocab_size, SPECIAL_TOKENS)
            step_derive, step_encode = "2/3", "3/3"
        else:
            step_derive, step_encode = "5/6", "6/6"

            # -- 1. Pre-tokenize once ---------------------------------------
            print(f"\n[1/6] Pre-tokenizing (workers={args.workers}) ...")
            t0 = time.time()
            if args.workers > 1:
                word_freq = get_word_freq_parallel(train_files, SPECIAL_TOKENS, args.workers)
            else:
                word_freq = get_word_freq_from_files(train_files, SPECIAL_TOKENS)
            t_pretok = time.time() - t0
            n_chars, n_bytes = corpus_size_from_word_freq(word_freq)
            print(f"      {len(word_freq):,} distinct pre-tokens, {n_bytes:,} bytes, "
                  f"{t_pretok / 60:.1f} min")

            merges_max, merges_primary, vocab_primary = run_study_and_choose(
                args, results, word_freq, base_size, corpus_bytes, t_pretok)
            if merges_max is None:
                return   # study only: no vocabulary size committed to yet


        # -- Derive and save both tokenizers ----------------------------------
        # The §3.5 second tokenizer is another truncation of the same merge
        # list, so it costs nothing beyond writing the file.
        vocab_second, merges_second = derive_vocab_and_merges(
            merges_max, args.second_vocab_size, SPECIAL_TOKENS)

        primary_paths = save_tokenizer(vocab_primary, merges_primary, args.out_dir, "")
        second_prefix = f"vocab{args.second_vocab_size}_"
        second_paths = save_tokenizer(vocab_second, merges_second, args.out_dir, second_prefix)
        print(f"\n[{step_derive}] Saved tokenizers (vocab_size {args.vocab_size} and "
              f"{args.second_vocab_size}):\n      {primary_paths[0]}\n      {second_paths[0]}")

        results["q4"] = describe_merges(vocab_primary, merges_primary)
        results["tokenizers"] = {
            "primary": {"vocab_size": args.vocab_size,
                        "vocab": os.path.basename(primary_paths[0]),
                        "merges": os.path.basename(primary_paths[1])},
            "secondary": {"vocab_size": args.second_vocab_size,
                          "vocab": os.path.basename(second_paths[0]),
                          "merges": os.path.basename(second_paths[1])},
        }

        print(f"\n[Q4] longest token: {results['q4']['longest_token']} "
              f"({results['q4']['longest_token_length_bytes']} bytes)")
        print("     first merges:", ", ".join(results["q4"]["first_merges"]))
        print("     last  merges:", ", ".join(results["q4"]["last_merges"]))

        # -- Encode the corpus with both tokenizers ---------------------------
        if args.skip_encode:
            print(f"\n[{step_encode}] Skipping corpus encoding (--skip_encode).")
        else:
            # Plain ASCII: the Windows console's default code page mangles
            # non-ASCII characters such as the section sign.
            print(f"\n[{step_encode}] Encoding corpus (section 3.5) ...")
            results["encoded"] = {}
            t_encode_start = time.time()
            for label, (vocab, merges), size in (
                    ("primary", (vocab_primary, merges_primary), args.vocab_size),
                    ("secondary", (vocab_second, merges_second), args.second_vocab_size)):
                tokenizer = BPETokenizer(vocab, merges, SPECIAL_TOKENS)
                suffix = "" if label == "primary" else f"_vocab{size}"
                for name, sources in (("train", train_files), ("valid", [args.valid_file])):
                    for i, source in enumerate(sources):
                        part = f"_part{i + 1}" if len(sources) > 1 else ""
                        out = os.path.join(args.out_dir, f"{name}{part}{suffix}.npy")
                        print(f"      {os.path.basename(source)} -> {os.path.basename(out)}")
                        meta = encode_corpus(source, out, tokenizer)
                        results["encoded"][os.path.basename(out)] = meta
            # With --from_merges there is no q1 block (the timings belong to the
            # earlier run that actually did the training).
            results.setdefault("q1", {})["encode_seconds"] = round(
                time.time() - t_encode_start, 1)

        # -- Q2 --------------------------------------------------------------
        if not args.skip_q2:
            slice_path = make_slice(train_files, os.path.join(tmp_dir, "q2_slice.txt"),
                                     args.q2_slice_mb * 1_000_000)
            run_q2(slice_path, 1000, results)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # -- save every number the report will quote ----------------------------
    results_path = save_results(results, args.out_dir)

    print("\n" + "=" * 66)
    print("TASK 1 COMPLETE")
    print("=" * 66)
    q1 = results.get("q1", {})
    if "train_total_seconds" in q1:
        print(f"[Q1] BPE training on {q1['corpus_bytes'] / 1e9:.2f} GB at vocab_size="
              f"{q1['vocab_size']}: {q1['train_total_seconds'] / 60:.1f} min "
              f"({q1['pretokenize_seconds'] / 60:.1f} pre-tokenize + "
              f"{q1['merge_seconds'] / 60:.1f} merge)")
    else:
        print(f"[Q1] Training timings come from the earlier run that produced "
              f"{results.get('reused_merges')}")
    if "encode_seconds" in q1:
        print(f"     Encoding: {q1['encode_seconds'] / 60:.1f} min")
    print(f"     Machine: {results['machine']['platform']} "
          f"({results['machine']['cpu_count']} CPUs)")
    print(f"\nAll numbers written to {results_path}")
    print("Still to write up by hand: the Q3 vocabulary-size justification and")
    print("the Q4 comment on whether the merge progression makes sense.")


if __name__ == "__main__":
    # Required for the `spawn` start method used by parallel pre-tokenization
    # on Windows and macOS.
    multiprocessing.freeze_support()
    main()
