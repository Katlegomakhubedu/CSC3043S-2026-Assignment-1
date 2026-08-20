import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Tokens per copied chunk. 32M uint16 is 64MB of buffer, small enough to stay
# out of the way and large enough that the copy is bound by disk, not Python.
CHUNK_TOKENS = 32 << 20

# Counts that add across parts. Anything else (rates, the eot id, timings)
# either has to be recomputed or does not survive the merge - see merge_meta.
SUMMABLE = ("n_tokens", "n_documents", "n_chars_excluding_delimiters",
            "n_chars_including_delimiters", "n_bytes_utf8_excluding_delimiters",
            "eot_count")


def part_paths(data_dir, suffix, n_parts=2):
    return [os.path.join(data_dir, f"train_part{i}{suffix}.npy")
            for i in range(1, n_parts + 1)]


def meta_path(npy_path):
    return os.path.splitext(npy_path)[0] + "_meta.json"


def read_meta(npy_path):
    path = meta_path(npy_path)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def merge_meta(metas, output_file, n_tokens):
    """Sum the per-part sidecars into one for the combined array.

    Rates are recomputed from the summed totals rather than averaged - a mean of
    two ratios is not the ratio of the sums unless the parts happen to be the
    same size, and BPC divides by this.
    """
    merged = {
        "output_file": os.path.basename(output_file),
        "input_file": [m.get("input_file") for m in metas],
        "combined_from": [m.get("output_file") for m in metas],
        "n_parts": len(metas),
    }
    for key in SUMMABLE:
        values = [m.get(key) for m in metas]
        if all(v is not None for v in values):
            merged[key] = sum(values)

    vocab_sizes = {m.get("vocab_size") for m in metas}
    if len(vocab_sizes) > 1:
        raise SystemExit(f"parts disagree on vocab_size: {sorted(vocab_sizes)} - "
                         f"they were not encoded with the same tokenizer")
    merged["vocab_size"] = vocab_sizes.pop()

    eot_ids = {m.get("eot_id") for m in metas}
    if len(eot_ids) > 1:
        raise SystemExit(f"parts disagree on eot_id: {sorted(eot_ids)}")
    merged["eot_id"] = eot_ids.pop()

    if merged.get("n_tokens") not in (None, n_tokens):
        raise SystemExit(
            f"sidecars claim {merged['n_tokens']:,} tokens but the arrays hold "
            f"{n_tokens:,} - the parts and their metadata are out of step")
    merged["n_tokens"] = n_tokens

    chars = merged.get("n_chars_excluding_delimiters")
    bytes_ = merged.get("n_bytes_utf8_excluding_delimiters")
    if chars:
        merged["chars_per_token"] = chars / n_tokens
    if bytes_:
        merged["bytes_per_token"] = bytes_ / n_tokens
    return merged


def combine(parts, output, chunk_tokens=CHUNK_TOKENS, verify=True):
    """Concatenate `parts` into `output`, streaming through a memmap.

    Order matters: the parts are documents 1..N of one corpus, and part 2
    continues where part 1 stopped.
    """
    arrays = [np.load(p, mmap_mode="r") for p in parts]

    dtypes = {a.dtype for a in arrays}
    if len(dtypes) > 1:
        raise SystemExit(f"parts have different dtypes: {sorted(map(str, dtypes))}")
    dtype = dtypes.pop()
    if dtype != np.uint16:
        print(f"  ! parts are {dtype}, not uint16 (§3.3 expects uint16)")

    total = sum(len(a) for a in arrays)
    print(f"Combining {len(parts)} parts -> {os.path.basename(output)}  "
          f"({total:,} tokens, {total * dtype.itemsize / 1e9:.2f} GB)")

    out = np.lib.format.open_memmap(output, mode="w+", dtype=dtype, shape=(total,))
    written = 0
    try:
        for path, array in zip(parts, arrays):
            print(f"  {os.path.basename(path)}: {len(array):,} tokens")
            for start in range(0, len(array), chunk_tokens):
                stop = min(start + chunk_tokens, len(array))
                out[written + start:written + stop] = array[start:stop]
            written += len(array)
        out.flush()
    finally:
        del out

    if verify:
        verify_combined(parts, output)
    return total


def verify_combined(parts, output):
    """Check the combined array really is the parts, end to end.

    Spot-checks rather than a full comparison: the boundaries are where a
    concatenation goes wrong, and reading 1.1GB twice to prove the middle is
    intact is not worth the minutes.
    """
    combined = np.load(output, mmap_mode="r")
    offset = 0
    for path in parts:
        part = np.load(path, mmap_mode="r")
        edge = min(4096, len(part))
        head_ok = np.array_equal(combined[offset:offset + edge], part[:edge])
        tail_ok = np.array_equal(combined[offset + len(part) - edge:offset + len(part)],
                                 part[-edge:])
        if not (head_ok and tail_ok):
            raise SystemExit(
                f"verification failed at {os.path.basename(path)} "
                f"(offset {offset:,}) - the combined array does not match the parts")
        offset += len(part)

    if offset != len(combined):
        raise SystemExit(f"length mismatch: parts total {offset:,}, "
                         f"combined holds {len(combined):,}")
    print(f"  verified: {len(parts)} part boundaries line up, "
          f"{len(combined):,} tokens total")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Combine Task 1's train_part*.npy into one train.npy.")
    p.add_argument("--suffix", default="",
                   help="Task 1 naming suffix, e.g. '_vocab1000' for the "
                        "section 7.3 tokenizer.")
    p.add_argument("--parts", nargs="+", default=None,
                   help="Explicit part paths, in order. Defaults to "
                        "train_part1/train_part2 with --suffix.")
    p.add_argument("--output", default=None,
                   help="Defaults to train<suffix>.npy next to the parts.")
    p.add_argument("--data_dir", default=REPO_ROOT)
    p.add_argument("--chunk_tokens", type=int, default=CHUNK_TOKENS)
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing output.")
    p.add_argument("--no_verify", dest="verify", action="store_false",
                   help="Skip the boundary check after writing.")
    p.add_argument("--remove_parts", action="store_true",
                   help="Delete the parts once the combined array verifies. "
                        "Re-encoding them costs an hour, so this is opt-in.")
    args = p.parse_args(argv)

    parts = args.parts or part_paths(args.data_dir, args.suffix)
    missing = [q for q in parts if not os.path.exists(q)]
    if missing:
        raise SystemExit(
            "missing:\n  " + "\n  ".join(os.path.basename(q) for q in missing) +
            f"\n\nTask 1 phase 2 writes these. If it is still running, wait for "
            f"it to finish; if it used different names, pass --parts explicitly.")

    output = args.output or os.path.join(args.data_dir, f"train{args.suffix}.npy")
    if os.path.exists(output) and not args.force:
        raise SystemExit(f"{os.path.basename(output)} already exists - "
                         f"pass --force to overwrite it.")

    free = None
    try:
        free = os.statvfs(os.path.dirname(output) or ".").f_bavail * \
            os.statvfs(os.path.dirname(output) or ".").f_frsize
    except (AttributeError, OSError):
        import shutil
        try:
            free = shutil.disk_usage(os.path.dirname(output) or ".").free
        except OSError:
            pass
    needed = sum(os.path.getsize(q) for q in parts)
    if free is not None and free < needed:
        raise SystemExit(
            f"need {needed / 1e9:.2f} GB for the combined array but only "
            f"{free / 1e9:.2f} GB is free")

    total = combine(parts, output, args.chunk_tokens, verify=args.verify)

    metas = [read_meta(q) for q in parts]
    if all(m is not None for m in metas):
        merged = merge_meta(metas, output, total)
        with open(meta_path(output), "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        print(f"  wrote {os.path.basename(meta_path(output))}: "
              f"{merged['n_documents']:,} documents, "
              f"{merged['n_chars_excluding_delimiters']:,} characters")
    else:
        print("  ! some parts have no _meta.json, so no sidecar was written. "
              "Section 6 needs the character count for BPC.")

    if args.remove_parts:
        if not args.verify:
            raise SystemExit("--remove_parts needs the verification it was told "
                             "to skip; drop --no_verify.")
        for q in parts:
            os.remove(q)
            if os.path.exists(meta_path(q)):
                os.remove(meta_path(q))
        print(f"  removed {len(parts)} parts and their sidecars")

    print(f"\nWrote {output}")
    print(f"Point training at it with --train_data {os.path.basename(output)}")
    return output


if __name__ == "__main__":
    main()
