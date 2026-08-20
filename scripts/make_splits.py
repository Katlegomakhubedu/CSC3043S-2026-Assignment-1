import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import json

from src.tokenizer import BPETokenizer
from scripts.encode_corpus import encode_corpus

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
END_OF_TEXT = "<|endoftext|>"


def split_documents(text, n_test_docs):
    """Split on the document delimiter, reserving the last `n_test_docs`.

    Trailing empty pieces are dropped: a file ending in a delimiter produces one
    after the final split, and counting it as a document would shift the
    boundary by one and leave an empty "document" in the test set.
    """
    docs = [d for d in text.split(END_OF_TEXT) if d.strip()]
    if len(docs) <= n_test_docs:
        raise SystemExit(
            f"the corpus has {len(docs)} documents, which is not more than the "
            f"{n_test_docs} being reserved for test - nothing would be left to "
            f"validate on")
    return docs[:-n_test_docs], docs[-n_test_docs:]


def write_split(docs, path):
    """Write documents back out with the delimiter between them.

    Each document ends with a delimiter, matching how the corpus itself is laid
    out, so re-encoding a split reproduces the same document boundaries the
    full file would have given.

    newline="" for the same reason the reader in tokenizer.py uses it: without
    it Windows rewrites every \\n as \\r\\n on the way out, which inflates the
    character count §6 divides by and changes how the text tokenizes. The split
    has to be byte-identical to the slice of the corpus it came from.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        for doc in docs:
            f.write(doc)
            f.write(END_OF_TEXT)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description="Section 2's validation/test split.")
    p.add_argument("--input", default=os.path.join(REPO_ROOT, "data",
                                                   "TinyStoriesV2-GPT4-valid.txt"))
    p.add_argument("--vocab", default=os.path.join(REPO_ROOT, "vocab.pkl"))
    p.add_argument("--merges", default=os.path.join(REPO_ROOT, "merges.pkl"))
    p.add_argument("--test_docs", type=int, default=2000,
                   help="Documents reserved for test (section 2 says 2,000).")
    p.add_argument("--suffix", default="",
                   help="Name suffix matching Task 1's convention, e.g. "
                        "'_vocab1000' for the second tokenizer.")
    p.add_argument("--out_dir", default=REPO_ROOT)
    p.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "data"))
    args = p.parse_args(argv)

    for path in (args.input, args.vocab, args.merges):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")

    print(f"Reading {args.input}")
    with open(args.input, encoding="utf-8") as f:
        text = f.read()

    val_docs, test_docs = split_documents(text, args.test_docs)
    print(f"  {len(val_docs):,} validation documents, "
          f"{len(test_docs):,} test documents")

    os.makedirs(args.data_dir, exist_ok=True)
    val_txt = write_split(val_docs, os.path.join(args.data_dir, "valid_split.txt"))
    test_txt = write_split(test_docs, os.path.join(args.data_dir, "test_split.txt"))

    tokenizer = BPETokenizer.from_files(args.vocab, args.merges, [END_OF_TEXT])

    metas = {}
    for name, source in (("valid_split", val_txt), ("test_split", test_txt)):
        out = os.path.join(args.out_dir, f"{name}{args.suffix}.npy")
        print(f"\nEncoding {os.path.basename(source)} -> {os.path.basename(out)}")
        metas[name] = encode_corpus(source, out, tokenizer, report_every=0)

    summary = {
        "source": os.path.basename(args.input),
        "test_docs_reserved": args.test_docs,
        "vocab_size": len(tokenizer.vocab),
        "splits": {k: {"tokens": v["n_tokens"], "documents": v["n_documents"],
                       "chars_excluding_delimiters": v["n_chars_excluding_delimiters"]}
                   for k, v in metas.items()},
    }
    summary_path = os.path.join(args.out_dir, "logs",
                                f"splits{args.suffix}.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    for name, meta in metas.items():
        print(f"  {name:<12} {meta['n_documents']:>7,} docs  "
              f"{meta['n_tokens']:>10,} tokens  "
              f"{meta['n_chars_excluding_delimiters']:>12,} chars")
    print("=" * 70)
    print(f"Wrote {summary_path}")
    print(f"\nTraining and model selection use valid_split{args.suffix}.npy from here on.")
    print(f"test_split{args.suffix}.npy is for Question 18 only - section 2 says once.")
    return summary


if __name__ == "__main__":
    main()
