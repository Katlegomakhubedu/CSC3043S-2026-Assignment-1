"""Encode a text corpus to a uint16 .npy token array (§3.5).

Also writes a JSON sidecar next to the array with the document, character and
byte counts of the encoded text. §6 needs the raw character count to turn a
cross-entropy loss into bits-per-character, so recording it here keeps BPC from
ever re-reading or re-tokenizing the corpus.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import argparse
from array import array

import numpy as np

from src.tokenizer import BPETokenizer, iter_document_pieces

END_OF_TEXT = "<|endoftext|>"


def encode_corpus(input_file, output_file, tokenizer, add_eot=True, report_every=200_000,
                  flush_threshold=None):
    """Encode `input_file` into a uint16 .npy token array at `output_file`.

    Documents are separated by the <|endoftext|> ID (§2), which the model needs
    in the training data to ever learn to emit the token `generate()` stops on.

    One pass: IDs accumulate in an `array('H')`, which grows in amortised O(1)
    and is already the uint16 layout `np.save` wants, so nothing is copied. It
    also raises OverflowError on any ID >= 65536, enforcing §3.3's uint16
    requirement where violating it would silently corrupt the training data.
    """
    eot_id = None
    if add_eot:
        eot_id = tokenizer.token_to_id[END_OF_TEXT.encode("utf-8")]

    ids = array("H")
    if ids.itemsize != 2:
        raise RuntimeError(f"array('H') is {ids.itemsize} bytes, expected 2 (uint16)")

    n_docs = 0
    n_chars = 0
    n_bytes = 0
    start = time.time()

    piece_kwargs = {} if flush_threshold is None else {"flush_threshold": flush_threshold}
    for text, ends_document in iter_document_pieces(
            input_file, tokenizer.special_tokens, **piece_kwargs):
        n_chars += len(text)
        n_bytes += len(text.encode("utf-8"))
        ids.extend(tokenizer.encode(text))

        if ends_document:
            # Only a real boundary gets a delimiter. ends_document=False means
            # the reader's safety valve cut mid-document, and a delimiter there
            # would plant a boundary inside a story.
            n_docs += 1
            if add_eot:
                ids.append(eot_id)

            if report_every and n_docs % report_every == 0:
                elapsed = time.time() - start
                print(f"  {n_docs:,} docs | {len(ids):,} tokens | "
                      f"{n_chars / max(elapsed, 1e-9) / 1e6:.1f} MChar/s | {elapsed / 60:.1f} min",
                      flush=True)

    elapsed = time.time() - start
    arr = np.frombuffer(ids, dtype=np.uint16)
    np.save(output_file, arr)

    meta = {
        "input_file": os.path.basename(input_file),
        "output_file": os.path.basename(output_file),
        "vocab_size": len(tokenizer.vocab),
        "n_tokens": int(arr.size),
        "n_documents": n_docs,
        # §6 requires stating whether <|endoftext|> is counted, so record both
        # conventions and let the report cite the one it uses.
        "n_chars_excluding_delimiters": n_chars,
        "n_chars_including_delimiters": n_chars + n_docs * len(END_OF_TEXT),
        "n_bytes_utf8_excluding_delimiters": n_bytes,
        "eot_id": eot_id,
        "eot_count": int((arr == eot_id).sum()) if add_eot else 0,
        "bytes_per_token": n_bytes / arr.size if arr.size else None,
        "chars_per_token": n_chars / arr.size if arr.size else None,
        "encode_seconds": round(elapsed, 1),
    }
    meta_path = os.path.splitext(output_file)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {output_file}  ({arr.size:,} tokens, {n_docs:,} docs, "
          f"{elapsed / 60:.1f} min)")
    print(f"Saved {meta_path}")

    if add_eot and meta["eot_count"] != n_docs:
        raise RuntimeError(
            f"expected {n_docs} <|endoftext|> tokens, found {meta['eot_count']}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Encode a text corpus into a uint16 .npy array.")
    parser.add_argument("--input", required=True, help="Path to the input .txt file")
    parser.add_argument("--output", required=True, help="Path to the output .npy file")
    parser.add_argument("--vocab", required=True, help="Path to the vocab pickle file")
    parser.add_argument("--merges", required=True, help="Path to the merges pickle file")
    parser.add_argument("--special", nargs="+", default=[END_OF_TEXT], help="Special tokens")
    parser.add_argument("--no_eot", dest="add_eot", action="store_false",
                        help="Do not emit <|endoftext|> between documents (not recommended: "
                             "§2 requires documents to be delimited).")
    args = parser.parse_args()

    tokenizer = BPETokenizer.from_files(args.vocab, args.merges, args.special)
    encode_corpus(args.input, args.output, tokenizer, add_eot=args.add_eot)

if __name__ == "__main__":
    main()