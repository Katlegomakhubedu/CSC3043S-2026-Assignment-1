import numpy as np
import re
import argparse
from src.tokenizer import BPETokenizer

def document_generator(file_path, special_tokens, chunk_size=1 << 20):
    """
    Yield documents (text chunks) separated by any special token.
    Special tokens themselves are stripped and not returned.
    """
    # Build regex that splits on any special token, keeping delimiters
    pattern = "(" + "|".join(re.escape(st) for st in special_tokens) + ")"
    buffer = ""

    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                # Flush remaining buffer
                if buffer:
                    parts = re.split(pattern, buffer)
                    for part in parts:
                        if part and part not in special_tokens:
                            yield part
                break

            buffer += chunk
            parts = re.split(pattern, buffer)
            # Yield all complete parts; keep the last (possibly incomplete) part in buffer
            for part in parts[:-1]:
                if part and part not in special_tokens:
                    yield part
            buffer = parts[-1]

def encode_corpus(input_file, output_file, tokenizer):
    """
    Encode a text file with the given tokenizer and save as uint16 .npy.
    """
    # 1. Count total tokens (first pass)
    total_tokens = 0
    for doc in document_generator(input_file, tokenizer.special_tokens):
        total_tokens += len(tokenizer.encode(doc))
    print(f"[{input_file}] Total tokens: {total_tokens}")

    # 2. Allocate and fill the array (second pass)
    arr = np.zeros(total_tokens, dtype=np.uint16)
    idx = 0
    for doc in document_generator(input_file, tokenizer.special_tokens):
        ids = tokenizer.encode(doc)
        n = len(ids)
        arr[idx:idx+n] = ids
        idx += n

    np.save(output_file, arr)
    print(f"Saved {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Encode a text corpus into a uint16 .npy array.")
    parser.add_argument("--input", required=True, help="Path to the input .txt file")
    parser.add_argument("--output", required=True, help="Path to the output .npy file")
    parser.add_argument("--vocab", required=True, help="Path to the vocab pickle file")
    parser.add_argument("--merges", required=True, help="Path to the merges pickle file")
    parser.add_argument("--special", nargs="+", default=["<|endoftext|>"], help="Special tokens")
    args = parser.parse_args()

    tokenizer = BPETokenizer.from_files(args.vocab, args.merges, args.special)
    encode_corpus(args.input, args.output, tokenizer)

if __name__ == "__main__":
    main()