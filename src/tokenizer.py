import re
import regex
from collections import Counter, defaultdict
from typing import Iterator
import pickle

def get_pretokens(documents: list[str]) -> list[tuple[bytes, ...]]:
    """Tokenize documents into a list of byte-tuples using the GPT-2 regex."""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens = []

    for doc in documents:
        # Skip empty documents
        if not doc:
            continue
        for match in regex.finditer(PAT, doc):
            # Encode match to utf-8, then split into a tuple of individual bytes
            token_bytes = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            pretokens.append(token_bytes)

    return pretokens

def count_pretokens(tokens: list[tuple[bytes, ...]]) -> dict[tuple[bytes, ...], int]:
    """Count frequency of byte-level pretokens."""
    counter = Counter(tokens)
    return dict(counter)

def count_pairs(word_freqs: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """Count frequency of adjacent byte pairs across the unique vocabulary."""
    counter = Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            counter[pair] += freq
    return counter

def merge_pair(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Merge the most frequent pair in a single pretoken."""
    merged_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            merged_word.append(word[i] + word[i + 1])
            i += 2
        else:
            merged_word.append(word[i])
            i += 1
    return tuple(merged_word)

def read_txt(input_path: str) -> str:
    with open(input_path, "r", encoding="utf-8") as file:
        return file.read()

def split_text(raw_text: str, special_tokens: list[str]) -> list[str]:
    # Split on special tokens to ensure they are hard boundaries
    if special_tokens:
        split_pattern = "(" + "|".join(re.escape(s) for s in special_tokens) + ")"
         # Filter out the special tokens themselves so they aren't pre-tokenized
        documents = [doc for doc in re.split(split_pattern, raw_text) if doc not in special_tokens]
    else:
        documents = [raw_text]

    return documents
    
def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """Byte-level vocabulary: special tokens first, then all 256 byte values."""
    vocab = {}
    for i, st in enumerate(special_tokens):
        vocab[i] = st.encode("utf-8")
    offset = len(special_tokens)
    for b in range(256):
        vocab[offset + b] = bytes([b])
    return vocab


def _build_pair_index(words: list[tuple[bytes, ...]], freqs: list[int]):
    """
    Build the incremental-merge bookkeeping used by `train_bpe_incremental`:
        pair_counts:   pair -> total (freq-weighted) occurrence count across the corpus
        pair_to_words: pair -> set of indices into `words` where the pair currently occurs
    """
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for idx, word in enumerate(words):
        freq = freqs[idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(idx)
    return pair_counts, pair_to_words


def train_bpe_incremental(word_freq: dict[tuple[bytes, ...], int], vocab: dict[int, bytes],
                           num_merges: int, track_token_counts: bool = False):
    """
    Learn up to `num_merges` BPE merges, updating pair counts incrementally rather
    than recounting the whole corpus after every merge (required by §3.2).

    After each merge, only the pre-tokens that actually contained the merged pair
    (tracked via `pair_to_words`) have their pair-count contributions removed and
    re-added - every other pre-token is untouched. This is what makes training
    scale to the full corpus instead of being O(merges * corpus_size).

    Mutates `vocab` in place (adds one entry per merge) and returns
    (merges, token_counts). `token_counts` is None unless track_token_counts=True,
    in which case it's a list of the corpus's total token count after each merge
    (starting with the pre-merge count at index 0) - free to compute because every
    merge reduces the token count by exactly the merged pair's occurrence count.
    """
    words = list(word_freq.keys())
    freqs = list(word_freq.values())
    pair_counts, pair_to_words = _build_pair_index(words, freqs)

    merges = []
    token_counts = None
    if track_token_counts:
        total_tokens = sum(freqs[i] * len(words[i]) for i in range(len(words)))
        token_counts = [total_tokens]

    for _ in range(num_merges):
        if not pair_counts:
            break

        # Tie-breaking: take the lexicographically greatest pair if counts are tied.
        best_count = max(pair_counts.values())
        best_pair = max(p for p, c in pair_counts.items() if c == best_count)

        # Only revisit pre-tokens that actually contain best_pair.
        affected = list(pair_to_words.get(best_pair, ()))
        for idx in affected:
            word = words[idx]
            freq = freqs[idx]

            # Remove this word's old pair contributions.
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                bucket = pair_to_words.get(p)
                if bucket is not None:
                    bucket.discard(idx)
                    if not bucket:
                        del pair_to_words[p]

            new_word = merge_pair(word, best_pair)
            words[idx] = new_word

            # Add the merged word's new pair contributions.
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_counts[p] = pair_counts.get(p, 0) + freq
                pair_to_words[p].add(idx)

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        if track_token_counts:
            total_tokens -= best_count
            token_counts.append(total_tokens)

    return merges, token_counts


def get_word_freq_from_files(input_paths: list[str], special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    Pre-tokenize one or more files and return their combined pre-token frequency
    table. Files are processed (and special-token-split) one at a time and their
    counts merged, rather than concatenating the raw text first, so this doesn't
    need to hold more than one file's text in memory at once - e.g. TinyStories'
    train split ships as two ~1.1GB part files that together form one corpus.
    (A pre-token straddling exactly the boundary between two files would be
    mis-split into two pre-tokens; with millions of documents in the corpus and
    at most `len(input_paths) - 1` such boundaries, the effect on merge
    statistics is negligible.)
    """
    word_freq = Counter()
    for path in input_paths:
        raw_text = read_txt(path)
        documents = split_text(raw_text, special_tokens)
        pretokens = get_pretokens(documents)
        for pretoken, count in count_pretokens(pretokens).items():
            word_freq[pretoken] += count
    return dict(word_freq)


def train_bpe(input_path: str | list[str], vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Returns (vocab: id -> token bytes, merges: ordered list of (left, right) byte pairs).
    `input_path` may be a single path or a list of paths making up one corpus."""
    input_paths = [input_path] if isinstance(input_path, str) else list(input_path)
    word_freq = get_word_freq_from_files(input_paths, special_tokens)

    vocab = init_vocab(special_tokens)
    num_merges = vocab_size - len(vocab)
    merges, _ = train_bpe_incremental(word_freq, vocab, num_merges)

    return vocab, merges
    
class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = list(special_tokens) if special_tokens else []
        
        # Invert the vocabulary to map bytes back to IDs
        self.token_to_id = {v: k for k, v in vocab.items()}
        
        # Store ranks to apply the earliest-learned merges first
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self._cache = {}
        
        # We need the pre-tokenizer regex for encoding new strings
        self.pat = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_path, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
        
    def _apply_merges(self, word_bytes: tuple[bytes, ...]) -> tuple[bytes, ...]:
        """Greedily applies merges to a sequence of bytes based on learned merge ranks."""
        word = list(word_bytes)
        while len(word) > 1:
            best_rank = float('inf')
            best_idx = -1
            
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i
                    
            if best_idx == -1:
                break
                
            merged_token = word[best_idx] + word[best_idx+1]
            word = word[:best_idx] + [merged_token] + word[best_idx+2:]
            
        return tuple(word)

    def encode(self, text: str) -> list[int]:
        ids = []
        # Keep the special tokens as delimiters so we can emit their IDs directly.
        if self.special_tokens:
            split_pattern = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            chunks = re.split(split_pattern, text)
        else:
            chunks = [text]
            
        for chunk in chunks:
            if not chunk:
                continue
                
            if chunk in self.special_tokens:
                # Emit special token IDs directly by converting them to their byte representation
                ids.append(self.token_to_id[chunk.encode("utf-8")])
            else:
                for match in self.pat.finditer(chunk):
                    token_text = match.group()
                    
                    if token_text not in self._cache:
                        # Convert to individual bytes
                        token_bytes = tuple(bytes([b]) for b in token_text.encode("utf-8"))
                        # Apply merges
                        merged_bytes = self._apply_merges(token_bytes)
                        # Store in cache
                        self._cache[token_text] = [self.token_to_id[b] for b in merged_bytes]
                        
                    ids.extend(self._cache[token_text])
        return ids
    
    def encode_iterable(self, iterable) -> Iterator[int]:
        """Memory-efficient streaming encoding."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of integer token IDs back into a string."""
        byte_sequence = b"".join(self.vocab[i] for i in ids)
        # Decode to string using errors="replace" to safely handle partial multi-byte characters
        return byte_sequence.decode("utf-8", errors="replace")


if __name__ == "__main__":
    # Smoke test: train a small tokenizer on samples.txt and round-trip the
    # validation file through it. Paths are relative to the repo root (the
    # parent of this file's directory) so this runs on any checkout.
    import argparse
    import os

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Quick BPE tokenizer smoke test.")
    parser.add_argument("--input", default=os.path.join(repo_root, "samples.txt"))
    parser.add_argument("--valid", default=os.path.join(repo_root, "data", "TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--vocab_size", type=int, default=8000)
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(args.input, args.vocab_size, special_tokens)
    print(f"Trained {len(merges)} merges, vocab size {len(vocab)}")

    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    val_text = read_txt(args.valid)
    encoded = tokenizer.encode(val_text)
    decoded = tokenizer.decode(encoded)
    print(f"Encoded {len(val_text)} chars into {len(encoded)} tokens; round-trip match: {decoded == val_text}")
