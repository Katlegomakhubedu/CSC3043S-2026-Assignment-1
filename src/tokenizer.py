import os
import re
import regex
import multiprocessing as mp
from collections import Counter, defaultdict
from typing import Iterator
import pickle

# GPT-2 pre-tokenizer regex (Appendix A). Compiled once at import: the corpus
# is streamed past it billions of characters at a time, so per-call compilation
# lookups are pure overhead.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT = regex.compile(PAT)

# Shared single-byte objects. `bytes([b])` allocates a fresh object on every
# call; the corpus has billions of byte positions, so pre-token tuples are built
# out of these 256 interned objects instead.
_BYTE = [bytes([i]) for i in range(256)]


def _to_byte_tuple(text: str) -> tuple[bytes, ...]:
    """Represent one pre-token as a tuple of single-byte objects."""
    return tuple(_BYTE[b] for b in text.encode("utf-8"))


def get_pretokens(documents) -> Iterator[tuple[bytes, ...]]:
    """Yield one byte-tuple per pre-token *occurrence*, using the GPT-2 regex.

    This is a generator, not a list. The full training corpus contains billions
    of pre-token occurrences but only ~10^4-10^5 distinct ones, so materialising
    the occurrence list (as this previously did) cost tens of GB of RAM to
    produce a frequency table that fits in a few MB. Feed this straight into
    `count_pretokens`, or prefer `count_pretokens_from_documents`, which avoids
    building a tuple per occurrence at all.
    """
    for doc in documents:
        # Skip empty documents
        if not doc:
            continue
        for match in _PAT.finditer(doc):
            yield _to_byte_tuple(match.group())


def count_pretokens_from_documents(documents) -> dict[tuple[bytes, ...], int]:
    """Pre-token frequency table for a stream of documents.

    Counts on the matched *strings* first - which are cheap to hash and are
    shared by every repeat occurrence - and only converts the distinct keys to
    byte tuples at the end. On a 20MB slice that is ~4.8M occurrences collapsing
    to ~13k distinct pre-tokens, so this does ~370x less tuple construction than
    counting `get_pretokens` occurrence-by-occurrence, for an identical result.
    """
    str_counts = Counter()
    for doc in documents:
        if not doc:
            continue
        str_counts.update(m.group() for m in _PAT.finditer(doc))
    return {_to_byte_tuple(s): c for s, c in str_counts.items()}


def iter_document_pieces(input_path: str, special_tokens: list[str],
                         chunk_size: int = 1 << 22,
                         flush_threshold: int | None = None) -> Iterator[tuple[str, bool]]:
    """Stream `input_path`, yielding `(text, ends_document)` with special tokens
    removed.

    Reads in chunks rather than pulling the whole file into memory (the training
    split is ~2.2GB, against ~12GB of Colab RAM that the merge bookkeeping also
    needs). Special tokens are hard boundaries (§3.1), so chunks are normally
    only cut at a delimiter match: a delimiter straddling a read boundary stays
    in the buffer until the rest of it arrives.

    `ends_document` is False only for the memory safety valve below, which cuts
    an over-long delimiter-free stretch at a whitespace boundary. Such a piece
    continues in the next yield. Callers that merely count pre-tokens can ignore
    the flag (a whitespace cut splits no pre-token); callers that insert a
    document delimiter must not insert one after a piece with `ends_document`
    False, or they would plant an <|endoftext|> in the middle of a document.
    """
    if special_tokens:
        # Longest-first, so a special token that is a prefix of another can't
        # shadow it.
        pattern = re.compile("|".join(
            re.escape(s) for s in sorted(special_tokens, key=len, reverse=True)))
        max_special = max(len(s) for s in special_tokens)
    else:
        pattern = None
        max_special = 0

    # If a file goes a long way with no delimiter, the buffer would grow without
    # bound. Past this point flush it at a whitespace boundary instead, which
    # the pre-tokenizer regex never splits across. Exposed as a parameter purely
    # so tests can drive this path on a small file.
    if flush_threshold is None:
        flush_threshold = max(chunk_size * 4, 1 << 20)

    # newline="": read the file's actual characters, never Python's
    # universal-newline translation of them (see read_txt).
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        buffer = ""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk

            if pattern is not None:
                last = 0
                for m in pattern.finditer(buffer):
                    # A delimiter follows, so this piece really is a whole
                    # document (an empty one, between adjacent delimiters, is
                    # dropped rather than yielded).
                    if m.start() > last:
                        yield buffer[last:m.start()], True
                    last = m.end()
                if last:
                    buffer = buffer[last:]

            if len(buffer) > flush_threshold:
                keep = max(max_special, 1)
                cut = buffer.rfind(" ", 0, len(buffer) - keep)
                if cut <= 0:
                    cut = len(buffer) - keep
                if cut > 0:
                    # Safety valve, not a document boundary.
                    yield buffer[:cut], False
                    buffer = buffer[cut:]

        # Trailing text after the final delimiter: EOF ends the document.
        if buffer:
            yield buffer, True


def iter_documents(input_path: str, special_tokens: list[str],
                   chunk_size: int = 1 << 22) -> Iterator[str]:
    """Stream `input_path`, yielding document text with special tokens removed.

    Consecutive pieces of one very long document may arrive as separate yields;
    use `iter_document_pieces` when that distinction matters.
    """
    for text, _ in iter_document_pieces(input_path, special_tokens, chunk_size):
        yield text

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
    # newline="" disables universal-newline translation. Without it Python's text
    # mode silently rewrites CRLF to LF on Windows, which would (a) make the
    # serial reader disagree with the binary-reading parallel one, (b) make the
    # learned merges platform-dependent, and (c) break the §3.1 guarantee that
    # decode(encode(s)) == s, since the decoded text could never reproduce the
    # CRLF the file actually contains.
    with open(input_path, "r", encoding="utf-8", newline="") as file:
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
    table. Each file is streamed a chunk at a time (`iter_documents`) and counted
    on the fly, so peak memory is set by the number of *distinct* pre-tokens
    (~10^5, a few MB) rather than by the corpus size - the whole point of §3.2's
    "well under an hour, within ~12GB" target.

    Files are counted one at a time and their counts merged, rather than
    concatenating the raw text first - e.g. TinyStories' train split ships as two
    ~1.1GB part files that together form one corpus. (A pre-token straddling
    exactly the boundary between two files would be mis-split into two
    pre-tokens; with millions of documents in the corpus and at most
    `len(input_paths) - 1` such boundaries, the effect on merge statistics is
    negligible.)
    """
    word_freq = Counter()
    for path in input_paths:
        for pretoken, count in count_pretokens_from_documents(
                iter_documents(path, special_tokens)).items():
            word_freq[pretoken] += count
    return dict(word_freq)


# ---------------------------------------------------------------------------
# Parallel pre-tokenization (§3.2, "recommended")
#
# Once merging is incremental, pre-tokenization is the bottleneck, and it is
# embarrassingly parallel: the corpus is split into byte ranges aligned to
# <|endoftext|> boundaries, each worker counts pre-token strings in its own
# range, and the parent merges the counters. Workers return Counters keyed by
# *strings* (~10^5 entries, a few MB) rather than byte tuples, which keeps the
# inter-process pickling cheap.
# ---------------------------------------------------------------------------

def find_chunk_boundaries(input_path: str, num_chunks: int,
                          delimiter: bytes = b"<|endoftext|>") -> list[int]:
    """Byte offsets splitting `input_path` into at most `num_chunks` ranges.

    Every boundary lands immediately *after* a delimiter, so no document - and
    therefore no pre-token - straddles two ranges, and each range decodes as
    valid UTF-8 on its own (the delimiter is ASCII).
    """
    size = os.path.getsize(input_path)
    if num_chunks <= 1 or size == 0:
        return [0, size]

    boundaries = [i * size // num_chunks for i in range(num_chunks + 1)]
    boundaries[0], boundaries[-1] = 0, size
    window_size = 1 << 20

    with open(input_path, "rb") as f:
        for i in range(1, len(boundaries) - 1):
            position = boundaries[i]
            while True:
                f.seek(position)
                window = f.read(window_size)
                if not window:
                    boundaries[i] = size
                    break
                found = window.find(delimiter)
                if found != -1:
                    boundaries[i] = position + found + len(delimiter)
                    break
                # Step forward, overlapping by len(delimiter)-1 bytes so a
                # delimiter spanning two windows is still found.
                position += max(len(window) - len(delimiter) + 1, 1)

    # Ranges can collapse to empty if delimiters are sparse; drop duplicates.
    return sorted(set(boundaries))


def _count_pretoken_strings_in_range(task):
    """Worker: count pre-token strings in one byte range. Module-level and
    picklable so it survives the `spawn` start method on Windows/macOS."""
    input_path, start, end, special_tokens = task
    with open(input_path, "rb") as f:
        f.seek(start)
        raw = f.read(end - start)

    # Boundaries are delimiter-aligned, so this is a whole number of documents.
    text = raw.decode("utf-8", errors="replace")
    counts = Counter()
    for doc in split_text(text, special_tokens):
        if doc:
            counts.update(m.group() for m in _PAT.finditer(doc))
    return counts


def get_word_freq_parallel(input_paths: list[str], special_tokens: list[str],
                           workers: int, tasks_per_worker: int = 4) -> dict[tuple[bytes, ...], int]:
    """Parallel equivalent of `get_word_freq_from_files`.

    Must be called from under an `if __name__ == "__main__":` guard, because the
    default start method on Windows re-imports the calling module in each worker.
    """
    tasks = []
    for path in input_paths:
        boundaries = find_chunk_boundaries(path, workers * tasks_per_worker)
        tasks.extend((path, start, end, special_tokens)
                     for start, end in zip(boundaries[:-1], boundaries[1:])
                     if end > start)

    str_counts = Counter()
    with mp.Pool(processes=workers) as pool:
        for counts in pool.imap_unordered(_count_pretoken_strings_in_range, tasks):
            str_counts.update(counts)

    return {_to_byte_tuple(s): c for s, c in str_counts.items()}


def train_bpe_naive(word_freq: dict[tuple[bytes, ...], int], vocab: dict[int, bytes],
                    num_merges: int) -> list[tuple[bytes, bytes]]:
    """Tutorial-1 style trainer: recount every pair from scratch after every merge.

    This is the O(num_merges * corpus_size) implementation that §3.2 asks you to
    replace. It is kept as the reference the optimised trainer is checked
    against, and as the "before" side of the Q2 speed comparison, so that both
    uses share one definition instead of drifting apart.
    """
    merges = []
    wf = dict(word_freq)
    for _ in range(num_merges):
        pair_counts = Counter()
        for word, freq in wf.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += freq
        if not pair_counts:
            break
        best_count = max(pair_counts.values())
        best_pair = max(p for p, c in pair_counts.items() if c == best_count)

        new_wf = {}
        for word, freq in wf.items():
            merged = merge_pair(word, best_pair)
            new_wf[merged] = new_wf.get(merged, 0) + freq
        wf = new_wf

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
    return merges


def derive_vocab_and_merges(merges: list[tuple[bytes, bytes]], vocab_size: int,
                            special_tokens: list[str]):
    """Build the (vocab, merges) pair for `vocab_size` from a longer merge list.

    BPE is greedy and order-dependent only on what came before, so the first k
    merges learned for a large target vocabulary *are* the merges a run with the
    smaller target would have learned. Training once to the largest size needed
    and truncating therefore gives exactly the same tokenizers as separate runs,
    for one pre-tokenization pass instead of several - which is what makes the
    §3.4 vocabulary study and the §3.5 second tokenizer nearly free.
    """
    vocab = init_vocab(special_tokens)
    num_merges = vocab_size - len(vocab)
    if num_merges < 0:
        raise ValueError(
            f"vocab_size={vocab_size} is smaller than the {len(vocab)} base tokens")
    if num_merges > len(merges):
        raise ValueError(
            f"vocab_size={vocab_size} needs {num_merges} merges but only "
            f"{len(merges)} were learned")

    kept = list(merges[:num_merges])
    for pair in kept:
        vocab[len(vocab)] = pair[0] + pair[1]
    return vocab, kept


def train_bpe(input_path: str | list[str], vocab_size: int, special_tokens: list[str],
              workers: int = 1) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Returns (vocab: id -> token bytes, merges: ordered list of (left, right) byte pairs).

    `input_path` may be a single path or a list of paths making up one corpus.
    `workers > 1` pre-tokenizes in parallel, and must be called from under an
    `if __name__ == "__main__":` guard.
    """
    input_paths = [input_path] if isinstance(input_path, str) else list(input_path)
    if workers and workers > 1:
        word_freq = get_word_freq_parallel(input_paths, special_tokens, workers)
    else:
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
        
        # Same compiled pre-tokenizer regex the trainer uses, so encoding can
        # never drift from the pre-tokenization the merges were learned over.
        self.pat = _PAT

    

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
                        token_bytes = _to_byte_tuple(token_text)
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
