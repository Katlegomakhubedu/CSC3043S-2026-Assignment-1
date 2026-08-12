import re
import regex
from collections import Counter
from typing import Iterator

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
    
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Returns (vocab: id -> token bytes, merges: ordered list of (left, right) byte pairs)."""
    raw_text = read_txt(input_path)
    documents = split_text(raw_text, special_tokens)

    # 2. Get initial frequencies
    pretokens = get_pretokens(documents)
    word_freq = count_pretokens(pretokens)

    # 3. Initialize Byte-Level Vocabulary (Special Tokens + 256 byte values)
    vocab = {}
    for i, st in enumerate(special_tokens):
        vocab[i] = st.encode("utf-8")
        
    offset = len(special_tokens)
    for b in range(256):
        vocab[offset + b] = bytes([b])

    # 4. Iteratively learn merges
    merges = []
    num_merges = vocab_size - len(vocab)
    
    for _ in range(num_merges):
        pair_counts = count_pairs(word_freq)
        if not pair_counts:
            break
            
        # Tie-breaking: take the lexicographically greatest pair if counts are tied
        best_count = max(pair_counts.values())
        best_pair = max(pair for pair, count in pair_counts.items() if count == best_count)

        # Apply merge across all unique words
        word_freq = {merge_pair(word, best_pair): freq for word, freq in word_freq.items()}
        
        # Record the merge
        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

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
        # The assignment does not mandate a specific serialization format, 
        # so this can be implemented using json or pickle depending on how you save.
        raise NotImplementedError("Implement file loading based on your chosen save format.")
        
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
    input_path = r"C:\Users\katle\OneDrive - University of Cape Town\Final Year\CS3043S\samples.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    raw_text = read_txt(input_path)
    documents = split_text(raw_text, special_tokens)
    pretokens = get_pretokens(documents)
    freq = count_pretokens(pretokens)

    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    val_path = r"C:\Users\katle\OneDrive - University of Cape Town\Final Year\CS3043S\data\TinyStoriesV2-GPT4-valid.txt"
    val_files = read_txt(val_path)
    
    encoded = tokenizer.encode(val_files)
    decoded = tokenizer.decode(encoded)
    

    print(f"Encoded IDs: {encoded}")
    print(f"Decoded: {decoded}")


    
