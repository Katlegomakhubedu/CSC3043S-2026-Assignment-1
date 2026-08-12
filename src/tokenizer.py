import regex
from collections import Counter

def train_bpe(input_path: str, vocab_size: int,
              special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Returns (vocab: id -> token bytes, merges: ordered list of (left, right) byte pairs)."""

    with open(input_path, encoding = "utf-8") as file:
        raw_text = file.read()
    
    documents = raw_text.split(special_tokens[0])  # Split on the first special token
    pretokens = get_pretokens(documents)
    freq = count_pretokens(pretokens)
    pairs = count_pairs(freq)
    


    
def get_pretokens(documents: list[str]) -> list[bytes]:
    """Tokenize documents into a list of tokens."""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pretokens = []

    for doc in documents:
        for match in regex.finditer(PAT, doc):
            token = match.group()
            pretokens.append(token.encode("utf-8"))

    return pretokens

def count_pretokens(tokens: list[bytes]) -> dict[tuple[bytes, bytes], int]:
    """Count frequency of adjacent byte pairs in the list of tokens."""
    counter = Counter()
    counter.update(tokens)
    return counter

def count_pairs(tokens: list[bytes]) -> dict[tuple[bytes, bytes], int]:
    """Count frequency of adjacent byte pairs in the list of tokens."""
    counter = Counter()
    for token in tokens:
        for i in range(len(token) - 1):
            pair = (token[i:i+1], token[i+1:i+2])
            counter[pair] += 1
    return counter

def merge_pair(tokens: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
    """Merge the most frequent pair in the list of tokens."""
    merged_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if i < len(tokens) - 1 and (token, tokens[i + 1]) == pair:
            merged_tokens.append(token + tokens[i + 1])
            i += 2
        else:
            merged_tokens.append(token)
            i += 1
    return merged_tokens

train_bpe("./example.txt", vocab_size=10000, special_tokens=["<|END_OF_TEXT|>", "</s>"])

class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
    @classmethod    
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        """Load vocab and merges from files."""
        with open(vocab_path, "rb") as f:
            vocab = {i: line.strip() for i, line in enumerate(f)}
        with open(merges_path, "rb") as f:
            merges = [tuple(line.strip().split(b" ")) for line in f]
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        tokens = self._tokenize(text)
        return [self.vocab[token] for token in tokens if token in self.vocab]
    
    def encode_iterable(self, iterable) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        tokens = [self.vocab[i] for i in ids if i in self.vocab]
        return self._detokenize(tokens)
