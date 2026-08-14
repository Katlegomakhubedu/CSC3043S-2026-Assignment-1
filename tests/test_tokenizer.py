"""
Sanity checks for the BPE tokenizer, per assignment §3.3:
 - round-trip on real documents
 - a document containing <|endoftext|> encodes to exactly one occurrence of that ID
 - every ID fits in uint16
 - merges are identical across two runs on the same input
 - (§3.2) the incremental trainer produces byte-identical merges to a naive,
   full-recount-every-merge reference implementation
"""
import os
import random
from collections import Counter

import pytest

from conftest import SAMPLES_FILE, VALID_FILE
from tokenizer import (
    BPETokenizer, train_bpe, train_bpe_incremental, init_vocab,
    read_txt, split_text, get_pretokens, count_pretokens, merge_pair,
)

SPECIAL_TOKENS = ["<|endoftext|>"]


def naive_train_bpe(word_freq, vocab, num_merges):
    """Reference tutorial-style BPE trainer: recounts every pair from scratch
    after every merge. Used only to check the incremental trainer agrees with it."""
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


@pytest.fixture(scope="module")
def small_tokenizer():
    """A small tokenizer trained on the repo's samples.txt - fast and deterministic."""
    vocab, merges = train_bpe(SAMPLES_FILE, vocab_size=500, special_tokens=SPECIAL_TOKENS)
    return BPETokenizer(vocab, merges, SPECIAL_TOKENS)


@pytest.fixture(scope="module")
def valid_documents():
    """First ~100 documents from the validation file (only reads a small
    prefix of the 22MB file so this stays fast)."""
    if not os.path.exists(VALID_FILE):
        pytest.skip(f"validation file not found at {VALID_FILE}")
    with open(VALID_FILE, "r", encoding="utf-8") as f:
        chunk = f.read(400_000)
    docs = [d for d in chunk.split("<|endoftext|>") if d]
    return docs[:100]


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_round_trip_on_validation_documents(small_tokenizer, valid_documents):
    assert len(valid_documents) > 0
    for doc in valid_documents:
        assert small_tokenizer.decode(small_tokenizer.encode(doc)) == doc


def test_round_trip_synthetic_strings(small_tokenizer):
    cases = [
        "",
        "hello world",
        "  leading and trailing spaces  ",
        "line1\nline2\n\ttabbed",
        "Punctuation! Does it work? Yes; (probably) - 100%.",
        "unicode: café, 你好, \U0001F600",  # accented latin, CJK, emoji (surrogate pair in UTF-8)
        "repeated repeated repeated repeated",
        "a" * 500,
    ]
    for s in cases:
        assert small_tokenizer.decode(small_tokenizer.encode(s)) == s


def test_round_trip_of_truncated_multibyte_sequence_does_not_crash(small_tokenizer):
    # Decoding a prefix of a token sequence can cut a multi-byte UTF-8 character
    # in half. That must degrade gracefully (errors="replace"), not raise.
    ids = small_tokenizer.encode("café naïve \U0001F600 done")
    for k in range(1, len(ids)):
        out = small_tokenizer.decode(ids[:k])  # must not raise
        assert isinstance(out, str)


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

def test_endoftext_encodes_to_exactly_one_id(small_tokenizer):
    eot_id = small_tokenizer.token_to_id["<|endoftext|>".encode("utf-8")]
    text = "Once upon a time.<|endoftext|>Another story starts here."
    ids = small_tokenizer.encode(text)
    assert ids.count(eot_id) == 1


def test_special_token_never_split_by_merges(small_tokenizer):
    # The special token must come through as a single ID, never merged with
    # surrounding text or split into byte tokens.
    ids = small_tokenizer.encode("abc<|endoftext|>xyz")
    eot_id = small_tokenizer.token_to_id["<|endoftext|>".encode("utf-8")]
    assert eot_id in ids


# ---------------------------------------------------------------------------
# uint16 range
# ---------------------------------------------------------------------------

def test_ids_fit_in_uint16(small_tokenizer, valid_documents):
    for doc in valid_documents[:20]:
        for token_id in small_tokenizer.encode(doc):
            assert 0 <= token_id < 65536


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_merges_deterministic_across_runs():
    vocab_a, merges_a = train_bpe(SAMPLES_FILE, vocab_size=400, special_tokens=SPECIAL_TOKENS)
    vocab_b, merges_b = train_bpe(SAMPLES_FILE, vocab_size=400, special_tokens=SPECIAL_TOKENS)
    assert merges_a == merges_b
    assert vocab_a == vocab_b


# ---------------------------------------------------------------------------
# §3.2: incremental trainer must match the naive reference exactly
# ---------------------------------------------------------------------------

def test_incremental_matches_naive_reference():
    raw_text = read_txt(SAMPLES_FILE)
    documents = split_text(raw_text, SPECIAL_TOKENS)
    word_freq = count_pretokens(get_pretokens(documents))

    vocab_inc = init_vocab(SPECIAL_TOKENS)
    merges_inc, _ = train_bpe_incremental(word_freq, vocab_inc, num_merges=300)

    vocab_naive = init_vocab(SPECIAL_TOKENS)
    merges_naive = naive_train_bpe(word_freq, vocab_naive, num_merges=300)

    assert merges_inc == merges_naive
    assert vocab_inc == vocab_naive


def test_incremental_token_count_matches_recount():
    """token_counts[i] tracked incrementally (§3.4: every merge reduces the
    corpus token count by exactly the merged pair's count) must equal a fresh
    recount of the corpus after applying the same merges."""
    raw_text = read_txt(SAMPLES_FILE)
    documents = split_text(raw_text, SPECIAL_TOKENS)
    word_freq = count_pretokens(get_pretokens(documents))

    vocab = init_vocab(SPECIAL_TOKENS)
    merges, token_counts = train_bpe_incremental(word_freq, vocab, num_merges=50, track_token_counts=True)

    # Recount from scratch after applying the same merges, in order.
    wf = dict(word_freq)
    assert token_counts[0] == sum(f * len(w) for w, f in wf.items())
    for i, pair in enumerate(merges):
        new_wf = {}
        for word, freq in wf.items():
            merged = merge_pair(word, pair)
            new_wf[merged] = new_wf.get(merged, 0) + freq
        wf = new_wf
        recounted = sum(f * len(w) for w, f in wf.items())
        assert token_counts[i + 1] == recounted


# ---------------------------------------------------------------------------
# Tie-breaking
# ---------------------------------------------------------------------------

def test_tie_breaking_is_lexicographically_greatest_pair():
    # "aa aa bb bb" -> pretokens (with GPT-2 pre-tokenizer, leading space kept):
    # ('a','a'), (' ','a','a'), (' ','b','b'), (' ','b','b') each freq 1 (first "aa"
    # has no leading space). Pairs ('a','a') and ('b','b') both occur twice (bb bb
    # both preceded by space, each is its own pretoken with 1 internal pair, but
    # there are two such pretokens): counts tie, so the greater pair, (b'b', b'b'),
    # must be merged first over (b'a', b'a').
    word_freq = count_pretokens(get_pretokens(["aa aa bb bb"]))
    vocab = init_vocab(SPECIAL_TOKENS)
    merges, _ = train_bpe_incremental(word_freq, vocab, num_merges=1)
    assert len(merges) == 1
    assert merges[0] == max((b'a', b'a'), (b'b', b'b'))
    assert merges[0] == (b'b', b'b')


# ---------------------------------------------------------------------------
# Save / load round trip
# ---------------------------------------------------------------------------

def test_from_files_round_trip(tmp_path, small_tokenizer):
    import pickle
    vocab_path = tmp_path / "vocab.pkl"
    merges_path = tmp_path / "merges.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(small_tokenizer.vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(small_tokenizer.merges, f)

    reloaded = BPETokenizer.from_files(str(vocab_path), str(merges_path), SPECIAL_TOKENS)
    text = "Round trip through pickle, with <|endoftext|> inside."
    assert reloaded.encode(text) == small_tokenizer.encode(text)
    assert reloaded.decode(reloaded.encode(text)) == text
