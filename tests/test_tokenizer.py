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

import pytest

from conftest import SAMPLES_FILE, VALID_FILE
from tokenizer import (
    BPETokenizer, train_bpe, train_bpe_incremental, init_vocab,
    read_txt, split_text, get_pretokens, count_pretokens, merge_pair,
    iter_documents, iter_document_pieces, count_pretokens_from_documents,
    get_word_freq_from_files, get_word_freq_parallel, find_chunk_boundaries,
    train_bpe_naive, derive_vocab_and_merges,
)

SPECIAL_TOKENS = ["<|endoftext|>"]


# The tutorial-style reference trainer lives in the tokenizer module so that the
# equivalence test here and the Q2 speed comparison in scripts/ measure the same
# implementation rather than two copies that can drift apart.
naive_train_bpe = train_bpe_naive


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
# §3.2: the streaming/chunked corpus reader must agree with the whole-file path
#
# `get_word_freq_from_files` streams the corpus and counts pre-token *strings*,
# converting only the distinct keys to byte tuples, so that the full 2.2GB
# training file fits in Colab's RAM. That is only a safe optimisation if it
# produces exactly the frequency table the naive whole-file path produces -
# the merges, and therefore the whole tokenizer, depend on these counts.
# ---------------------------------------------------------------------------

def _reference_word_freq(path, special_tokens):
    """Whole-file, occurrence-by-occurrence reference (the original approach)."""
    return count_pretokens(get_pretokens(split_text(read_txt(path), special_tokens)))


def test_streaming_counter_matches_occurrence_counter():
    reference = _reference_word_freq(SAMPLES_FILE, SPECIAL_TOKENS)
    streamed = count_pretokens_from_documents(
        iter_documents(SAMPLES_FILE, SPECIAL_TOKENS))
    assert streamed == reference


def test_get_word_freq_from_files_matches_reference():
    reference = _reference_word_freq(SAMPLES_FILE, SPECIAL_TOKENS)
    assert get_word_freq_from_files([SAMPLES_FILE], SPECIAL_TOKENS) == reference


@pytest.mark.parametrize("chunk_size", [16, 64, 512, 1 << 16])
def test_iter_documents_is_invariant_to_chunk_size(chunk_size):
    """A special token straddling a read boundary must not be mis-split, and a
    pre-token must never be cut in half by the buffer flush. Tiny chunk sizes
    force both boundary cases that a 4MB chunk would essentially never hit."""
    reference = _reference_word_freq(SAMPLES_FILE, SPECIAL_TOKENS)
    streamed = count_pretokens_from_documents(
        iter_documents(SAMPLES_FILE, SPECIAL_TOKENS, chunk_size=chunk_size))
    assert streamed == reference


def test_flush_valve_splits_only_between_pretokens_and_is_flagged():
    """Force the delimiter-free memory safety valve to fire on a small file.

    Two guarantees matter: the pieces must still reassemble into the same
    pre-token counts (the cut lands at whitespace, never inside a pre-token),
    and every mid-document cut must be flagged ends_document=False so that
    `encode_corpus` does not plant an <|endoftext|> inside a story.
    """
    pieces = list(iter_document_pieces(
        SAMPLES_FILE, SPECIAL_TOKENS, chunk_size=256, flush_threshold=512))

    assert any(not ends for _, ends in pieces), "safety valve never fired"

    streamed = count_pretokens_from_documents(text for text, _ in pieces)
    assert streamed == _reference_word_freq(SAMPLES_FILE, SPECIAL_TOKENS)


def test_iter_documents_strips_every_special_token():
    streamed = "".join(iter_documents(SAMPLES_FILE, SPECIAL_TOKENS))
    for special in SPECIAL_TOKENS:
        assert special not in streamed


def test_streaming_and_reference_train_identical_merges():
    """The end-to-end consequence: identical counts must give identical merges."""
    reference = _reference_word_freq(SAMPLES_FILE, SPECIAL_TOKENS)
    streamed = get_word_freq_from_files([SAMPLES_FILE], SPECIAL_TOKENS)

    vocab_ref = init_vocab(SPECIAL_TOKENS)
    merges_ref, _ = train_bpe_incremental(reference, vocab_ref, num_merges=200)
    vocab_str = init_vocab(SPECIAL_TOKENS)
    merges_str, _ = train_bpe_incremental(streamed, vocab_str, num_merges=200)

    assert merges_ref == merges_str
    assert vocab_ref == vocab_str


# ---------------------------------------------------------------------------
# §3.2: parallel pre-tokenization must not change the answer
# ---------------------------------------------------------------------------

def test_chunk_boundaries_align_to_delimiters_and_tile_the_file():
    raw = open(SAMPLES_FILE, "rb").read()
    boundaries = find_chunk_boundaries(SAMPLES_FILE, 4)

    assert boundaries[0] == 0
    assert boundaries[-1] == len(raw)
    assert boundaries == sorted(boundaries)
    # Interior boundaries sit immediately after a delimiter, so no document is
    # ever cut across two ranges.
    for boundary in boundaries[1:-1]:
        assert raw[:boundary].endswith(b"<|endoftext|>")
    # The ranges tile the file exactly - nothing dropped, nothing counted twice.
    assert b"".join(raw[s:e] for s, e in zip(boundaries[:-1], boundaries[1:])) == raw


def test_parallel_word_freq_matches_serial():
    serial = get_word_freq_from_files([SAMPLES_FILE], SPECIAL_TOKENS)
    parallel = get_word_freq_parallel([SAMPLES_FILE], SPECIAL_TOKENS, workers=2)
    assert parallel == serial


def test_reading_does_not_translate_line_endings(tmp_path):
    """Python's text mode rewrites CRLF to LF on Windows unless newline="" is
    passed. That would make the serial reader disagree with the binary-reading
    parallel one, make the learned merges platform-dependent, and break the
    round-trip guarantee - so the readers must see the file's actual bytes."""
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"line one\r\nline two<|endoftext|>second doc\r\n")

    assert read_txt(str(path)).count("\r") == 2
    streamed = "".join(iter_documents(str(path), SPECIAL_TOKENS))
    assert streamed.count("\r") == 2

    serial = get_word_freq_from_files([str(path)], SPECIAL_TOKENS)
    parallel = get_word_freq_parallel([str(path)], SPECIAL_TOKENS, workers=2)
    assert serial == parallel


def test_crlf_text_round_trips_exactly(tmp_path):
    path = tmp_path / "crlf.txt"
    text = "a line\r\nanother line\r\n"
    path.write_bytes(text.encode("utf-8"))

    tokenizer = BPETokenizer(*train_bpe(str(path), 300, SPECIAL_TOKENS), SPECIAL_TOKENS)
    assert tokenizer.decode(tokenizer.encode(text)) == text


# ---------------------------------------------------------------------------
# §3.4/§3.5: deriving a smaller tokenizer by truncating a longer merge list
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_run():
    """One training run to a large vocabulary, to derive smaller ones from."""
    return train_bpe(SAMPLES_FILE, vocab_size=600, special_tokens=SPECIAL_TOKENS)


@pytest.mark.parametrize("vocab_size", [300, 450, 600])
def test_derived_tokenizer_is_identical_to_a_direct_run(full_run, vocab_size):
    """The vocabulary study and the §3.5 second tokenizer both rely on training
    once to the largest size and truncating. That is only sound if truncation
    reproduces a direct run exactly - if it does not, every derived tokenizer is
    subtly wrong and so is the compression curve built from it."""
    _, merges_full = full_run
    vocab_direct, merges_direct = train_bpe(SAMPLES_FILE, vocab_size, SPECIAL_TOKENS)
    vocab_derived, merges_derived = derive_vocab_and_merges(
        merges_full, vocab_size, SPECIAL_TOKENS)

    assert merges_derived == merges_direct
    assert vocab_derived == vocab_direct


def test_derived_tokenizer_encodes_identically(full_run):
    _, merges_full = full_run
    direct = BPETokenizer(*train_bpe(SAMPLES_FILE, 400, SPECIAL_TOKENS), SPECIAL_TOKENS)
    derived = BPETokenizer(*derive_vocab_and_merges(merges_full, 400, SPECIAL_TOKENS),
                           SPECIAL_TOKENS)
    text = "Once upon a time, a small dog.<|endoftext|>Then a cat appeared."
    assert derived.encode(text) == direct.encode(text)


def test_derive_rejects_impossible_vocab_sizes(full_run):
    _, merges_full = full_run
    with pytest.raises(ValueError):
        derive_vocab_and_merges(merges_full, 10, SPECIAL_TOKENS)      # below the byte floor
    with pytest.raises(ValueError):
        derive_vocab_and_merges(merges_full, 10_000, SPECIAL_TOKENS)  # more merges than learned


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
