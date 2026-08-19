"""Checks for the §3.4 compression study.

The study reads the corpus size off the pre-token frequency table instead of
making a second pass over a 2.2GB file. That shortcut is only valid if the
GPT-2 pre-tokenizer regex tiles the text completely - every character belonging
to exactly one pre-token - so that is tested here rather than assumed.
"""
import sys

import pytest

from conftest import REPO_ROOT, SAMPLES_FILE
from src.tokenizer import (read_txt, split_text, iter_documents, init_vocab,
                       get_word_freq_from_files, train_bpe_incremental, _PAT)

sys.path.insert(0, REPO_ROOT)
from scripts.vocab_study import (  # noqa: E402
    corpus_size_from_word_freq, compute_compression_metrics, train_bpe_with_counts,
)

SPECIAL_TOKENS = ["<|endoftext|>"]


def test_pretokenizer_tiles_the_text_completely():
    documents = split_text(read_txt(SAMPLES_FILE), SPECIAL_TOKENS)
    for doc in documents:
        rebuilt = "".join(m.group() for m in _PAT.finditer(doc))
        assert rebuilt == doc, "the regex left a gap; corpus size cannot be read off word_freq"


def test_corpus_size_from_word_freq_matches_a_direct_count():
    word_freq = get_word_freq_from_files([SAMPLES_FILE], SPECIAL_TOKENS)
    n_chars, n_bytes = corpus_size_from_word_freq(word_freq)

    documents = list(iter_documents(SAMPLES_FILE, SPECIAL_TOKENS))
    assert n_chars == sum(len(d) for d in documents)
    assert n_bytes == sum(len(d.encode("utf-8")) for d in documents)


def test_initial_token_count_equals_the_corpus_byte_count():
    """Before any merge every token is one byte, so the two must agree - this is
    what makes `token_counts` a compression-ratio curve."""
    word_freq = get_word_freq_from_files([SAMPLES_FILE], SPECIAL_TOKENS)
    _, n_bytes = corpus_size_from_word_freq(word_freq)

    vocab = init_vocab(SPECIAL_TOKENS)
    _, token_counts = train_bpe_incremental(word_freq, vocab, 50, track_token_counts=True)
    assert token_counts[0] == n_bytes


def test_metrics_are_reported_at_the_requested_vocab_sizes():
    _, _, token_counts, initial_size, word_freq = train_bpe_with_counts(
        SAMPLES_FILE, 500, SPECIAL_TOKENS)
    targets = [300, 400, 500]
    metrics, _, _ = compute_compression_metrics(
        word_freq, token_counts, initial_size, targets)

    assert [m["vocab_size"] for m in metrics] == targets
    # A bigger vocabulary can never compress worse.
    ratios = [m["bytes_per_token"] for m in metrics]
    assert ratios == sorted(ratios)
    # The parameter cost is the part of the budget that scales with vocab size.
    assert metrics[0]["embed_lm_head_params"] == 2 * 300 * 512


def test_out_of_range_vocab_sizes_are_skipped_not_silently_wrong():
    """The previous version fell back to 'closest available size', which
    reported a number for a vocabulary it had not actually trained."""
    _, _, token_counts, initial_size, word_freq = train_bpe_with_counts(
        SAMPLES_FILE, 400, SPECIAL_TOKENS)
    metrics, _, _ = compute_compression_metrics(
        word_freq, token_counts, initial_size, [300, 100_000])

    assert [m["vocab_size"] for m in metrics] == [300]
