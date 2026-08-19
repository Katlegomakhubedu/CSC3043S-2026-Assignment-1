"""Regression tests for corpus encoding (§3.5).

The bug these exist to prevent: the original encoder split the corpus on
<|endoftext|>, discarded the delimiter, and never re-emitted it. The encoded
array therefore contained *zero* <|endoftext|> tokens - stories ran together
with no boundary, so the model could never learn to emit the token that
`generate()` stops on. Nothing failed loudly; the arrays looked fine.
"""
import json
import os
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT, SAMPLES_FILE
from src.tokenizer import BPETokenizer, train_bpe

sys.path.insert(0, REPO_ROOT)
from scripts.encode_corpus import encode_corpus, END_OF_TEXT  # noqa: E402

SPECIAL_TOKENS = [END_OF_TEXT]


@pytest.fixture(scope="module")
def tokenizer():
    vocab, merges = train_bpe(SAMPLES_FILE, vocab_size=500, special_tokens=SPECIAL_TOKENS)
    return BPETokenizer(vocab, merges, SPECIAL_TOKENS)


@pytest.fixture
def corpus(tmp_path):
    """Three documents, delimiter-separated, with no trailing delimiter."""
    docs = ["First story about a dog.", "Second story about a cat.", "Third story."]
    path = tmp_path / "corpus.txt"
    path.write_text(END_OF_TEXT.join(docs), encoding="utf-8")
    return path, docs


def _encode(tmp_path, tokenizer, corpus_path, **kwargs):
    out = tmp_path / "encoded.npy"
    meta = encode_corpus(str(corpus_path), str(out), tokenizer, report_every=0, **kwargs)
    return np.load(out), meta


def test_documents_are_delimited_by_endoftext(tmp_path, tokenizer, corpus):
    path, docs = corpus
    arr, meta = _encode(tmp_path, tokenizer, path)

    eot_id = tokenizer.token_to_id[END_OF_TEXT.encode("utf-8")]
    assert int((arr == eot_id).sum()) == len(docs)
    assert meta["n_documents"] == len(docs)
    assert meta["eot_count"] == len(docs)


def test_encoded_array_round_trips_to_the_original_documents(tmp_path, tokenizer, corpus):
    path, docs = corpus
    arr, _ = _encode(tmp_path, tokenizer, path)

    decoded = tokenizer.decode(arr.tolist())
    # Every document is delimited, including the last, so the decode ends with a
    # trailing delimiter that the source file does not have.
    assert decoded == END_OF_TEXT.join(docs) + END_OF_TEXT


def test_array_is_uint16(tmp_path, tokenizer, corpus):
    path, _ = corpus
    arr, meta = _encode(tmp_path, tokenizer, path)
    assert arr.dtype == np.uint16
    assert meta["n_tokens"] == arr.size


def test_no_eot_flag_emits_no_delimiters(tmp_path, tokenizer, corpus):
    path, _ = corpus
    arr, meta = _encode(tmp_path, tokenizer, path, add_eot=False)

    eot_id = tokenizer.token_to_id[END_OF_TEXT.encode("utf-8")]
    assert int((arr == eot_id).sum()) == 0
    assert meta["eot_count"] == 0


def test_meta_sidecar_records_char_counts_for_bpc(tmp_path, tokenizer, corpus):
    """§6 computes BPC from the raw character count of the evaluated text, so the
    sidecar must record it exactly - both with and without the delimiters, since
    the spec requires stating which convention was used."""
    path, docs = corpus
    out = tmp_path / "encoded.npy"
    encode_corpus(str(path), str(out), tokenizer, report_every=0)

    meta = json.loads((tmp_path / "encoded_meta.json").read_text(encoding="utf-8"))
    assert meta["n_chars_excluding_delimiters"] == sum(len(d) for d in docs)
    assert meta["n_chars_including_delimiters"] == (
        sum(len(d) for d in docs) + len(docs) * len(END_OF_TEXT))
    assert meta["chars_per_token"] == pytest.approx(
        meta["n_chars_excluding_delimiters"] / meta["n_tokens"])


def test_long_document_gets_no_interior_delimiter(tmp_path, tokenizer):
    """The streaming reader cuts an over-long delimiter-free stretch to bound
    memory. That cut is not a document boundary, and must not produce an
    <|endoftext|> in the middle of a document."""
    long_doc = " ".join(f"word{i}" for i in range(4000))
    path = tmp_path / "long.txt"
    path.write_text(long_doc, encoding="utf-8")

    # flush_threshold well below the document length forces the safety valve.
    arr, meta = _encode(tmp_path, tokenizer, path, flush_threshold=512)

    eot_id = tokenizer.token_to_id[END_OF_TEXT.encode("utf-8")]
    assert meta["n_documents"] == 1
    assert int((arr == eot_id).sum()) == 1, "safety-valve cut produced an interior delimiter"
    assert arr[-1] == eot_id, "the only delimiter should be the terminating one"
    assert tokenizer.decode(arr.tolist()) == long_doc + END_OF_TEXT
