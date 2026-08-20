"""Task 5 (§8) and §2's validation/test split.

Two things here are worth more than the rest.

The **split** (§2: "reserve the last 2,000 documents of the validation file as
a test set that you touch exactly once") has to be lossless and has to be
byte-identical to the slice of the corpus it came from. It was not: writing the
split files without newline="" let Windows rewrite every \\n as \\r\\n, which
inflated the character count §6 divides by and changed how the text tokenized.
That is a silent wrong answer for BPC, so it has a regression test.

The **truncation** in `generate` is the other one. Top-k and top-p both keep a
subset of the distribution, and an off-by-one at the boundary does not crash -
it just makes top_p=0.9 behave like 0.95. These check the cutoff against
distributions whose answer is known by hand.
"""
import json
import os

import numpy as np
import pytest
import torch

from src.generate import truncate_distribution
from scripts.make_splits import split_documents, write_split
from scripts.run_task5_questions import (
    DECODING_SETTINGS, DEGENERATE_SETTING, repetition_metrics, first_lines,
    read_ledger, record_test_touch, guard_test_set, build_parser, load_final_model)

END_OF_TEXT = "<|endoftext|>"


class FakeTokenizer:
    """Whitespace tokenizer - enough for the repetition metrics, which only
    need a consistent token count and identity."""
    def encode(self, text):
        return [hash(w) % 1000 for w in text.split()]


# ---------------------------------------------------------------------------
# §2 - the validation/test split
# ---------------------------------------------------------------------------

def corpus(n_docs):
    return "".join(f"story number {i} " + END_OF_TEXT for i in range(n_docs))


def test_the_last_n_documents_are_reserved_for_test():
    """§2: the *last* 2,000, not a random sample - the split has to be
    reproducible from the corpus alone."""
    val, test = split_documents(corpus(50), n_test_docs=10)
    assert len(val) == 40 and len(test) == 10
    assert "story number 49" in test[-1]
    assert "story number 39" in val[-1]
    assert "story number 40" in test[0]


def test_split_drops_only_blank_pieces():
    """Splitting on the delimiter leaves an empty piece after a trailing
    delimiter; counting it as a document shifts the boundary by one."""
    val, test = split_documents(corpus(20), n_test_docs=5)
    assert len(val) + len(test) == 20
    assert all(d.strip() for d in val + test)


def test_reserving_everything_is_rejected():
    with pytest.raises(SystemExit, match="nothing would be left"):
        split_documents(corpus(10), n_test_docs=10)
    with pytest.raises(SystemExit):
        split_documents(corpus(10), n_test_docs=25)


def test_written_splits_do_not_translate_newlines(tmp_path):
    """The regression that mattered: without newline="" Windows turns every \\n
    into \\r\\n on write, so the split is no longer the same bytes as the slice
    of the corpus it came from - inflating the character count BPC divides by."""
    docs = ["line one\nline two\n", "another\ndocument\n"]
    path = write_split(docs, str(tmp_path / "split.txt"))

    raw = open(path, "rb").read()
    assert b"\r\n" not in raw, "newlines were translated on write"
    assert raw.decode("utf-8") == "".join(d + END_OF_TEXT for d in docs)


def test_the_split_round_trips_losslessly(tmp_path):
    """Every non-delimiter character of the corpus survives into exactly one of
    the two splits - the whole basis for scoring them separately."""
    text = corpus(30)
    val, test = split_documents(text, n_test_docs=7)
    val_path = write_split(val, str(tmp_path / "v.txt"))
    test_path = write_split(test, str(tmp_path / "t.txt"))

    def chars_excluding_delimiters(s):
        return len(s) - s.count(END_OF_TEXT) * len(END_OF_TEXT)

    written = (chars_excluding_delimiters(open(val_path, encoding="utf-8", newline="").read())
               + chars_excluding_delimiters(open(test_path, encoding="utf-8", newline="").read()))
    assert written == chars_excluding_delimiters(text)


# ---------------------------------------------------------------------------
# §4.3 - decoding
# ---------------------------------------------------------------------------

def test_top_k_keeps_exactly_k_tokens():
    probs = torch.tensor([0.4, 0.3, 0.2, 0.07, 0.03])
    out = truncate_distribution(probs, top_k=2)

    assert torch.count_nonzero(out) == 2
    assert out[0] > 0 and out[1] > 0
    assert out[2] == 0 and out[3] == 0 and out[4] == 0
    assert out.sum() == pytest.approx(1.0)
    # Kept mass is renormalised, and the ratio between survivors is preserved.
    assert out[0] / out[1] == pytest.approx(0.4 / 0.3)


def test_top_k_of_one_is_greedy():
    probs = torch.tensor([0.1, 0.6, 0.3])
    out = truncate_distribution(probs, top_k=1)
    assert out.argmax() == 1
    assert out[1] == pytest.approx(1.0)
    assert torch.count_nonzero(out) == 1


def test_top_k_larger_than_the_vocabulary_keeps_everything():
    probs = torch.tensor([0.5, 0.3, 0.2])
    out = truncate_distribution(probs, top_k=99)
    assert torch.count_nonzero(out) == 3
    assert out.sum() == pytest.approx(1.0)


def test_top_p_keeps_the_smallest_set_reaching_p():
    # Cumulative: 0.5, 0.8, 0.95, 1.0. p=0.9 needs three tokens.
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    out = truncate_distribution(probs, top_p=0.9)
    assert torch.count_nonzero(out) == 3
    assert out[3] == 0


def test_top_p_boundary_landing_exactly_on_p():
    """A prefix summing exactly to top_p is already enough; taking one more
    would quietly make top_p=0.8 behave like 0.95."""
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    out = truncate_distribution(probs, top_p=0.8)
    assert torch.count_nonzero(out) == 2


def test_top_p_always_keeps_at_least_one_token():
    """A p below the largest single probability must not empty the
    distribution - there would be nothing to sample."""
    probs = torch.tensor([0.9, 0.05, 0.05])
    out = truncate_distribution(probs, top_p=0.1)
    assert torch.count_nonzero(out) == 1
    assert out[0] == pytest.approx(1.0)


def test_top_k_and_top_p_compose():
    """top-k first, then top-p narrows what survived."""
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    both = truncate_distribution(probs, top_k=3, top_p=0.8)
    # top-k=3 keeps 0.4/0.3/0.2 renormalised to 4/9, 3/9, 2/9; cumulative
    # 0.444, 0.777, 1.0 - so reaching 0.8 takes all three.
    assert torch.count_nonzero(both) == 3
    assert both.sum() == pytest.approx(1.0)


def test_no_truncation_returns_the_distribution_unchanged():
    probs = torch.tensor([0.5, 0.3, 0.2])
    assert torch.equal(truncate_distribution(probs), probs)
    # top_p=1.0 is not a truncation either.
    assert torch.equal(truncate_distribution(probs, top_p=1.0), probs)


def test_truncated_distribution_is_a_distribution():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(200), dim=-1)
    for kwargs in ({"top_k": 5}, {"top_p": 0.9}, {"top_k": 50, "top_p": 0.8}):
        out = truncate_distribution(probs, **kwargs)
        assert out.sum() == pytest.approx(1.0, abs=1e-6)
        assert (out >= 0).all()


# ---------------------------------------------------------------------------
# §8.1 - what the questions require
# ---------------------------------------------------------------------------

def test_q19_settings_cover_top_k_and_top_p():
    """§8.1 Q19: three settings, at least one using top-p and one using top-k."""
    assert len(DECODING_SETTINGS) >= 3
    assert any("top_k" in s for s in DECODING_SETTINGS)
    assert any("top_p" in s for s in DECODING_SETTINGS)
    assert all(s.get("label") for s in DECODING_SETTINGS), "each sample must be labelled"


def test_the_degenerate_setting_is_not_one_of_the_recommended_three():
    """Q20's degenerate setting is a separate exhibit, not one of Q19's."""
    q19 = [{k: v for k, v in s.items() if k != "label"} for s in DECODING_SETTINGS]
    degenerate = {k: v for k, v in DEGENERATE_SETTING.items() if k != "label"}
    assert degenerate not in q19


def test_repetition_metrics_detect_a_loop():
    """Q20's claim about repetition is measured, not asserted."""
    looped = repetition_metrics("the cat sat " * 20, FakeTokenizer())
    varied = repetition_metrics(
        " ".join(f"word{i}" for i in range(60)), FakeTokenizer())

    assert looped["distinct_token_ratio"] < varied["distinct_token_ratio"]
    assert looped["most_repeated_trigram_count"] > 5
    assert varied["most_repeated_trigram_count"] == 1


def test_repetition_metrics_handle_empty_text():
    m = repetition_metrics("", FakeTokenizer())
    assert m["distinct_token_ratio"] == 0.0
    assert m["most_repeated_trigram_count"] == 0


def test_first_lines_caps_the_quoted_evidence():
    """§8.1 Q20 allows at most three lines of generated text as evidence."""
    text = "\n".join(f"line {i}" for i in range(10))
    assert len(first_lines(text, n=3)) == 3
    assert first_lines("x" * 500, n=1)[0] == "x" * 100


# ---------------------------------------------------------------------------
# §2 - the test set is touched once
# ---------------------------------------------------------------------------

def make_args(tmp_path, *extra):
    return build_parser().parse_args(
        ["--log_dir", str(tmp_path), "--data_dir", str(tmp_path), *extra])


def test_the_first_test_evaluation_is_allowed(tmp_path):
    args = make_args(tmp_path)
    assert read_ledger(args) == []
    guard_test_set(args)          # must not raise


def test_a_second_test_evaluation_is_refused(tmp_path):
    """§2 says the test set is touched exactly once, so a second attempt has to
    be an explicit decision rather than an accident of re-running a script."""
    args = make_args(tmp_path)
    record_test_touch(args, {"when": "2026-01-01T00:00:00+00:00",
                             "run_name": "final_model", "loss": 1.234})

    with pytest.raises(SystemExit, match="reserves the test set"):
        guard_test_set(args)


def test_a_second_evaluation_can_be_asked_for_explicitly(tmp_path):
    args = make_args(tmp_path, "--touch_test_again")
    record_test_touch(args, {"when": "2026-01-01T00:00:00+00:00",
                             "run_name": "final_model", "loss": 1.234})
    guard_test_set(args)          # must not raise


def test_every_touch_is_recorded(tmp_path):
    args = make_args(tmp_path)
    record_test_touch(args, {"when": "a", "run_name": "r", "loss": 1.0})
    record_test_touch(args, {"when": "b", "run_name": "r", "loss": 2.0})

    entries = read_ledger(args)
    assert [e["when"] for e in entries] == ["a", "b"]
    assert os.path.exists(os.path.join(str(tmp_path), "test_set_ledger.json"))


# ---------------------------------------------------------------------------
# no results from an untrained model
# ---------------------------------------------------------------------------

def test_an_untrained_model_is_refused_by_default(tmp_path):
    """The previous version of this script had the checkpoint load commented
    out, so every number and sample came from random initialisation. Producing
    that silently is the failure being prevented."""
    args = make_args(tmp_path)
    with pytest.raises(SystemExit, match="No trained final model found"):
        load_final_model(args, torch.device("cpu"))


def test_untrained_is_allowed_only_when_asked_for(tmp_path):
    args = make_args(tmp_path, "--allow_untrained", "--vocab_size", "64",
                     "--context_length", "16")
    model, config, record = load_final_model(args, torch.device("cpu"))
    assert record is None, "an untrained run must not look like a real one"
    assert config["vocab_size"] == 64


def test_a_run_record_is_loaded_when_present(tmp_path):
    """The final model comes from Task 4's run record, so its config never has
    to be restated - and so it cannot drift from what was actually trained."""
    from src.model import TransformerLM, TransformerConfig
    from src.training_helpers.manage_checkpoint import save_checkpoint
    from torch.optim import AdamW

    config = dict(vocab_size=64, context_length=16, n_layers=1, d_model=32,
                  n_heads=4, d_ff=64)
    model = TransformerLM(TransformerConfig(**config))
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    ckpt = str(tmp_path / "final.pt")
    save_checkpoint(model, optimizer, scheduler, 4321, ckpt, config={"num_steps": 4321})

    with open(tmp_path / "final_model_run.json", "w") as f:
        json.dump({"name": "final_model", "model_config": config,
                   "checkpoint": ckpt}, f)

    args = make_args(tmp_path)
    loaded, loaded_config, record = load_final_model(args, torch.device("cpu"))

    assert record["name"] == "final_model"
    assert loaded_config == config
    for (name, a), (_, b) in zip(model.named_parameters(), loaded.named_parameters()):
        assert torch.equal(a, b), f"{name} was not restored from the checkpoint"


# ---------------------------------------------------------------------------
# §5.5 - the training corpus is two files addressed as one
# ---------------------------------------------------------------------------

def test_concatenated_parts_index_as_one_sequence():
    """Task 1 encodes the training split as two arrays because the corpus ships
    as two files. The training loop wants one flat sequence, and np.concatenate
    would materialise 1.1GB of uint16 that §5.5 says to keep on disk."""
    from src.data import ConcatTokens

    a = np.arange(0, 10, dtype=np.uint16)
    b = np.arange(10, 25, dtype=np.uint16)
    joined = ConcatTokens([a, b])
    expected = np.arange(0, 25, dtype=np.uint16)

    assert len(joined) == 25
    assert joined[0] == 0 and joined[9] == 9 and joined[10] == 10 and joined[24] == 24
    assert joined[-1] == 24
    np.testing.assert_array_equal(joined[:], expected)


def test_a_slice_spanning_the_boundary_is_contiguous():
    """The only interesting case: a window that starts in one part and ends in
    the next must come back as though the parts were one array."""
    from src.data import ConcatTokens

    a = np.arange(0, 10, dtype=np.uint16)
    b = np.arange(10, 25, dtype=np.uint16)
    joined = ConcatTokens([a, b])

    np.testing.assert_array_equal(joined[7:14], np.arange(7, 14, dtype=np.uint16))
    np.testing.assert_array_equal(joined[9:11], np.array([9, 10], dtype=np.uint16))
    # Entirely inside one part, on each side of the boundary.
    np.testing.assert_array_equal(joined[2:5], np.arange(2, 5, dtype=np.uint16))
    np.testing.assert_array_equal(joined[12:15], np.arange(12, 15, dtype=np.uint16))


def test_concatenated_parts_work_with_the_batching_path():
    """get_batch slices the array directly, so the wrapper has to satisfy it."""
    from src.data import ConcatTokens
    from src.training_helpers.get_batch import get_batch

    joined = ConcatTokens([np.arange(0, 50, dtype=np.uint16),
                           np.arange(50, 120, dtype=np.uint16)])
    x, y = get_batch(joined, batch_size=4, context_length=8,
                     device=torch.device("cpu"), rng=np.random.default_rng(0))

    assert x.shape == (4, 8) and y.shape == (4, 8)
    # Targets are inputs shifted one position, across the boundary too.
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_load_token_parts_returns_a_plain_array_for_one_path(tmp_path):
    """One part needs no wrapper, and the common case should carry no
    indirection."""
    from src.data import ConcatTokens, load_token_parts

    one = tmp_path / "a.npy"
    np.save(one, np.arange(10, dtype=np.uint16))
    assert not isinstance(load_token_parts([str(one)]), ConcatTokens)

    two = tmp_path / "b.npy"
    np.save(two, np.arange(10, 20, dtype=np.uint16))
    joined = load_token_parts([str(one), str(two)])
    assert isinstance(joined, ConcatTokens)
    assert len(joined) == 20


def test_task4_finds_task1s_actual_file_names(tmp_path):
    """Task 1 writes train_part1/train_part2 and suffixes the second
    tokenizer's outputs. Guessing a different convention means Task 4 reports
    the corpus as missing when it is sitting right there."""
    from scripts.run_task4_questions import corpus_paths

    for name in ("train_part1.npy", "train_part2.npy",
                 "train_part1_vocab1000.npy", "train_part2_vocab1000.npy"):
        (tmp_path / name).write_bytes(b"")

    primary = corpus_paths(str(tmp_path), 4000, 4000)
    assert [os.path.basename(p) for p in primary["train"]] == [
        "train_part1.npy", "train_part2.npy"]
    assert os.path.basename(primary["valid"]) == "valid_split.npy"

    second = corpus_paths(str(tmp_path), 1000, 4000)
    assert [os.path.basename(p) for p in second["train"]] == [
        "train_part1_vocab1000.npy", "train_part2_vocab1000.npy"]
    assert os.path.basename(second["valid"]) == "valid_split_vocab1000.npy"
