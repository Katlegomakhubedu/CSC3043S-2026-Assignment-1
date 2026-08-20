"""Task 4 (§7): the experiment harness and the §6 metrics it reports.

§7's rule is that when you vary one thing, everything else - seed, tokens
processed, validation batches - is held fixed. A comparison in which two things
changed is not informative, and the failure is silent: you get a number, it
just isn't the number you think it is. Most of these tests are about that
invariant rather than about any single computation.

The rest cover §6's metrics, where the mistakes are also quiet ones: perplexity
that isn't exp(loss), BPC that divides by the wrong character count, and
overlapping evaluation windows that flatter one model over another.
"""
import json
import math
import os

import numpy as np
import pytest
import torch

from src.model import TransformerLM, TransformerConfig
from src.evaluate import (evaluate, evaluate_by_position, n_windows,
                          chars_from_meta)
from src.train import train
from scripts.run_task4_questions import (
    Experiment, BASE_CONFIG, STANDARD_RUN, RELU_D_FF, build_model,
    ablation_experiments, baseline_experiment, sweep_experiments, reduced_lr,
    build_parser, ensure_run, load_run, synthetic_corpus)

CONTEXT = 16
VOCAB = 64


def tiny_config(**overrides):
    cfg = dict(vocab_size=VOCAB, context_length=CONTEXT, n_layers=2, d_model=32,
               n_heads=4, d_ff=64)
    cfg.update(overrides)
    return cfg


def tiny_model(seed=0, **overrides):
    torch.manual_seed(seed)
    return TransformerLM(TransformerConfig(**tiny_config(**overrides)))


def tokens(n=1000, seed=0):
    return np.random.default_rng(seed).integers(0, VOCAB, size=n, dtype=np.uint16)


# ---------------------------------------------------------------------------
# §7 - one thing at a time
# ---------------------------------------------------------------------------

def test_each_ablation_changes_exactly_one_thing():
    """§7.2: an ablation is one standard run differing from the baseline in the
    single feature under test. Anything else that drifts makes the gap
    uninterpretable."""
    baseline = baseline_experiment(1e-3)
    base_cfg = baseline.model_config()

    expected = {
        "ablation_no_rmsnorm": {"use_rmsnorm"},
        "ablation_no_rmsnorm_lowlr": {"use_rmsnorm"},
        "ablation_nope": {"use_rope"},
        # The ReLU arm changes d_ff too, and must: §7.2 sets d_ff=2048 precisely
        # so the parameter counts match. Holding d_ff fixed would confound the
        # activation with a 33% parameter cut.
        "ablation_relu": {"ffn_type", "d_ff"},
    }

    for exp in ablation_experiments(best_lr=1e-3, reduced_lr=3e-4):
        differing = {k for k, v in exp.model_config().items() if base_cfg[k] != v}
        assert differing == expected[exp.name], (
            f"{exp.name} differs from the baseline in {differing}, "
            f"expected {expected[exp.name]}")


def test_ablations_inherit_the_standard_run_configuration():
    """Batch size, steps, warmup, decay, clipping and seed come from §7's table
    for every arm - none of them is restated per experiment, so none can drift."""
    for exp in ablation_experiments(best_lr=1e-3, reduced_lr=3e-4):
        kwargs = exp.run_kwargs()
        for key, value in STANDARD_RUN.items():
            assert kwargs[key] == value, f"{exp.name} changed {key}"


def test_only_the_low_lr_arm_changes_the_learning_rate():
    arms = {e.name: e for e in ablation_experiments(best_lr=1e-3, reduced_lr=3e-4)}
    assert arms["ablation_no_rmsnorm"].lr == 1e-3
    assert arms["ablation_nope"].lr == 1e-3
    assert arms["ablation_relu"].lr == 1e-3
    assert arms["ablation_no_rmsnorm_lowlr"].lr == 3e-4


def test_reduced_lr_is_below_the_best_lr():
    """§7.2's second no-RMSNorm run is at a *reduced* learning rate. A fixed
    default lands above the baseline whenever the sweep picks something smaller,
    which would quietly make it an increased-lr run instead."""
    args = build_parser().parse_args([])
    for best in (1e-2, 1e-3, 1e-4, 3e-5):
        assert reduced_lr(args, best) < best

    explicit = build_parser().parse_args(["--reduced_lr", "5e-5"])
    assert reduced_lr(explicit, 1e-3) == 5e-5


def test_experiment_overrides_do_not_leak_into_the_shared_config():
    """Experiments copy BASE_CONFIG rather than mutating it, so building one
    ablation cannot change what the next one inherits."""
    before = dict(BASE_CONFIG)
    exp = Experiment(name="x", phase="p", lr=1e-3,
                     config_overrides={"use_rope": False, "d_ff": 99})
    exp.model_config()
    exp.run_kwargs()
    assert BASE_CONFIG == before


def test_models_for_two_arms_start_from_identical_weights():
    """§7: hold the seed fixed. Two arms of a comparison must differ by design,
    not by initialisation."""
    device = torch.device("cpu")
    a = build_model(tiny_config(), device)
    b = build_model(tiny_config(), device)
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), f"{name} differs between two fresh models"


def test_relu_ablation_is_parameter_matched_at_the_assignment_dimensions():
    """§7.2: three matrices at 1344 against two at 2048, within 2%."""
    base = baseline_experiment(1e-3).model_config()
    relu = {e.name: e for e in ablation_experiments(1e-3, 3e-4)}["ablation_relu"].model_config()

    swiglu_params = sum(p.numel() for p in TransformerLM(TransformerConfig(**base)).parameters())
    relu_params = sum(p.numel() for p in TransformerLM(TransformerConfig(**relu)).parameters())

    assert relu["d_ff"] == RELU_D_FF
    assert abs(relu_params - swiglu_params) / swiglu_params < 0.02


def test_sweep_spans_two_orders_of_magnitude_with_a_matched_schedule():
    """§7.1: at least five rates over at least two orders of magnitude, and if
    the sweep is shortened the cosine period is shortened with it."""
    args = build_parser().parse_args(["--sweep_steps", "500"])
    exps = sweep_experiments(args)

    lrs = [e.lr for e in exps]
    assert len(lrs) >= 5
    assert max(lrs) / min(lrs) >= 100

    for e in exps:
        kwargs = e.run_kwargs()
        assert kwargs["num_steps"] == 500, "sweep must run at the reduced length"
        assert kwargs["warmup_steps"] < kwargs["num_steps"], (
            "warmup must fit inside the shortened run, or the cosine period "
            "goes negative")


# ---------------------------------------------------------------------------
# §7.1 - divergence
# ---------------------------------------------------------------------------

def test_a_diverging_run_is_flagged_and_stopped_early(tmp_path):
    """§7.1 requires the sweep to contain a divergent run, and requires killing
    it as soon as it is visibly diverged - the remaining steps teach nothing and
    the budget is not free."""
    train_ids, val_ids = tokens(), tokens(seed=1)
    history = train(tiny_model(), train_ids, val_ids, num_steps=60, batch_size=4,
                    learning_rate=50.0, warmup_steps=1, eval_every=5, save_every=0,
                    log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
                    run_name="boom", seed=0)

    assert history["diverged"] is True
    assert history["diverged_at"] is not None
    assert history["completed_steps"] < 60, "a diverged run should stop early"
    assert history["diverged_reason"]


def test_a_healthy_run_is_not_flagged_as_diverged(tmp_path):
    train_ids, val_ids = tokens(), tokens(seed=1)
    history = train(tiny_model(), train_ids, val_ids, num_steps=30, batch_size=4,
                    learning_rate=1e-3, warmup_steps=5, eval_every=10, save_every=0,
                    log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
                    run_name="fine", seed=0)

    assert history["diverged"] is False
    assert history["completed_steps"] == 30


def test_divergence_can_be_left_to_run(tmp_path):
    """The kill is the default, not the only option - Q10's figure may want the
    full divergent curve."""
    train_ids, val_ids = tokens(), tokens(seed=1)
    history = train(tiny_model(), train_ids, val_ids, num_steps=25, batch_size=4,
                    learning_rate=50.0, warmup_steps=1, eval_every=5, save_every=0,
                    log_dir=str(tmp_path / "logs"), checkpoint_dir=str(tmp_path / "ckpt"),
                    run_name="boom_full", seed=0, stop_on_divergence=False)

    assert history["diverged"] is True
    assert history["completed_steps"] == 25


# ---------------------------------------------------------------------------
# §6 - metrics
# ---------------------------------------------------------------------------

def test_windows_are_whole_and_non_overlapping():
    """§6: score whole non-overlapping windows. Each needs context_length + 1
    tokens so the shifted target of the last position exists."""
    assert n_windows(100, 10) == 9      # 99 usable tokens after the shift
    assert n_windows(101, 10) == 10
    assert n_windows(10, 10) == 0       # no room for the shift
    assert n_windows(11, 10) == 1


def test_perplexity_is_exp_of_the_mean_loss():
    model, data = tiny_model(), tokens(2000)
    m = evaluate(model, data, batch_size=4, context_length=CONTEXT,
                 device=torch.device("cpu"), total_chars=8000)
    assert m["perplexity"] == pytest.approx(math.exp(m["loss"]))


def test_bpc_is_loss_over_ln2_times_characters_per_token():
    """§6: BPC = loss / (ln2 x characters-per-token). This is the identity that
    makes two different vocabularies comparable, so it is worth pinning."""
    model, data = tiny_model(), tokens(2000)
    m = evaluate(model, data, batch_size=4, context_length=CONTEXT,
                 device=torch.device("cpu"), total_chars=8000)

    assert m["bpc"] == pytest.approx(
        m["loss"] / (math.log(2) * m["chars_per_token"]), rel=1e-9)


def test_bpc_counts_only_the_characters_actually_scored():
    """The trailing partial window is not evaluated, so C must be scaled to the
    portion that was - otherwise BPC is quietly diluted by text nobody scored."""
    model = tiny_model()
    # 2005 tokens at context 16 leaves 125 whole windows = 2000 scored tokens.
    data = tokens(2005)
    m = evaluate(model, data, batch_size=4, context_length=CONTEXT,
                 device=torch.device("cpu"), total_chars=10025)

    assert m["n_tokens"] == 125 * CONTEXT
    assert m["chars_used"] == pytest.approx(10025 * (m["n_tokens"] / 2005))


def test_batching_does_not_change_the_reported_loss():
    """Batches at the end of the data are smaller, so the total has to be summed
    rather than averaged over batches."""
    model, data = tiny_model(), tokens(1500)
    losses = [evaluate(model, data, batch_size=bs, context_length=CONTEXT,
                       device=torch.device("cpu"))["loss"]
              for bs in (1, 7, 32)]
    assert losses[0] == pytest.approx(losses[1], rel=1e-6)
    assert losses[0] == pytest.approx(losses[2], rel=1e-6)


def test_position_wise_losses_average_to_the_overall_loss():
    """Q17's breakdown must be a decomposition of the aggregate, not a
    differently-computed number that happens to look similar."""
    model, data = tiny_model(), tokens(2000)
    device = torch.device("cpu")

    overall = evaluate(model, data, 4, CONTEXT, device)
    by_pos = evaluate_by_position(model, data, 4, CONTEXT, device, n_buckets=4)

    assert len(by_pos["per_position"]) == CONTEXT
    assert float(np.mean(by_pos["per_position"])) == pytest.approx(
        overall["loss"], rel=1e-5)


def test_position_buckets_tile_the_window():
    model, data = tiny_model(), tokens(1000)
    buckets = evaluate_by_position(model, data, 4, CONTEXT, torch.device("cpu"),
                                   n_buckets=4)["buckets"]
    assert [b["start"] for b in buckets] == [0, 4, 8, 12]
    assert [b["end"] for b in buckets] == [3, 7, 11, 15]


def test_evaluate_restores_training_mode():
    model = tiny_model()
    model.train()
    evaluate(model, tokens(500), 4, CONTEXT, torch.device("cpu"))
    assert model.training
    evaluate_by_position(model, tokens(500), 4, CONTEXT, torch.device("cpu"))
    assert model.training


def test_evaluate_rejects_data_too_short_to_score():
    with pytest.raises(ValueError, match="no complete window"):
        evaluate(tiny_model(), tokens(CONTEXT), 4, CONTEXT, torch.device("cpu"))


def test_chars_from_meta_reports_which_convention_it_used(tmp_path):
    """§6 requires stating whether <|endoftext|> is counted, so the reader gets
    the convention back rather than a bare number."""
    npy = tmp_path / "corpus.npy"
    np.save(npy, tokens(10))
    (tmp_path / "corpus_meta.json").write_text(json.dumps({
        "n_chars_excluding_delimiters": 1000,
        "n_chars_including_delimiters": 1100}))

    chars, convention = chars_from_meta(str(npy))
    assert (chars, "excluding") == (1000, convention.split()[0])

    chars, convention = chars_from_meta(str(npy), include_delimiters=True)
    assert chars == 1100 and convention.startswith("including")


def test_chars_from_meta_explains_a_missing_sidecar(tmp_path):
    npy = tmp_path / "nometa.npy"
    np.save(npy, tokens(10))
    chars, reason = chars_from_meta(str(npy))
    assert chars is None and "no sidecar" in reason


# ---------------------------------------------------------------------------
# the run cache
# ---------------------------------------------------------------------------

def make_args(tmp_path, *extra):
    return build_parser().parse_args([
        "--smoke", "--log_dir", str(tmp_path / "logs"),
        "--checkpoint_dir", str(tmp_path / "ckpt"), "--out_dir", str(tmp_path),
        *extra])


def test_a_finished_run_is_reused_rather_than_retrained(tmp_path, monkeypatch):
    """These are 5,000-step GPU jobs and Q14 alone reads six of them. Re-running
    the analysis must not re-run the training."""
    import scripts.run_task4_questions as t4

    args = make_args(tmp_path)
    os.makedirs(args.log_dir, exist_ok=True)
    monkeypatch.setitem(t4.BASE_CONFIG, "context_length", CONTEXT)
    monkeypatch.setitem(t4.BASE_CONFIG, "vocab_size", VOCAB)
    monkeypatch.setitem(t4.BASE_CONFIG, "n_layers", 1)
    monkeypatch.setitem(t4.BASE_CONFIG, "d_model", 32)
    monkeypatch.setitem(t4.BASE_CONFIG, "n_heads", 4)
    monkeypatch.setitem(t4.BASE_CONFIG, "d_ff", 64)
    for key, value in (("num_steps", 6), ("batch_size", 4), ("warmup_steps", 1),
                       ("eval_every", 3), ("save_every", 0)):
        monkeypatch.setitem(t4.STANDARD_RUN, key, value)

    corpus = synthetic_corpus(VOCAB)
    exp = Experiment(name="cached_run", phase="test", lr=1e-3)
    device = torch.device("cpu")

    first = ensure_run(args, exp, corpus, device)
    assert load_run(args, "cached_run") is not None

    calls = []
    monkeypatch.setattr(t4, "train", lambda *a, **k: calls.append(1))
    second = ensure_run(args, exp, corpus, device)

    assert not calls, "a cached run must not retrain"
    assert second["final_val_loss"] == first["final_val_loss"]


def write_cached_run(args, name="already"):
    """A run record as ensure_run writes one: a JSON file and the checkpoint
    it points at, since the analysis loads the weights back."""
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint = os.path.join(args.checkpoint_dir, f"{name}_final.pt")
    with open(checkpoint, "wb") as f:
        f.write(b"not a real checkpoint, but it exists")
    with open(os.path.join(args.log_dir, f"{name}_run.json"), "w") as f:
        json.dump({"name": name, "final_val_loss": 1.0,
                   "checkpoint": checkpoint}, f)
    return checkpoint


def test_force_retrains_a_cached_run(tmp_path, monkeypatch):
    import scripts.run_task4_questions as t4

    args = make_args(tmp_path)
    write_cached_run(args)

    assert load_run(args, "already") is not None
    args.force = True
    assert load_run(args, "already") is None


def test_a_run_whose_checkpoint_is_gone_is_not_cached(tmp_path):
    """Deleting checkpoints to reclaim disk must not leave records that skip
    the training and then fail when the analysis loads the weights."""
    args = make_args(tmp_path)
    checkpoint = write_cached_run(args)

    assert load_run(args, "already") is not None
    os.remove(checkpoint)
    assert load_run(args, "already") is None
