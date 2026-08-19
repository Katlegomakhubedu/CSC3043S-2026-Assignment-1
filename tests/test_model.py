"""
Model sanity checks:
 - forward pass runs (regression test for the TransformerBlock/use_cache crash)
 - output shape and finiteness
 - parameter count breakdown (embedding / lm_head / non-embedding), per §4.4 Q5
 - SwiGLU vs ReLU FFN variants have matching total parameter counts, per §4.4 Q5
 - config ablations (use_rmsnorm, use_rope, ffn_type) all build and run
 - causal masking: changing a future token must not change earlier logits
"""
import torch
import pytest

from src.model import TransformerLM, TransformerConfig


def make_config(**overrides):
    cfg = dict(vocab_size=64, context_length=16, n_layers=2, d_model=32, n_heads=4, d_ff=64)
    cfg.update(overrides)
    return TransformerConfig(**cfg)


def test_forward_pass_runs_and_shape_is_correct():
    torch.manual_seed(0)
    model = TransformerLM(make_config())
    x = torch.randint(0, 64, (2, 10))
    logits = model(x)
    assert logits.shape == (2, 10, 64)
    assert torch.isfinite(logits).all()


def test_forward_pass_with_use_cache_flag_does_not_crash():
    # Regression test: TransformerBlock.forward used to not accept `use_cache`,
    # so ANY call to model(x, use_cache=...) - including the default
    # use_cache=False - raised TypeError.
    torch.manual_seed(0)
    model = TransformerLM(make_config())
    x = torch.randint(0, 64, (1, 5))
    _ = model(x, use_cache=False)
    _ = model(x, use_cache=True)


@pytest.mark.parametrize("overrides", [
    {"use_rmsnorm": False},
    {"use_rope": False},
    {"use_qk_norm": False},
    {"ffn_type": "relu"},
    {"use_rmsnorm": False, "use_rope": False, "ffn_type": "relu"},
])
def test_ablation_configs_build_and_run(overrides):
    torch.manual_seed(0)
    model = TransformerLM(make_config(**overrides))
    x = torch.randint(0, 64, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(logits).all()


def test_parameter_count_breakdown():
    cfg = make_config()
    model = TransformerLM(cfg)

    embedding_params = model.token_embeddings.weight.numel()
    lm_head_params = model.lm_head.weight.numel()
    total = model.num_parameters(non_embedding=False)
    non_embedding = model.num_parameters(non_embedding=True)

    assert embedding_params == cfg.vocab_size * cfg.d_model
    assert lm_head_params == cfg.vocab_size * cfg.d_model
    assert non_embedding == total - embedding_params - lm_head_params
    assert total == sum(p.numel() for p in model.parameters())


def test_swiglu_and_relu_ffn_variants_match_in_total_parameters():
    """§4.4 Q5 / §7.2: the parameter-matched ReLU ablation.

    §7.2 pairs SwiGLU at d_ff=1344 with ReLU at d_ff = 4 * d_model = 2048, and
    states the parameter counts then "match to within 2%" - three matrices at
    1344 versus two at 2048. So this asserts a 2% band, not equality, and it
    uses the real §4.1 dimensions: the tolerance is a claim about those specific
    numbers and means nothing at the toy sizes make_config uses.
    """
    base = dict(vocab_size=4000, context_length=256, n_layers=4, d_model=512, n_heads=8)
    swiglu_model = TransformerLM(TransformerConfig(**base, d_ff=1344, ffn_type="swiglu"))
    relu_model = TransformerLM(TransformerConfig(**base, d_ff=2048, ffn_type="relu"))

    swiglu_total = swiglu_model.num_parameters()
    relu_total = relu_model.num_parameters()
    relative_difference = abs(relu_total - swiglu_total) / swiglu_total
    assert relative_difference < 0.02, (
        f"SwiGLU {swiglu_total:,} vs ReLU {relu_total:,} differ by "
        f"{relative_difference:.2%}, outside the 2% §7.2 promises")

    # Everything outside the FFN sub-layers must be identical, so the ablation
    # changes the FFN and nothing else.
    swiglu_other = sum(p.numel() for n, p in swiglu_model.named_parameters() if ".ffn." not in n)
    relu_other = sum(p.numel() for n, p in relu_model.named_parameters() if ".ffn." not in n)
    assert swiglu_other == relu_other


def test_d_ff_is_used_as_given_for_both_ffn_types():
    """Regression test: TransformerBlock used to silently rescale d_ff by 1.5
    for ReLU, so a caller asking for §7.2's d_ff=2048 got a 3072-wide FFN."""
    cfg = make_config(ffn_type="relu", d_ff=64)
    model = TransformerLM(cfg)
    w1 = model.layers[0].ffn.w1.weight
    assert w1.shape == (64, cfg.d_model)


def test_causal_masking_future_tokens_do_not_affect_earlier_logits():
    torch.manual_seed(0)
    model = TransformerLM(make_config())
    model.eval()
    x = torch.randint(0, 64, (1, 8))

    with torch.no_grad():
        logits_full = model(x)

    x_perturbed = x.clone()
    x_perturbed[0, -1] = (x_perturbed[0, -1] + 1) % 64  # change only the last token
    with torch.no_grad():
        logits_perturbed = model(x_perturbed)

    # Every position except the last must be unaffected by a change to the last token.
    assert torch.allclose(logits_full[:, :-1, :], logits_perturbed[:, :-1, :], atol=1e-5)
