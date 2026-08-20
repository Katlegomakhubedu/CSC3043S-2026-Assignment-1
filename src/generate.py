import torch
import torch.nn.functional as F

# No set_seed() at import: seeding is `generate(seed=...)`'s job, per call.
# Doing it here made merely importing this module reseed the caller's RNG and
# switch torch into deterministic-algorithms mode as a side effect.

END_OF_TEXT = "<|endoftext|>"


def truncate_distribution(probs, top_k=None, top_p=None):
    """Apply top-k and/or top-p truncation to `probs` and renormalise (§4.3).

    Both keep a subset of the distribution and zero the rest; applied together,
    top-k runs first and top-p then narrows what survives.

    params:
        probs: (vocab_size,) probabilities, summing to 1
        top_k: keep only the k most probable tokens
        top_p: keep the smallest set whose cumulative probability reaches top_p
    returns:
        a renormalised distribution over the kept tokens

    Separate from `generate` so the cutoff can be tested against a distribution
    whose answer is known by hand. The top-p boundary in particular is easy to
    get wrong by one either way: too few tokens and a distribution whose prefix
    sums land exactly on top_p loses its last legitimate token; too many and
    top_p=0.9 quietly behaves like 0.95.
    """
    truncated = False

    if top_k is not None and top_k > 0:
        top_k_vals, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
        probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_vals)
        truncated = True

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        # The smallest prefix whose cumulative probability reaches top_p: the
        # leftmost index that gets there, plus one to make it a count.
        # right=False keeps a prefix sum landing exactly on top_p from pulling
        # in one extra token.
        cutoff = torch.searchsorted(
            cum_probs, torch.tensor(top_p, device=cum_probs.device), right=False) + 1
        keep_indices = sorted_indices[:cutoff]      # always at least one token
        mask = torch.zeros_like(probs).scatter_(-1, keep_indices, 1.0)
        probs = probs * mask
        truncated = True

    if truncated:
        probs = probs / probs.sum()
    return probs

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 1.0, top_k: int | None = None,
            top_p: float | None = None, seed: int | None = None, use_cache: bool = True) -> str:
    """Autoregressively generate a continuation of `prompt` (§4.3).

    temperature == 0 means greedy decoding. Generation stops at <|endoftext|>,
    at max_new_tokens, or - on the cached path - when the fixed-size KV buffer
    reaches model.config.context_length (§4.2 permits stopping there).
    """
    model.eval()
    device = next(model.parameters()).device
    context_length = model.config.context_length
    # token_to_id is keyed by token bytes, not str.
    eot_id = tokenizer.token_to_id[END_OF_TEXT.encode("utf-8")]

    # Clear the caches so a previous generate() call's state cannot leak in.
    if use_cache:
        for layer in model.layers:
            layer.attn.reset_cache()

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("generate() requires a non-empty prompt: there must be at least "
                          "one token to seed the (KV-cached) decode loop.")

    # 1. Prefill: the whole prompt in one forward pass. Its last-position logits
    # already predict the first generated token, so keep them (`pending_logits`);
    # re-feeding the last prompt token instead would duplicate an already-cached
    # token at a fresh RoPE position and corrupt the cache.
    prompt_tail = token_ids[-context_length:]
    cache_len = 0
    pending_logits = None
    if use_cache:
        prompt_input = torch.tensor([prompt_tail], dtype=torch.long, device=device)
        logits = model(prompt_input, use_cache=True)
        pending_logits = logits[0, -1, :].float()
        cache_len = len(prompt_tail)

    # 2. Decode loop
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    for _ in range(max_new_tokens):
        if use_cache:
            if pending_logits is not None:
                # First step: reuse the prefill's own prediction rather than
                # running another forward pass.
                next_logits = pending_logits
                pending_logits = None
            else:
                # The cache is a fixed-size buffer; stop on reaching it rather
                # than overflowing (§4.2 allows this).
                if cache_len >= context_length:
                    break
                input_tensor = torch.tensor([[token_ids[-1]]], dtype=torch.long, device=device)
                logits = model(input_tensor, use_cache=True)
                next_logits = logits[0, -1, :].float()
                cache_len += 1
        else:
            input_tensor = torch.tensor([token_ids[-context_length:]], dtype=torch.long, device=device)
            logits = model(input_tensor, use_cache=False)
            next_logits = logits[0, -1, :].float()  # (vocab_size,)

        # --- Temperature ---
        if temperature == 0:
            next_id = int(next_logits.argmax())
        else:
            scaled_logits = next_logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)

            probs = truncate_distribution(probs, top_k=top_k, top_p=top_p)

            # Sample
            probs_cpu = probs.cpu()
            next_id = int(torch.multinomial(probs_cpu, num_samples=1, generator=generator))

        token_ids.append(next_id)
        if next_id == eot_id:
            break

    return tokenizer.decode(token_ids)
