import torch
import torch.nn.functional as F
from .seed import set_seed

set_seed(42)

END_OF_TEXT = "<|endoftext|>"

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 1.0, top_k: int | None = None,
            top_p: float | None = None, seed: int | None = None, use_cache: bool = True) -> str:
    """Autoregressively generate a continuation of `prompt`.

    temperature == 0 means greedy decoding.

    If use_cache, generation stops once the KV cache reaches model.config.context_length
    (a fixed-size buffer), rather than raising - see the context-length guard below.
    """
    model.eval()
    device = next(model.parameters()).device
    context_length = model.config.context_length
    # token_to_id is keyed by the raw token bytes (see BPETokenizer.encode), not str.
    eot_id = tokenizer.token_to_id[END_OF_TEXT.encode("utf-8")]

    # Reset caches if using them, so a previous generate() call's leftovers can't leak in.
    if use_cache:
        for layer in model.layers:
            layer.attn.reset_cache()

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("generate() requires a non-empty prompt: there must be at least "
                          "one token to seed the (KV-cached) decode loop.")

    # 1. Prefill: process the entire prompt in one forward pass to fill the cache.
    # Its last-position logits already predict the first token to generate - keep
    # them (`pending_logits`) instead of discarding them. The old code discarded
    # this output and then re-fed the *last prompt token* as the first decode
    # step's input: that duplicated an already-cached token under a fresh RoPE
    # position, corrupting the cache for every step after it.
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
                # First iteration: reuse the prefill's own prediction, no new
                # forward pass (and no new token has been fed to the cache yet).
                next_logits = pending_logits
                pending_logits = None
            else:
                # Cached decode steps hit a fixed-size buffer of length
                # context_length - stop cleanly rather than overflowing it
                # (documented in §4.2: "simply stopping generation there is
                # acceptable, as long as it is documented").
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

            # --- Top‑k truncation ---
            if top_k is not None and top_k > 0:
                # Keep only the top_k tokens, set others to 0
                top_k_vals, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
                probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_vals)

            # --- Top‑p (nucleus) truncation ---
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                # Smallest k such that the top-k cumulative probability is >= top_p:
                # the leftmost index whose cumulative prob reaches top_p, plus one
                # (searchsorted's left/right=False side finds that leftmost index;
                # right=True would over-include one extra token whenever a prefix
                # sum lands exactly on top_p).
                cutoff = torch.searchsorted(cum_probs, torch.tensor(top_p, device=cum_probs.device), right=False) + 1
                # Keep only the tokens up to cutoff (at least one)
                keep_indices = sorted_indices[:cutoff]
                # Zero out all others
                mask = torch.zeros_like(probs).scatter_(-1, keep_indices, 1.0)
                probs = probs * mask

            # --- Renormalise (if any truncation was applied) ---
            if top_k is not None or (top_p is not None and 0.0 < top_p < 1.0):
                probs = probs / probs.sum()

            # Sample
            probs_cpu = probs.cpu()
            next_id = int(torch.multinomial(probs_cpu, num_samples=1, generator=generator))

        token_ids.append(next_id)
        if next_id == eot_id:
            break

    return tokenizer.decode(token_ids)
