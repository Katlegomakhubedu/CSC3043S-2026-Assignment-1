import torch
import torch.nn.functional as F
from tokenizer import BPETokenizer
from .seed import set_seed

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

END_OF_TEXT = "<|endoftext|>"

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 1.0, top_k: int | None = None,
            top_p: float | None = None, seed: int | None = None, use_cache: bool = True) -> str:
    """
    Autoregressively generate a continuation of `prompt`.
    """
    model.eval()
    context_length = model.config.context_length
    eot_id = tokenizer.token_to_id[END_OF_TEXT]

    # Reset caches if using them
    if use_cache:
        for layer in model.transformer_blocks:
            layer.attn.k_cache = None
            layer.attn.v_cache = None
            layer.attn.cache_len = 0

    token_ids = tokenizer.encode(prompt)

    # 1. Prefill: process the entire prompt to fill the cache
    if use_cache and len(token_ids) > 0:
        prompt_input = torch.tensor([token_ids[-context_length:]],
                                    dtype=torch.long, device=device)
        _ = model(prompt_input, use_cache=True)

    # 2. Decode loop
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    for _ in range(max_new_tokens):
        if use_cache:
            if len(token_ids) == 1:
                input_tensor = torch.tensor([[token_ids[0]]], dtype=torch.long, device=device)
            else:
                input_tensor = torch.tensor([[token_ids[-1]]], dtype=torch.long, device=device)
        else:
            input_tensor = torch.tensor([token_ids[-context_length:]],
                                        dtype=torch.long, device=device)

        logits = model(input_tensor, use_cache=use_cache)
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
                # Find the index where cumulative probability exceeds top_p
                cutoff = torch.searchsorted(cum_probs, top_p, right=True) + 1
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