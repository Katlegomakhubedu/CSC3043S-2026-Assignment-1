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
    params:
        model, tokenizer: the trained model and its tokenizer
        prompt:           starting string
        max_new_tokens:   maximum number of tokens to generate
        temperature:      sampling temperature; 0 means greedy decoding
        seed:             optional seed for reproducible sampling
    returns:
        the generated string, including the prompt
    """
    model.eval()
    context_length = model.config.context_length
    eot_id = tokenizer.token_to_id[END_OF_TEXT]

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    token_ids = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        # Feed the whole context so far, truncated to what the model can handle
        model_input = torch.tensor([token_ids[-context_length:]], dtype=torch.long, device=device)
        logits = model(model_input)

        # Step 1: take the logits at the FINAL position - that is the next-token prediction
        next_logits = logits[0, -1, :].float()

        # Step 2: sample (or take the argmax if temperature is 0)
        if temperature == 0:
            next_id = int(next_logits.argmax())
        else:
            probs = F.softmax(next_logits / temperature, dim=-1).cpu()
            next_id = int(torch.multinomial(probs, num_samples=1, generator=generator))

        # Step 3: append it and stop if it ends the document
        token_ids.append(next_id)
        if next_id == eot_id:
            break

    return tokenizer.decode(token_ids)