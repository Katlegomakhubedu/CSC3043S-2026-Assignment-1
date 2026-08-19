"""§6 evaluation metrics: perplexity and bits-per-character, plus the
position-wise breakdown Q17 asks for.

§6 fixes the procedure: score whole **non-overlapping** windows of
`context_length`, and be consistent about it across every model being compared.
Overlapping or sliding windows would give each token more left-context and a
lower loss, so a model evaluated that way cannot be compared against one that
was not.

Perplexity is exp(mean cross-entropy in nats). BPC divides the same total by the
number of *characters* rather than tokens, which is what makes two models with
different vocabularies comparable at all - a 1,000-token model and a 4,000-token
model are not even counting the same events, so their perplexities are not on
the same scale.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


def n_windows(n_tokens, context_length):
    """How many whole non-overlapping windows fit, leaving room for the shift.

    Each window needs context_length + 1 tokens: the inputs, plus one more so
    the last target exists.
    """
    return max(0, (n_tokens - 1) // context_length)


def _window_batches(token_ids, context_length, batch_size, limit=None):
    """Yield (inputs, targets) int64 arrays over non-overlapping windows."""
    total = n_windows(len(token_ids), context_length)
    if limit is not None:
        total = min(total, limit)

    for start in range(0, total, batch_size):
        idx = range(start, min(start + batch_size, total))
        # np.asarray forces the memmap slice into memory one batch at a time,
        # which is the point - the full array never is.
        rows = np.stack([np.asarray(token_ids[i * context_length:
                                              i * context_length + context_length + 1])
                         for i in idx])
        yield rows[:, :-1].astype(np.int64), rows[:, 1:].astype(np.int64)


@torch.no_grad()
def evaluate(model, token_ids, batch_size, context_length, device,
             total_chars=None, amp_enabled=False, max_windows=None):
    """Mean loss, perplexity and BPC over non-overlapping windows (§6).

    params:
        token_ids:   1-D token array (memory-mapped is fine)
        total_chars: characters in the text these tokens encode. Required for
                     BPC; see `chars_from_meta`. Scaled to the portion actually
                     scored, since the trailing partial window is skipped.
        max_windows: cap for a quick estimate; None scores everything.
    returns:
        dict with loss, perplexity, bpc, n_tokens, n_windows, chars_used
    """
    was_training = model.training
    model.eval()

    total_loss, total_tokens, windows = 0.0, 0, 0
    for x, y in _window_batches(token_ids, context_length, batch_size, max_windows):
        xb = torch.from_numpy(x).to(device)
        yb = torch.from_numpy(y).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled):
            logits = model(xb)
            # sum, not mean: batches at the end may be smaller, and a mean of
            # means would then weight the last batch too heavily.
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   yb.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += yb.numel()
        windows += xb.size(0)

    if was_training:
        model.train()

    if total_tokens == 0:
        raise ValueError(
            f"no complete window of {context_length} tokens in an array of "
            f"{len(token_ids)} - nothing to evaluate")

    mean_loss = total_loss / total_tokens
    result = {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "n_tokens": total_tokens,
        "n_windows": windows,
        "context_length": context_length,
    }

    if total_chars is not None:
        # §6: count C over exactly the text evaluated. The trailing partial
        # window is skipped, so the character count is scaled by the fraction of
        # tokens actually scored rather than taken over the whole file.
        chars_used = total_chars * (total_tokens / len(token_ids))
        result["bpc"] = total_loss / (chars_used * math.log(2))
        result["chars_used"] = chars_used
        result["chars_per_token"] = chars_used / total_tokens

    return result


@torch.no_grad()
def evaluate_by_position(model, token_ids, batch_size, context_length, device,
                         n_buckets=8, amp_enabled=False, max_windows=None):
    """Mean loss at each position in the context window, and bucketed (Q17).

    Computed with reduction='none' and summed into a per-position accumulator,
    so one pass over the data gives all `context_length` positions. Scoring each
    position with its own cross_entropy call instead - the previous approach -
    costs a Python loop of context_length calls per window.

    returns:
        dict with per_position (length context_length) and buckets, a list of
        {start, end, loss} covering the window in n_buckets equal spans.
    """
    was_training = model.training
    model.eval()

    position_loss = torch.zeros(context_length, dtype=torch.float64, device=device)
    windows = 0
    for x, y in _window_batches(token_ids, context_length, batch_size, max_windows):
        xb = torch.from_numpy(x).to(device)
        yb = torch.from_numpy(y).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled):
            logits = model(xb)
        per_token = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                    yb.reshape(-1), reduction="none")
        position_loss += per_token.view(yb.shape).sum(dim=0).double()
        windows += xb.size(0)

    if was_training:
        model.train()

    if windows == 0:
        raise ValueError("no complete window to evaluate")

    per_position = (position_loss / windows).tolist()

    # §7.5's Q17 asks for buckets (0-31, 32-63, ...) rather than 256 noisy points.
    width = context_length // n_buckets
    buckets = [{"start": b * width,
                "end": b * width + width - 1,
                "loss": float(np.mean(per_position[b * width:(b + 1) * width]))}
               for b in range(n_buckets)]

    return {"per_position": per_position, "buckets": buckets, "n_windows": windows}


def chars_from_meta(npy_path, include_delimiters=False):
    """Character count for an encoded array, from the sidecar Task 1 wrote.

    §6 requires stating whether the <|endoftext|> delimiters are counted. The
    default here is to exclude them - they are tokenizer bookkeeping rather than
    text - and the choice is returned so it can be reported.

    returns:
        (n_chars, convention) or (None, reason) when no sidecar exists.
    """
    meta_path = os.path.splitext(npy_path)[0] + "_meta.json"
    if not os.path.exists(meta_path):
        return None, f"no sidecar at {os.path.basename(meta_path)}"

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    key = ("n_chars_including_delimiters" if include_delimiters
           else "n_chars_excluding_delimiters")
    if key not in meta:
        return None, f"{os.path.basename(meta_path)} has no {key}"
    return meta[key], ("including <|endoftext|>" if include_delimiters
                       else "excluding <|endoftext|>")
