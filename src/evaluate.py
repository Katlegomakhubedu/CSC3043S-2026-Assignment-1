
import torch
import numpy as np

def evaluate(model, token_ids, batch_size, context_length, device, total_chars):
    """
    Calculates and returns the evaluation metrics for the model.
    """
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(0, len(token_ids) - context_length, context_length):
            x = torch.tensor(token_ids[i:i+context_length], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(token_ids[i+1:i+context_length+1], dtype=torch.long, device=device).unsqueeze(0)
            
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            val_losses.append(loss.item())
            
    avg_loss = np.mean(val_losses)
    perplexity = np.exp(avg_loss)
    bpc = avg_loss / np.log(2) * (len(token_ids) / total_chars)
    
    model.train()
    return avg_loss, perplexity, bpc

def evaluate_by_position(model, token_ids, batch_size, context_length, device):
    """
    Calculates and returns the average loss for each position in the context window.
    """
    model.eval()
    positional_losses = [[] for _ in range(context_length)]
    with torch.no_grad():
        for i in range(0, len(token_ids) - context_length, context_length):
            x = torch.tensor(token_ids[i:i+context_length], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(token_ids[i+1:i+context_length+1], dtype=torch.long, device=device).unsqueeze(0)
            
            logits = model(x)
            for pos in range(context_length):
                loss = torch.nn.functional.cross_entropy(logits[:, pos, :], y[:, pos])
                positional_losses[pos].append(loss.item())
    
    avg_positional_losses = [np.mean(losses) for losses in positional_losses]
    model.train()
    return avg_positional_losses
