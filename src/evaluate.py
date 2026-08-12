import math
import torch
import torch.nn.functional as F

@torch.no_grad()
def evaluate(model, token_ids, batch_size, context_length, device, total_chars):
    """
    Evaluates the model on a dataset computing Loss, Perplexity, and BPC.
    Evaluates over whole non-overlapping windows of context_length.
    """
    model.eval()
    
    # We need context_length + 1 tokens per sequence to get context_length inputs and targets
    chunk_size = context_length + 1 
    num_chunks = len(token_ids) // chunk_size
    
    total_loss = 0.0
    total_batches = 0
    evaluated_tokens = 0
    
    # Iterate over the data in non-overlapping batches
    for i in range(0, num_chunks, batch_size):
        batch_chunks = []
        for j in range(batch_size):
            if i + j < num_chunks:
                start_idx = (i + j) * chunk_size
                batch_chunks.append(token_ids[start_idx : start_idx + chunk_size])
        
        if not batch_chunks:
            break
            
        # Convert to tensor and slice inputs (x) and targets (y)
        batch_tensor = torch.tensor(batch_chunks, dtype=torch.long, device=device)
        x = batch_tensor[:, :-1]
        y = batch_tensor[:, 1:]
        
        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        
        total_loss += loss.item()
        total_batches += 1
        evaluated_tokens += x.numel() # Add the number of tokens evaluated in this batch

    # Calculate final metrics
    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    
    # Perplexity
    perplexity = math.exp(avg_loss)
    
    # Bits Per Character (BPC)
    # BPC = (N * L) / (C * ln(2))
    bpc = (evaluated_tokens * avg_loss) / (total_chars * math.log(2)) if total_chars > 0 else 0.0
    
    # Return model to training mode
    model.train()
    
    return avg_loss, perplexity, bpc