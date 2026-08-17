"""
This script provides a command-line interface for evaluating a trained Transformer
language model.

It calculates and prints the evaluation metrics (Loss, Perplexity, BPC)
for the model on the validation set.
"""
import sys
import os
import argparse
import torch
import numpy as np

# Add the root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TransformerLM, TransformerConfig
from src.tokenizer import BPETokenizer
from src.evaluate import evaluate


def get_model_and_tokenizer():
    """Initializes the model and tokenizer."""
    # --- Model Configuration ---
    # Using the base configuration from the assignment PDF (Section 4.1)
    config = TransformerConfig(
        vocab_size=4000,
        context_length=256,
        n_layers=4,
        d_model=512,
        n_heads=8,
        d_ff=1344,
        rope_theta=10000.0,
        use_qk_norm=True
    )
    model = TransformerLM(config)

    # --- Tokenizer ---
    # Load the pre-trained tokenizer if available, otherwise use a dummy tokenizer
    try:
        vocab_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vocab.pkl')
        merges_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'merges.pkl')
        tokenizer = BPETokenizer.from_files(vocab_path, merges_path)
    except FileNotFoundError:
        print("Warning: vocab.pkl or merges.pkl not found. Using a dummy byte-level tokenizer.")
        vocab = {i: str(i).encode() for i in range(config.vocab_size)}
        merges = []
        tokenizer = BPETokenizer(vocab, merges)

    return model, tokenizer, config

def run_evaluation(args):
    """
    Calculates and prints the evaluation metrics for the model.
    """
    print("--- Running Model Evaluation ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model, tokenizer, config = get_model_and_tokenizer()
    model.to(device)

    # Load validation data
    val_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'TinyStoriesV2-GPT4-valid.txt')
    
    print(f"Loading validation data from: {val_data_path}")
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_text = f.read()
    
    total_chars = len(val_text)
    
    # Tokenize the validation data
    print("Tokenizing validation data...")
    val_token_ids = tokenizer.encode(val_text)
    
    # Run evaluation
    print("Running evaluation...")
    avg_loss, perplexity, bpc = evaluate(
        model,
        val_token_ids,
        batch_size=args.batch_size,
        context_length=config.context_length,
        device=device,
        total_chars=total_chars
    )

    print("\n[Result] Model Evaluation Metrics:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Perplexity:     {perplexity:.4f}")
    print(f"  Bits Per Character (BPC): {bpc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation.")
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation.')
    
    args = parser.parse_args()
    
    run_evaluation(args)


if __name__ == "__main__":
    main()

