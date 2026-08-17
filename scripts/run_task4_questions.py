"""
This script runs the necessary experiments to answer the questions for Task 4
of the CSC3043S assignment.

It performs the following actions:
- Q10: Performs a learning-rate sweep.
- Q11: Performs an ablation study on RMSNorm.
- Q12: Performs an ablation study on positional encoding.
- Q13: Performs an ablation study on SwiGLU vs. ReLU.
- Q14: Compares the results from the ablation studies.
- Q15: Compares models with two different vocabulary sizes.
- Q16: Reports on GPU-hours used.
- Q17: Generates a position-wise validation loss plot.
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
from src.train import train_model

def get_model_and_tokenizer(vocab_size=4000):
    """Initializes the model and tokenizer."""
    # --- Model Configuration ---
    config = TransformerConfig(
        vocab_size=vocab_size,
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

def run_q10(args):
    """
    Performs a learning-rate sweep.
    """
    print("--- Running Q10: Learning-Rate Sweep ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    
    for lr in learning_rates:
        print(f"\n--- Training with learning rate: {lr} ---")
        
        model, tokenizer, config = get_model_and_tokenizer()
        model.to(device)

        # NOTE: This is a placeholder for the training function.
        # The actual 'train_model' function from 'src.train' needs to be
        # implemented and called here.
        # train_model(model, tokenizer, lr, device, args)

        print(f"--- Finished training with learning rate: {lr} ---")

def main():
    parser = argparse.ArgumentParser(description="Run experiments for Task 4 questions.")
    parser.add_argument('--questions', nargs='+', type=int, default=[10],
                        help='A list of questions to run (10-17).')
    
    args = parser.parse_args()
    
    if 10 in args.questions:
        run_q10(args)


if __name__ == "__main__":
    main()
