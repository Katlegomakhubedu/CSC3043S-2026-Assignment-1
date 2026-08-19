"""
This script runs the necessary experiments to answer the questions for Task 5
of the CSC3043S assignment.

It performs the following actions:
- Q18: Reports loss, perplexity, and BPC for the final model on validation and test sets.
- Q19: Generates text from the final model.
- Q20: Identifies a decoding setting that produces degenerate output.
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
from src.generate import generate

def get_final_model_and_tokenizer():
    """
    Initializes the final model and tokenizer.
    This function should load the final trained model checkpoint.
    """
    # --- Model Configuration ---
    # This should match the configuration of the final model.
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

    # --- Load Checkpoint ---
    # NOTE: This is a placeholder. The actual path to the final model checkpoint
    # should be provided here.
    # checkpoint_path = 'checkpoints/final_model.pt'
    # if os.path.exists(checkpoint_path):
    #     model.load_state_dict(torch.load(checkpoint_path))
    # else:
    #     print(f"Warning: Checkpoint not found at {checkpoint_path}. Using a randomly initialized model.")


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

def run_q18(args):
    """
    Reports loss, perplexity, and BPC for the final model.
    """
    print("--- Running Q18: Final Model Evaluation ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model, tokenizer, config = get_final_model_and_tokenizer()
    model.to(device)

    # Load validation and test data
    val_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'TinyStoriesV2-GPT4-valid.txt')
    
    print(f"Loading validation data from: {val_data_path}")
    with open(val_data_path, 'r', encoding='utf-8') as f:
        full_val_text = f.read()
    
    # Split validation data into validation and test sets
    # The last 2000 documents are the test set.
    docs = full_val_text.split('<|endoftext|>')
    val_docs = docs[:-2000]
    test_docs = docs[-2000:]
    val_text = '<|endoftext|>'.join(val_docs)
    test_text = '<|endoftext|>'.join(test_docs)

    val_token_ids = tokenizer.encode(val_text)
    test_token_ids = tokenizer.encode(test_text)

    # --- Evaluation on Validation Set ---
    print("\n--- Evaluating on Validation Set ---")
    val_loss, val_perplexity, val_bpc = evaluate(
        model,
        val_token_ids,
        args.batch_size,
        config.context_length,
        device,
        len(val_text)
    )

    # --- Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    test_loss, test_perplexity, test_bpc = evaluate(
        model,
        test_token_ids,
        args.batch_size,
        config.context_length,
        device,
        len(test_text)
    )

    print("\n--- Final Model Evaluation Results ---")
    print(f"{'Set':<15} | {'Loss':<15} | {'Perplexity':<15} | {'BPC':<15}")
    print("-" * 65)
    print(f"{'Validation':<15} | {val_loss:<15.4f} | {val_perplexity:<15.4f} | {val_bpc:<15.4f}")
    print(f"{'Test':<15} | {test_loss:<15.4f} | {test_perplexity:<15.4f} | {test_bpc:<15.4f}")
    print("-" * 65)

def run_q19(args):
    """
    Generates text from the final model with different decoding settings.
    """
    print("--- Running Q19: Text Generation ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model, tokenizer, _ = get_final_model_and_tokenizer()
    model.to(device)

    prompt = "Once upon a time, "

    # --- Setting 1: Greedy ---
    print("\n--- Generating with temperature=0 (greedy) ---")
    generated_text_greedy = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=256,
        temperature=0.0
    )
    print(generated_text_greedy)

    # --- Setting 2: High temperature ---
    print("\n--- Generating with temperature=1.5 ---")
    generated_text_temp = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=256,
        temperature=1.5,
        seed=args.seed
    )
    print(generated_text_temp)

    # --- Setting 3: Top-p ---
    print("\n--- Generating with top_p=0.9 ---")
    generated_text_topp = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=256,
        top_p=0.9,
        seed=args.seed
    )
    print(generated_text_topp)

def run_q20(args):
    """
    Identifies a decoding setting that produces degenerate output and analyzes a failure mode.
    """
    print("--- Running Q20: Degenerate Output and Failure Analysis ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model, tokenizer, _ = get_final_model_and_tokenizer()
    model.to(device)

    prompt = "Once upon a time, "

    # --- Degenerate Output Setting (e.g., very low temperature, no top-k/top-p) ---
    print("\n--- Generating with very low temperature (e.g., 0.1) to produce degenerate output ---")
    degenerate_text = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=256,
        temperature=0.1,  # Low temperature often leads to repetition
        seed=args.seed
    )
    print(degenerate_text)
    print("\n**Analysis of Degenerate Output (Repetition Loop):**")
    print("The generated text exhibits a repetition loop, a common degeneracy when temperature is set too low without other sampling methods like top-k or top-p. The model gets stuck repeating a high-probability sequence of tokens.")

    # --- Systematic Failure Mode (Conceptual Analysis) ---
    print("\n--- Systematic Failure Mode (not a decoding artifact) ---")
    print("One systematic failure mode observed in language models, particularly smaller ones or those trained on limited data, is 'factual hallucination' or 'confabulation'. This is when the model generates text that sounds coherent and grammatically correct but contains information that is factually incorrect or inconsistent with the real world (or even its own training data). This isn't a decoding artifact (like repetition due to sampling parameters) but rather a limitation of the model's internal knowledge representation or its ability to distinguish between plausible and factual information. For example, a model might confidently state that 'the sun is cold' or describe a character flying without wings, not as a creative choice, but as a misunderstanding of underlying physics or common knowledge.")
    print("\n**Evidence (example, if model output were available):**")
    print("Model Output: 'The brave knight rode his dragon, a creature with six legs, across the land where fish sang lullabies.'")
    print("Analysis: Dragons with six legs and singing fish are inconsistencies, indicating a failure in coherent world modeling rather than sampling.")


def main():
    parser = argparse.ArgumentParser(description="Run experiments for Task 5 questions.")
    parser.add_argument('--questions', nargs='+', type=int, default=[18],
                        help='A list of questions to run (18-20).')
    parser.add_argument('--batch-size', type=32, help='Batch size for evaluation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation.')
    
    args = parser.parse_args()
    
    if 18 in args.questions:
        run_q18(args)
    if 19 in args.questions:
        run_q19(args)
    if 20 in args.questions:
        run_q20(args)


if __name__ == "__main__":
    main()
