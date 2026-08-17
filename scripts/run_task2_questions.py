"""
This script runs the necessary experiments to answer the questions for Task 2
of the CSC3043S assignment.

It performs the following actions:
- Q5: Calculates and prints the parameter counts for the model with both
      SwiGLU and ReLU FFN variants.
- Q6: Verifies the correctness of the KV cache by comparing the outputs and
      logits of cached and uncached generation.
- Q7: Measures and plots the generation throughput (tokens/second) with and
      without the KV cache, and reports the speedup.
"""
import sys
import os
import time
import argparse
import torch
import matplotlib.pyplot as plt

# Add the root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TransformerLM, TransformerConfig
from src.tokenizer import BPETokenizer
from src.generate import generate

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

def run_q5():
    """
    Calculates and prints the parameter counts for SwiGLU and ReLU FFN variants.
    """
    print("--- Running Q5: Model Parameter Count ---")

    # SwiGLU configuration (from assignment section 4.1)
    swiglu_config = TransformerConfig(vocab_size=4000, context_length=256, n_layers=4, d_model=512, n_heads=8, d_ff=1344, ffn_type='swiglu')
    swiglu_model = TransformerLM(swiglu_config)

    # ReLU configuration (from assignment section 7.2)
    relu_config = TransformerConfig(vocab_size=4000, context_length=256, n_layers=4, d_model=512, n_heads=8, d_ff=2048, ffn_type='relu')
    relu_model = TransformerLM(relu_config)

    def get_params(model):
        total = model.num_parameters()
        embedding = model.token_embeddings.weight.numel()
        lm_head = model.lm_head.weight.numel()
        non_embedding = total - embedding - lm_head
        return total, embedding, lm_head, non_embedding

    swiglu_params = get_params(swiglu_model)
    relu_params = get_params(relu_model)

    print("\n[Q5 Result] Parameter Count Comparison:")
    print(f"{'Parameter':<20} {'SwiGLU (d_ff=1344)':<25} {'ReLU (d_ff=2048)':<20}")
    print("-" * 65)
    print(f"{'Embedding':<20} {swiglu_params[1]:<25,} {relu_params[1]:<20,}")
    print(f"{'LM Head':<20} {swiglu_params[2]:<25,} {relu_params[2]:<20,}")
    print(f"{'Non-embedding':<20} {swiglu_params[3]:<25,} {relu_params[3]:<20,}")
    print("-" * 65)
    print(f"{'Total':<20} {swiglu_params[0]:<25,} {relu_params[0]:<20,}")
    print("\nNote: SwiGLU uses 3 weight matrices in the FFN, while ReLU uses 2.")


def run_q6(model, tokenizer):
    """
    Verifies the correctness of the KV cache.
    """
    print("\n--- Running Q6: KV Cache Correctness ---")
    prompt = "The quick brown fox jumps over the lazy dog"
    
    # Generate with cache
    out_cached = generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0, use_cache=True, seed=42)
    
    # Generate without cache
    out_uncached = generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.0, use_cache=False, seed=42)

    # To get logits, we have to manually run the model
    with torch.no_grad():
        # Uncached logits
        uncached_ids = tokenizer.encode(prompt)
        uncached_logits_list = []
        for i in range(50):
            input_ids = torch.tensor([uncached_ids[-model.config.context_length:]])
            logits = model(input_ids, use_cache=False)
            next_id = logits[0, -1, :].argmax().item()
            uncached_logits_list.append(logits[0, -1, :])
            uncached_ids.append(next_id)

        # Cached logits
        model.layers[0].attn.reset_cache()
        cached_ids = tokenizer.encode(prompt)
        cached_logits_list = []
        input_ids = torch.tensor([cached_ids])
        logits = model(input_ids, use_cache=True) # Prefill
        
        # First token from prefill
        next_id = logits[0, -1, :].argmax().item()
        cached_logits_list.append(logits[0, -1, :])
        cached_ids.append(next_id)

        for i in range(49):
            input_ids = torch.tensor([[next_id]])
            logits = model(input_ids, use_cache=True)
            next_id = logits[0, -1, :].argmax().item()
            cached_logits_list.append(logits[0, -1, :])
            cached_ids.append(next_id)

    max_abs_diff = 0
    for l1, l2 in zip(uncached_logits_list, cached_logits_list):
        diff = (l1 - l2).abs().max().item()
        if diff > max_abs_diff:
            max_abs_diff = diff

    print("\n[Q6 Result] KV Cache Correctness Check:")
    print(f"  Greedy generations identical: {out_cached == out_uncached}")
    print(f"  Maximum absolute logit difference: {max_abs_diff:.6f}")
    
def run_q7(model, tokenizer):
    """
    Measures and plots the generation throughput.
    """
    print("\n--- Running Q7: Generation Throughput ---")
    prompt = "Once upon a time"
    token_counts = [16, 32, 64, 128, 256]
    
    throughputs_cached = []
    throughputs_uncached = []

    for count in token_counts:
        print(f"  Benchmarking with max_new_tokens = {count}...")
        
        # Cached
        start_time = time.time()
        generate(model, tokenizer, prompt, max_new_tokens=count, use_cache=True, temperature=0.0)
        end_time = time.time()
        throughputs_cached.append(count / (end_time - start_time))

        # Uncached
        start_time = time.time()
        generate(model, tokenizer, prompt, max_new_tokens=count, use_cache=False, temperature=0.0)
        end_time = time.time()
        throughputs_uncached.append(count / (end_time - start_time))

    # Plotting
    plt.figure()
    plt.plot(token_counts, throughputs_cached, 'o-', label='With KV Cache')
    plt.plot(token_counts, throughputs_uncached, 'o-', label='Without KV Cache')
    plt.title('Generation Throughput vs. Number of Tokens')
    plt.xlabel('Number of Tokens Generated')
    plt.ylabel('Throughput (tokens/second)')
    plt.legend()
    plt.grid(True)
    
    plot_path = 'task2_q7_throughput.png'
    plt.savefig(plot_path)
    print(f"\n[Q7 Result] Plot saved to '{plot_path}'")

    speedup = throughputs_cached[-1] / throughputs_uncached[-1]
    print(f"[Q7 Result] Speedup at 256 tokens: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Run experiments for Task 2 questions.")
    parser.add_argument('--questions', nargs='+', type=int, default=[5, 6, 7],
                        help='A list of questions to run (5, 6, 7).')
    args = parser.parse_args()

    model, tokenizer, config = get_model_and_tokenizer()
    
    if 5 in args.questions:
        run_q5()
    
    if 6 in args.questions:
        run_q6(model, tokenizer)

    if 7 in args.questions:
        run_q7(model, tokenizer)


if __name__ == "__main__":
    main()
