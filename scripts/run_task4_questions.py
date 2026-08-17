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
import pandas as pd
import matplotlib.pyplot as plt

# Add the root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TransformerLM, TransformerConfig
from src.tokenizer import BPETokenizer
from src.train import train
from src.evaluate import evaluate, evaluate_by_position

def get_model_and_tokenizer(vocab_size=4000, use_rmsnorm=True, use_rope=True, ffn_type='swiglu'):
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
        use_qk_norm=True,
        use_rmsnorm=use_rmsnorm,
        use_rope=use_rope,
        ffn_type=ffn_type
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

    # Load data
    train_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'TinyStoriesV2-GPT4-train.txt')
    val_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'TinyStoriesV2-GPT4-valid.txt')
    
    print(f"Loading training data from: {train_data_path}")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_text = f.read()

    print(f"Loading validation data from: {val_data_path}")
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_text = f.read()

    # Get tokenizer
    _, tokenizer, _ = get_model_and_tokenizer()

    # Tokenize data
    print("Tokenizing training data...")
    train_ids = tokenizer.encode(train_text)
    print("Tokenizing validation data...")
    val_ids = tokenizer.encode(val_text)

    learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    
    for lr in learning_rates:
        print(f"\n--- Training with learning rate: {lr} ---")
        
        model, _, config = get_model_and_tokenizer()
        model.to(device)

        run_name = f"lr_sweep_{lr:.0e}"

        train_args = {
            'num_steps': args.steps,
            'batch_size': args.batch_size,
            'learning_rate': lr,
            'warmup_steps': args.warmup_steps,
            'weight_decay': args.weight_decay,
            'grad_clip': args.grad_clip,
            'eval_every': args.eval_every,
            'save_every': args.save_every,
            'checkpoint_dir': args.checkpoint_dir,
            'log_dir': args.log_dir,
            'run_name': run_name,
            'seed': args.seed,
            'resume_from': args.resume_from,
            'use_amp': args.use_amp,
        }

        train(
            model,
            np.array(train_ids),
            np.array(val_ids),
            **train_args
        )

        print(f"--- Finished training with learning rate: {lr} ---")

def run_q11(args):
    """
    Performs an ablation study on RMSNorm.
    """
    print("--- Running Q11: RMSNorm Ablation Study ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load and tokenize data
    _, tokenizer, _ = get_model_and_tokenizer()
    train_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'TinyStoriesV2-GPT4-train.txt')
    val_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'TinyStoriesV2-GPT4-valid.txt')
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_text = f.read()
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    train_args = vars(args).copy()
    train_args.pop('questions')

    # --- Baseline Model (with RMSNorm) ---
    print("\n--- Training baseline model (with RMSNorm) ---")
    model_baseline, _, _ = get_model_and_tokenizer(use_rmsnorm=True)
    model_baseline.to(device)
    
    train_args['run_name'] = 'rmsnorm_baseline'
    train(model_baseline, np.array(train_ids), np.array(val_ids), **train_args)

    # --- Ablation Model (without RMSNorm) ---
    print("\n--- Training ablation model (without RMSNorm) ---")
    model_ablation, _, _ = get_model_and_tokenizer(use_rmsnorm=False)
    model_ablation.to(device)

    train_args['run_name'] = 'rmsnorm_ablation'
    train(model_ablation, np.array(train_ids), np.array(val_ids), **train_args)

def run_q12(args):
    """
    Performs an ablation study on positional encoding.
    """
    print("--- Running Q12: Positional Encoding Ablation Study ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load and tokenize data
    _, tokenizer, _ = get_model_and_tokenizer()
    train_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'TinyStoriesV2-GPT4-train.txt')
    val_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'TinyStoriesV2-GPT4-valid.txt')
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_text = f.read()
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    train_args = vars(args).copy()
    train_args.pop('questions')

    # --- Baseline Model (with RoPE) ---
    print("\n--- Training baseline model (with RoPE) ---")
    model_baseline, _, _ = get_model_and_tokenizer(use_rope=True)
    model_baseline.to(device)
    
    train_args['run_name'] = 'rope_baseline'
    train(model_baseline, np.array(train_ids), np.array(val_ids), **train_args)

    # --- Ablation Model (without RoPE) ---
    print("\n--- Training ablation model (without RoPE) ---")
    model_ablation, _, _ = get_model_and_tokenizer(use_rope=False)
    model_ablation.to(device)

    train_args['run_name'] = 'rope_ablation'
    train(model_ablation, np.array(train_ids), np.array(val_ids), **train_args)

def run_q13(args):
    """
    Performs an ablation study on SwiGLU vs. ReLU.
    """
    print("--- Running Q13: SwiGLU vs. ReLU Ablation Study ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load and tokenize data
    _, tokenizer, _ = get_model_and_tokenizer()
    train_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'TinyStoriesV2-GPT4-train.txt')
    val_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'TinyStoriesV2-GPT4-valid.txt')
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_text = f.read()
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    train_args = vars(args).copy()
    train_args.pop('questions')

    # --- Baseline Model (with SwiGLU) ---
    print("\n--- Training baseline model (with SwiGLU) ---")
    model_baseline, _, _ = get_model_and_tokenizer(ffn_type='swiglu')
    model_baseline.to(device)
    
    train_args['run_name'] = 'swiglu_baseline'
    train(model_baseline, np.array(train_ids), np.array(val_ids), **train_args)

    # --- Ablation Model (with ReLU) ---
    print("\n--- Training ablation model (with ReLU) ---")
    model_ablation, _, _ = get_model_and_tokenizer(ffn_type='relu')
    model_ablation.to(device)

    train_args['run_name'] = 'relu_ablation'
    train(model_ablation, np.array(train_ids), np.array(val_ids), **train_args)

def run_q14(args):
    """
    Compares the performance gaps from the completed ablations.
    """
    print("--- Running Q14: Ablation Performance Comparison ---")

    log_dir = args.log_dir
    
    # --- Learning Rate Sweep ---
    lr_logs = [f for f in os.listdir(log_dir) if f.startswith('lr_sweep_')]
    lr_losses = {}
    for log in lr_logs:
        try:
            lr = float(log.split('_')[-1].replace('.csv', ''))
            df = pd.read_csv(os.path.join(log_dir, log))
            if not df.empty:
                lr_losses[lr] = df['val_loss'].iloc[-1]
        except (ValueError, IndexError):
            print(f"Could not parse learning rate from log file name: {log}")
            continue

    if len(lr_losses) < 2:
        print("Not enough learning rate sweep logs to compare.")
        lr_gap = 'N/A'
    else:
        sorted_lrs = sorted(lr_losses.items(), key=lambda item: item[1])
        best_lr, best_loss = sorted_lrs[0]
        second_best_lr, second_best_loss = sorted_lrs[1]
        lr_gap = second_best_loss - best_loss
    
    # --- RMSNorm Ablation ---
    try:
        rmsnorm_baseline_df = pd.read_csv(os.path.join(log_dir, 'rmsnorm_baseline.csv'))
        rmsnorm_ablation_df = pd.read_csv(os.path.join(log_dir, 'rmsnorm_ablation.csv'))
        rmsnorm_gap = rmsnorm_ablation_df['val_loss'].iloc[-1] - rmsnorm_baseline_df['val_loss'].iloc[-1]
    except FileNotFoundError:
        rmsnorm_gap = 'N/A'

    # --- RoPE Ablation ---
    try:
        rope_baseline_df = pd.read_csv(os.path.join(log_dir, 'rope_baseline.csv'))
        rope_ablation_df = pd.read_csv(os.path.join(log_dir, 'rope_ablation.csv'))
        rope_gap = rope_ablation_df['val_loss'].iloc[-1] - rope_baseline_df['val_loss'].iloc[-1]
    except FileNotFoundError:
        rope_gap = 'N/A'

    # --- SwiGLU/ReLU Ablation ---
    try:
        swiglu_baseline_df = pd.read_csv(os.path.join(log_dir, 'swiglu_baseline.csv'))
        relu_ablation_df = pd.read_csv(os.path.join(log_dir, 'relu_ablation.csv'))
        swiglu_gap = relu_ablation_df['val_loss'].iloc[-1] - swiglu_baseline_df['val_loss'].iloc[-1]
    except FileNotFoundError:
        swiglu_gap = 'N/A'

    print("\n--- Performance Gaps ---")
    print(f"{'Experiment':<25} | {'Validation Loss Gap':<25}")
    print("-" * 50)
    if isinstance(lr_gap, float):
        print(f"{'Learning Rate (second best)':<25} | {lr_gap:<25.4f}")
    else:
        print(f"{'Learning Rate (second best)':<25} | {'N/A'}")
    if isinstance(rmsnorm_gap, float):
        print(f"{'RMSNorm Ablation':<25} | {rmsnorm_gap:<25.4f}")
    else:
        print(f"{'RMSNorm Ablation':<25} | {'N/A'}")
    if isinstance(rope_gap, float):
        print(f"{'RoPE Ablation':<25} | {rope_gap:<25.4f}")
    else:
        print(f"{'RoPE Ablation':<25} | {'N/A'}")
    if isinstance(swiglu_gap, float):
        print(f"{'SwiGLU/ReLU Ablation':<25} | {swiglu_gap:<25.4f}")
    else:
        print(f"{'SwiGLU/ReLU Ablation':<25} | {'N/A'}")
    print("-" * 50)

def run_q15(args):
    """
    Compares models with two different vocabulary sizes.
    """
    print("--- Running Q15: Vocabulary Size Comparison ---")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Baseline Model (vocab size 4000) ---
    print("\n--- Training baseline model (vocab size 4000) ---")
    model_baseline, tokenizer_baseline, _ = get_model_and_tokenizer(vocab_size=4000)
    model_baseline.to(device)

    # Load and tokenize data
    train_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'TinyStoriesV2-GPT4-train.txt')
    val_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'TinyStoriesV2-GPT4-valid.txt')
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_text = f.read()
    train_ids_baseline = tokenizer_baseline.encode(train_text)
    val_ids_baseline = tokenizer_baseline.encode(val_text)

    train_args = vars(args).copy()
    train_args.pop('questions')
    train_args['run_name'] = 'vocab_4000_baseline'
    
    train(
        model_baseline,
        np.array(train_ids_baseline),
        np.array(val_ids_baseline),
        **train_args
    )

    # --- Ablation Model (vocab size 1000) ---
    print("\n--- Training ablation model (vocab size 1000) ---")
    model_ablation, tokenizer_ablation, _ = get_model_and_tokenizer(vocab_size=1000)
    model_ablation.to(device)

    train_ids_ablation = tokenizer_ablation.encode(train_text)
    val_ids_ablation = tokenizer_ablation.encode(val_text)

    train_args['run_name'] = 'vocab_1000_ablation'

    train(
        model_ablation,
        np.array(train_ids_ablation),
        np.array(val_ids_ablation),
        **train_args
    )

    # --- Evaluation ---
    print("\n--- Evaluating baseline model (vocab size 4000) ---")
    avg_loss_baseline, perplexity_baseline, bpc_baseline = evaluate(
        model_baseline,
        val_ids_baseline,
        batch_size=args.batch_size,
        context_length=model_baseline.config.context_length,
        device=device,
        total_chars=len(val_text)
    )

    print("\n--- Evaluating ablation model (vocab size 1000) ---")
    avg_loss_ablation, perplexity_ablation, bpc_ablation = evaluate(
        model_ablation,
        val_ids_ablation,
        batch_size=args.batch_size,
        context_length=model_ablation.config.context_length,
        device=device,
        total_chars=len(val_text)
    )

    print("\n--- Vocabulary Size Comparison ---")
    print(f"{'Metric':<20} | {'Vocab Size 4000':<20} | {'Vocab Size 1000':<20}")
    print("-" * 66)
    print(f"{'Perplexity':<20} | {perplexity_baseline:<20.4f} | {perplexity_ablation:<20.4f}")
    print(f"{'BPC':<20} | {bpc_baseline:<20.4f} | {bpc_ablation:<20.4f}")
    print(f"{'Total Parameters':<20} | {model_baseline.num_parameters():<20} | {model_ablation.num_parameters():<20}")
    print("-" * 66)

def run_q16(args):
    """
    Reports the GPU-hours used.
    """
    print("--- Running Q16: GPU-Hours Report ---")

    log_dir = args.log_dir
    total_wall_time = 0
    
    for log_file in os.listdir(log_dir):
        if log_file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(log_dir, log_file))
                if 'wall_time' in df.columns and not df.empty:
                    total_wall_time += df['wall_time'].iloc[-1]
            except pd.errors.EmptyDataError:
                print(f"Log file is empty: {log_file}")


    gpu_hours = total_wall_time / 3600
    
    print(f"\n--- Total GPU-Hours Used ---")
    print(f"Total Wall Time: {total_wall_time:.2f} seconds")
    print(f"Total GPU-Hours: {gpu_hours:.4f} hours")

def run_q17(args):
    """
    Generates a position-wise validation loss plot.
    """
    print("--- Running Q17: Position-wise Validation Loss Plot ---")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- RoPE Model ---
    print("\n--- Evaluating RoPE model ---")
    model_rope, tokenizer, config = get_model_and_tokenizer(use_rope=True)
    model_rope.to(device)

    # Load validation data
    val_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'TinyStoriesV2-GPT4-valid.txt')
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_text = f.read()
    val_ids = tokenizer.encode(val_text)
    
    rope_losses = evaluate_by_position(
        model_rope,
        val_ids,
        batch_size=args.batch_size,
        context_length=config.context_length,
        device=device
    )

    # --- NoPE Model ---
    print("\n--- Evaluating NoPE model ---")
    model_nope, _, _ = get_model_and_tokenizer(use_rope=False)
    model_nope.to(device)

    nope_losses = evaluate_by_position(
        model_nope,
        val_ids,
        batch_size=args.batch_size,
        context_length=config.context_length,
        device=device
    )

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(rope_losses, label='RoPE')
    plt.plot(nope_losses, label='NoPE')
    plt.xlabel('Position in Context Window')
    plt.ylabel('Validation Loss')
    plt.title('Position-wise Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('position_wise_loss.png')
    print("\nPlot saved to position_wise_loss.png")

def main():
    parser = argparse.ArgumentParser(description="Run experiments for Task 4 questions.")
    parser.add_argument('--questions', nargs='+', type=int, default=[10],
                        help='A list of questions to run (10-17).')
    parser.add_argument('--steps', type=int, default=1000, help='Number of training steps.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--warmup_steps', type=int, default=200, help='Number of warmup steps.')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay.')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--eval_every', type=int, default=50, help='Evaluate every N steps.')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint every N steps.')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint.')
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', dest='use_amp', action='store_false')
    
    args = parser.parse_args()
    
    if 10 in args.questions:
        run_q10(args)
    if 11 in args.questions:
        run_q11(args)
    if 12 in args.questions:
        run_q12(args)
    if 13 in args.questions:
        run_q13(args)
    if 14 in args.questions:
        run_q14(args)
    if 15 in args.questions:
        run_q15(args)
    if 16 in args.questions:
        run_q16(args)
    if 17 in args.questions:
        run_q17(args)


if __name__ == "__main__":
    main()
