"""
This script runs the necessary experiments to answer the questions for Task 3
of the CSC3043S assignment.

This script will:
1.  Generate small dummy data files to run the training on.
2.  Q8: Run short training sessions in fp32 and bf16 to compare step times.
3.  Q9: Run training sessions with and without warmup, then plot the
    validation curves and report gradient norms.
"""
import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import torch

# Add the root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TransformerLM, TransformerConfig
from src.train import train

def prepare_dummy_data(data_dir='data', train_size=100000, val_size=10000, vocab_size=4000):
    """Creates dummy .npy files for training and validation."""
    print("--- Preparing dummy data ---")
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, 'dummy_train.npy')
    val_path = os.path.join(data_dir, 'dummy_valid.npy')

    if not os.path.exists(train_path):
        train_data = np.random.randint(0, vocab_size, size=train_size, dtype=np.uint16)
        np.save(train_path, train_data)
        print(f"Saved dummy training data to {train_path}")

    if not os.path.exists(val_path):
        val_data = np.random.randint(0, vocab_size, size=val_size, dtype=np.uint16)
        np.save(val_path, val_data)
        print(f"Saved dummy validation data to {val_path}")

    return train_path, val_path

def run_q8(model, train_ids, val_ids, base_config):
    """Runs training in fp32 and bf16 to compare step times."""
    print("\n--- Running Q8: Step Time Comparison (fp32 vs bf16) ---")
    num_steps = 100
    
    # Run with bf16
    print("  Running with bf16...")
    train(model, train_ids, val_ids, num_steps=num_steps, **base_config, use_amp=True, run_name="q8_bf16")

    # Run with fp32
    print("\n  Running with fp32...")
    train(model, train_ids, val_ids, num_steps=num_steps, **base_config, use_amp=False, run_name="q8_fp32")
    
    # Parse logs and report
    log_bf16 = pd.read_csv('logs/q8_bf16.csv')
    log_fp32 = pd.read_csv('logs/q8_fp32.csv')

    time_bf16 = log_bf16['wall_time'].iloc[-1]
    time_fp32 = log_fp32['wall_time'].iloc[-1]

    mean_step_bf16 = time_bf16 / num_steps
    mean_step_fp32 = time_fp32 / num_steps

    print("\n[Q8 Result] Mean Step Time:")
    print(f"  bf16 (autocast): {mean_step_bf16*1000:.2f} ms/step")
    print(f"  fp32:            {mean_step_fp32*1000:.2f} ms/step")


def run_q9(model, train_ids, val_ids, base_config):
    """Runs training with and without warmup and plots validation curves."""
    print("\n--- Running Q9: Warmup Comparison ---")
    num_steps = 200
    
    # Run with warmup
    print("  Running with warmup...")
    train(model, train_ids, val_ids, num_steps=num_steps, **base_config, warmup_steps=50, run_name="q9_with_warmup")

    # Run without warmup
    print("\n  Running without warmup...")
    train(model, train_ids, val_ids, num_steps=num_steps, **base_config, warmup_steps=0, run_name="q9_no_warmup")

    # Parse logs
    log_warmup = pd.read_csv('logs/q9_with_warmup.csv')
    log_no_warmup = pd.read_csv('logs/q9_no_warmup.csv')

    # Plotting
    plt.figure()
    plt.plot(log_warmup['step'], log_warmup['val_loss'], label='With Warmup')
    plt.plot(log_no_warmup['step'], log_no_warmup['val_loss'], label='Without Warmup')
    plt.title('Validation Loss vs. Steps')
    plt.xlabel('Step')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plot_path = 'task3_q9_warmup.png'
    plt.savefig(plot_path)
    print(f"\n[Q9 Result] Plot saved to '{plot_path}'")

    # Report gradient norms
    final_step = log_warmup['step'].iloc[-1]
    
    def get_grad_norms(log_df, run_name):
        grad_log = pd.read_csv(f"logs/{run_name}_grad_norm.csv")
        norm_step1 = grad_log[grad_log['step'] == 1]['grad_norm'].iloc[0]
        norm_step50 = grad_log[grad_log['step'] == 50]['grad_norm'].iloc[0]
        norm_final = grad_log[grad_log['step'] == final_step]['grad_norm'].iloc[0]
        return norm_step1, norm_step50, norm_final

    norms_warmup = get_grad_norms(log_warmup, "q9_with_warmup")
    norms_no_warmup = get_grad_norms(log_no_warmup, "q9_no_warmup")

    print("\n[Q9 Result] Pre-clipping Gradient Norms:")
    print(f"{'Run':<15} {'Step 1':<10} {'Step 50':<10} {'Final Step':<10}")
    print("-" * 45)
    print(f"{'With Warmup':<15} {norms_warmup[0]:<10.4f} {norms_warmup[1]:<10.4f} {norms_warmup[2]:<10.4f}")
    print(f"{'Without Warmup':<15} {norms_no_warmup[0]:<10.4f} {norms_no_warmup[1]:<10.4f} {norms_no_warmup[2]:<10.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run experiments for Task 3 questions.")
    parser.add_argument('--questions', nargs='+', type=int, default=[8, 9],
                        help='A list of questions to run (8, 9).')
    args = parser.parse_args()

    train_path, val_path = prepare_dummy_data()
    train_ids = np.load(train_path, mmap_mode='r')
    val_ids = np.load(val_path, mmap_mode='r')

    config = TransformerConfig(
        vocab_size=4000, context_length=256, n_layers=4, d_model=512,
        n_heads=8, d_ff=1344, use_qk_norm=True, use_rmsnorm=True, use_rope=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(config).to(device)

    base_config = {
        'batch_size': 16,
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'grad_clip': 1.0,
        'eval_every': 50,
        'save_every': 1000,
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'seed': 42
    }

    if 8 in args.questions:
        run_q8(model, train_ids, val_ids, base_config)

    if 9 in args.questions:
        run_q9(model, train_ids, val_ids, base_config)

if __name__ == "__main__":
    main()
