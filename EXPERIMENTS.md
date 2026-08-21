# Experiment log

Every training run launched for this assignment, including the one that diverged. Generated from `logs/*_run.json` by `scripts/make_experiments_table.py` -- every number is traceable to the JSON file named in the last column.

**12 runs, 1.67 GPU-hours total** (single CUDA device, AMP enabled).

Fixed across every run unless the configuration column says otherwise: context_length=256, n_layers=4, d_model=512, n_heads=8, SwiGLU FFN, RoPE (theta=10000), QK-norm, AdamW with weight_decay=0.1, grad_clip=1.0, cosine schedule, seed=42.

## Baseline

| Run | Configuration | Steps | Final val loss | Params | Wall-clock | GPU-hours | What I learned | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | vocab=4000, lr=0.003, bs=32, warmup=200, d_ff=1344 | 5000 | 1.6140 | 16,552,960 | 10m 30s | 0.18 | TODO | `logs/baseline_run.json` |

## Task 4 Q10 - learning-rate sweep

| Run | Configuration | Steps | Final val loss | Params | Wall-clock | GPU-hours | What I learned | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lr_sweep_1e-04` | vocab=4000, lr=0.0001, bs=32, warmup=100, d_ff=1344 | 1000 | 3.1000 | 16,552,960 | 2m 04s | 0.03 | TODO | `logs/lr_sweep_1e-04_run.json` |
| `lr_sweep_3e-04` | vocab=4000, lr=0.0003, bs=32, warmup=100, d_ff=1344 | 1000 | 2.5255 | 16,552,960 | 2m 02s | 0.03 | TODO | `logs/lr_sweep_3e-04_run.json` |
| `lr_sweep_1e-03` | vocab=4000, lr=0.001, bs=32, warmup=100, d_ff=1344 | 1000 | 2.1039 | 16,552,960 | 2m 03s | 0.03 | TODO | `logs/lr_sweep_1e-03_run.json` |
| `lr_sweep_3e-03` | vocab=4000, lr=0.003, bs=32, warmup=100, d_ff=1344 | 1000 | 1.9868 | 16,552,960 | 2m 03s | 0.03 | TODO | `logs/lr_sweep_3e-03_run.json` |
| `lr_sweep_1e-02` | vocab=4000, lr=0.01, bs=32, warmup=100, d_ff=1344 | 1000 | 2.0339 | 16,552,960 | 2m 03s | 0.03 | TODO | `logs/lr_sweep_1e-02_run.json` |

## Task 4 Q11-Q13 - architecture ablations

| Run | Configuration | Steps | Final val loss | Params | Wall-clock | GPU-hours | What I learned | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ablation_no_rmsnorm_lowlr` | vocab=4000, lr=0.001, bs=32, warmup=200, **no RMSNorm**, d_ff=1344 | 5000 | 1.6386 | 16,548,352 | 8m 31s | 0.14 | TODO | `logs/ablation_no_rmsnorm_lowlr_run.json` |
| `ablation_no_rmsnorm` | vocab=4000, lr=0.003, bs=32, warmup=200, **no RMSNorm**, d_ff=1344 | 150 / 5000 (**stopped**) | 131.4532 (diverged) | 16,548,352 | 0m 16s | 0.00 | TODO | `logs/ablation_no_rmsnorm_run.json` |
| `ablation_nope` | vocab=4000, lr=0.003, bs=32, warmup=200, **no RoPE**, d_ff=1344 | 5000 | 1.7393 | 16,552,960 | 9m 11s | 0.15 | TODO | `logs/ablation_nope_run.json` |
| `ablation_relu` | vocab=4000, lr=0.003, bs=32, warmup=200, **ffn=relu**, d_ff=2048 | 5000 | 1.5961 | 16,684,032 | 9m 47s | 0.16 | TODO | `logs/ablation_relu_run.json` |

## Task 4 - vocabulary-size study

| Run | Configuration | Steps | Final val loss | Params | Wall-clock | GPU-hours | What I learned | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vocab_1000` | vocab=1000, lr=0.003, bs=32, warmup=200, d_ff=1344 | 5000 | 1.3566 | 13,480,960 | 9m 50s | 0.16 | TODO | `logs/vocab_1000_run.json` |

## Task 5 - final model

| Run | Configuration | Steps | Final val loss | Params | Wall-clock | GPU-hours | What I learned | Log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `final_model` | vocab=4000, lr=0.003, bs=32, warmup=200, d_ff=1344 | 20000 | 1.4157 | 16,552,960 | 41m 44s | 0.70 | TODO | `logs/final_model_run.json` |

## Runs that failed

- `ablation_no_rmsnorm` stopped at step 150: validation loss 131.4532 exceeded 1.5x its step-1 value 9.8355. Cost: 0m 16s (0.00 GPU-hours).
