# CSC3043S 2026 — Assignment 1: Training a Transformer Language Model

Byte-level BPE tokenizer, a dense Transformer LM, and the training/evaluation
pipeline used for the experiments in the report.

## Environment

Developed and tested on **Python 3.13.6** with **PyTorch 2.8.0**.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On a CUDA machine, install torch from the PyTorch index instead so you get a GPU
build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Data

Download the TinyStories V2 files from Amathuba (Resources → Datasets) into
`data/`:

```
data/TinyStoriesV2-GPT4-train.txt
data/TinyStoriesV2-GPT4-train-part2.txt     # the train split ships in two parts
data/TinyStoriesV2-GPT4-valid.txt
```

The dataset, the encoded `.npy` arrays and model checkpoints are gitignored
(§10.1). The tokenizer `vocab.pkl` / `merges.pkl` and everything under `logs/`
are committed.

## Tests

```bash
python -m pytest tests/ -q
```

Task 1 has 40 tests covering the §3.3 sanity checks (round-trip, single
`<|endoftext|>` ID, uint16 range, determinism) plus the equivalences that the
performance work depends on — that the streaming/parallel pre-tokenizer produces
byte-identical merges to the naive whole-file path, and that a tokenizer derived
by truncating a longer merge list is identical to one trained directly.

Task 2 adds the model, KV-cache and throughput tests. `src/` is a package and
its modules import each other relatively, so everything imports it as
`src.model`, `src.tokenizer`, … with the repo root on `sys.path` — tests and
scripts now share that one convention.

## Task 1 — tokenizer (§3, Q1–Q4)

The corpus is pre-tokenized **once** and everything else is derived from that
single pass: the compression curve, both tokenizers, and the Q1 timings. Because
BPE is greedy, the first *k* merges of a long run *are* the merges a shorter run
would learn, so every vocabulary size is a truncation of one merge list.

§3.4 is the step that **decides** the vocabulary size, so this runs in two
phases — the curve first, the choice second.

```bash
# Smoke test first (~2 min) — exercises every step on a 30MB slice
python scripts/run_task1.py --limit_mb 30 --workers 4 --vocab_size 4000 \
    --out_dir /tmp/task1_smoke

# Phase 1: pre-tokenize + build the compression curve, then stop.
# Omitting --vocab_size is what makes it stop.
python scripts/run_task1.py --workers 8

# Phase 2: having read the curve, commit to a size. --from_merges reuses the
# saved merge list, so this does NOT repeat the pre-tokenization pass.
python scripts/run_task1.py --vocab_size 4000 --from_merges merges_upto16000.pkl
```

Phase 2 carries the Q1/Q3 numbers forward from phase 1 rather than overwriting
them, so `logs/task1_results.json` ends up holding all of Q1–Q4.

Useful flags: `--vocab_size` (omit to stop after the study), `--second_vocab_size`
(the §7.3 comparison tokenizer, default 1000), `--study_sizes` (Q3 curve points,
default 1000/2000/4000/8000/16000), `--from_merges`, `--skip_encode`,
`--skip_q2`, `--limit_mb`.

Outputs:

| Output | Answers |
|---|---|
| `logs/task1_results.json` | every number quoted for Q1–Q4 |
| `logs/vocab_study.csv` | the Q3 table |
| `compression_ratio.png` | **the Q3 figure** |
| `merges_upto<N>.pkl` | full merge list; any smaller vocab is a truncation |
| `vocab.pkl`, `merges.pkl` | primary tokenizer |
| `vocab1000_vocab.pkl`, `vocab1000_merges.pkl` | second tokenizer (§3.5) |
| `train*.npy`, `valid*.npy` + `*_meta.json` | encoded corpus (§3.5) |

Each encoded array gets a `_meta.json` sidecar recording token/document counts
and the **raw character count** of the encoded text, which §6 needs to convert a
loss into bits-per-character without re-reading the corpus. It records the count
both with and without the `<|endoftext|>` delimiters, since §6 requires stating
which convention was used.

To re-encode a corpus with an existing tokenizer:

```bash
python scripts/encode_corpus.py \
    --input data/TinyStoriesV2-GPT4-valid.txt \
    --output valid.npy --vocab vocab.pkl --merges merges.pkl
```

To re-run only the vocabulary-size study:

```bash
python scripts/vocab_study.py --input data/TinyStoriesV2-GPT4-valid.txt --workers 8
```

## Task 2 — model and inference (§4, Q5–Q7)

```bash
python scripts/run_task2_questions.py                    # all of Q5–Q7
python scripts/run_task2_questions.py --questions 7 --repeats 5
```

**No trained weights and no GPU time are required.** Parameter counts are a
property of the architecture, KV-cache correctness is an identity that must hold
for any weights, and throughput is set by the shape of the computation rather
than its values — so this runs on a freshly initialised model (seeded, so the
numbers reproduce). Running a subset with `--questions` merges into the existing
results rather than overwriting the other answers.

Outputs:

| Output | Answers |
|---|---|
| `logs/task2_results.json` | every number quoted for Q5–Q7 |
| `task2_q7_throughput.png` | **the Q7 figure** |

Two things worth knowing about the numbers:

*§4.1's "roughly 17M parameters excluding the embedding and LM head" does not
match this model.* The measured non-embedding count is 12.46M; 16.55M is the
total *including* the embedding and LM head at `vocab_size=4000`. Q5 reports the
measured split.

*`generate(max_new_tokens=N)` does not always produce N tokens.* The cached path
decodes into a fixed-size `context_length` buffer and stops on reaching it (§4.2
permits this); the uncached path slides a window and has no such limit. At
`max_new_tokens=256` with a 4-token prompt the cached path produces 253 tokens
and the uncached 256, so dividing both by 256 would compare different amounts of
work. Q7 divides by the true count from `tokens_actually_generated`, which
`tests/test_task2_questions.py` checks against real generation.

## Tasks 3–5

Not yet reproducible from a single command — see `EXPERIMENTS.md` for the run
log. Entry points as they stand:

| Script | Covers |
|---|---|
| `src/train.py` | training loop, optimiser/schedule, checkpointing, logging (CLI) |
| `src/evaluate.py` | perplexity, BPC, position-wise loss |
| `src/generate.py` | sampling + KV-cache generation |
| `scripts/run_task3_questions.py` | Q8–Q9 |
| `scripts/run_task4_questions.py` | Q10–Q17 |
| `scripts/run_task5_questions.py` | Q18–Q20 |

> **Known gap:** `src/train.py` hardcodes `use_rmsnorm`, `use_rope` and
> `ffn_type` when it builds the config, so the §7.2 ablations are not yet
> reachable from the command line even though `TransformerConfig` supports all
> three. Needs three CLI flags before the Task 7 runs.

## Repository layout

```
src/
  tokenizer.py          train_bpe, BPETokenizer, streaming + parallel pre-tokenization
  model.py              RMSNorm, SwiGLU, RoPE, attention, block, TransformerLM, KV cache
  data.py               memmap loader, batching
  train.py              training loop, optimiser/schedule, checkpointing, logging, CLI
  evaluate.py           perplexity, BPC
  generate.py           sampling + KV-cache generation
scripts/
  run_task1.py          Task 1 end to end (Q1-Q4)
  run_task2_questions.py  Task 2 end to end (Q5-Q7)
  encode_corpus.py      corpus -> uint16 .npy + metadata sidecar
  vocab_study.py        compression ratio vs vocabulary size
  make_plots.py         figures from run logs
  run_task{2,3,4,5}_questions.py
tests/
logs/                   one CSV/JSONL per run
configs/
```
