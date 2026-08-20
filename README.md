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

## §2 — the validation/test split

§2: *"Reserve the last 2,000 documents of the validation file as a test set that
you touch exactly once."* That split has to exist as **data**, not as a slice
taken inside whichever script needs it — otherwise every run that validates on
`valid_encoded.npy` scores the reserved documents, and model selection has been
reading the test set all along.

```bash
python scripts/make_splits.py --vocab vocab.pkl --merges merges.pkl
```

Writes `valid_split.npy` (25,630 docs) and `test_split.npy` (2,000 docs), each
with a `_meta.json` sidecar carrying the character count §6 needs. Tasks 3 and 4
train and validate against `valid_split.npy`; only Q18 reads the test split. For
the §7.3 comparison tokenizer, pass `--suffix _vocab1000` to match Task 1's
naming.

Task 1 encodes the training split as **two** arrays (`train_part1.npy`,
`train_part2.npy`), because the corpus ships as two files. Either merge them
into one file, or leave them as they are — both work, and Task 4 prefers
`train.npy` when it exists:

```bash
python scripts/combine_train_parts.py                     # -> train.npy (1.12 GB)
python scripts/combine_train_parts.py --suffix _vocab1000 # the section 7.3 tokenizer
python scripts/combine_train_parts.py --remove_parts      # reclaim 1.12 GB, after verifying
```

The merge streams through `np.lib.format.open_memmap` in chunks, so it costs a
64MB buffer rather than 1.1GB of RAM, checks both part boundaries afterwards,
and merges the sidecars — summing the counts and **recomputing** chars-per-token
from the sums, since a mean of two ratios is not the ratio of the sums and §6
divides by that number.

Left unmerged, `src/data.py`'s `ConcatTokens` addresses the parts as one
sequence of 559.8M tokens without copying, for the same §5.5 reason.

Each split is **encoded separately** rather than sliced out of the already-encoded
array, because a token-index slice cannot tell you how many characters it covers
and §6 needs C for exactly the text scored. The split is verified lossless
against re-encoding the whole file with the current code: −2 tokens and −1
document, both the whitespace-only tail piece that is not a document.

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

Task 3 adds `tests/test_task3_training.py`, which turns §5.6's "before you
spend GPU hours" checklist into tests — loss at initialisation near
ln(vocab_size), overfitting a single batch, a resumed run reproducing the
uninterrupted trajectory, a short run that neither NaNs nor stalls — plus the
parts of §5.1–5.3 that are quietly wrong-able: which parameters weight decay
reaches, whether the cosine period matches the run length, and whether the
logged gradient norm is the pre-clipping one.

Task 5 adds `tests/test_task5_final_model.py`, covering §2's split (lossless,
and byte-identical to the corpus slice — a regression test for the newline
translation that silently inflated the character count BPC divides by), the
top-k/top-p cutoffs against distributions whose answer is known by hand, and
the test-set ledger.

Task 4 adds `tests/test_task4_experiments.py`. Most of it guards §7's
one-thing-at-a-time rule — that each ablation differs from the baseline in
exactly the feature under test and inherits everything else, that both arms
start from identical weights, and that the reduced learning rate is actually
below the baseline's. The rest pins §6's metric identities: perplexity is
exp(loss), BPC is loss / (ln2 × chars-per-token), the position-wise losses
average back to the aggregate, and batching does not move any of them.

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

## Task 3 — training (§5, Q8–Q9)

`src/train.py` is the §5 deliverable: AdamW with decoupled decay on the weight
matrices only, linear warmup into cosine decay, gradient clipping, bf16
autocast, checkpoint/resume, and CSV logging. Every hyperparameter is a flag
(§5.5), including the §7.2 architecture switches the ablations need:

```bash
python -m src.train --train_data train_encoded.npy --valid_data valid_encoded.npy     --steps 5000 --lr 3e-3 --warmup_steps 200 --run_name baseline

# the §7.2 ablations, from the command line
python -m src.train ... --no_rmsnorm
python -m src.train ... --no_rope
python -m src.train ... --ffn_type relu --d_ff 2048

# resume an interrupted job (§5.5) — keep --steps the same, or the restored
# scheduler is one built for a different run length
python -m src.train ... --resume checkpoints/baseline_step4000.pt --steps 5000
```

**GPU required for the real runs.** A step of the §4.1 model at
`batch_size=32` takes ~12s on this CPU, so Q8 alone is ~1.4 hours and Q9 ~13,
and §5.4 puts bf16 on CUDA only — on CPU both Q8 arms run fp32 and the
comparison is vacuous. The script says so rather than reporting the two equal
numbers as a result. Check the pipeline first:

```bash
python scripts/run_task3_questions.py --smoke    # ~1 min, tiny model, into logs/smoke/
python scripts/run_task3_questions.py            # the real Q8 and Q9
python scripts/run_task3_questions.py --questions 9 --lr 1e-3
```

Set `--lr` to the Q10 sweep winner before quoting Q9; the default is §5's
starting value, not a tuned one.

Outputs:

| Output | Answers |
|---|---|
| `logs/task3_results.json` | every number quoted for Q8–Q9 |
| `logs/<run>.csv` | §5.5's row per evaluation: step, wall time, tokens, lr, losses, grad norm |
| `logs/<run>_steps.csv` | per-step pre-clip gradient norm and step time |
| `task3_q9_warmup.png` | **the Q9 figure** |

Three things that quietly invalidate these answers, and what the code does
about them:

*Each arm gets a freshly initialised model.* Reusing one model object across
runs — what the script used to do — means the second arm starts from weights
the first arm already trained, so Q9 compares warmup against "no warmup, plus
200 steps of training in hand". Both arms now begin from identical weights.

*Step time is measured around the forward/backward/step only*, inside the
training loop, with a CUDA sync on each side. Dividing total wall-clock by the
step count instead folds in validation passes, and without the sync the timer
measures kernel enqueue rather than work — bf16 and fp32 come out identical.

*The batch at step N is seeded from `(seed, step)`, not drawn from a generator
advanced once per step.* A long-lived generator is rebuilt from `seed` when a
run resumes while the loop restarts mid-run, so a job resumed at step 26
replayed the batches from steps 1–25 and drifted off the uninterrupted
trajectory §5.6 requires it to match.

## Task 4 — experiments (§7, Q10–Q17)

Every run is an `Experiment` naming **only what it changes**; everything else
comes from §7's standard configuration. That is the whole design of this
script, because §7's rule — vary one thing, hold the rest fixed — fails
silently when it is broken: you still get a number, it just isn't measuring what
you think.

```bash
python scripts/run_task4_questions.py --plan      # what would train, and roughly what it costs
python scripts/run_task4_questions.py --smoke     # whole pipeline, tiny, ~2 min
python scripts/run_task4_questions.py --questions 10          # the sweep
python scripts/run_task4_questions.py --questions 11 12 13 14 # ablations, reusing the baseline
python scripts/run_task4_questions.py --final_model           # §7.4, for Task 5
```

**Runs are cached.** Each finished run writes `logs/<name>_run.json`, and asking
for it again reuses it. Q14 reads six runs and Q17 reads two; without the cache,
re-running the analysis would re-run the training. `--force` retrains.

Q10 finds the best learning rate and everything downstream uses it, so run it
first — or pass `--best_lr` to skip it. The sweep runs at `--sweep_steps`
(default 1000) with the cosine period shortened to match, which §7.1 permits.

Outputs:

| Output | Answers |
|---|---|
| `logs/task4_results.json` | every number quoted for Q10–Q17 |
| `logs/<name>_run.json` | one record per run: config, wall time, divergence, final losses |
| `task4_q10_lr_sweep.png` | **the Q10 figure** |
| `task4_q11_rmsnorm.png`, `task4_q12_nope.png`, `task4_q13_swiglu_relu.png` | **Q11–Q13** |
| `task4_q17_position.png` | **the Q17 figure** |

Three things worth knowing:

*Diverged runs are detected and killed.* §7.1 requires the sweep to contain a
divergent run and requires killing it once it is visibly diverged. `train()`
stops when the loss goes non-finite or the validation loss exceeds 1.5× its
step-1 value, records why, and Q10 reads that back — divergence is a recorded
outcome, not a crash. On the Q10 figure the axis is scaled to the runs that
trained, so the divergent one leaves the top of the plot rather than flattening
everything else into one band.

*The no-RMSNorm ablation's reduced learning rate is relative to the best one*
(`best_lr / 3`, or `--reduced_lr`). A fixed value lands *above* the baseline
whenever the sweep picks something smaller, which would quietly make §7.2's
"same ablation at a reduced learning rate" an increased-rate run instead.

*BPC, not perplexity, decides Q15.* A 1,000-token model and a 4,000-token model
are not counting the same events, so their perplexities are not on the same
scale. `src/evaluate.py` scores whole non-overlapping windows (§6) and counts
characters over exactly the tokens it scored, excluding `<|endoftext|>` — the
convention is reported alongside the number, as §6 requires.

### Prerequisites

Task 4 trains on the **encoded corpus from Task 1 §3.5** and never re-tokenizes.
It currently needs files that Task 1 phase 2 has not yet produced:

```bash
python scripts/run_task_1.py --vocab_size 4000 --from_merges merges_upto16000.pkl   # train_encoded.npy
python scripts/run_task_1.py --vocab_size 1000 --from_merges merges_upto16000.pkl   # §7.3's second tokenizer, for Q15
```

The script names the missing file and the command that makes it, rather than
falling back to something that would look like a result.

## Task 5 — final model and generation (§8, Q18–Q20)

```bash
python scripts/run_task5_questions.py --questions 19 20   # generation, no test set
python scripts/run_task5_questions.py --questions 18      # touches the test set
python scripts/run_task5_questions.py --smoke             # plumbing, untrained
```

Evaluates and generates from §7.4's `final_model` checkpoint, which it reads
from Task 4's run record — so the model config never has to be restated and
cannot drift from what was actually trained.

Outputs: `logs/task5_results.json`, `task5_samples.txt` (every generation in full,
next to the settings that produced it), and `logs/test_set_ledger.json`.

**The test set is touched once.** §2 reserves the last 2,000 validation
documents for exactly one evaluation. Every touch is appended to
`logs/test_set_ledger.json` with the checkpoint and the numbers, and a second
one is refused unless `--touch_test_again` is passed. The ledger is the evidence
the rule was kept — the marking scheme has a line for it.

**Nothing runs on an untrained model.** The checkpoint load has to succeed;
`--allow_untrained` exists for plumbing checks and stamps everything it
produces as not-a-result.

**Q20's evidence is generated, not written.** The degenerate sample is produced
and its repetition *measured* (distinct-token ratio, most-repeated trigram)
against a healthy setting, and the candidate failure mode is probed across
several prompts under the recommended sampler — so anything that survives is
the model rather than the decoding. Read the samples and name the failure mode
yourself; the script gathers evidence, it does not write your analysis.

## Repository layout

```
src/
  tokenizer.py          train_bpe, BPETokenizer, streaming + parallel pre-tokenization
  model.py              RMSNorm, SwiGLU, RoPE, attention, block, TransformerLM, KV cache
  data.py               memmap loader; ConcatTokens joins the two train parts
  train.py              training loop, optimiser/schedule, checkpointing, logging, CLI
  training_helpers/     batching, checkpoint save/load
  evaluate.py           perplexity, BPC, position-wise loss
  generate.py           sampling + KV-cache generation
scripts/
  run_task_1.py         Task 1 end to end (Q1-Q4)
  run_task2_questions.py  Task 2 end to end (Q5-Q7)
  run_task3_questions.py  Task 3 end to end (Q8-Q9)
  run_task4_questions.py  Task 4 end to end (Q10-Q17)
  run_task5_questions.py  Task 5 end to end (Q18-Q20)
  encode_corpus.py      corpus -> uint16 .npy + metadata sidecar
  vocab_study.py        compression ratio vs vocabulary size
  make_splits.py        section 2's validation/test split
  combine_train_parts.py  train_part1 + train_part2 -> train.npy
  run_experiments.sh    prep (CPU) and train (GPU) phases, end to end
tests/
logs/                   one CSV/JSONL per run
```
