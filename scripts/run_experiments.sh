#!/bin/bash

# Exit immediately if any command exits with a non-zero status, if an unset
# variable is read, or if any stage of a pipeline fails.
set -euo pipefail

# Every path below is repo-relative, so run from the repo root regardless of
# where the script was invoked from.
cd "$(dirname "$0")/.."

# -u keeps stdout unbuffered, so `bash scripts/run_experiments.sh | tee log`
# shows progress as it happens instead of in one burst at the end.
# Many distros (e.g. WSL/Ubuntu) only ship `python3`, not a bare `python`.
if [ -n "${PYTHON:-}" ]; then
    DEFAULT_PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY=python3
elif command -v python >/dev/null 2>&1; then
    DEFAULT_PY=python
else
    echo "Error: no python interpreter found on PATH (tried python3, python)." >&2
    exit 1
fi
PY="$DEFAULT_PY -u"

# The training stages need a GPU: the section 4.1 model runs at roughly
# 10 s/step on CPU, and stages 5-7 are ~50,000 steps. Set SKIP_TRAINING=1 to
# run only the data-preparation stages (1-4), which are CPU work by design.
SKIP_TRAINING="${SKIP_TRAINING:-0}"

VOCAB_SIZE=4000
SECOND_VOCAB_SIZE=1000
# Stage 1 names its merge list after the largest size it studied, so the two
# have to agree; deriving the name here keeps them from drifting apart.
STUDY_SIZES=(1000 2000 4000 8000 16000)
MAX_STUDY=$(printf '%s\n' "${STUDY_SIZES[@]}" | sort -n | tail -1)
MERGES_UPTO="merges_upto${MAX_STUDY}.pkl"

echo "Starting full pipeline execution..."

# ---------------------------------------------------------
# Stage 1 — Task 1, phase 1: the vocabulary-size study
# ---------------------------------------------------------
echo "=== Stage 1: Pre-tokenizing and vocabulary study ==="
$PY scripts/run_task_1.py --workers 8 --study_sizes "${STUDY_SIZES[@]}"

# ---------------------------------------------------------
# Stage 2 — Task 1, phase 2: commit to a size
# ---------------------------------------------------------
echo "=== Stage 2: Deriving vocab and merges (size ${VOCAB_SIZE}) ==="
$PY scripts/run_task_1.py --vocab_size "$VOCAB_SIZE" --from_merges "$MERGES_UPTO" \
    --second_vocab_size "$SECOND_VOCAB_SIZE"

# ---------------------------------------------------------
# Stage 3 — The split, and merging the train parts
# ---------------------------------------------------------
# Both tokenizers need this: section 7.3 (Q15) trains an arm on the second
# tokenizer's corpus, and load_corpus() looks for valid_split_vocab1000.npy and
# train_vocab1000.npy by name. Each split has to be encoded with *its own*
# tokenizer -- pointing --vocab/--merges at the primary files while passing
# --suffix _vocab1000 would write 4,000-vocab tokens under the 1,000-vocab name.
echo "=== Stage 3: Making test/valid splits and combining train arrays ==="
$PY scripts/make_splits.py --vocab vocab.pkl --merges merges.pkl
$PY scripts/combine_train_parts.py --force

$PY scripts/make_splits.py \
    --vocab "vocab${SECOND_VOCAB_SIZE}_vocab.pkl" \
    --merges "vocab${SECOND_VOCAB_SIZE}_merges.pkl" \
    --suffix "_vocab${SECOND_VOCAB_SIZE}"
$PY scripts/combine_train_parts.py --suffix "_vocab${SECOND_VOCAB_SIZE}" --force

# ---------------------------------------------------------
# Stage 4 — Task 2 (Q5–Q7), off to the side
# ---------------------------------------------------------
echo "=== Stage 4: Running Task 2 (Model sweeps, no training) ==="
# The default prompt ("Once upon a time") hits an early <|endoftext|> at some
# token counts under the fixed seed=0 random init, which Q7 treats as a fatal
# error (it would make the throughput denominator wrong). This prompt was
# checked to run clean through all of --token_counts (16-256) with repeats=3.
$PY scripts/run_task2_questions.py --prompt "The quick brown fox jumps over the lazy dog."

if [ "$SKIP_TRAINING" = "1" ]; then
    echo "SKIP_TRAINING=1 - stopping after the data-preparation stages."
    echo "Stages 5-7 (Tasks 3-5) need a GPU; run them there."
    exit 0
fi

# ---------------------------------------------------------
# Stage 5 — Task 3 (Q8–Q9)
# ---------------------------------------------------------
echo "=== Stage 5: Running Task 3 (Training arms) ==="
echo "Running smoke test..."
$PY scripts/run_task3_questions.py --smoke
echo "Running full Task 3..."
$PY scripts/run_task3_questions.py --train_data train.npy --valid_data valid_split.npy

# ---------------------------------------------------------
# Stage 6 — Task 4 (Q10–Q17) and the final model
# ---------------------------------------------------------
echo "=== Stage 6: Running Task 4 (Hyperparameters and final model) ==="
$PY scripts/run_task4_questions.py --plan
echo "Running smoke test..."
$PY scripts/run_task4_questions.py --smoke --final_model

# Q10 gates Task 4 and determines best_lr
echo "Running Q10 (Learning Rate Sweep)..."
$PY scripts/run_task4_questions.py --questions 10

# --final_model rides along with Q11-Q17 rather than in a call of its own:
# --questions defaults to all of 10-17, so a bare `--final_model` run would
# replay every question's analysis before training anything.
echo "Running Q11-Q17 (Experiments) and training the final model..."
$PY scripts/run_task4_questions.py --questions 11 12 13 14 15 16 17 --final_model

# ---------------------------------------------------------
# Stage 7 — Task 5 (Q18–Q20)
# ---------------------------------------------------------
echo "=== Stage 7: Running Task 5 (Evaluation and Generation) ==="
echo "Running smoke test..."
$PY scripts/run_task5_questions.py --smoke
# Q19/Q20 first, so that section 2's "touch the test set exactly once" is the
# last thing the pipeline does.
$PY scripts/run_task5_questions.py --questions 19 20
$PY scripts/run_task5_questions.py --questions 18

echo "Pipeline execution successfully completed!"
