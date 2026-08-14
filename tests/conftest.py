import os
import sys

# src/*.py import each other with bare names (e.g. `from tokenizer import ...`,
# `from model_components.RMSNorm import ...`), which only resolves if `src/`
# itself is on sys.path - not the repo root. Match that convention here so
# tests can `import tokenizer`, `import model`, etc. regardless of the
# directory pytest is invoked from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATA_DIR = os.path.join(REPO_ROOT, "data")
SAMPLES_FILE = os.path.join(REPO_ROOT, "samples.txt")
VALID_FILE = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")
