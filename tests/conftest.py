import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_DIR = os.path.join(REPO_ROOT, "data")
SAMPLES_FILE = os.path.join(REPO_ROOT, "samples.txt")
VALID_FILE = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")
