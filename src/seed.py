"""Central place to fix random seeds across the whole project.

Import and call `set_seed()` once, as early as possible (before building
the dataset, model, or dataloaders), to make runs reproducible.

Usage:
    from seed import set_seed
    set_seed
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + CUDA) with the same value.

    Args:
        seed: the seed to use everywhere.
        deterministic: if True, ask cuDNN/torch for deterministic
            algorithms. This can slow things down a bit but makes GPU
            runs reproducible too. Set to False if you need max speed
            and don't care about bit-for-bit GPU reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no GPU / multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Makes torch raise instead of silently using a non-deterministic op.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older torch versions may not support this call.
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """Pass as `worker_init_fn` to DataLoader so each worker is seeded too.

    Example:
        DataLoader(..., worker_init_fn=seed_worker,
                   generator=torch.Generator().manual_seed(seed))
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
