import random
import torch
import numpy as np

# Single canonical checkpoint schema, used by both this module and train.py.
# (Previously train.py wrote its own inline checkpoints with different key
# names than this module's save/load pair used - which didn't even agree
# with each other. This is now the one place that format is defined.)


def save_checkpoint(model, optimizer, scheduler, step, path, config=None):
    """Save full training state needed to resume a run exactly."""
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'torch_rng': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'config': config,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=None, restore_rng=True):
    """
    Restore model (and optionally optimizer/scheduler) state from `path`.
    Only meant for checkpoints this codebase produced itself, so we don't
    restrict to `weights_only=True` (that would reject the RNG state).
    returns:
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if restore_rng:
        if 'torch_rng' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng'])
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])
    return checkpoint['step'], checkpoint.get('config')
