import torch
from dataclasses import asdict

def save_checkpoint(model, optimizer, iteration, path):
    """Write everything needed to resume training to `path`."""
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        # Store the config as a plain dict rather than the dataclass object, so that loading
        # the checkpoint does not require the class definition to be importable.
        "config": asdict(model.config),
    }, path)


def load_checkpoint(path, model, optimizer=None):
    """
    Restore model (and optionally optimizer) state from `path`.
    returns:
        the iteration number stored in the checkpoint
    """
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]