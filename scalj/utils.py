"""Utility functions for the MACE-LJ fitting workflow."""

from pathlib import Path

import torch


def perturb_tensor(
    tensor: torch.Tensor, frac: float = 0.2, mode: str = "add", seed: int = 0
) -> torch.Tensor:
    """
    Perturb a tensor by a random factor.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to perturb.
    frac : float
        The fraction of the tensor to perturb.
    mode : str
        The mode of perturbation. Can be "add" or "mul".
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor
        The perturbed tensor (absolute values).
    """
    assert mode in ["add", "mul"], "Mode must be 'add' or 'mul'"
    mode_ref = 1.0 if mode == "mul" else 0.0
    factor_low = mode_ref - frac
    factor_high = mode_ref + frac

    # Fix random seed for reproducibility
    torch.manual_seed(seed)
    noise = torch.empty_like(tensor).uniform_(factor_low, factor_high)

    with torch.no_grad():
        if mode == "add":
            tensor.add_(noise)
        elif mode == "mul":
            tensor.mul_(noise)

    return tensor.abs()


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : Path
        The directory path to ensure exists.

    Returns
    -------
    Path
        The created or existing directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
