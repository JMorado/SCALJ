"""Utility functions for the MACE-LJ fitting workflow."""

import copy
from pathlib import Path

import torch
from openff.units import unit as offunit


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


def create_off_forcefield_from_tensor(forcefield, tensor_ff):
    """
    Get an OpenFF ForceField object with updated parameters from a tensor-based force field.

    Notes
    -----
    Currently this only works for the vdW parameters, and specifically using the 
    Double Exponential potential or the Lennard-Jones potential.

    Parameters
    ----------
    forcefield : openff.toolkit.typing.engines.smirnoff.ForceField\
        The OpenFF force field to update.
    tensor_ff : smee._models.TensorForceField
        The tensor-based force field containing the new parameters.

    Returns
    -------
    openff.toolkit.typing.engines.smirnoff.ForceField
        The updated OpenFF force field.
    """
    forcefield = copy.deepcopy(forcefield)
    tag = (
        "vdW"
        if forcefield.get_parameter_handler("vdW").parameters
        else "DoubleExponential"
    )
    potential_vdw = tensor_ff.potentials_by_type["vdW"]
    off_potential_vdw = forcefield.get_parameter_handler(tag)
    for i in range(potential_vdw.parameters.shape[1]):
        col = potential_vdw.parameter_cols[i]
        for j in range(potential_vdw.parameters.shape[0]):
            smirk_id = potential_vdw.parameter_keys[j].id
            val = potential_vdw.parameters[j, i]
            unit = (
                offunit.kilocalories_per_mole if col == "epsilon" else offunit.angstrom
            )
            param = off_potential_vdw.get_parameter({"smirks": smirk_id})[0]
            setattr(param, col, val.item() * unit)

    return forcefield
