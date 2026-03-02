"""Type aliases and dataclasses for the training module."""

from dataclasses import dataclass
from typing import Literal

import smee
import torch

ReferenceMode = Literal["mean", "min", "none", "infinite"]
WeightingMethod = Literal["uniform", "boltzmann", "mixed"]


@dataclass
class LossConfig:
    """
    Configuration for loss computation parameters.

    Parameters
    ----------
    energy_weight : float
        Weight for energy loss term.
    force_weight : float
        Weight for force loss term.
    reference : ReferenceMode
        Reference energy mode: "mean", "min", "none", or "infinite".
    energy_cutoff : float, optional
        Energy cutoff in kcal/mol for filtering high-energy conformers.
    weighting_method : WeightingMethod
        Conformer weighting method: "uniform", "boltzmann", or "mixed".
    weighting_temperature : float
        Temperature in Kelvin for Boltzmann weighting.
    compute_forces : bool
        Whether to compute forces (for tracking).
    """

    energy_weight: float = 1.0
    force_weight: float = 1.0
    reference: ReferenceMode = "none"
    energy_cutoff: float | None = None
    weighting_method: WeightingMethod = "uniform"
    weighting_temperature: float = 298.15
    compute_forces: bool = True


@dataclass
class EntryData:
    """
    Prepared data for a single dataset entry.

    Parameters
    ----------
    energy_ref : torch.Tensor
        Reference energies, normalized by n_mols. Shape: [n_conformers].
    forces_ref : torch.Tensor
        Reference forces. Shape: [n_conformers, n_atoms, 3].
    coords : torch.Tensor
        Coordinates. Shape: [n_conformers, n_atoms, 3].
    box_vectors : torch.Tensor
        Box vectors. Shape: [n_conformers, 3, 3].
    system : smee.TensorSystem
        Tensor system for energy computation.
    n_mols : int
        Number of molecules in the system.
    n_atoms : int
        Number of atoms in the system.
    """

    energy_ref: torch.Tensor
    forces_ref: torch.Tensor
    coords: torch.Tensor
    box_vectors: torch.Tensor
    system: smee.TensorSystem
    n_mols: int
    n_atoms: int


@dataclass
class ConformerWeights:
    """
    Conformer weights and valid indices after filtering.

    Parameters
    ----------
    valid_indices : torch.Tensor
        Indices of conformers that pass the energy cutoff filter.
    weights : torch.Tensor
        Normalized weights for valid conformers. Shape: [n_valid].
    weights_forces : torch.Tensor
        Expanded weights for forces. Shape: [n_valid, n_atoms, 3].
    energy_var : torch.Tensor
        Variance of shifted reference energies (scalar).
    forces_var : torch.Tensor
        Variance of reference forces (scalar).
    energy_ref_0 : torch.Tensor
        Reference energy offset (scalar).
    reference_idx : int
        Index of the conformer used for the reference energy.
    """

    valid_indices: torch.Tensor
    weights: torch.Tensor
    weights_forces: torch.Tensor
    energy_var: torch.Tensor
    forces_var: torch.Tensor
    energy_ref_0: torch.Tensor
    reference_idx: int


@dataclass
class BatchResult:
    """
    Result from processing a single conformer batch.

    Parameters
    ----------
    grad : torch.Tensor
        Gradient w.r.t. parameters for this batch.
    weighted_energy_sse : torch.Tensor
        Weighted sum of squared energy errors.
    weighted_force_sse : torch.Tensor
        Weighted sum of squared force errors.
    dloss_d_pred0 : torch.Tensor
        Derivative of loss w.r.t. energy_pred_0 for this batch.
    """

    grad: torch.Tensor
    weighted_energy_sse: torch.Tensor
    weighted_force_sse: torch.Tensor
    dloss_d_pred0: torch.Tensor


@dataclass
class ReferenceOffsetGradient:
    """
    Gradient information for minimum-energy reference offset.

    Parameters
    ----------
    energy_pred_0 : torch.Tensor
        Predicted energy at the minimum reference conformer (scalar).
    grad : torch.Tensor
        Gradient of energy_pred_0 w.r.t. parameters.
    """

    energy_pred_0: torch.Tensor
    grad: torch.Tensor
