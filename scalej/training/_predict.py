"""Prediction functions for energies and forces."""

import logging
import datasets
import smee
import smee.utils
import torch
from tqdm import tqdm

from ..models import PredictionResult
from ._loss import _compute_kbt
from ._types import ReferenceMode, WeightingMethod


log = logging.getLogger(__name__)


def predict_energies_forces(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    tensor_systems: dict[str, smee.TensorSystem],
    reference: ReferenceMode = "none",
    energy_cutoff: float | None = None,
    weighting_method: WeightingMethod = "uniform",
    weighting_temperature: float = 298.15,
    device: str = "cpu",
) -> PredictionResult:
    """
    Predict energies and forces using the force field.

    Computes classical force field energies and forces via automatic
    differentiation, with optional filtering and weighting.

    Parameters
    ----------
    dataset : datasets.Dataset
        Dataset with reference energies and forces.
    force_field : smee.TensorForceField
        Force field for energy/force computation.
    tensor_systems : dict[str, smee.TensorSystem]
        Dictionary mapping mixture IDs to tensor systems.
    reference : ReferenceMode
        Reference energy mode.
    energy_cutoff : float, optional
        Energy cutoff in kcal/mol to filter high-energy conformers.
    weighting_method : WeightingMethod
        Method to weight conformers in loss computation.
    weighting_temperature : float
        Temperature in Kelvin for Boltzmann weighting.
    device : str
        Device for computation.

    Returns
    -------
    PredictionResult
        Predicted and reference energies, forces, weights, and masks.

    Examples
    --------
    >>> result = predict_energies_forces(
    ...     dataset, force_field, systems, reference="mean"
    ... )
    """
    energy_ref_all = []
    energy_pred_all = []
    forces_ref_all = []
    forces_pred_all = []
    weights_all = []
    weights_forces_all = []
    all_mask_idxs = []

    for entry in tqdm(dataset, desc="Predicting", leave=False):
        mixture_id = entry["mixture_id"]

        energy_ref = entry["energy"].to(device)
        forces_ref = entry["forces"].reshape(len(energy_ref), -1, 3).to(device)

        coords_flat = smee.utils.tensor_like(
            entry["coords"], force_field.potentials[0].parameters
        )

        coords = (
            (coords_flat.reshape(len(energy_ref), -1, 3))
            .to(device)
            .requires_grad_(True)
        )

        box_vectors_flat = smee.utils.tensor_like(
            entry["box_vectors"], force_field.potentials[0].parameters
        )
        box_vectors = (
            (box_vectors_flat.reshape(len(energy_ref), 3, 3))
            .to(device)
            .detach()
            .requires_grad_(False)
        )

        system = tensor_systems[mixture_id].to(device)

        # Compute energies and forces using one-by-one pattern to save memory
        energies = []
        forces = []
        for coord, box_vector in tqdm(
            zip(coords, box_vectors),
            total=len(coords),
            desc="Predicting energies/forces",
            leave=False,
        ):
            # Compute energy and its gradient (force) for a single conformer
            try:
                e = smee.compute_energy(system, force_field, coord, box_vector)
                g = torch.autograd.grad(e, coord, allow_unused=True)[0]
                energies.append(e.detach())
                forces.append(-g.detach())
            except Exception as ex:
                log.error(f"Error computing energy/forces: {ex}")
                raise ValueError("Error computing energy/forces")

        energy_pred = torch.stack(energies)
        forces_pred = torch.stack(forces)

        # Normalize energies by the number of molecules
        n_mols = sum(system.n_copies)
        energy_ref = energy_ref / n_mols
        energy_pred = energy_pred / n_mols

        # Determine reference energy offset
        if reference.lower() == "mean":
            energy_ref_0 = energy_ref.mean()
            energy_pred_0 = energy_pred.mean()
        elif reference.lower() == "min":
            min_idx = energy_ref.argmin()
            energy_ref_0 = energy_ref[min_idx]
            energy_pred_0 = energy_pred[min_idx]
        elif reference.lower() == "infinite":
            energy_ref_0 = energy_ref[-1]
            energy_pred_0 = energy_pred[-1]
        elif reference.lower() == "none":
            energy_ref_0 = 0
            energy_pred_0 = 0
        else:
            raise NotImplementedError(f"invalid reference energy {reference}")

        # Filtering mask
        mask = torch.ones_like(energy_ref, dtype=torch.bool)
        if energy_cutoff is not None:
            energy_ref_min = energy_ref.min()
            mask = (energy_ref - energy_ref_min) <= energy_cutoff

        # Apply weights
        weights = torch.ones_like(energy_ref)
        if weighting_method == "boltzmann":
            kb_t = _compute_kbt(weighting_temperature)
            e_rel = energy_ref - energy_ref.min()
            weights = torch.exp(-e_rel / kb_t)

        # Apply mask to everything
        mask_idx = torch.where(mask)[0]

        energy_ref_masked = energy_ref[mask_idx]
        energy_pred_masked = energy_pred[mask_idx]

        forces_ref_masked = forces_ref[mask_idx]
        forces_pred_masked = forces_pred[mask_idx]
        weights_masked = weights[mask_idx]

        # Expand weights for forces
        n_atoms = forces_ref.shape[1]
        weights_forces_masked = weights_masked.repeat_interleave(n_atoms)

        energy_ref_all.append(energy_ref_masked - energy_ref_0)
        forces_ref_all.append(forces_ref_masked.reshape(-1, 3))

        energy_pred_all.append(energy_pred_masked - energy_pred_0)
        forces_pred_all.append(forces_pred_masked.reshape(-1, 3))

        weights_all.append(weights_masked)
        weights_forces_all.append(weights_forces_masked)

        all_mask_idxs.append(mask_idx)

    if not energy_pred_all:
        raise ValueError("No valid conformers found after filtering")

    energy_pred_all = torch.cat(energy_pred_all)
    forces_pred_all = torch.cat(forces_pred_all)

    energy_ref_all = torch.cat(energy_ref_all)
    energy_ref_all = smee.utils.tensor_like(energy_ref_all, energy_pred_all)

    forces_ref_all = torch.cat(forces_ref_all)
    forces_ref_all = smee.utils.tensor_like(forces_ref_all, forces_pred_all)

    weights_all = torch.cat(weights_all)
    weights_all = smee.utils.tensor_like(weights_all, energy_pred_all)

    weights_forces_all = torch.cat(weights_forces_all)
    weights_forces_all = weights_forces_all.unsqueeze(1)
    weights_forces_all = smee.utils.tensor_like(weights_forces_all, forces_pred_all)

    # Normalize weights
    weights_energy_sum = weights_all.sum()
    if weights_energy_sum > 0:
        weights_all = weights_all / weights_energy_sum

    weights_forces_sum = weights_forces_all.sum()
    if weights_forces_sum > 0:
        weights_forces_all = weights_forces_all / weights_forces_sum

    return PredictionResult(
        energy_ref=energy_ref_all,
        energy_pred=energy_pred_all,
        forces_ref=forces_ref_all,
        forces_pred=forces_pred_all,
        weights_energy=weights_all,
        weights_forces=weights_forces_all,
        mask_idxs=all_mask_idxs,
    )
