"""Loss computation functions for training."""

from typing import Any

import descent.train
import openmm.unit
import smee
import smee.utils
import torch
from tqdm import tqdm

from ._types import (
    BatchResult,
    ConformerWeights,
    EntryData,
    LossConfig,
    ReferenceMode,
    ReferenceOffsetGradient,
    WeightingMethod,
)


def _prepare_entry_data(
    entry: dict[str, Any],
    tensor_systems: dict[str, smee.TensorSystem],
    device: str,
) -> EntryData:
    """
    Prepare and reshape data from a dataset entry.

    Parameters
    ----------
    entry : dict[str, Any]
        Dataset entry with coords, energy, forces, box_vectors.
    tensor_systems : dict[str, smee.TensorSystem]
        Dictionary mapping mixture IDs to tensor systems.
    device : str
        Device to move tensors to.

    Returns
    -------
    EntryData
        Prepared entry data with normalized energies.
    """
    mixture_id = entry["mixture_id"]
    n_conformers = len(entry["energy"])

    # Load and reshape tensors
    energy_ref = entry["energy"].to(device)
    forces_ref = entry["forces"].reshape(n_conformers, -1, 3).to(device)
    coords = entry["coords"].reshape(n_conformers, -1, 3).to(device)
    box_vectors = entry["box_vectors"].reshape(n_conformers, 3, 3).to(device)

    # Get system
    system = tensor_systems[mixture_id].to(device)
    n_mols = sum(system.n_copies)
    n_atoms = coords.shape[1]

    # Normalize energies and forces by number of molecules
    energy_ref = energy_ref / n_mols
    forces_ref = forces_ref / n_mols

    return EntryData(
        energy_ref=energy_ref,
        forces_ref=forces_ref,
        coords=coords,
        box_vectors=box_vectors,
        system=system,
        n_mols=n_mols,
        n_atoms=n_atoms,
    )


def _compute_kbt(temperature: float) -> float:
    """
    Compute kB*T in kcal/mol.

    Parameters
    ----------
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    float
        kB*T in kcal/mol.
    """
    return (
        openmm.unit.AVOGADRO_CONSTANT_NA
        * openmm.unit.BOLTZMANN_CONSTANT_kB
        * temperature
    ).value_in_unit(openmm.unit.kilocalories_per_mole)


def _compute_conformer_weights(
    entry_data: EntryData,
    config: LossConfig,
    device: str,
) -> ConformerWeights | None:
    """
    Compute conformer weights and filter by energy cutoff.

    Parameters
    ----------
    entry_data : EntryData
        Prepared entry data.
    config : LossConfig
        Training configuration.
    device : str
        Device for tensors.

    Returns
    -------
    ConformerWeights or None
        Weights and valid indices, or None if no valid conformers.
    """
    energy_ref = entry_data.energy_ref
    forces_ref = entry_data.forces_ref
    n_conformers = len(energy_ref)
    n_atoms = entry_data.n_atoms

    # Apply energy cutoff filter
    # This uses the global minimum energy conformer to define the cutoff
    mask = torch.ones(n_conformers, dtype=torch.bool, device=device)
    if config.energy_cutoff is not None:
        energy_ref_min = energy_ref.min()
        mask = (energy_ref - energy_ref_min) <= config.energy_cutoff

    valid_indices = torch.where(mask)[0]
    n_valid = len(valid_indices)

    if n_valid == 0:
        raise ValueError("No valid conformers after applying energy cutoff filter.")

    # Compute weights for valid conformers
    energy_ref_valid = energy_ref[valid_indices]
    weights = torch.ones(n_valid, device=device)

    # Compute reference offset
    ref_idx = -1
    min_idx = energy_ref.argmin().item()
    if config.reference.lower() == "mean":
        energy_ref_0 = energy_ref_valid.mean().detach()
    elif config.reference.lower() == "min":
        ref_idx = min_idx
        energy_ref_0 = energy_ref[ref_idx].detach()
    elif config.reference.lower() == "infinite":
        # We assume that the infinite separation (r->inf) idx is the last one
        ref_idx = len(energy_ref) - 1
        energy_ref_0 = energy_ref[ref_idx].detach()
    elif config.reference.lower() == "none":
        energy_ref_0 = torch.tensor(0.0, device=device)
        ref_idx = min_idx
    else:
        raise NotImplementedError(f"Unknown reference mode: {config.reference}")

    # Compute weights for valid conformers
    # We use relative energies to avoid numerical issues
    if config.weighting_method == "boltzmann":
        kb_t = _compute_kbt(config.weighting_temperature)
        e_rel = energy_ref_valid - energy_ref_valid.min()
        weights = torch.exp(-e_rel / kb_t)
    elif config.weighting_method == "mixed":
        # Boltzmann weights for conformers below the minimum index, uniform above
        kb_t = _compute_kbt(config.weighting_temperature)
        min_valid_idx = (valid_indices == min_idx).nonzero(as_tuple=True)[0].item()
        e_rel = energy_ref_valid - energy_ref_valid[min_valid_idx]
        weights = torch.where(
            torch.arange(n_valid, device=device) < min_valid_idx,
            torch.exp(-e_rel / kb_t),
            torch.ones(n_valid, device=device),
        )

    # Normalize weights to sum to 1
    weights = (weights / weights.sum()).detach()

    # Expand weights for forces
    weights_forces = weights.view(-1, 1, 1).expand(n_valid, n_atoms, 3)
    weights_forces = (weights_forces / weights_forces.sum()).detach()

    # Compute variances for normalization
    energy_ref_shifted = energy_ref_valid - energy_ref_0
    forces_ref_valid = forces_ref[valid_indices]
    energy_var = torch.var(energy_ref_shifted).detach()
    forces_var = torch.var(forces_ref_valid).detach()

    return ConformerWeights(
        valid_indices=valid_indices,
        weights=weights,
        weights_forces=weights_forces,
        energy_var=energy_var,
        forces_var=forces_var,
        energy_ref_0=energy_ref_0,
        reference_idx=ref_idx,
    )


def _compute_reference_gradient(
    params: torch.Tensor,
    trainable: descent.train.Trainable,
    entry_data: EntryData,
    ref_idx: int,
    device: str,
) -> ReferenceOffsetGradient:
    """
    Compute gradient of reference energy offset.

    Parameters
    ----------
    params : torch.Tensor
        Current parameter values.
    trainable : descent.train.Trainable
        Trainable object for parameter mapping.
    entry_data : EntryData
        Prepared entry data.
    ref_idx : int
        Index of reference energy conformer.
    device : str
        Device for computation.

    Returns
    -------
    ReferenceOffsetGradient
        Energy at reference conformer and its gradient.
    """
    force_field_ref = trainable.to_force_field(params.abs()).to(device)

    coords_ref = smee.utils.tensor_like(
        entry_data.coords[ref_idx], force_field_ref.potentials[0].parameters
    )
    box_ref = smee.utils.tensor_like(
        entry_data.box_vectors[ref_idx], force_field_ref.potentials[0].parameters
    )

    energy_pred_0 = (
        smee.compute_energy(entry_data.system, force_field_ref, coords_ref, box_ref)
        / entry_data.n_mols
    )

    # Compute gradient of energy_pred_0 w.r.t. params
    (grad,) = torch.autograd.grad(
        energy_pred_0, params, create_graph=False, retain_graph=False
    )
    grad = grad.detach()
    energy_pred_0 = energy_pred_0.detach()

    del force_field_ref

    return ReferenceOffsetGradient(energy_pred_0=energy_pred_0, grad=grad)


def _compute_batch_energies(
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    coords_batch: torch.Tensor,
    box_vectors_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Compute energies for a batch of conformers.

    Parameters
    ----------
    system : smee.TensorSystem
        Tensor system.
    force_field : smee.TensorForceField
        Force field for energy computation.
    coords_batch : torch.Tensor
        Batch coordinates. Shape: [batch_size, n_atoms, 3].
    box_vectors_batch : torch.Tensor
        Batch box vectors. Shape: [batch_size, 3, 3].

    Returns
    -------
    torch.Tensor
        Energies for each conformer. Shape: [batch_size].
    """
    energies = []
    for i in range(len(coords_batch)):
        energies.append(
            smee.compute_energy(
                system, force_field, coords_batch[i], box_vectors_batch[i]
            )
        )
    return torch.stack(energies)


def _compute_batch_forces(
    energy_pred_batch: torch.Tensor,
    coords_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Compute forces from energies via autograd.

    Parameters
    ----------
    energy_pred_batch : torch.Tensor
        Predicted energies.
    coords_batch : torch.Tensor
        Coordinates (must have requires_grad=True).

    Returns
    -------
    torch.Tensor
        Predicted forces (negative gradient of energy).
    """
    return -torch.autograd.grad(
        energy_pred_batch.sum(),
        coords_batch,
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]


def _process_conformer_batch(
    params: torch.Tensor,
    trainable: descent.train.Trainable,
    entry_data: EntryData,
    conf_weights: ConformerWeights,
    batch_start: int,
    batch_end: int,
    energy_pred_0: torch.Tensor,
    config: LossConfig,
    device: str,
) -> BatchResult:
    """
    Process a single batch of conformers.

    Creates a fresh force field for gradient computation, computes energies
    and forces, then computes and detaches gradient.

    Parameters
    ----------
    params : torch.Tensor
        Current parameter values.
    trainable : descent.train.Trainable
        Trainable object for parameter mapping.
    entry_data : EntryData
        Prepared entry data.
    conf_weights : ConformerWeights
        Conformer weights and filtering info.
    batch_start : int
        Start index within valid_indices.
    batch_end : int
        End index within valid_indices.
    energy_pred_0 : torch.Tensor
        Predicted energy offset (for reference subtraction).
    config : LossConfig
        Training configuration.
    device : str
        Device for computation.

    Returns
    -------
    BatchResult
        Gradients and loss contributions from this batch.
    """
    batch_indices = conf_weights.valid_indices[batch_start:batch_end]

    # Create fresh force_field for this batch (new computation graph)
    force_field = trainable.to_force_field(params.abs()).to(device)
    ff_dtype = force_field.potentials[0].parameters.dtype

    # Get batch weights with correct dtype
    batch_weights = conf_weights.weights[batch_start:batch_end].to(dtype=ff_dtype)

    # Prepare batch data with proper dtype matching
    coords_batch = smee.utils.tensor_like(
        entry_data.coords[batch_indices], force_field.potentials[0].parameters
    ).requires_grad_(True)
    box_vectors_batch = smee.utils.tensor_like(
        entry_data.box_vectors[batch_indices], force_field.potentials[0].parameters
    ).detach()
    energy_ref_batch = entry_data.energy_ref[batch_indices].to(dtype=ff_dtype)
    forces_ref_batch = entry_data.forces_ref[batch_indices].to(dtype=ff_dtype)

    # Compute energies
    energy_pred_batch = _compute_batch_energies(
        entry_data.system, force_field, coords_batch, box_vectors_batch
    )
    energy_pred_batch = energy_pred_batch / entry_data.n_mols

    # Compute forces
    if config.compute_forces:
        forces_pred_batch = _compute_batch_forces(energy_pred_batch, coords_batch)
    else:
        forces_pred_batch = None

    # Apply reference offset
    energy_ref_shifted = energy_ref_batch - conf_weights.energy_ref_0
    energy_pred_shifted = energy_pred_batch - energy_pred_0

    # Compute batch contributions
    energy_diff = energy_pred_shifted - energy_ref_shifted

    # Energy: weighted SSE contribution
    batch_weighted_energy_sse = torch.sum(batch_weights * energy_diff**2)

    # Forces: weighted SSE contribution
    if config.compute_forces:
        batch_weights_forces = conf_weights.weights_forces[batch_start:batch_end].to(
            dtype=ff_dtype
        )
        force_diff_sq = (forces_pred_batch - forces_ref_batch) ** 2
        batch_weighted_force_sse = torch.sum(batch_weights_forces * force_diff_sq)
    else:
        batch_weighted_force_sse = torch.zeros(1, device=device)

    # Combine energy and force losses for gradient computation
    # Weights already sum to 1 so no division by n_valid needed
    batch_energy_loss = batch_weighted_energy_sse / conf_weights.energy_var
    if config.compute_forces:
        batch_force_loss = batch_weighted_force_sse / conf_weights.forces_var
    else:
        batch_force_loss = torch.zeros(1, device=device, dtype=ff_dtype)
    batch_total_loss = (
        config.energy_weight * batch_energy_loss
        + config.force_weight * batch_force_loss
    )

    # Compute gradient w.r.t. params and free graph
    (batch_grad,) = torch.autograd.grad(
        batch_total_loss, params, create_graph=False, retain_graph=False
    )
    batch_grad = batch_grad.detach()

    # Compute d(loss)/d(energy_pred_0) for chain rule correction
    # dL/dE_{0,pred} = -2 * w_E * sum_i(w_i * diff_i) / var_E
    dloss_d_pred0 = (
        -2.0
        * config.energy_weight
        * torch.sum(batch_weights * energy_diff)
        / conf_weights.energy_var
    ).detach()

    return BatchResult(
        grad=batch_grad,
        weighted_energy_sse=batch_weighted_energy_sse.detach(),
        weighted_force_sse=batch_weighted_force_sse.detach(),
        dloss_d_pred0=dloss_d_pred0,
    )


def _aggregate_losses(
    total_weighted_energy_sse: torch.Tensor,
    total_weighted_force_sse: torch.Tensor,
    energy_var: torch.Tensor,
    forces_var: torch.Tensor,
    config: LossConfig,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aggregate batch losses into final loss values.

    Parameters
    ----------
    total_weighted_energy_sse : torch.Tensor
        Accumulated weighted sum of squared energy errors.
    total_weighted_force_sse : torch.Tensor
        Accumulated weighted sum of squared force errors.
    energy_var : torch.Tensor
        Reference energy variance.
    forces_var : torch.Tensor
        Reference forces variance.
    config : LossConfig
        Training configuration.
    device : str
        Device for tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Total loss, energy loss, force loss.
    """
    total_energy_loss = total_weighted_energy_sse / energy_var

    if config.compute_forces:
        total_force_loss = total_weighted_force_sse / forces_var
    else:
        total_force_loss = torch.zeros(1, device=device)

    total_loss = (
        config.energy_weight * total_energy_loss
        + config.force_weight * total_force_loss
    )

    return total_loss, total_energy_loss, total_force_loss


def _apply_reference_gradient_correction(
    accumulated_grad: torch.Tensor,
    total_dloss_d_energy_pred_0: torch.Tensor,
    ref_offset_grad: ReferenceOffsetGradient | None,
) -> torch.Tensor:
    r"""
    Apply chain rule correction for minimum-reference gradient.

    Final gradient assembly:
    \nabla_{\theta} L = \nabla_{\theta} L|_{batch}
        + (\partial L / \partial E_{ref}) * \nabla_{\theta} E_{ref}

    Parameters
    ----------
    accumulated_grad : torch.Tensor
        Accumulated gradient from batch processing.
    total_dloss_d_energy_pred_0 : torch.Tensor
        Total derivative of loss w.r.t. energy_pred_0.
    ref_offset_grad : ReferenceOffsetGradient or None
        Gradient info for reference configuration, or None if not using reference.
        \nabla_{\theta} E_{ref}

    Returns
    -------
    torch.Tensor
        Corrected gradient.
    """
    if ref_offset_grad is not None:
        return accumulated_grad + total_dloss_d_energy_pred_0 * ref_offset_grad.grad
    return accumulated_grad


def get_losses(
    params: torch.Tensor,
    trainable: descent.train.Trainable,
    entry: dict[str, Any],
    tensor_systems: dict[str, smee.TensorSystem],
    conformer_batch_size: int = 8,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    reference: ReferenceMode = "none",
    energy_cutoff: float | None = None,
    weighting_method: WeightingMethod = "uniform",
    weighting_temperature: float = 298.15,
    device: str = "cuda",
    compute_forces: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute losses for an entry.

    Parameters
    ----------
    params : torch.Tensor
        Current parameter values.
    trainable : descent.train.Trainable
        Trainable object for parameter mapping.
    entry : dict[str, Any]
        Single dataset entry with coords, energy, forces, box_vectors.
    tensor_systems : dict[str, smee.TensorSystem]
        Dictionary mapping mixture IDs to tensor systems.
    conformer_batch_size : int
        Number of conformers to process at once before accumulating gradient.
    energy_weight : float
        Weight for energy loss term.
    force_weight : float
        Weight for force loss term.
    reference : ReferenceMode
        Reference energy mode.
    energy_cutoff : float, optional
        Energy cutoff for filtering.
    weighting_method : WeightingMethod
        Conformer weighting method.
    weighting_temperature : float
        Temperature for Boltzmann weighting.
    device : str
        Device for computations.
    compute_forces : bool
        Whether to compute forces.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple of (total_loss, energy_loss, force_loss, grad).
    """
    # Build config from parameters
    config = LossConfig(
        energy_weight=energy_weight,
        force_weight=force_weight,
        reference=reference,
        energy_cutoff=energy_cutoff,
        weighting_method=weighting_method,
        weighting_temperature=weighting_temperature,
        compute_forces=compute_forces,
    )

    # Prepare entry data
    entry_data = _prepare_entry_data(entry, tensor_systems, device)

    # Compute weights and filter conformers
    conf_weights = _compute_conformer_weights(entry_data, config, device)

    if conf_weights is None:
        # No valid conformers
        zero_loss = torch.zeros(1, device=device)
        zero_grad = torch.zeros_like(params)
        return zero_loss, zero_loss.clone(), zero_loss.clone(), zero_grad

    n_valid = len(conf_weights.valid_indices)

    # Compute reference offset gradient for "min" or "infinite" reference
    if config.reference.lower() in ["min", "infinite"]:
        ref_offset_grad = _compute_reference_gradient(
            params, trainable, entry_data, conf_weights.reference_idx, device
        )
        energy_pred_0 = ref_offset_grad.energy_pred_0
    elif config.reference.lower() in ["mean", "none"]:
        ref_offset_grad = None
        energy_pred_0 = torch.tensor(0.0, device=device)
    else:
        raise ValueError(f"Unknown reference mode: {config.reference}")

    # Initialize accumulators
    total_weighted_energy_sse = torch.zeros(1, device=device)
    total_weighted_force_sse = torch.zeros(1, device=device)
    total_dloss_d_energy_pred_0 = torch.zeros(1, device=device)
    accumulated_grad = None

    # Process conformers in batches
    n_batches = (n_valid + conformer_batch_size - 1) // conformer_batch_size
    for batch_start in tqdm(
        range(0, n_valid, conformer_batch_size),
        total=n_batches,
        desc="Conformer batches",
        leave=False,
    ):
        batch_end = min(batch_start + conformer_batch_size, n_valid)

        batch_result = _process_conformer_batch(
            params=params,
            trainable=trainable,
            entry_data=entry_data,
            conf_weights=conf_weights,
            batch_start=batch_start,
            batch_end=batch_end,
            energy_pred_0=energy_pred_0,
            config=config,
            device=device,
        )

        # Accumulate gradient
        if accumulated_grad is None:
            accumulated_grad = batch_result.grad
        else:
            accumulated_grad = accumulated_grad + batch_result.grad

        # Accumulate d(loss)/d(energy_pred_0) for chain rule
        if ref_offset_grad is not None:
            total_dloss_d_energy_pred_0 = (
                total_dloss_d_energy_pred_0 + batch_result.dloss_d_pred0
            )

        # Accumulate loss contributions
        total_weighted_energy_sse = (
            total_weighted_energy_sse + batch_result.weighted_energy_sse
        )
        total_weighted_force_sse = (
            total_weighted_force_sse + batch_result.weighted_force_sse
        )

    # Apply reference offset gradient correction
    accumulated_grad = _apply_reference_gradient_correction(
        accumulated_grad, total_dloss_d_energy_pred_0, ref_offset_grad
    )

    # Compute final losses
    total_loss, total_energy_loss, total_force_loss = _aggregate_losses(
        total_weighted_energy_sse=total_weighted_energy_sse,
        total_weighted_force_sse=total_weighted_force_sse,
        energy_var=conf_weights.energy_var,
        forces_var=conf_weights.forces_var,
        config=config,
        device=device,
    )

    return total_loss, total_energy_loss, total_force_loss, accumulated_grad
