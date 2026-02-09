"""Parameter training and optimization."""

import typing

import datasets
import descent.train
import openmm
import smee
import smee.utils
import torch
from tqdm import tqdm

from ..config import ParameterConfig, TrainingConfig


def create_trainable(
    force_field: smee.TensorForceField,
    parameter_config: ParameterConfig,
    training_config: TrainingConfig,
) -> descent.train.Trainable:
    """
    Create a trainable object for parameter optimization.

    Parameters
    ----------
    force_field : smee.TensorForceField
        The force field with parameters to train
    parameter_config : ParameterConfig
        Configuration for trainable parameters

    Returns
    -------
    descent.train.Trainable
        Trainable object
    """
    vdw_parameter_config = descent.train.ParameterConfig(
        cols=parameter_config.cols,
        scales=parameter_config.scales,
        limits=parameter_config.limits,
    )

    # Ensure vdW parameters require gradients
    vdw_potential = force_field.potentials_by_type["vdW"]
    vdw_potential.parameters.requires_grad = True

    trainable = descent.train.Trainable(
        force_field=force_field.to(training_config.device),
        parameters={"vdW": vdw_parameter_config},
        attributes={},
    )

    return trainable


def predict(
    dataset: datasets.Dataset,
    composite_force_field: smee.TensorForceField,
    all_tensor_systems: dict[str, smee.TensorSystem],
    reference: typing.Literal["mean", "min", "none"] = "none",
    normalize: bool = True,
    energy_cutoff: float | None = None,
    weighting_method: typing.Literal["uniform", "boltzmann"] = "uniform",
    weighting_temperature: float = 298.15,
    device: str = "cpu",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
]:
    """
    Predict the relative energies per molecule [kcal/mol] and forces [kcal/mol/Å] of a dataset.

    Parameters
    -----------
    dataset : datasets.Dataset
        The dataset to predict the energies and forces of.
    composite_force_field : smee.TensorForceField
        The force field to use to predict the energies and forces.
    all_tensor_systems : dict[str, smee.TensorSystem]
        The systems of the molecules in the dataset.
        Each key should be the system name.
    reference : typing.Literal["mean", "min", "none"], optional
        The reference energy to compute the relative energies with respect
            to. This should be either the "mean" energy of all conformers, or the
            energy of the conformer with the lowest reference energy ("min").
    normalize : bool, optional
        Whether to scale the relative energies by ``1/n_confs_i``    and the forces
        by ``1/n_confs_i * n_atoms_per_conf_i * 3)``. This is useful when
        wanting to compute the MSE per entry.
    energy_cutoff : float, optional
        Energy cutoff in kcal/mol to filter high-energy conformers.
    weighting_method : typing.Literal["uniform", "boltzmann"], optional
        Method to weight conformers in loss function.
    weighting_temperature : float, optional
        Temperature in Kelvin for Boltzmann weighting.
    device : str, optional
        The device to use for the prediction.

    Returns
    -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
            The predicted and reference relative energies [kcal/mol],
            predicted and reference forces [kcal/mol/Å], weights for energies,
            and weights for forces.
    """
    energy_ref_all, energy_pred_all = [], []
    forces_ref_all, forces_pred_all = [], []
    weights_all = []
    weights_forces_all = []
    all_mask_idxs = []

    for entry in dataset:
        mixture_id = entry["mixture_id"]
        energy_ref = entry["energy"].to(device)
        forces_ref = (entry["forces"].reshape(len(energy_ref), -1, 3)).to(device)

        coords_flat = smee.utils.tensor_like(
            entry["coords"], composite_force_field.potentials[0].parameters
        )

        coords = (
            (coords_flat.reshape(len(energy_ref), -1, 3))
            .to(device)
            .requires_grad_(True)
        )

        box_vectors_flat = smee.utils.tensor_like(
            entry["box_vectors"], composite_force_field.potentials[0].parameters
        )
        box_vectors = (
            (box_vectors_flat.reshape(len(energy_ref), 3, 3))
            .to(device)
            .detach()
            .requires_grad_(False)
        )

        system = all_tensor_systems[mixture_id].to(device)

        energy_pred = torch.zeros_like(energy_ref)
        for i, (coord, box_vector) in tqdm(
            enumerate(zip(coords, box_vectors)),
            total=len(coords),
            desc="Predicting energies/forces",
            leave=False,
        ):
            energy_pred[i] = smee.compute_energy(
                system, composite_force_field, coord, box_vector
            )

        forces_pred = -torch.autograd.grad(
            energy_pred.sum(),
            coords,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]

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
        elif reference.lower() == "none":
            energy_ref_0 = 0
            energy_pred_0 = 0
        else:
            raise NotImplementedError(f"invalid reference energy {reference}")

        # Filtering mask
        mask = torch.ones_like(energy_ref, dtype=torch.bool)
        if energy_cutoff is not None:
            # We filter based on reference energy relative to its minimum
            # Note that we already normalize by the number molecules so that the
            # energy cutoff is physically meaningful
            energy_ref_min = energy_ref.min()
            mask = (energy_ref - energy_ref_min) <= energy_cutoff

        # Apply weights
        weights = torch.ones_like(energy_ref)
        if weighting_method == "boltzmann":
            kBT = (
                openmm.unit.AVOGADRO_CONSTANT_NA
                * openmm.unit.BOLTZMANN_CONSTANT_kB
                * weighting_temperature
            ).value_in_unit(openmm.unit.kilocalories_per_mole)
            e_rel = energy_ref - energy_ref.min()
            weights = torch.exp(-e_rel / kBT)

        # Apply mask to everything
        mask_idx = torch.where(mask)[0]

        energy_ref_masked = energy_ref[mask_idx]
        energy_pred_masked = energy_pred[mask_idx]

        forces_ref_masked = forces_ref[mask_idx]
        forces_pred_masked = forces_pred[mask_idx]
        weights_masked = weights[mask_idx]

        # Expand weights for forces: (n_confs,) -> (n_confs * n_atoms,)
        # forces_ref has shape (n_confs, n_atoms, 3)
        n_atoms = forces_ref.shape[1]
        weights_forces_masked = weights_masked.repeat_interleave(n_atoms)

        scale_energy, scale_forces = 1.0, 1.0

        if normalize:
            n_confs = len(mask_idx)
            if n_confs > 0:
                scale_energy = 1.0 / energy_ref_masked.numel()
                scale_forces = 1.0 / forces_ref_masked.numel()

        energy_ref_all.append(scale_energy * (energy_ref_masked - energy_ref_0))
        forces_ref_all.append(scale_forces * forces_ref_masked.reshape(-1, 3))

        energy_pred_all.append(scale_energy * (energy_pred_masked - energy_pred_0))
        forces_pred_all.append(scale_forces * forces_pred_masked.reshape(-1, 3))

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

    return (
        energy_ref_all,
        energy_pred_all,
        forces_ref_all,
        forces_pred_all,
        weights_all,
        weights_forces_all,
        all_mask_idxs,
    )


def _compute_loss(
    energy_ref: torch.Tensor,
    energy_pred: torch.Tensor,
    forces_ref: torch.Tensor,
    forces_pred: torch.Tensor,
    weights_energy: torch.Tensor | None = None,
    weights_forces: torch.Tensor | None = None,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute weighted loss for energies and forces.

    Parameters
    ----------
    energy_ref : torch.Tensor
        Reference energies
    energy_pred : torch.Tensor
        Predicted energies
    forces_ref : torch.Tensor
        Reference forces
    forces_pred : torch.Tensor
        Predicted forces
    weights_energy : torch.Tensor, optional
        Weights for each energy point, by default None (all 1.0)
    weights_forces : torch.Tensor, optional
        Weights for each force component, by default None (all 1.0)
    energy_weight : float, optional
        Weight for energy loss, by default 1.0
    force_weight : float, optional
        Weight for force loss, by default 1.0

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple of (total_loss, energy_loss, force_loss)
    """
    if weights_energy is None:
        weights_energy = torch.ones_like(energy_ref)

    if weights_forces is None:
        weights_forces = torch.ones_like(forces_ref)

    energy_loss = torch.mean(
        weights_energy * (energy_pred - energy_ref) ** 2
    ) / torch.var(energy_ref)

    force_loss = torch.mean(
        weights_forces * (forces_pred - forces_ref) ** 2
    ) / torch.var(forces_ref)

    total_loss = energy_weight * energy_loss + force_weight * force_loss

    return total_loss, energy_loss, force_loss


def train_parameters(
    trainable: descent.train.Trainable,
    dataset: datasets.Dataset,
    all_tensor_systems: dict[str, smee.TensorSystem],
    config: TrainingConfig,
) -> tuple[list[float], list[float]]:
    """
    Train force field parameters to match reference data.

    Parameters
    ----------
    trainable : descent.train.Trainable
        Trainable object with parameters to optimize
    dataset : datasets.Dataset
        Training dataset
    all_tensor_systems : dict[str, smee.TensorSystem]
        The systems of the molecules in the dataset.
        Each key should be the system name.
    config : TrainingConfig
        Training configuration

    Returns
    -------
    tuple[list[float], list[float]]
        Tuple of (energy_losses, force_losses)
    """
    params = trainable.to_values().to(config.device).detach().requires_grad_(True)

    # Initially perturb the parameters
    eps = 0.2
    with torch.no_grad():
        params += torch.empty_like(params).uniform_(-eps, eps)

    optimizer = torch.optim.Adam([params], lr=config.learning_rate, amsgrad=False)

    energy_losses = []
    force_losses = []

    for epoch in range(config.n_epochs):
        optimizer.zero_grad()

        # Predict with current parameters
        (
            energy_ref,
            energy_pred,
            forces_ref,
            forces_pred,
            weights_energy,
            weights_forces,
            mask_idx,
        ) = predict(
            dataset,
            trainable.to_force_field(params.abs()).to(
                config.device
            ),  # Note the absolute value, as the LJ parameters are positive
            all_tensor_systems,
            reference=config.reference,
            normalize=config.normalize,
            energy_cutoff=config.energy_cutoff,
            weighting_method=config.weighting_method,
            weighting_temperature=config.weighting_temperature,
            device=config.device,
        )

        # Compute loss
        total_loss, energy_loss, force_loss = _compute_loss(
            energy_ref,
            energy_pred,
            forces_ref,
            forces_pred,
            weights_energy=weights_energy,
            weights_forces=weights_forces,
            energy_weight=config.energy_weight,
            force_weight=config.force_weight,
        )

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Record losses
        energy_losses.append(energy_loss.item())
        force_losses.append(force_loss.item())

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: loss_energy = {energy_loss.item():.4e}, "
                f"loss_forces = {force_loss.item():.4e}"
            )

    return params.abs(), energy_losses, force_losses
