"""Parameter training and optimization."""

import typing

import datasets
import descent.train
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
    force_field: smee.TensorForceField,
    systems: dict[str, smee.TensorSystem],
    reference: typing.Literal["mean", "min", "none"] = "none",
    normalize: bool = True,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predict the relative energies [kcal/mol] and forces [kcal/mol/Å] of a dataset.

    Parameters
    -----------
    dataset : datasets.Dataset
        The dataset to predict the energies and forces of.
    force_field : smee.TensorForceField
        The force field to use to predict the energies and forces.
    systems : dict[str, smee.TensorSystem]
        The systems of the molecules in the dataset. Each key should be
            a fully indexed SMILES string.
    reference : typing.Literal["mean", "min", "none"], optional
        The reference energy to compute the relative energies with respect
            to. This should be either the "mean" energy of all conformers, or the
            energy of the conformer with the lowest reference energy ("min").
    normalize : bool, optional
        Whether to scale the relative energies by ``1/n_confs_i``    and the forces
        by ``1/n_confs_i * n_atoms_per_conf_i * 3)``. This is useful when
        wanting to compute the MSE per entry.
    device : str, optional
        The device to use for the prediction.

    Returns
    -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The predicted and reference relative energies [kcal/mol] with
            ``shape=(n_confs,)``, and predicted and reference forces [kcal/mol/Å]
            with ``shape=(n_confs * n_atoms_per_conf, 3)``.
    """
    energy_ref_all, energy_pred_all = [], []
    forces_ref_all, forces_pred_all = [], []
    for entry in dataset:
        smiles = entry["smiles"]
        energy_ref = entry["energy"].to(device)
        forces_ref = (entry["forces"].reshape(len(energy_ref), -1, 3)).to(device)

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

        system = systems[smiles].to(device)

        energy_pred = torch.zeros_like(energy_ref)
        for i, (coord, box_vector) in tqdm(
            enumerate(zip(coords, box_vectors)),
            total=len(coords),
            desc="Predicting energies/forces",
            leave=False,
        ):
            energy_pred[i] = smee.compute_energy(system, force_field, coord, box_vector)

        forces_pred = -torch.autograd.grad(
            energy_pred.sum(),
            coords,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]

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

        scale_energy, scale_forces = 1.0, 1.0

        if normalize:
            scale_energy = 1.0 / torch.tensor(energy_pred.numel(), device=device)
            scale_forces = 1.0 / torch.tensor(forces_pred.numel(), device=device)

        energy_ref_all.append(scale_energy * (energy_ref - energy_ref_0))
        forces_ref_all.append(scale_forces * forces_ref.reshape(-1, 3))

        energy_pred_all.append(scale_energy * (energy_pred - energy_pred_0))
        forces_pred_all.append(scale_forces * forces_pred.reshape(-1, 3))

    energy_pred_all = torch.cat(energy_pred_all)
    forces_pred_all = torch.cat(forces_pred_all)

    energy_ref_all = torch.cat(energy_ref_all)
    energy_ref_all = smee.utils.tensor_like(energy_ref_all, energy_pred_all)

    forces_ref_all = torch.cat(forces_ref_all)
    forces_ref_all = smee.utils.tensor_like(forces_ref_all, forces_pred_all)

    return (
        energy_ref_all,
        energy_pred_all,
        forces_ref_all,
        forces_pred_all,
    )


def _compute_loss(
    energy_ref: torch.Tensor,
    energy_pred: torch.Tensor,
    forces_ref: torch.Tensor,
    forces_pred: torch.Tensor,
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
    energy_weight : float, optional
        Weight for energy loss, by default 1.0
    force_weight : float, optional
        Weight for force loss, by default 1.0

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple of (total_loss, energy_loss, force_loss)
    """
    energy_loss = torch.mean((energy_pred - energy_ref) ** 2)
    force_loss = torch.mean((forces_pred - forces_ref) ** 2)

    total_loss = energy_weight * energy_loss + force_weight * force_loss

    return total_loss, energy_loss, force_loss


def train_parameters(
    trainable: descent.train.Trainable,
    dataset: datasets.Dataset,
    systems: dict[str, smee.TensorSystem],
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
    systems : dict[str, smee.TensorSystem]
        Molecular systems
    config : TrainingConfig
        Training configuration

    Returns
    -------
    tuple[list[float], list[float]]
        Tuple of (energy_losses, force_losses)
    """
    params = trainable.to_values().to(config.device).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=config.learning_rate)

    energy_losses = []
    force_losses = []

    for epoch in range(config.n_epochs):
        optimizer.zero_grad()

        # Predict with current parameters
        energy_ref, energy_pred, forces_ref, forces_pred = predict(
            dataset,
            trainable.to_force_field(params.abs()).to(
                config.device
            ),  # Note the absolute value, as the LJ parameters are positive
            systems,
            reference=config.reference,
            normalize=config.normalize,
            device=config.device,
        )

        # Compute loss
        total_loss, energy_loss, force_loss = _compute_loss(
            energy_ref,
            energy_pred,
            forces_ref,
            forces_pred,
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

    return params, energy_losses, force_losses
