"""Parameter training functions."""

from typing import TYPE_CHECKING, Literal

import datasets
import descent.train
import openmm.unit
import smee
import smee.utils
import torch
from tqdm import tqdm

from .models import LossResult, PredictionResult, TrainingResult

if TYPE_CHECKING:
    from .config import ParameterConfig, TrainingConfig


def create_trainable(
    force_field: smee.TensorForceField,
    cols: list[str] = ["epsilon", "sigma"],
    scales: dict[str, float] | None = None,
    limits: dict[str, tuple[float, float]] | None = None,
    device: str = "cpu",
) -> descent.train.Trainable:
    """Create a trainable object for parameter optimization.

    Parameters
    ----------
    force_field : smee.TensorForceField
        The force field with parameters to train.
    cols : list[str]
        Parameter columns to optimize (e.g., ["epsilon", "sigma"]).
    scales : dict[str, float], optional
        Scaling factors for each parameter type.
    limits : dict[str, tuple[float, float]], optional
        Min/max limits for each parameter type.
    device : str
        Device to use for training.

    Returns
    -------
    descent.train.Trainable
        Trainable object for optimization.

    Examples
    --------
    >>> trainable = create_trainable(force_field, cols=["epsilon", "sigma"])
    """
    if scales is None:
        scales = {}
    if limits is None:
        limits = {}

    # Create trainable parameter config for vdW parameters
    vdw_parameter_config = descent.train.ParameterConfig(
        cols=cols,
        scales=scales,
        limits=limits,
    )

    # Ensure vdW parameters require gradients
    vdw_potential = force_field.potentials_by_type["vdW"]
    vdw_potential.parameters.requires_grad = True

    # Create trainable object
    trainable = descent.train.Trainable(
        force_field=force_field.to(device),
        parameters={"vdW": vdw_parameter_config},
        attributes={},
    )

    return trainable


def predict_energies_forces(
    dataset: datasets.Dataset,
    force_field: smee.TensorForceField,
    tensor_systems: dict[str, smee.TensorSystem],
    reference: Literal["mean", "min", "none"] = "none",
    normalize: bool = True,
    energy_cutoff: float | None = None,
    weighting_method: Literal["uniform", "boltzmann"] = "uniform",
    weighting_temperature: float = 298.15,
    device: str = "cpu",
) -> PredictionResult:
    """Predict energies and forces using the force field.

    Computes classical force field energies and forces via automatic
    differentiation, with optional filtering and weighting.

    Parameters
    ----------
    dataset : datasets.Dataset
        Dataset containing reference energies and forces.
    force_field : smee.TensorForceField
        Force field for predictions.
    tensor_systems : dict[str, smee.TensorSystem]
        Dictionary mapping mixture IDs to tensor systems.
    reference : Literal["mean", "min", "none"]
        Reference energy mode:
        - "mean": Subtract mean energy
        - "min": Subtract minimum energy
        - "none": Use absolute energies
    normalize : bool
        Whether to normalize by number of conformers/atoms.
    energy_cutoff : float, optional
        Energy cutoff in kcal/mol to filter high-energy conformers.
    weighting_method : Literal["uniform", "boltzmann"]
        Method to weight conformers in loss computation.
    weighting_temperature : float
        Temperature in Kelvin for Boltzmann weighting.
    device : str
        Device for computations.

    Returns
    -------
    PredictionResult
        Prediction results including energies, forces, and weights.

    Examples
    --------
    >>> result = predict_energies_forces(
    ...     dataset, force_field, tensor_systems, reference="mean"
    ... )
    >>> result.energy_pred.shape
    torch.Size([100])
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

        # Expand weights for forces
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

    return PredictionResult(
        energy_ref=energy_ref_all,
        energy_pred=energy_pred_all,
        forces_ref=forces_ref_all,
        forces_pred=forces_pred_all,
        weights_energy=weights_all,
        weights_forces=weights_forces_all,
        mask_idxs=all_mask_idxs,
    )


def compute_loss(
    energy_ref: torch.Tensor,
    energy_pred: torch.Tensor,
    forces_ref: torch.Tensor,
    forces_pred: torch.Tensor,
    weights_energy: torch.Tensor | None = None,
    weights_forces: torch.Tensor | None = None,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
) -> LossResult:
    """Compute weighted loss for energies and forces.

    Parameters
    ----------
    energy_ref : torch.Tensor
        Reference energies.
    energy_pred : torch.Tensor
        Predicted energies.
    forces_ref : torch.Tensor
        Reference forces.
    forces_pred : torch.Tensor
        Predicted forces.
    weights_energy : torch.Tensor, optional
        Weights for each energy point.
    weights_forces : torch.Tensor, optional
        Weights for each force component.
    energy_weight : float
        Weight for energy loss term.
    force_weight : float
        Weight for force loss term.

    Returns
    -------
    LossResult
        Result containing total, energy, and force losses.

    Examples
    --------
    >>> loss = compute_loss(e_ref, e_pred, f_ref, f_pred)
    >>> loss.total_loss.backward()
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

    return LossResult(
        total_loss=total_loss,
        energy_loss=energy_loss,
        force_loss=force_loss,
    )


def train_parameters(
    trainable: descent.train.Trainable,
    dataset: datasets.Dataset,
    tensor_systems: dict[str, smee.TensorSystem],
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    reference: Literal["mean", "min", "none"] = "none",
    normalize: bool = True,
    energy_cutoff: float | None = None,
    weighting_method: Literal["uniform", "boltzmann"] = "uniform",
    weighting_temperature: float = 298.15,
    device: str = "cpu",
    initial_perturbation: float = 0.2,
    verbose: bool = True,
) -> TrainingResult:
    """Train force field parameters to match reference data.

    Uses gradient-based optimization to fit LJ parameters (epsilon, sigma)
    to reproduce reference energies and forces from an ML potential.

    Parameters
    ----------
    trainable : descent.train.Trainable
        Trainable object with parameters to optimize.
    dataset : datasets.Dataset
        Training dataset with reference energies and forces.
    tensor_systems : dict[str, smee.TensorSystem]
        Dictionary mapping mixture IDs to tensor systems.
    n_epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for Adam optimizer.
    energy_weight : float
        Weight for energy loss term.
    force_weight : float
        Weight for force loss term.
    reference : Literal["mean", "min", "none"]
        Reference energy mode.
    normalize : bool
        Whether to normalize losses.
    energy_cutoff : float, optional
        Energy cutoff for filtering.
    weighting_method : Literal["uniform", "boltzmann"]
        Conformer weighting method.
    weighting_temperature : float
        Temperature for Boltzmann weighting.
    device : str
        Device for computations.
    initial_perturbation : float
        Magnitude of initial parameter perturbation.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    TrainingResult
        Training results including initial/final parameters and loss history.

    Examples
    --------
    >>> result = train_parameters(
    ...     trainable, dataset, tensor_systems,
    ...     n_epochs=100, learning_rate=0.01
    ... )
    >>> result.energy_losses[-1]  # Final energy loss
    0.0023
    """
    initial_params = trainable.to_values().clone()
    params = trainable.to_values().to(device).detach().requires_grad_(True)

    # Initially perturb the parameters
    with torch.no_grad():
        params += torch.empty_like(params).uniform_(
            -initial_perturbation, initial_perturbation
        )

    optimizer = torch.optim.Adam([params], lr=learning_rate, amsgrad=False)

    energy_losses = []
    force_losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # Predict with current parameters
        prediction = predict_energies_forces(
            dataset,
            trainable.to_force_field(params.abs()).to(device),
            tensor_systems,
            reference=reference,
            normalize=normalize,
            energy_cutoff=energy_cutoff,
            weighting_method=weighting_method,
            weighting_temperature=weighting_temperature,
            device=device,
        )

        # Compute loss
        loss = compute_loss(
            prediction.energy_ref,
            prediction.energy_pred,
            prediction.forces_ref,
            prediction.forces_pred,
            weights_energy=prediction.weights_energy,
            weights_forces=prediction.weights_forces,
            energy_weight=energy_weight,
            force_weight=force_weight,
        )

        # Backpropagation
        loss.total_loss.backward()
        optimizer.step()

        # Record losses
        energy_losses.append(loss.energy_loss.item())
        force_losses.append(loss.force_loss.item())

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch}: loss_energy = {loss.energy_loss.item():.4e}, "
                f"loss_forces = {loss.force_loss.item():.4e}"
            )

    return TrainingResult(
        initial_parameters=initial_params,
        trained_parameters=params.abs(),
        energy_losses=energy_losses,
        force_losses=force_losses,
    )
