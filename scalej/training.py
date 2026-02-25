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
    from .config import AttributeConfig, ParameterConfig, TrainingConfig


def create_trainable(
    force_field: smee.TensorForceField,
    parameters_cols: list[str] = ["epsilon", "sigma"],
    parameters_scales: dict[str, float] | None = None,
    parameters_limits: dict[str, tuple[float, float]] | None = None,
    attributes_cols: list[str] = [],
    attributes_scales: dict[str, float] | None = None,
    attributes_limits: dict[str, tuple[float, float]] | None = None,
    device: str = "cpu",
) -> descent.train.Trainable:
    """Create a trainable object for parameter optimization.

    Parameters
    ----------
    force_field : smee.TensorForceField
        The force field with parameters to train.
    parameters_cols : list[str]
        Parameter columns to optimize (e.g., ["epsilon", "sigma"]).
    parameters_scales : dict[str, float], optional
        Scaling factors for each parameter type.
    parameters_limits : dict[str, tuple[float, float]], optional
        Min/max limits for each parameter type.
    attributes_cols : list[str]
        Attribute columns to optimize (e.g., ["charge"]).
    attributes_scales : dict[str, float], optional
        Scaling factors for each attribute type.
    attributes_limits : dict[str, tuple[float, float]], optional
        Min/max limits for each attribute type.
    device : str
        Device to use for training.

    Returns
    -------
    descent.train.Trainable
        Trainable object for optimization.

    Examples
    --------
    >>> trainable = create_trainable(force_field, parameters_cols=["epsilon", "sigma"])
    """
    parameters_scales = parameters_scales or {}
    parameters_limits = parameters_limits or {}
    attributes_scales = attributes_scales or {}
    attributes_limits = attributes_limits or {}

    # Create trainable parameter config for vdW parameters
    vdw_parameter_config = descent.train.ParameterConfig(
        cols=parameters_cols,
        scales=parameters_scales,
        limits=parameters_limits,
    )

    vdw_attribute_config = descent.train.AttributeConfig(
        cols=attributes_cols,
        scales=attributes_scales,
        limits=attributes_limits,
    )

    # Ensure vdW parameters require gradients
    vdw_potential = force_field.potentials_by_type["vdW"]
    vdw_potential.parameters.requires_grad = True

    # Create trainable object
    trainable = descent.train.Trainable(
        force_field=force_field.to(device),
        parameters={"vdW": vdw_parameter_config},
        attributes={"vdW": vdw_attribute_config},
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
    """
    Predict energies and forces using the force field.

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
        coords_raw = coords_flat.reshape(len(energy_ref), -1, 3).to(device)

        box_vectors_flat = smee.utils.tensor_like(
            entry["box_vectors"], force_field.potentials[0].parameters
        )
        box_vectors = (
            box_vectors_flat.reshape(len(energy_ref), 3, 3).to(device).detach()
        )

        system = tensor_systems[mixture_id].to(device)

        energy_pred_list = []
        forces_pred_list = []
        for coord_raw, box_vector in tqdm(
            zip(coords_raw, box_vectors),
            total=len(coords_raw),
            desc="Predicting energies/forces",
            leave=False,
        ):
            coord = coord_raw.detach().requires_grad_(True)
            e = smee.compute_energy(system, force_field, coord, box_vector)
            f = -torch.autograd.grad(
                e, coord, create_graph=True, retain_graph=True, allow_unused=False
            )[0]
            energy_pred_list.append(e)
            forces_pred_list.append(f)

        energy_pred = torch.stack(energy_pred_list)
        forces_pred = torch.stack(forces_pred_list)

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
    energy_ref_all = torch.cat(energy_ref_all).to(energy_pred_all)

    forces_pred_all = torch.cat(forces_pred_all)
    forces_ref_all = torch.cat(forces_ref_all).to(forces_pred_all)

    weights_all = torch.cat(weights_all).to(energy_pred_all)
    weights_all = weights_all / weights_all.sum().clamp(min=1e-16)

    weights_forces_all = torch.cat(weights_forces_all).unsqueeze(1).to(forces_pred_all)
    weights_forces_all = weights_forces_all / weights_forces_all.sum().clamp(min=1e-16)

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
    """Compute weighted MSE loss for energies and forces.

    Parameters
    ----------
    energy_ref, energy_pred : torch.Tensor
        Reference and predicted per-molecule energies, shape ``[N]``.
    forces_ref, forces_pred : torch.Tensor
        Reference and predicted forces, shape ``[N, n_atoms, 3]``.
    weights_energy : torch.Tensor, optional
        Per-conformer weights for energy loss.  Defaults to uniform.
    weights_forces : torch.Tensor, optional
        Per-element weights for force loss.  Defaults to uniform.
    energy_weight : float
        Global scale for the energy loss term.
    force_weight : float
        Global scale for the force loss term.

    Returns
    -------
    LossResult
        ``total_loss``, ``energy_loss``, and ``force_loss``.
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


def _train_on_entry(
    entry: dict,
    params: torch.Tensor,
    trainable: "descent.train.Trainable",
    tensor_systems: dict[str, smee.TensorSystem],
    device: str,
    reference: str,
    normalize: bool,
    energy_cutoff: float | None,
    weighting_method: str,
    weighting_temperature: float,
    energy_weight: float,
    force_weight: float,
    frame_batch_size: int = 8,
) -> tuple[float, float]:
    """
    Train on one dataset entry.

    Notes
    -----
    This uses mini-batched for the forward and backward passes.
    Configurations are processes in ``frame_batch_size``.

    Parameters
    ----------
    entry : dict
        Dataset entry.
    params : torch.Tensor
        Parameters of the force field.
    trainable : descent.train.Trainable
        Trainable object.
    tensor_systems : dict[str, smee.TensorSystem]
        Tensor systems.
    device : str
        Device to use.
    reference : str
        Reference method.
    normalize : bool
        Normalize energies and forces.
    energy_cutoff : float | None
        Energy cutoff.
    weighting_method : str
        Weighting method.
    weighting_temperature : float
        Weighting temperature.
    energy_weight : float
        Energy weight.
    force_weight : float
        Force weight.
    frame_batch_size : int
        Frame batch size.

    Returns
    -------
    tuple[float, float]
        Energy and force loss.
    """
    mixture_id = entry["mixture_id"]
    energy_ref_raw = entry["energy"].to(device)
    forces_ref_raw = entry["forces"].reshape(len(energy_ref_raw), -1, 3).to(device)

    _dtype = params.dtype
    coords_raw = (
        torch.as_tensor(entry["coords"], dtype=_dtype, device=device)
        .clone()
        .detach()
        .reshape(len(energy_ref_raw), -1, 3)
    )
    box_vectors = (
        torch.as_tensor(entry["box_vectors"], dtype=_dtype, device=device)
        .clone()
        .detach()
        .reshape(len(energy_ref_raw), 3, 3)
    )

    system = tensor_systems[mixture_id].to(device)
    n_mols = sum(system.n_copies)
    n_atoms = forces_ref_raw.shape[1]
    n_frames = len(energy_ref_raw)

    energy_ref_n = (energy_ref_raw / n_mols).detach()

    # Pre-pass to compute reference offsets, Boltzmann weights, normalisers.
    needs_model = reference.lower() in ("mean", "min")
    with torch.no_grad():
        if needs_model:
            ff_ng = trainable.to_force_field(params.detach().abs()).to(device)
            energy_pred_ng = (
                torch.stack(
                    [
                        smee.compute_energy(system, ff_ng, c.detach(), b)
                        for c, b in zip(coords_raw, box_vectors)
                    ]
                )
                / n_mols
            )
            del ff_ng
        else:
            energy_pred_ng = None

        if reference.lower() == "mean":
            e_ref_0 = energy_ref_n.mean().item()
            e_pred_0 = energy_pred_ng.mean().item()
        elif reference.lower() == "min":
            mi = energy_ref_n.argmin().item()
            e_ref_0 = energy_ref_n[mi].item()
            e_pred_0 = energy_pred_ng[mi].item()
        else:
            e_ref_0, e_pred_0 = 0.0, 0.0

        if energy_cutoff is not None:
            mask = (energy_ref_n - energy_ref_n.min()) <= energy_cutoff
        else:
            mask = torch.ones(n_frames, dtype=torch.bool, device=device)
        valid_idx = torch.where(mask)[0]
        n_valid = len(valid_idx)

        if n_valid == 0:
            return 0.0, 0.0

        if weighting_method == "boltzmann":
            kBT = (
                openmm.unit.AVOGADRO_CONSTANT_NA
                * openmm.unit.BOLTZMANN_CONSTANT_kB
                * weighting_temperature
            ).value_in_unit(openmm.unit.kilocalories_per_mole)
            e_rel = energy_ref_n[valid_idx] - energy_ref_n[valid_idx].min()
            w = torch.exp(-e_rel / kBT)
        else:
            w = torch.ones(n_valid, device=device)
        w = w / w.sum()

        energy_ref_shifted = energy_ref_n[valid_idx] - e_ref_0
        forces_ref_valid = forces_ref_raw[valid_idx]

        var_e = float(energy_ref_shifted.var().clamp(min=1e-16))
        var_f = float(forces_ref_valid.var().clamp(min=1e-16))

        inv_n = 1.0 / n_valid if normalize else 1.0
        inv_fn = 1.0 / (n_valid * n_atoms * 3) if normalize else 1.0

    # Mini-batch grad pass
    total_e_loss = 0.0
    total_f_loss = 0.0
    n_batches = (n_valid + frame_batch_size - 1) // frame_batch_size
    for batch_num in tqdm(
        range(n_batches),
        desc=f"  batches [{mixture_id}]",
        leave=False,
    ):
        # Slice tensors
        start = batch_num * frame_batch_size
        end = min(start + frame_batch_size, n_valid)
        batch_frame_ids = valid_idx[start:end]
        w_b = w[start:end].to(dtype=params.dtype)

        # Get force field for this batch
        ff = trainable.to_force_field(params.abs()).to(device)

        # Get coordinates and boxes for this batch
        coords_b = [
            coords_raw[fi].detach().requires_grad_(True)
            for fi in batch_frame_ids.tolist()
        ]
        boxes_b = [box_vectors[fi] for fi in batch_frame_ids.tolist()]

        e_batch = torch.stack(
            [
                smee.compute_energy(system, ff, c, b) / n_mols
                for c, b in zip(coords_b, boxes_b)
            ]
        )

        forces_tuple = torch.autograd.grad(
            e_batch.sum(),
            inputs=coords_b,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )
        f_batch = torch.stack([-f for f in forces_tuple])

        e_ref_b = energy_ref_n[batch_frame_ids] - e_ref_0
        f_ref_b = forces_ref_raw[batch_frame_ids]

        diff_e = e_batch - e_pred_0 - e_ref_b
        diff_f = f_batch - f_ref_b

        batch_e = energy_weight * (w_b * diff_e**2).sum() * inv_n / var_e
        batch_f = force_weight * (w_b[:, None, None] * diff_f**2).sum() * inv_fn / var_f
        batch_loss = batch_e + batch_f
        total_e_loss += batch_e.detach().item()
        total_f_loss += batch_f.detach().item()
        batch_loss.backward(inputs=[params])

        del (
            ff,
            coords_b,
            e_batch,
            forces_tuple,
            f_batch,
            e_ref_b,
            f_ref_b,
            w_b,
            diff_e,
            diff_f,
            batch_e,
            batch_f,
            batch_loss,
        )

    return total_e_loss, total_f_loss


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
    frame_batch_size: int = 8,
    verbose: bool = True,
) -> TrainingResult:
    """Train force field parameters to match reference data.

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
        Weight for energy loss term (λ_E).
    force_weight : float
        Weight for force loss term (λ_F).
    reference : Literal["mean", "min", "none"]
        Reference energy subtraction mode.
    normalize : bool
        Whether to normalise losses by variance.
    energy_cutoff : float, optional
        kcal/mol above minimum; frames above are excluded.
    weighting_method : Literal["uniform", "boltzmann"]
        Per-conformer weighting scheme.
    weighting_temperature : float
        Temperature in K for Boltzmann weighting.
    device : str
        Device for computations.
    initial_perturbation : float
        Uniform noise magnitude applied to the initial parameters.
    frame_batch_size : int
        Conformers processed per backward pass; tune to fill VRAM.
    verbose : bool
        Print loss every 10 epochs.

    Returns
    -------
    TrainingResult
        Loss history and initial/final parameters.
    """
    initial_params = trainable.to_values().clone()
    params = trainable.to_values().to(device).detach().requires_grad_(True)

    # Perturb initial parameters
    with torch.no_grad():
        params += torch.empty_like(params).uniform_(
            -initial_perturbation, initial_perturbation
        )

    optimizer = torch.optim.Adam([params], lr=learning_rate, amsgrad=False)

    energy_losses = []
    force_losses = []

    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        optimizer.zero_grad()

        epoch_energy_loss = 0.0
        epoch_force_loss = 0.0
        n_mixtures = 0

        for entry in tqdm(dataset, desc=f"Epoch {epoch} mixtures", leave=False):
            e_loss, f_loss = _train_on_entry(
                entry=entry,
                params=params,
                trainable=trainable,
                tensor_systems=tensor_systems,
                device=device,
                reference=reference,
                normalize=normalize,
                energy_cutoff=energy_cutoff,
                weighting_method=weighting_method,
                weighting_temperature=weighting_temperature,
                energy_weight=energy_weight,
                force_weight=force_weight,
                frame_batch_size=frame_batch_size,
            )
            epoch_energy_loss += e_loss
            epoch_force_loss += f_loss
            n_mixtures += 1

        optimizer.step()

        if n_mixtures > 0:
            epoch_energy_loss /= n_mixtures
            epoch_force_loss /= n_mixtures

        energy_losses.append(epoch_energy_loss)
        force_losses.append(epoch_force_loss)

        if verbose and epoch % 1 == 0:
            print(
                f"Epoch {epoch}: loss_energy = {epoch_energy_loss:.4e}, "
                f"loss_forces = {epoch_force_loss:.4e}"
            )

    return TrainingResult(
        initial_parameters=initial_params,
        trained_parameters=params.abs(),
        energy_losses=energy_losses,
        force_losses=force_losses,
    )


def _worker_fn_thread(
    rank: int,
    world_size: int,
    params_snapshot: torch.Tensor,
    trainable_bytes: bytes,
    dataset: datasets.Dataset,
    tensor_systems: dict,
    reference: str,
    normalize: bool,
    energy_cutoff,
    weighting_method: str,
    weighting_temperature: float,
    energy_weight: float,
    force_weight: float,
    frame_batch_size: int,
    results: list,
) -> None:
    """
    Thread worker executed on ``cuda:{rank}``.

    Parameters
    ----------
    rank : int
        Rank of the current thread.
    world_size : int
        Total number of threads.
    params_snapshot : torch.Tensor
        Snapshot of the parameters to use for this epoch.
    trainable_bytes : bytes
        Serialized trainable object.
        This is the result of ``torch.save(trainable, buf)`` in the main thread.
        Each worker deserialises its own copy with ``torch.load`` so that
        ``to_force_field()`` calls are fully isolated (no shared mutable state
        between threads) and non-leaf tensors inside the trainable are safely
        reconstructed as plain leaf tensors.
    dataset : datasets.Dataset
        Training dataset.
    tensor_systems : dict
        Dictionary of tensor systems.
    reference : str
        Reference energy subtraction mode.
    normalize : bool
        Whether to normalize losses by variance.
    energy_cutoff : float
        Energy cutoff for excluding frames.
    weighting_method : str
        Weighting method for conformers.
    weighting_temperature : float
        Temperature for Boltzmann weighting.
    energy_weight : float
        Weight for energy loss term.
    force_weight : float
        Weight for force loss term.
    frame_batch_size : int
        Number of frames to process per batch.
    results : list
        List to store results.
    """
    import io as _io

    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    # Deserialise a thread-local trainable with all tensors on CPU
    local_trainable = torch.load(
        _io.BytesIO(trainable_bytes), weights_only=False, map_location="cpu"
    )

    # Keep params on CPU. This will be moved to cuda:rank inside _train_on_entry
    params_local = params_snapshot.detach().clone().requires_grad_(True)

    total_e_loss = 0.0
    total_f_loss = 0.0
    n_entries = 0

    # Interleaved slicing distributes entries evenly across threads.
    indices = list(range(rank, len(dataset), world_size))

    for idx in indices:
        entry = dataset[idx]
        e_loss, f_loss = _train_on_entry(
            entry=entry,
            params=params_local,
            trainable=local_trainable,
            tensor_systems=tensor_systems,
            device=device,
            reference=reference,
            normalize=normalize,
            energy_cutoff=energy_cutoff,
            weighting_method=weighting_method,
            weighting_temperature=weighting_temperature,
            energy_weight=energy_weight,
            force_weight=force_weight,
            frame_batch_size=frame_batch_size,
        )
        total_e_loss += e_loss
        total_f_loss += f_loss
        n_entries += 1

    grad_cpu = (
        params_local.grad.detach().cpu()
        if params_local.grad is not None
        else torch.zeros_like(params_snapshot)
    )

    results[rank] = (total_e_loss, total_f_loss, n_entries, grad_cpu)


def train_parameters_ddp(
    trainable: "descent.train.Trainable",
    dataset: datasets.Dataset,
    tensor_systems: dict[str, smee.TensorSystem],
    n_gpus: int = 2,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    reference: Literal["mean", "min", "none"] = "none",
    normalize: bool = True,
    energy_cutoff: float | None = None,
    weighting_method: Literal["uniform", "boltzmann"] = "uniform",
    weighting_temperature: float = 298.15,
    initial_perturbation: float = 0.2,
    frame_batch_size: int = 8,
    verbose: bool = True,
) -> TrainingResult:
    """
    Train parameters using data-parallel multi-GPU evaluation.

    Notes
    -----
    Dataset entries are distributed across ``n_gpus`` CUDA devices in an
    interleaved fashion (``rank, rank + n_gpus, rank + 2*n_gpus, ...``).
    Each GPU runs in its own Python thread; the GIL is released during CUDA
    kernel calls and autograd, so GPU computation proceeds in parallel.
    Each thread holds its own ``deepcopy`` of *trainable* and its own clone
    of the parameter tensor, so there is no shared mutable state between
    threads.  Gradients are averaged on the main thread before each
    optimizer step.

    Parameters
    ----------
    trainable : descent.train.Trainable
        Trainable object with parameters to optimize.
    dataset : datasets.Dataset
        Training dataset with reference energies and forces.
    tensor_systems : dict[str, smee.TensorSystem]
        Dictionary mapping mixture IDs to tensor systems.
    n_gpus : int
        Number of CUDA GPUs to use.  Must be <= ``torch.cuda.device_count()``.
    n_epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for Adam optimizer.
    energy_weight : float
        Weight for energy loss term.
    force_weight : float
        Weight for force loss term.
    reference : Literal["mean", "min", "none"]
        Reference energy subtraction mode.
    normalize : bool
        Whether to normalise losses by variance.
    energy_cutoff : float, optional
        kcal/mol above minimum; frames above are excluded.
    weighting_method : Literal["uniform", "boltzmann"]
        Per-conformer weighting scheme.
    weighting_temperature : float
        Temperature in K for Boltzmann weighting.
    initial_perturbation : float
        Uniform noise magnitude applied to the initial parameters.
    frame_batch_size : int
        Conformers processed per backward pass per thread.
    verbose : bool
        Print loss every 10 epochs.

    Returns
    -------
    TrainingResult
        Loss history and initial/final parameters.
    """
    import io as _io
    import threading

    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError(
            "train_parameters_ddp requires at least one CUDA GPU, but none were found."
        )
    if n_gpus > available:
        raise ValueError(
            f"Requested n_gpus={n_gpus} but only {available} CUDA device(s) available."
        )

    # Get initial parameters
    initial_params = trainable.to_values().clone()
    params = initial_params.clone().detach().cpu().requires_grad_(True)

    # Add noise to initial parameters
    with torch.no_grad():
        params.data += torch.empty_like(params).uniform_(
            -initial_perturbation, initial_perturbation
        )

    # Set up optimizer
    optimizer = torch.optim.Adam([params], lr=learning_rate, amsgrad=False)

    # Serialise trainable once in the main thread.
    _buf = _io.BytesIO()
    torch.save(trainable, _buf)
    trainable_bytes = _buf.getvalue()
    del _buf

    # Pre-initialise CUDA on every target device serially in the main thread.
    for _rank in range(n_gpus):
        with torch.cuda.device(_rank):
            _ = torch.zeros(1, device=f"cuda:{_rank}")
            _det = torch.det(torch.eye(3, device=f"cuda:{_rank}"))
            torch.cuda.synchronize(_rank)
    del _, _det

    # Lists to store loss history
    energy_losses: list[float] = []
    force_losses: list[float] = []

    # Training loop
    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        optimizer.zero_grad()

        # Take a detached snapshot of the current params for this epoch.
        # Each thread clones it onto its own device independently.
        params_snapshot = params.detach().clone().cpu()

        # Pre-allocate result slots; threads write directly by rank index.
        results: list = [None] * n_gpus

        threads = []
        for rank in range(n_gpus):
            t = threading.Thread(
                target=_worker_fn_thread,
                args=(
                    rank,
                    n_gpus,
                    params_snapshot,
                    trainable_bytes,
                    dataset,
                    tensor_systems,
                    reference,
                    normalize,
                    energy_cutoff,
                    weighting_method,
                    weighting_temperature,
                    energy_weight,
                    force_weight,
                    frame_batch_size,
                    results,
                ),
                daemon=True,
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Average gradients over workers that processed at least one entry.
        all_grads = [r[3] for r in results]
        n_workers_with_entries = sum(1 for r in results if r[2] > 0)

        if n_workers_with_entries > 0:
            avg_grad = torch.stack(all_grads).sum(dim=0) / n_workers_with_entries
        else:
            avg_grad = torch.zeros_like(params)

        params.grad = avg_grad
        optimizer.step()

        # Epoch-level loss
        # r[0] = energy loss
        # r[1] = force loss
        # r[2] = number of entries
        total_e = sum(r[0] for r in results)
        total_f = sum(r[1] for r in results)
        total_n = sum(r[2] for r in results)

        epoch_e = total_e / max(total_n, 1)
        epoch_f = total_f / max(total_n, 1)
        energy_losses.append(epoch_e)
        force_losses.append(epoch_f)

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch}: loss_energy = {epoch_e:.4e}, "
                f"loss_forces = {epoch_f:.4e}  "
                f"[{n_workers_with_entries}/{n_gpus} GPU(s) active]"
            )

    return TrainingResult(
        initial_parameters=initial_params,
        trained_parameters=params.abs(),
        energy_losses=energy_losses,
        force_losses=force_losses,
    )
