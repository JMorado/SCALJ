"""Single-GPU training loop."""

import datasets
import descent.train
import descent.utils.loss
import smee
import torch
from loguru import logger
from tqdm import tqdm

from ..types import TrainingResult
from ._loss import get_losses
from ._types import LossConfig, ReferenceMode, WeightingMethod


def _initialize_training(
    trainable: descent.train.Trainable,
    learning_rate: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.optim.Adam]:
    """
    Initialize parameters and optimizer for training.

    Parameters
    ----------
    trainable : descent.train.Trainable
        Trainable object.
    learning_rate : float
        Learning rate for optimizer.
    device : str
        Device for parameters.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.optim.Adam]
        Initial params, trainable params, and optimizer.
    """
    initial_params = trainable.to_values().clone()
    params = trainable.to_values().to(device).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=learning_rate, amsgrad=False)

    return initial_params, params, optimizer


def _run_epoch(
    params: torch.Tensor,
    trainable: descent.train.Trainable,
    dataset: datasets.Dataset,
    tensor_systems: dict[str, smee.TensorSystem],
    config: LossConfig,
    conformer_batch_size: int,
    device: str,
    epoch: int,
    verbose: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run a single training epoch.

    Parameters
    ----------
    params : torch.Tensor
        Current parameters.
    trainable : descent.train.Trainable
        Trainable object.
    dataset : datasets.Dataset
        Training dataset.
    tensor_systems : dict[str, smee.TensorSystem]
        Tensor systems.
    config : LossConfig
        Training configuration.
    conformer_batch_size : int
        Batch size for conformers.
    device : str
        Device for computation.
    epoch : int
        Current epoch number.
    verbose : bool
        Whether to show progress.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Accumulated gradient and epoch losses.
    """
    n_entries = len(dataset)

    # Initialize epoch accumulators.
    epoch_loss = torch.zeros(1, device=device)
    epoch_energy_loss = torch.zeros(1, device=device)
    epoch_force_loss = torch.zeros(1, device=device)
    accumulated_grad = None

    # Shuffle dataset indices.
    shuffled_indices = torch.randperm(n_entries).tolist()

    # Process each entry.
    for entry_idx in tqdm(
        shuffled_indices,
        total=n_entries,
        desc=f"Epoch {epoch}",
        leave=False,
        disable=not verbose,
    ):
        entry = dataset[entry_idx]
        entry = {k: v.to(device) if hasattr(v, "to") else v for k, v in entry.items()}

        # Compute loss and gradient for this entry.
        entry_loss, entry_energy_loss, entry_force_loss, entry_grad = get_losses(
            params,
            trainable,
            entry,
            tensor_systems,
            conformer_batch_size=conformer_batch_size,
            energy_weight=config.energy_weight,
            force_weight=config.force_weight,
            reference=config.reference,
            energy_cutoff=config.energy_cutoff,
            weighting_method=config.weighting_method,
            weighting_temperature=config.weighting_temperature,
            device=device,
            compute_forces=config.compute_forces,
        )

        # Accumulate gradient.
        if accumulated_grad is None:
            accumulated_grad = entry_grad
        else:
            accumulated_grad = accumulated_grad + entry_grad

        # Accumulate losses.
        epoch_loss += entry_loss
        epoch_energy_loss += entry_energy_loss
        epoch_force_loss += entry_force_loss

    return accumulated_grad, epoch_loss, epoch_energy_loss, epoch_force_loss


def train_parameters(
    trainable: descent.train.Trainable,
    dataset: datasets.Dataset,
    tensor_systems: dict[str, smee.TensorSystem],
    n_epochs: int = 100,
    conformer_batch_size: int = 2,
    learning_rate: float = 0.01,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    reference: ReferenceMode = "none",
    energy_cutoff: float | None = None,
    weighting_method: WeightingMethod = "uniform",
    weighting_temperature: float = 298.15,
    device: str = "cuda",
    verbose: bool = True,
    compute_forces: bool = True,
) -> TrainingResult:
    """
    Train force field parameters to match reference data.

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
    conformer_batch_size : int
        Number of conformers to process at once within each entry before
        accumulating gradients. Smaller values use less memory.
    learning_rate : float
        Learning rate for Adam optimizer.
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
    verbose : bool
        Whether to print progress.
    compute_forces : bool
        Whether to compute forces.

    Returns
    -------
    TrainingResult
        Training results including initial/final parameters and loss history.

    Examples
    --------
    >>> result = train_parameters(
    ...     trainable, dataset, tensor_systems,
    ...     n_epochs=100, conformer_batch_size=8, learning_rate=0.01
    ... )
    >>> result.energy_losses[-1]  # Final energy loss
    0.0023
    """
    # Initialize training.
    initial_params, params, optimizer = _initialize_training(
        trainable, learning_rate, device
    )

    # Build config.
    config = LossConfig(
        energy_weight=energy_weight,
        force_weight=force_weight,
        reference=reference,
        energy_cutoff=energy_cutoff,
        weighting_method=weighting_method,
        weighting_temperature=weighting_temperature,
        compute_forces=compute_forces,
    )

    energy_losses = []
    force_losses = []
    total_losses = []
    n_entries = len(dataset)

    for epoch in range(n_epochs):
        accumulated_grad, epoch_loss, epoch_energy_loss, epoch_force_loss = _run_epoch(
            params=params,
            trainable=trainable,
            dataset=dataset,
            tensor_systems=tensor_systems,
            config=config,
            conformer_batch_size=conformer_batch_size,
            device=device,
            epoch=epoch,
            verbose=verbose,
        )

        # Update parameters.
        params.grad = accumulated_grad
        optimizer.step()
        optimizer.zero_grad()

        # Record losses.
        avg_energy_loss = epoch_energy_loss.item() / n_entries
        avg_force_loss = epoch_force_loss.item() / n_entries
        avg_total_loss = epoch_loss.item() / n_entries

        energy_losses.append(avg_energy_loss)
        force_losses.append(avg_force_loss)
        total_losses.append(avg_total_loss)

        if verbose:
            logger.info(
                f"Epoch {epoch}: total_loss={avg_total_loss:.4e}, "
                f"energy_loss={avg_energy_loss:.4e}, "
                f"force_loss={avg_force_loss:.4e}"
            )

    return TrainingResult(
        initial_parameters=initial_params,
        trained_parameters=params.abs(),
        energy_losses=energy_losses,
        force_losses=force_losses,
        combined_losses=total_losses,
    )


def train_from_closure(
    trainable: descent.train.Trainable,
    closure_fn: descent.utils.loss.ClosureFn,
    n_epochs: int = 100,
    learning_rate: float = 0.01,
    device: str = "cuda",
    verbose: bool = True,
) -> TrainingResult:
    """
    Train using a pre-built closure function (e.g. from ``combine_closures``).

    Parameters
    ----------
    trainable : descent.train.Trainable
        Trainable object with parameters to optimize.
    closure_fn : ClosureFn
        A callable ``(x, compute_gradient, compute_hessian) -> (loss, grad, hessian)``.
        Typically built with ``to_scalej_closure``, ``combine_closures``,
        or ``dimers.default_closure``.
    n_epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for the Adam optimizer.
    device : str
        Compute device for parameters.
    verbose : bool
        Whether to log per-epoch loss.

    Returns
    -------
    TrainingResult
        Training results. ``energy_losses`` and ``force_losses`` are empty lists
        because individual target losses are not separately tracked here;
        use ``verbose=True`` in ``combine_closures`` for per-target breakdown.

    Examples
    --------
    >>> from scalej.training import to_scalej_closure, train_from_closure, LossConfig
    >>> from descent.targets import dimers
    >>> from descent.utils.loss import combine_closures
    >>>
    >>> config = LossConfig(energy_weight=1.0, force_weight=1.0, reference="min")
    >>> scalej_closure = to_scalej_closure(trainable, dataset, tensor_systems, config)
    >>> dimer_closure  = dimers.default_closure(trainable, topologies, dimer_dataset)
    >>>
    >>> combined = combine_closures(
    ...     {"scalej": scalej_closure, "dimers": dimer_closure},
    ...     weights={"scalej": 1.0, "dimers": 0.5},
    ...     verbose=True,
    ... )
    >>> result = train_from_closure(trainable, combined, n_epochs=100)
    """
    initial_params, params, optimizer = _initialize_training(
        trainable, learning_rate, device
    )

    total_losses = []

    for epoch in range(n_epochs):
        loss, grad, _ = closure_fn(params, compute_gradient=True, compute_hessian=False)

        params.grad = grad
        optimizer.step()
        optimizer.zero_grad()

        total_losses.append(loss.item())

        if verbose:
            parts = [f"Epoch {epoch}: total_loss={loss.item():.4e}"]
            per_target = getattr(closure_fn, "last_losses", None)
            if per_target:
                parts.extend(f"{k}={v:.4e}" for k, v in per_target.items())
            logger.info(" | ".join(parts))

    # We don't track energy and force losses separately here.
    # The total combined loss is given in combined_losses.
    return TrainingResult(
        initial_parameters=initial_params,
        trained_parameters=params.abs(),
        energy_losses=[],
        force_losses=[],
        combined_losses=total_losses,
    )
