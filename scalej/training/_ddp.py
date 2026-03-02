"""Multi-GPU data-parallel (DDP) training loop."""

import datasets
import descent.train
import smee
import torch
from loguru import logger
from tqdm import tqdm

from ..models import TrainingResult
from ._loss import get_losses
from ._types import LossConfig, ReferenceMode, WeightingMethod
from ._train import _initialize_training


def _worker_fn_ddp(
    rank: int,
    world_size: int,
    params_snapshot: torch.Tensor,
    trainable_bytes: bytes,
    dataset: datasets.Dataset,
    tensor_systems: dict[str, smee.TensorSystem],
    entry_indices: list[int],
    conformer_batch_size: int,
    config: LossConfig,
    results: list,
) -> None:
    """
    Worker thread for DDP training on cuda:{rank}.

    Each worker processes a subset of dataset entries and accumulates gradients
    using the memory-efficient pattern.

    Parameters
    ----------
    rank : int
        GPU rank (0, 1, 2, ...).
    world_size : int
        Total number of GPUs.
    params_snapshot : torch.Tensor
        Snapshot of current parameters (CPU tensor).
    trainable_bytes : bytes
        Serialized trainable object (via torch.save).
    dataset : datasets.Dataset
        Training dataset.
    tensor_systems : dict[str, smee.TensorSystem]
        Dictionary of tensor systems.
    entry_indices : list[int]
        Indices of entries this worker should process.
    conformer_batch_size : int
        Conformers per batch for memory efficiency.
    config : LossConfig
        Training configuration.
    results : list
        Shared list to store results (indexed by rank).
    """
    import io as _io
    import traceback as _traceback

    try:
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)

        # Deserialize a thread-local trainable to the target device
        local_trainable = torch.load(
            _io.BytesIO(trainable_bytes), weights_only=False, map_location=device
        )

        # Clone params to this device
        params_local = params_snapshot.detach().clone().to(device).requires_grad_(True)

        # Accumulators
        total_loss = torch.zeros(1, device=device)
        total_energy_loss = torch.zeros(1, device=device)
        total_force_loss = torch.zeros(1, device=device)
        accumulated_grad = None
        n_entries_processed = 0

        pbar = tqdm(entry_indices, desc=f"GPU {rank}", leave=False)
        for idx in pbar:
            entry = dataset[idx]
            mixture_id = entry.get("mixture_id", f"entry_{idx}")
            pbar.set_description(f"GPU {rank}: {mixture_id}")
            entry = {
                k: v.to(device) if hasattr(v, "to") else v for k, v in entry.items()
            }

            entry_loss, entry_energy_loss, entry_force_loss, entry_grad = (
                get_losses(
                    params_local,
                    local_trainable,
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
            )

            # Accumulate
            if accumulated_grad is None:
                accumulated_grad = entry_grad
            else:
                accumulated_grad = accumulated_grad + entry_grad

            total_loss += entry_loss
            total_energy_loss += entry_energy_loss
            total_force_loss += entry_force_loss
            n_entries_processed += 1

        # Move gradient to CPU for aggregation
        grad_cpu = (
            accumulated_grad.detach().cpu()
            if accumulated_grad is not None
            else torch.zeros_like(params_snapshot)
        )

        results[rank] = (
            total_loss.item(),
            total_energy_loss.item(),
            total_force_loss.item(),
            n_entries_processed,
            grad_cpu,
        )
    except Exception as e:
        print(f"Worker {rank} failed with error: {e}")
        _traceback.print_exc()
        raise


def train_parameters_ddp(
    trainable: descent.train.Trainable,
    dataset: datasets.Dataset,
    tensor_systems: dict[str, smee.TensorSystem],
    n_gpus: int = 2,
    n_epochs: int = 100,
    conformer_batch_size: int = 2,
    learning_rate: float = 0.01,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    reference: ReferenceMode = "none",
    energy_cutoff: float | None = None,
    weighting_method: WeightingMethod = "uniform",
    weighting_temperature: float = 298.15,
    initial_perturbation: float = 0.0,
    verbose: bool = True,
    compute_forces: bool = True,
) -> TrainingResult:
    """
    Train parameters using data-parallel multi-GPU evaluation.

    Dataset entries are distributed across GPUs. Each GPU processes its assigned
    entries using memory-efficient gradient accumulation. Gradients are aggregated
    on the main thread before each optimizer step.

    Architecture:
    - Main thread: holds optimizer, aggregates gradients, updates parameters
    - Worker threads: each runs on cuda:{rank}, processes subset of entries
    - Communication: via shared list (results), gradients moved to CPU

    Parameters
    ----------
    trainable : descent.train.Trainable
        Trainable object with parameters to optimize.
    dataset : datasets.Dataset
        Training dataset with reference energies and forces.
    tensor_systems : dict[str, smee.TensorSystem]
        Dictionary mapping mixture IDs to tensor systems.
    n_gpus : int
        Number of CUDA GPUs to use.
    n_epochs : int
        Number of training epochs.
    conformer_batch_size : int
        Conformers per batch for memory efficiency within each entry.
    learning_rate : float
        Learning rate for Adam optimizer.
    energy_weight : float
        Weight for energy loss term.
    force_weight : float
        Weight for force loss term.
    reference : ReferenceMode
        Reference energy mode.
    energy_cutoff : float, optional
        Energy cutoff for filtering high-energy conformers.
    weighting_method : WeightingMethod
        Conformer weighting scheme.
    weighting_temperature : float
        Temperature for Boltzmann weighting.
    initial_perturbation : float
        Magnitude of initial parameter perturbation.
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
    >>> result = train_parameters_ddp(
    ...     trainable, dataset, tensor_systems,
    ...     n_gpus=4, n_epochs=100, conformer_batch_size=4
    ... )
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

    # Initialize training
    initial_params, params, optimizer = _initialize_training(
        trainable, learning_rate, device="cpu"
    )

    # Build config
    config = LossConfig(
        energy_weight=energy_weight,
        force_weight=force_weight,
        reference=reference,
        energy_cutoff=energy_cutoff,
        weighting_method=weighting_method,
        weighting_temperature=weighting_temperature,
        compute_forces=compute_forces,
    )

    # Serialize trainable once (each worker deserializes its own copy)
    _buf = _io.BytesIO()
    torch.save(trainable, _buf)
    trainable_bytes = _buf.getvalue()
    del _buf

    # Pre-initialize CUDA on all target devices
    for _rank in range(n_gpus):
        with torch.cuda.device(_rank):
            _ = torch.zeros(1, device=f"cuda:{_rank}")
            torch.cuda.synchronize(_rank)

    n_entries = len(dataset)

    if verbose:
        logger.info(
            f"DDP training: {n_entries} entries across {n_gpus} GPUs "
            f"(~{n_entries // n_gpus} entries/GPU)"
        )
        if n_entries < n_gpus:
            logger.warning(
                f"Dataset has fewer entries ({n_entries}) than GPUs ({n_gpus}). "
                f"Only {n_entries} GPU(s) will be active. Consider using "
                f"train_parameters() with device='cuda' for small datasets."
            )

    energy_losses: list[float] = []
    force_losses: list[float] = []
    total_losses: list[float] = []

    pbar = tqdm(range(n_epochs), desc="Epoch 0", disable=not verbose)
    for epoch in pbar:
        optimizer.zero_grad()

        # Shuffle and distribute entries across GPUs
        shuffled_indices = torch.randperm(n_entries).tolist()

        # Interleaved distribution
        gpu_indices = [[] for _ in range(n_gpus)]
        for i, idx in enumerate(shuffled_indices):
            gpu_indices[i % n_gpus].append(idx)

        # Take detached snapshot for workers
        params_snapshot = params.detach().clone().cpu()

        # Pre-allocate results
        results: list = [None] * n_gpus

        # Launch worker threads
        threads = []
        for rank in range(n_gpus):
            t = threading.Thread(
                target=_worker_fn_ddp,
                args=(
                    rank,
                    n_gpus,
                    params_snapshot,
                    trainable_bytes,
                    dataset,
                    tensor_systems,
                    gpu_indices[rank],
                    conformer_batch_size,
                    config,
                    results,
                ),
                daemon=True,
            )
            t.start()
            threads.append(t)

        # Wait for all workers
        for t in threads:
            t.join()

        # Check if any worker failed
        failed_workers = [i for i, r in enumerate(results) if r is None]
        if failed_workers:
            raise RuntimeError(
                f"Worker(s) {failed_workers} failed during epoch {epoch}. "
                "Check the traceback above for details."
            )

        # Aggregate results
        all_grads = [r[4] for r in results]
        total_entries = sum(r[3] for r in results)
        n_workers_with_entries = sum(1 for r in results if r[3] > 0)

        if n_workers_with_entries > 0:
            avg_grad = torch.stack(all_grads).sum(dim=0)
        else:
            avg_grad = torch.zeros_like(params)

        params.grad = avg_grad
        optimizer.step()

        # Compute epoch losses
        epoch_total_loss = sum(r[0] for r in results)
        epoch_energy_loss = sum(r[1] for r in results)
        epoch_force_loss = sum(r[2] for r in results)

        avg_total = epoch_total_loss / max(total_entries, 1)
        avg_energy = epoch_energy_loss / max(total_entries, 1)
        avg_force = epoch_force_loss / max(total_entries, 1)

        total_losses.append(avg_total)
        energy_losses.append(avg_energy)
        force_losses.append(avg_force)

        if verbose:
            logger.info(
                f"Epoch {epoch + 1}/{n_epochs}: total_loss={avg_total:.4e}, "
                f"energy_loss={avg_energy:.4e}, force_loss={avg_force:.4e} "
                f"[{n_workers_with_entries}/{n_gpus} GPUs active]"
            )

    return TrainingResult(
        initial_parameters=initial_params,
        trained_parameters=params.detach().to(initial_params.device).abs(),
        energy_losses=energy_losses,
        force_losses=force_losses,
    )
