"""Base classes for workflow nodes with shared functionality.

This module provides base classes that extend WorkflowNode with common
functionality used across multiple nodes, avoiding code duplication.

Note: The actual computation functions have been moved to the API modules
(scalej.energy, scalej.training). This module now provides thin wrappers.
"""

import typing

import datasets
import numpy as np
import openmm
import openmm.unit
import smee
import smee.converters
import smee.utils
import torch
from openmmml import MLPotential
from tqdm import tqdm

# Import API functions
from ..energy import (
    compute_mlp_energies_forces,
    run_mlp_relaxation,
    setup_mlp_simulation,
)
from ..models import PredictionResult
from ..training import predict_energies_forces
from .node import WorkflowNode


class MLPotentialBaseNode(WorkflowNode):
    """Base class for nodes that use ML potential simulations.

    Provides static methods that delegate to the scalej.energy module.
    """

    @staticmethod
    def _setup_mlp_simulation(
        tensor_system: smee.TensorSystem,
        mlp_name: str,
        temperature: openmm.unit.Quantity = 300 * openmm.unit.kelvin,
        friction_coeff: openmm.unit.Quantity = 1.0 / openmm.unit.picoseconds,
        timestep: openmm.unit.Quantity = 1.0 * openmm.unit.femtoseconds,
        mlp_device: str = "cuda",
        platform: str = "CPU",
    ) -> openmm.app.Simulation:
        """Setup ML potential simulation. Delegates to scalej.energy.setup_mlp_simulation."""
        return setup_mlp_simulation(
            tensor_system,
            mlp_name,
            temperature=temperature,
            friction_coeff=friction_coeff,
            timestep=timestep,
            mlp_device=mlp_device,
            platform=platform,
        )

    @staticmethod
    def _run_mlp_simulation(
        mlp_simulation: openmm.app.Simulation,
        coords: np.ndarray,
        box_vectors: np.ndarray,
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run MLP simulation steps. Delegates to scalej.energy.run_mlp_relaxation."""
        return run_mlp_relaxation(mlp_simulation, coords, box_vectors, n_steps)

    @staticmethod
    def _compute_energies_forces(
        mlp_simulation: openmm.app.Simulation,
        coords_list: list,
        box_vectors_list: list,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute energies and forces. Delegates to scalej.energy.compute_mlp_energies_forces."""
        result = compute_mlp_energies_forces(
            mlp_simulation, coords_list, box_vectors_list, show_progress
        )
        return result.energies, result.forces


class PredictionBaseNode(WorkflowNode):
    """Base class for nodes that perform energy/force predictions.

    Provides static methods that delegate to the scalej.training module.
    """

    @staticmethod
    def _predict(
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
        """Predict energies and forces. Delegates to scalej.training.predict_energies_forces."""
        result = predict_energies_forces(
            dataset,
            composite_force_field,
            all_tensor_systems,
            reference=reference,
            normalize=normalize,
            energy_cutoff=energy_cutoff,
            weighting_method=weighting_method,
            weighting_temperature=weighting_temperature,
            device=device,
        )

        return (
            result.energy_ref,
            result.energy_pred,
            result.forces_ref,
            result.forces_pred,
            result.weights_energy,
            result.weights_forces,
            result.mask_idxs,
        )
