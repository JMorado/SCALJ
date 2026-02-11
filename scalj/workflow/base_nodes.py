"""Base classes for workflow nodes with shared functionality.

This module provides base classes that extend WorkflowNode with common
functionality used across multiple nodes, avoiding code duplication.
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

from .node import WorkflowNode


class MLPotentialBaseNode(WorkflowNode):
    """
    Base class for nodes that use ML potential simulations.
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
        """
        Setup ML potential simulation for energy/force computation or MD.

        Parameters
        ----------
        tensor_system : smee.TensorSystem
            The molecular system to simulate
        mlp_name : str
            Name of the ML potential model
        temperature : openmm.unit.Quantity, optional
            Simulation temperature (default: 300 K)
        friction_coeff : openmm.unit.Quantity, optional
            Langevin friction coefficient (default: 1.0 ps^-1)
        timestep : openmm.unit.Quantity, optional
            Integration timestep (default: 1.0 fs)
        mlp_device : str, optional
            Device for ML potential ('cuda' or 'cpu', default: 'cuda')
        platform : str, optional
            OpenMM platform ('CPU', 'CUDA', 'OpenCL', default: 'CPU')

        Returns
        -------
        openmm.app.Simulation
            Configured OpenMM simulation with ML potential
        """
        mlp = MLPotential(mlp_name)
        omm_topology = smee.converters.convert_to_openmm_topology(tensor_system)
        omm_topology.setPeriodicBoxVectors(np.eye(3) * 10 * openmm.unit.angstrom)

        omm_mlp_system = mlp.createSystem(
            omm_topology, removeCMMotion=False, device=mlp_device
        )
        omm_platform = openmm.Platform.getPlatformByName(platform)
        integrator = openmm.LangevinMiddleIntegrator(
            temperature,
            friction_coeff,
            timestep,
        )
        simulation_ml = openmm.app.Simulation(
            omm_topology, omm_mlp_system, integrator, platform=omm_platform
        )
        return simulation_ml

    @staticmethod
    def _run_mlp_simulation(
        mlp_simulation: openmm.app.Simulation,
        coords: np.ndarray,
        box_vectors: np.ndarray,
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run ML potential simulation steps.

        Parameters
        ----------
        mlp_simulation : openmm.app.Simulation
            Configured ML potential simulation
        coords : np.ndarray
            Initial coordinates
        box_vectors : np.ndarray
            Box vectors (3x3 array)
        n_steps : int
            Number of simulation steps to run

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Final coordinates and box vectors
        """
        mlp_simulation.context.setPeriodicBoxVectors(*box_vectors)
        mlp_simulation.context.setPositions(coords)
        mlp_simulation.step(n_steps)

        state = mlp_simulation.context.getState(getPositions=True)
        final_coords = state.getPositions(asNumpy=True)
        final_box_vectors = state.getPeriodicBoxVectors(asNumpy=True)

        return final_coords, final_box_vectors

    @staticmethod
    def _compute_energies_forces(
        mlp_simulation: openmm.app.Simulation,
        coords_list: list,
        box_vectors_list: list,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute energies and forces for a list of configurations using ML potential.

        Parameters
        ----------
        mlp_simulation : openmm.app.Simulation
            Configured ML potential simulation
        coords_list : list
            List of coordinate arrays
        box_vectors_list : list
            List of box vector arrays
        show_progress : bool, optional
            Whether to show progress bar (default: True)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Arrays of energies [kcal/mol] and forces [kcal/mol/Å]
        """
        energies = []
        forces = []

        iterator = zip(coords_list, box_vectors_list)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(coords_list),
                desc="Computing ML energies/forces",
            )

        for coord, box_vector in iterator:
            mlp_simulation.context.setPositions(coord * openmm.unit.angstrom)
            mlp_simulation.context.setPeriodicBoxVectors(
                *box_vector * openmm.unit.angstrom
            )
            state = mlp_simulation.context.getState(getEnergy=True, getForces=True)

            energy = state.getPotentialEnergy().value_in_unit(
                openmm.unit.kilocalorie_per_mole
            )
            force = state.getForces(asNumpy=True).value_in_unit(
                openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom
            )

            energies.append(energy)
            forces.append(force)

        return np.array(energies), np.array(forces)


class PredictionBaseNode(WorkflowNode):
    """
    Base class for nodes that perform energy/force predictions.
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
        """
        Predict relative energies per molecule [kcal/mol] and forces [kcal/mol/Å].

        This method uses smee for differentiable force field evaluation with
        automatic differentiation for computing forces.

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
            The reference energy to compute the relative energies with respect to.
            - "mean": Use mean energy of all conformers
            - "min": Use energy of lowest-energy conformer
            - "none": No reference (absolute energies)
        normalize : bool, optional
            Whether to scale the relative energies by ``1/n_confs_i`` and the forces
            by ``1/n_confs_i * n_atoms_per_conf_i * 3)``.
        energy_cutoff : float, optional
            Energy cutoff in kcal/mol to filter high-energy conformers.
        weighting_method : typing.Literal["uniform", "boltzmann"], optional
            Method to weight conformers in loss function.
        weighting_temperature : float, optional
            Temperature in Kelvin for Boltzmann weighting (default: 298.15 K).
        device : str, optional
            The device to use for the prediction ('cpu' or 'cuda').

        Returns
        -------
        tuple
            Tuple containing:
            - energy_ref_all: Reference energies [kcal/mol]
            - energy_pred_all: Predicted energies [kcal/mol]
            - forces_ref_all: Reference forces [kcal/mol/Å]
            - forces_pred_all: Predicted forces [kcal/mol/Å]
            - weights_all: Energy weights for loss computation
            - weights_forces_all: Force weights for loss computation
            - all_mask_idxs: Indices of conformers kept after filtering

        Raises
        ------
        ValueError
            If no valid conformers found after filtering
        NotImplementedError
            If invalid reference energy method specified
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

        return (
            energy_ref_all,
            energy_pred_all,
            forces_ref_all,
            forces_pred_all,
            weights_all,
            weights_forces_all,
            all_mask_idxs,
        )
