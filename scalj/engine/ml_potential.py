"""ML potential energy and force computation."""

import numpy as np
import openmm
import openmm.unit
import smee
import smee.converters
import torch
from openmmml import MLPotential
from tqdm import tqdm


def setup_mlp_simulation(tensor_system, mlp_name, simulation_config):
    """
    Run a molecular dynamics simulation using an MLP via openmm-ml.

    Parameters
    ----------
    tensor_system : smee.TensorSystem
        The molecular system to simulate
    mlp_name : str
        Name of the MLP to use
    coords : torch.Tensor
        Initial coordinates
    box_vectors : torch.Tensor
        Box vectors
    simulation_config : SimulationConfig
        Simulation configuration

    Returns
    -------
    openmm.app.Simulation
        Simulation object ready to be used.
    """
    # Create MLP and convert topology
    mlp = MLPotential(mlp_name)
    omm_topology = smee.converters.convert_to_openmm_topology(tensor_system)

    # Random box vectors, so that the simulation is periodic
    omm_topology.setPeriodicBoxVectors(np.eye(3) * 10 * openmm.unit.angstrom)

    # Create system, platform, integrator and simulation
    omm_mlp_system = mlp.createSystem(
        omm_topology, removeCMMotion=False, device=simulation_config.mlp_device
    )
    platform = openmm.Platform.getPlatformByName(simulation_config.platform)
    integrator = openmm.LangevinMiddleIntegrator(
        simulation_config.temperature,
        simulation_config.friction_coeff,
        simulation_config.timestep,
    )
    simulation_ml = openmm.app.Simulation(
        omm_topology, omm_mlp_system, integrator, platform=platform
    )
    return simulation_ml


def compute_energies_forces(
    mlp_simulation,
    coords_list: list[torch.Tensor],
    box_vectors_list: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute energies and forces for a list of configurations using an ML potential.

    Parameters
    ----------
    mlp_simulation : openmm.app.Simulation
        Simulation object ready to be used
    coords_list : list[torch.Tensor]
        List of coordinate tensors
    box_vectors_list : list[torch.Tensor]
        List of box vector tensors

    Returns
    -------
    Tuple of (energies, forces) as numpy arrays
    """
    # Convert to CPU numpy arrays
    coords = [c for c in coords_list]
    box_vectors = [bv for bv in box_vectors_list]
    energies = []
    forces = []
    for coord, box_vector in tqdm(
        zip(coords, box_vectors),
        total=len(coords),
        desc="Computing ML energies/forces",
    ):
        print("coord", coord)
        print("box_vector", box_vector)
        mlp_simulation.context.setPositions(coord * openmm.unit.angstrom)
        mlp_simulation.context.setPeriodicBoxVectors(*box_vector * openmm.unit.angstrom)
        state = mlp_simulation.context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(
            openmm.unit.kilocalorie_per_mole
        )
        force = state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom
        )
        energies.append(energy)
        forces.append(force)
        print(energies)

    return np.array(energies), np.array(forces)
