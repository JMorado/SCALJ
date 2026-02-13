"""Energy and force computation functions."""

from typing import TYPE_CHECKING

import numpy as np
import openmm
import openmm.app
import openmm.unit
from tqdm import tqdm

from .models import EnergyForceResult

if TYPE_CHECKING:
    import smee


def setup_mlp_simulation(
    tensor_system: "smee.TensorSystem",
    mlp_name: str,
    temperature: openmm.unit.Quantity = 300 * openmm.unit.kelvin,
    friction_coeff: openmm.unit.Quantity = 1.0 / openmm.unit.picoseconds,
    timestep: openmm.unit.Quantity = 1.0 * openmm.unit.femtoseconds,
    mlp_device: str = "cuda",
    platform: str = "CPU",
) -> openmm.app.Simulation:
    """Setup ML potential simulation for energy/force computation or MD.

    Creates an OpenMM simulation using a machine learning potential (ANI, MACE, etc.)
    for computing energies and forces or running dynamics.

    Parameters
    ----------
    tensor_system : smee.TensorSystem
        The molecular system to simulate.
    mlp_name : str
        Name of the ML potential model (e.g., "ani2x", "mace-off-small").
    temperature : openmm.unit.Quantity
        Simulation temperature (default: 300 K).
    friction_coeff : openmm.unit.Quantity
        Langevin friction coefficient (default: 1.0 ps^-1).
    timestep : openmm.unit.Quantity
        Integration timestep (default: 1.0 fs).
    mlp_device : str
        Device for ML potential ('cuda' or 'cpu').
    platform : str
        OpenMM platform ('CPU', 'CUDA', 'OpenCL').

    Returns
    -------
    openmm.app.Simulation
        Configured OpenMM simulation with ML potential.

    Examples
    --------
    >>> import smee
    >>> # Assuming you have a tensor_system
    >>> sim = setup_mlp_simulation(tensor_system, "ani2x", mlp_device="cuda")
    """
    import smee.converters
    from openmmml import MLPotential

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
    simulation = openmm.app.Simulation(
        omm_topology, omm_mlp_system, integrator, platform=omm_platform
    )
    return simulation


def run_mlp_relaxation(
    mlp_simulation: openmm.app.Simulation,
    coords: np.ndarray,
    box_vectors: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ML potential simulation steps for structure relaxation.

    Parameters
    ----------
    mlp_simulation : openmm.app.Simulation
        Configured ML potential simulation.
    coords : np.ndarray
        Initial coordinates in OpenMM-compatible units.
    box_vectors : np.ndarray
        Box vectors (3x3 array).
    n_steps : int
        Number of simulation steps to run.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Final coordinates and box vectors.

    Examples
    --------
    >>> sim = setup_mlp_simulation(tensor_system, "ani2x")
    >>> final_coords, final_box = run_mlp_relaxation(sim, coords, box, n_steps=100)
    """
    mlp_simulation.context.setPeriodicBoxVectors(*box_vectors)
    mlp_simulation.context.setPositions(coords)
    mlp_simulation.step(n_steps)

    state = mlp_simulation.context.getState(getPositions=True)
    final_coords = state.getPositions(asNumpy=True)
    final_box_vectors = state.getPeriodicBoxVectors(asNumpy=True)

    return final_coords, final_box_vectors


def compute_mlp_energies_forces(
    mlp_simulation: openmm.app.Simulation,
    coords_list: list[np.ndarray],
    box_vectors_list: list[np.ndarray],
    show_progress: bool = True,
) -> EnergyForceResult:
    """Compute energies and forces for configurations using ML potential.

    Evaluates the ML potential energy and forces for each configuration
    in the provided lists.

    Parameters
    ----------
    mlp_simulation : openmm.app.Simulation
        Configured ML potential simulation.
    coords_list : list[np.ndarray]
        List of coordinate arrays in Å, each with shape (n_atoms, 3).
    box_vectors_list : list[np.ndarray]
        List of box vector arrays in Å, each with shape (3, 3).
    show_progress : bool
        Whether to show progress bar.

    Returns
    -------
    EnergyForceResult
        Result containing energies [kcal/mol] and forces [kcal/mol/Å].

    Examples
    --------
    >>> sim = setup_mlp_simulation(tensor_system, "ani2x")
    >>> result = compute_mlp_energies_forces(sim, coords_list, box_vectors_list)
    >>> result.energies.shape
    (100,)  # For 100 configurations
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

    return EnergyForceResult(
        energies=np.array(energies),
        forces=np.array(forces),
    )


def compute_mlp_energies_forces_single(
    tensor_system: "smee.TensorSystem",
    coords_list: list[np.ndarray],
    box_vectors_list: list[np.ndarray],
    mlp_name: str = "ani2x",
    mlp_device: str = "cuda",
    platform: str = "CPU",
    show_progress: bool = True,
) -> EnergyForceResult:
    """Convenience function to compute MLP energies/forces in one call.

    Sets up the ML potential simulation and computes energies and forces
    for all provided configurations.

    Parameters
    ----------
    tensor_system : smee.TensorSystem
        The molecular system.
    coords_list : list[np.ndarray]
        List of coordinate arrays in Å.
    box_vectors_list : list[np.ndarray]
        List of box vector arrays in Å.
    mlp_name : str
        Name of the ML potential model.
    mlp_device : str
        Device for ML potential ('cuda' or 'cpu').
    platform : str
        OpenMM platform.
    show_progress : bool
        Whether to show progress bar.

    Returns
    -------
    EnergyForceResult
        Result containing energies [kcal/mol] and forces [kcal/mol/Å].

    Examples
    --------
    >>> result = compute_mlp_energies_forces_single(
    ...     tensor_system, coords_list, box_vectors_list, mlp_name="ani2x"
    ... )
    """
    simulation = setup_mlp_simulation(
        tensor_system,
        mlp_name,
        mlp_device=mlp_device,
        platform=platform,
    )

    return compute_mlp_energies_forces(
        simulation,
        coords_list,
        box_vectors_list,
        show_progress=show_progress,
    )
