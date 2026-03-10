"""ML potential (MLP) functions — setup, energy/force computation, relaxation."""

from typing import TYPE_CHECKING

import numpy as np
import openmm
import openmm.app
import openmm.unit
from tqdm import tqdm

from ..models import EnergyForceResult

if TYPE_CHECKING:
    import ase
    import smee

_EV_TO_KCAL_MOL = 23.06054194533


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


def relax_with_mlp(
    tensor_system: "smee.TensorSystem",
    coords: np.ndarray,
    box_vectors: np.ndarray,
    mlp_name: str = "ani2x",
    n_steps: int = 100,
    temperature: openmm.unit.Quantity = 300 * openmm.unit.kelvin,
    friction_coeff: openmm.unit.Quantity = 1.0 / openmm.unit.picoseconds,
    timestep: openmm.unit.Quantity = 1.0 * openmm.unit.femtoseconds,
    mlp_device: str = "cuda",
    platform: str = "CPU",
) -> tuple[np.ndarray, np.ndarray]:
    """Relax coordinates using an ML potential.

    Runs short dynamics with an ML potential to relax structures
    from classical MD to the MLP potential energy surface.

    Parameters
    ----------
    tensor_system : smee.TensorSystem
        The molecular system.
    coords : np.ndarray
        Initial coordinates (may have OpenMM units).
    box_vectors : np.ndarray
        Initial box vectors (may have OpenMM units).
    mlp_name : str
        Name of the ML potential model.
    n_steps : int
        Number of relaxation steps.
    temperature : openmm.unit.Quantity
        Relaxation temperature.
    friction_coeff : openmm.unit.Quantity
        Langevin friction coefficient.
    timestep : openmm.unit.Quantity
        Integration timestep.
    mlp_device : str
        Device for ML potential.
    platform : str
        OpenMM platform.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Relaxed coordinates and box vectors.

    Examples
    --------
    >>> relaxed_coords, relaxed_box = relax_with_mlp(
    ...     tensor_system, coords, box_vectors, n_steps=100
    ... )
    """
    simulation = setup_mlp_simulation(
        tensor_system,
        mlp_name,
        temperature=temperature,
        friction_coeff=friction_coeff,
        timestep=timestep,
        mlp_device=mlp_device,
        platform=platform,
    )

    return run_mlp_relaxation(simulation, coords, box_vectors, n_steps)


def atoms_template_from_tensor_system(
    tensor_system: "smee.TensorSystem",
) -> "ase.Atoms":
    """Build an ASE Atoms template from a smee TensorSystem.

    Reconstructs the chemical species (atomic numbers) in the same order as
    they appear in the system without assigning positions — positions are set
    per-frame inside the compute functions.

    Parameters
    ----------
    tensor_system : smee.TensorSystem
        The molecular system whose topologies define the atom ordering.

    Returns
    -------
    ase.Atoms
        Atoms object with correct atomic numbers and zeroed positions.

    Examples
    --------
    >>> atoms = atoms_template_from_tensor_system(tensor_system)
    """
    import ase

    atomic_nums = []
    for topology, n_copy in zip(tensor_system.topologies, tensor_system.n_copies):
        atomic_nums.extend(topology.atomic_nums.tolist() * n_copy)

    n_atoms = len(atomic_nums)
    return ase.Atoms(numbers=atomic_nums, positions=np.zeros((n_atoms, 3)))


def compute_ase_energies_forces(
    tensor_system: "smee.TensorSystem",
    calculator,
    coords_list: list[np.ndarray],
    box_vectors_list: list[np.ndarray],
    charge: int = 0,
    spin: int = 1,
    external_field: list[float] | None = None,
    show_progress: bool = True,
) -> EnergyForceResult:
    """
    Compute energies and forces for configurations using an ASE calculator.

    Parameters
    ----------
    tensor_system : smee.TensorSystem
        The molecular system from which the ASE atoms template is derived.
    calculator : ase.calculators.calculator.Calculator
        Any ASE-compatible calculator.
    coords_list : list[np.ndarray]
        List of coordinate arrays in Angstrom, each with shape (n_atoms, 3).
    box_vectors_list : list[np.ndarray]
        List of box vector arrays in Angstrom, each with shape (3, 3).
    charge : int
        Total charge passed via ``atoms.info["charge"]``.
    spin : int
        Spin multiplicity passed via ``atoms.info["spin"]``.
    external_field : list[float] | None
        External field vector [Fx, Fy, Fz] passed via
        ``atoms.info["external_field"]``.  Defaults to [0.0, 0.0, 0.0].
    show_progress : bool
        Whether to show a progress bar.

    Returns
    -------
    EnergyForceResult
        Result containing energies [kcal/mol] and forces [kcal/mol/Å].

    Examples
    --------
    >>> calc = setup_mace_polar_calculator()
    >>> result = compute_ase_energies_forces(
    ...     tensor_system, calc, coords_list, box_vectors_list
    ... )
    """
    atoms = atoms_template_from_tensor_system(tensor_system)

    if external_field is None:
        external_field = [0.0, 0.0, 0.0]

    energies = []
    forces = []

    iterator = zip(coords_list, box_vectors_list)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(coords_list),
            desc="Computing ASE energies/forces",
        )

    for coords, box_vectors in iterator:
        atoms_frame = atoms.copy()
        atoms_frame.set_positions(coords)
        atoms_frame.set_cell(box_vectors)
        atoms_frame.set_pbc(True)
        atoms_frame.info["charge"] = charge
        atoms_frame.info["spin"] = spin
        atoms_frame.info["external_field"] = external_field
        atoms_frame.calc = calculator

        energies.append(atoms_frame.get_potential_energy() * _EV_TO_KCAL_MOL)
        forces.append(atoms_frame.get_forces() * _EV_TO_KCAL_MOL)

    return EnergyForceResult(
        energies=np.array(energies),
        forces=np.array(forces),
    )
