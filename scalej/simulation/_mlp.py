"""MLP functions."""

import ase
import numpy as np
import openmm
import openmm.app
import openmm.unit
import smee
from tqdm import tqdm

from ..constants import EV_TO_KCAL_MOL
from ..types import EnergyForceResult


def setup_mlp_simulation(
    tensor_system: smee.TensorSystem,
    mlp_name: str,
    temperature: openmm.unit.Quantity = 300 * openmm.unit.kelvin,
    friction_coeff: openmm.unit.Quantity = 1.0 / openmm.unit.picoseconds,
    timestep: openmm.unit.Quantity = 1.0 * openmm.unit.femtoseconds,
    mlp_device: str = "cuda",
    platform: str = "CPU",
) -> openmm.app.Simulation:
    """
    Setup ML potential simulation.

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
    """
    import smee.converters
    from openmmml import MLPotential

    mlp = MLPotential(mlp_name)
    omm_topology = smee.converters.convert_to_openmm_topology(tensor_system)
    # We need to set some box vectors here for openmm-ml to recognise the system as periodic,
    # but they will be overwritten later.
    omm_topology.setPeriodicBoxVectors(np.eye(3) * 10 * openmm.unit.angstrom)
    omm_mlp_system = mlp.createSystem(
        omm_topology, removeCMMotion=False, device=mlp_device
    )
    omm_platform = openmm.Platform.getPlatformByName(platform)
    omm_integrator = openmm.LangevinMiddleIntegrator(
        temperature,
        friction_coeff,
        timestep,
    )
    simulation = openmm.app.Simulation(
        omm_topology, omm_mlp_system, omm_integrator, platform=omm_platform
    )
    return simulation


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


def ase_atoms_from_tensor_system(
    tensor_system: smee.TensorSystem,
) -> ase.Atoms:
    """
    Build an ASE Atoms object from a smee TensorSystem.

    Parameters
    ----------
    tensor_system : smee.TensorSystem
        The molecular system whose topologies define the atom ordering.

    Returns
    -------
    ase.Atoms
        Atoms object with correct atomic numbers and zeroed positions.
    """
    import ase

    atomic_nums = []
    for topology, n_copy in zip(
        tensor_system.topologies, tensor_system.n_copies, strict=True
    ):
        atomic_nums.extend(topology.atomic_nums.tolist() * n_copy)

    n_atoms = len(atomic_nums)
    return ase.Atoms(numbers=atomic_nums, positions=np.zeros((n_atoms, 3)))


def compute_ase_energies_forces(
    tensor_system: smee.TensorSystem,
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
    """
    atoms = ase_atoms_from_tensor_system(tensor_system)

    if external_field is None:
        external_field = [0.0, 0.0, 0.0]

    energies = []
    forces = []

    iterator = zip(coords_list, box_vectors_list, strict=True)
    if show_progress:  # pragma: no cover
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

        energies.append(atoms_frame.get_potential_energy() * EV_TO_KCAL_MOL)
        forces.append(atoms_frame.get_forces() * EV_TO_KCAL_MOL)

    return EnergyForceResult(
        energies=np.array(energies),
        forces=np.array(forces),
    )
