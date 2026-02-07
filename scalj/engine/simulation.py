"""Molecular dynamics simulation setup and execution."""

from pathlib import Path

import openmm
import openmm.unit
import smee
import smee.converters
import smee.mm
import torch
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule
from tqdm import tqdm

from ..config import SimulationConfig
from ..utils import ensure_directory


def setup_force_field(
    ff_name: str,
    smiles: str,
) -> tuple[smee.TensorForceField, list[smee.TensorTopology]]:
    """
    Set up force field and topology from SMILES.

    Args:
        ff_name: Force field file name (e.g., "openff_unconstrained-2.2.1.offxml")
        smiles: SMILES string for the molecule

    Returns:
        Tuple of (force_field, topologies)
    """
    force_field = ForceField(ff_name, load_plugins=True)
    interchange = Interchange.from_smirnoff(
        force_field,
        [Molecule.from_smiles(smiles)],
    )

    tensor_ff, topologies = smee.converters.convert_interchange(interchange)

    return tensor_ff, topologies


def create_system(
    topology: smee.TensorTopology,
    n_molecules: int,
    is_periodic: bool = True,
) -> smee.TensorSystem:
    """
    Create a tensor system for simulation.

    Args:
        topology: Molecular topology
        n_molecules: Number of molecules in the system
        is_periodic: Whether to use periodic boundary conditions

    Returns:
        TensorSystem for simulation
    """
    return smee.TensorSystem([topology], [n_molecules], is_periodic=is_periodic)


def run_simulation(
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    output_path: Path,
    config: SimulationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a molecular dynamics simulation and save the trajectory.

    Parameters
    ----------
    system : smee.TensorSystem
        The molecular system to simulate
    force_field : smee.TensorForceField
        The force field to use
    output_path : Path
        Path to save the trajectory
    config : SimulationConfig
        Simulation configuration

    Returns
    -------
    Tuple of (initial_coords, initial_box_vectors)
    """
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    # Compute beta for NPT ensemble
    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * config.temperature)

    # Equilibration configurations
    equilibrate_config = [
        smee.mm.MinimizationConfig(),
        # Short NVT equilibration
        smee.mm.SimulationConfig(
            temperature=config.temperature,
            pressure=None,
            n_steps=config.n_equilibration_nvt_steps,
            timestep=config.timestep,
        ),
        # Short NPT equilibration
        smee.mm.SimulationConfig(
            temperature=config.temperature,
            pressure=config.pressure,
            n_steps=config.n_equilibration_npt_steps,
            timestep=config.timestep,
        ),
    ]

    # Production configuration
    production_config = smee.mm.SimulationConfig(
        temperature=config.temperature,
        pressure=config.pressure,
        n_steps=config.n_production_steps,
        timestep=config.timestep,
    )

    # Generate initial coordinates
    initial_coords, box_vectors = smee.mm.generate_system_coords(system, force_field)

    # Set up reporters
    pdb_reporter_file = output_path.parent / "trajectory.pdb"
    pdb_reporter = openmm.app.PDBReporter(
        pdb_reporter_file.as_posix(), config.report_interval
    )

    with smee.mm.tensor_reporter(
        output_path, config.report_interval, beta, config.pressure
    ) as tensor_reporter:
        smee.mm.simulate(
            system,
            force_field,
            initial_coords,
            box_vectors,
            equilibrate_config,
            production_config,
            [tensor_reporter, pdb_reporter],
        )

    return initial_coords, box_vectors


def run_mlp_simulation(mlp_simulation, coords, box_vectors, simulation_config):
    """
    Run a molecular dynamics simulation using an MLP via openmm-ml.

    Parameters
    ----------
    mlp_simulation : openmm.app.Simulation
        Simulation object ready to be used
    coords : torch.Tensor
        Initial coordinates
    box_vectors : torch.Tensor
        Box vectors
    simulation_config : SimulationConfig
        Simulation configuration

    Returns
    -------
    Tuple of (final_coords, final_box_vectors)
    """
    # Set box vectors and positions
    mlp_simulation.context.setPeriodicBoxVectors(*box_vectors)
    mlp_simulation.context.setPositions(coords)

    # Run short MLP simulation
    mlp_simulation.step(simulation_config.n_mlp_steps)

    # Get final coords and box vectors
    state = mlp_simulation.context.getState(getPositions=True)
    final_coords = state.getPositions(asNumpy=True)
    final_box_vectors = state.getPeriodicBoxVectors(asNumpy=True)

    return final_coords, final_box_vectors, mlp_simulation


def load_trajectory(
    trajectory_path: Path,
    subsample_freq: int = 1,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Load trajectory frames from a saved trajectory file.

    Args:
        trajectory_path: Path to the trajectory file
        subsample_freq: Frequency to subsample frames (1 = all frames)

    Returns:
        Tuple of (coords_list, box_vectors_list)
    """
    coords = []
    box_vectors = []

    with open(trajectory_path, "rb") as f:
        for i, (coord, box_vector, _, kinetic) in tqdm(
            enumerate(smee.mm._reporters.unpack_frames(f)), desc="Loading trajectory"
        ):
            if i % subsample_freq != 0:
                continue

            coords.append(coord)
            box_vectors.append(box_vector)

    return coords, box_vectors


def load_last_frame(
    trajectory_path: Path,
) -> tuple[openmm.unit.Quantity, openmm.unit.Quantity]:
    """
    Load the last frame from a trajectory file.

    Args:
        trajectory_path: Path to the trajectory file

    Returns:
        Tuple of (coords, box_vectors) as OpenMM quantities in Angstroms.
    """
    coords_list, box_vectors_list = load_trajectory(trajectory_path)

    if not coords_list:
        raise ValueError(f"No frames found in trajectory: {trajectory_path}")

    last_coords = coords_list[-1]
    last_box_vectors = box_vectors_list[-1]

    coords_quantity = last_coords.detach().cpu().numpy() * openmm.unit.angstrom
    box_vectors_quantity = last_box_vectors.detach().cpu().numpy() * openmm.unit.angstrom

    return coords_quantity, box_vectors_quantity
