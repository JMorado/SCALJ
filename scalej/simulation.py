"""Molecular dynamics simulation functions."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import openmm
import openmm.app
import openmm.unit
import smee.mm
from tqdm import tqdm

from .models import TrajectoryFrames

if TYPE_CHECKING:
    import smee


def run_md_simulation(
    tensor_system: "smee.TensorSystem",
    tensor_forcefield: "smee.TensorForceField",
    output_path: Path | str,
    temperature: openmm.unit.Quantity = 300 * openmm.unit.kelvin,
    pressure: openmm.unit.Quantity = 1.0 * openmm.unit.atmosphere,
    timestep: openmm.unit.Quantity = 1.0 * openmm.unit.femtoseconds,
    n_equilibration_nvt_steps: int = 50_000,
    n_equilibration_npt_steps: int = 50_000,
    n_production_steps: int = 1_000_000,
    report_interval: int = 1000,
    save_pdb: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run classical MD simulation with equilibration and production phases.

    Performs energy minimization, NVT equilibration, NPT equilibration,
    and production dynamics, saving trajectory frames to a DCD file.

    Parameters
    ----------
    tensor_system : smee.TensorSystem
        The molecular system to simulate.
    tensor_forcefield : smee.TensorForceField
        The force field for the simulation.
    output_path : Path | str
        Path for output trajectory file (DCD format).
    temperature : openmm.unit.Quantity
        Simulation temperature.
    pressure : openmm.unit.Quantity
        Simulation pressure.
    timestep : openmm.unit.Quantity
        Integration timestep.
    n_equilibration_nvt_steps : int
        Number of NVT equilibration steps.
    n_equilibration_npt_steps : int
        Number of NPT equilibration steps.
    n_production_steps : int
        Number of production MD steps.
    report_interval : int
        Interval for saving trajectory frames.
    save_pdb : bool
        Whether to save a PDB trajectory file as well.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Initial coordinates and box vectors.

    Examples
    --------
    >>> import smee
    >>> coords, box = run_md_simulation(
    ...     tensor_system, tensor_forcefield,
    ...     "trajectory.dcd",
    ...     n_production_steps=100_000
    ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute beta for tensor reporter
    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temperature)

    # Equilibration configurations
    equilibrate_config = [
        smee.mm.MinimizationConfig(),
        smee.mm.SimulationConfig(
            temperature=temperature,
            pressure=None,
            n_steps=n_equilibration_nvt_steps,
            timestep=timestep,
        ),
        smee.mm.SimulationConfig(
            temperature=temperature,
            pressure=pressure,
            n_steps=n_equilibration_npt_steps,
            timestep=timestep,
        ),
    ]

    # Production configuration
    production_config = smee.mm.SimulationConfig(
        temperature=temperature,
        pressure=pressure,
        n_steps=n_production_steps,
        timestep=timestep,
    )

    # Generate initial coordinates
    initial_coords, box_vectors = smee.mm.generate_system_coords(
        tensor_system, tensor_forcefield
    )

    # Set up reporters
    reporters = []

    if save_pdb:
        pdb_reporter_file = output_path.parent / f"trajectory_{output_path.stem}.pdb"
        pdb_reporter = openmm.app.PDBReporter(
            pdb_reporter_file.as_posix(), report_interval
        )
        reporters.append(pdb_reporter)

    with smee.mm.tensor_reporter(
        output_path, report_interval, beta, pressure
    ) as tensor_reporter:
        reporters.insert(0, tensor_reporter)
        smee.mm.simulate(
            tensor_system,
            tensor_forcefield,
            initial_coords,
            box_vectors,
            equilibrate_config,
            production_config,
            reporters,
        )

    return initial_coords, box_vectors


def load_trajectory_frames(
    trajectory_path: Path | str,
    n_frames: int = 1,
    from_end: bool = True,
) -> TrajectoryFrames:
    """Load frames from a trajectory file.

    Parameters
    ----------
    trajectory_path : Path | str
        Path to the trajectory file (DCD format with SMEE metadata).
    n_frames : int
        Number of frames to load.
    from_end : bool
        If True, load the last n_frames. If False, load the first n_frames.

    Returns
    -------
    TrajectoryFrames
        Container with coordinates, box vectors, and frame count.

    Raises
    ------
    ValueError
        If trajectory has fewer frames than requested.

    Examples
    --------
    >>> frames = load_trajectory_frames("trajectory.dcd", n_frames=10)
    >>> frames.coords.shape
    (10, 100, 3)  # 10 frames, 100 atoms, 3 dimensions
    """
    coords_list = []
    box_vectors_list = []

    with open(trajectory_path, "rb") as f:
        for coord, box_vector, _, kinetic in tqdm(
            smee.mm._reporters.unpack_frames(f), desc="Loading trajectory"
        ):
            coords_list.append(coord)
            box_vectors_list.append(box_vector)

    if not coords_list:
        raise ValueError(f"No frames found in trajectory: {trajectory_path}")

    total_frames = len(coords_list)
    if n_frames > total_frames:
        raise ValueError(
            f"Requested {n_frames} frames but trajectory only has "
            f"{total_frames} frames. Please reduce n_frames."
        )

    # Select frames
    if from_end:
        selected_coords = coords_list[-n_frames:]
        selected_box_vectors = box_vectors_list[-n_frames:]
    else:
        selected_coords = coords_list[:n_frames]
        selected_box_vectors = box_vectors_list[:n_frames]

    if n_frames == 1:
        # Return single frame without batch dimension
        coords_array = selected_coords[0].detach().cpu().numpy()
        box_vectors_array = selected_box_vectors[0].detach().cpu().numpy()
    else:
        # Return multiple frames with batch dimension
        coords_array = np.stack(
            [c.detach().cpu().numpy() for c in selected_coords], axis=0
        )
        box_vectors_array = np.stack(
            [b.detach().cpu().numpy() for b in selected_box_vectors], axis=0
        )

    return TrajectoryFrames(
        coords=coords_array,
        box_vectors=box_vectors_array,
        n_frames=n_frames,
    )


def generate_initial_coords(
    tensor_system: "smee.TensorSystem",
    tensor_forcefield: "smee.TensorForceField",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate initial coordinates for a system using SMEE packmol.

    Parameters
    ----------
    tensor_system : smee.TensorSystem
        The molecular system.
    tensor_forcefield : smee.TensorForceField
        The force field.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Initial coordinates and box vectors.

    Examples
    --------
    >>> coords, box = generate_initial_coords(tensor_system, tensor_forcefield)
    """
    return smee.mm.generate_system_coords(tensor_system, tensor_forcefield)


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
    from .energy import run_mlp_relaxation, setup_mlp_simulation

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
