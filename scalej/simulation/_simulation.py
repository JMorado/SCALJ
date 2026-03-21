"""Molecular dynamics simulation functions."""

from pathlib import Path

import numpy as np
import openmm
import openmm.app
import openmm.unit
import smee.mm
from tqdm import tqdm

from ..types import TrajectoryFrames


def run_simulation_smee(
    tensor_system: smee.TensorSystem,
    tensor_forcefield: smee.TensorForceField,
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
    """
    Run classical MD simulation with equilibration and production phases using smee.

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
    """
    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temperature)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Equilibration configuration.
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

    # Production configuration.
    production_config = smee.mm.SimulationConfig(
        temperature=temperature,
        pressure=pressure,
        n_steps=n_production_steps,
        timestep=timestep,
    )

    initial_coords, box_vectors = smee.mm.generate_system_coords(
        tensor_system, tensor_forcefield
    )

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


def run_simulation_omm(
    simulation: openmm.app.Simulation,
    coords: np.ndarray,
    box_vectors: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run OpenMM simulation.

    Parameters
    ----------
    simulation : openmm.app.Simulation
        Configured OpenMM simulation.
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
    """
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(coords)
    simulation.step(n_steps)

    # Get final state.
    state = simulation.context.getState(getPositions=True)
    final_coords = state.getPositions(asNumpy=True)
    final_box_vectors = state.getPeriodicBoxVectors(asNumpy=True)

    return final_coords, final_box_vectors


def load_trajectory_frames_smee(
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

    # Select frames.
    if from_end:
        selected_coords = coords_list[-n_frames:]
        selected_box_vectors = box_vectors_list[-n_frames:]
    else:
        selected_coords = coords_list[:n_frames]
        selected_box_vectors = box_vectors_list[:n_frames]

    if n_frames == 1:
        # Return single frame without batch dimension.
        coords_array = selected_coords[0].detach().cpu().numpy()
        box_vectors_array = selected_box_vectors[0].detach().cpu().numpy()
    else:
        # Return multiple frames with batch dimension.
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
