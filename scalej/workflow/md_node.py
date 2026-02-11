"""Molecular dynamics simulation node."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import openmm
import openmm.app
import openmm.unit
import smee
import smee.converters
import smee.mm
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule
from tqdm import tqdm

from ..cli.utils import create_configs_from_dict, load_config
from ._utils import save_pickle
from .base_nodes import MLPotentialBaseNode


class MDNode(MLPotentialBaseNode):
    """
    MD node for running molecular dynamics simulations and optional MLP simulations.

    Inputs (from config):
    - system definitions (SMILES, composition)
    - force field name
    - simulation parameters

    Outputs:
    - trajectory_{system}.dcd: DCD trajectory file
    - system_{system}.pkl: System state with tensor_system, tensor_forcefield, coords, box_vectors
    """

    @classmethod
    def name(cls) -> str:
        return "md"

    @classmethod
    def description(cls) -> str:
        return """MD node for running molecular dynamics simulations and optional MLP simulations.

Inputs (from config):
- system definitions (SMILES, composition)
- force field name
- simulation parameters

Outputs:
- trajectory_{system}.dcd: DCD trajectory file
- system_{system}.pkl: System state with tensor_system, tensor_forcefield, coords, box_vectors"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--system-name",
            type=str,
            help="Process only this system (default: all systems in config)",
        )
        parser.add_argument(
            "--skip-mlp", action="store_true", help="Skip MLP simulation steps"
        )

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Execute MD simulation."""
        print("=" * 80)
        print("MDNode: Molecular Dynamics Simulation")
        print("=" * 80)

        # Load configuration
        config_dict = load_config(args.config)
        general_config, simulation_config, scaling_config, _, _ = (
            create_configs_from_dict(config_dict)
        )

        self._ensure_output_dir(args.output_dir)

        # Load force field
        force_field = ForceField(general_config.force_field_name, load_plugins=True)
        print(f"Force field: {general_config.force_field_name}")

        # Filter systems if requested
        systems = general_config.systems
        if args.system_name:
            systems = [s for s in systems if s.name == args.system_name]
            if not systems:
                raise ValueError(f"System '{args.system_name}' not found in config")

        results = {}

        for system in systems:
            print(f"\n{'=' * 80}")
            print(f"Processing system: {system.name}")
            print(f"{'=' * 80}")

            # Create molecules and interchanges
            mols = [Molecule.from_smiles(comp.smiles) for comp in system.components]
            interchanges = [
                Interchange.from_smirnoff(force_field, [mol]) for mol in mols
            ]

            # Create tensor forcefield and topologies
            tensor_forcefield, topologies = smee.converters.convert_interchange(
                interchanges
            )

            # Create tensor system
            nmol_list = [comp.nmol for comp in system.components]
            tensor_system = smee.TensorSystem(topologies, nmol_list, is_periodic=True)

            print("Components:")
            for comp in system.components:
                print(f"  - {comp.smiles}: {comp.nmol} molecules")

            # Check if trajectory already exists
            trajectory_path = self._output_path(
                args.output_dir, f"trajectory_{system.name}.dcd"
            )

            if system.trajectory_path:
                # Load existing trajectory
                print(f"Loading existing trajectory: {system.trajectory_path}")
                traj_path = Path(system.trajectory_path)
                if not traj_path.exists():
                    raise FileNotFoundError(f"Trajectory not found: {traj_path}")
                coords, box_vectors = self._load_last_frames(
                    traj_path, scaling_config.n_frames
                )
            else:
                # Run simulation
                print("Running MD simulation...")
                print(f"  Platform: {simulation_config.platform}")
                print(f"  Temperature: {simulation_config.temperature}")
                print(f"  Pressure: {simulation_config.pressure}")
                print(f"  Production steps: {simulation_config.n_production_steps}")

                self._run_md_simulation(
                    tensor_system,
                    tensor_forcefield,
                    trajectory_path,
                    simulation_config,
                )
                print(f"  Trajectory saved: {trajectory_path}")

                # Load last frames
                coords, box_vectors = self._load_last_frames(
                    trajectory_path, scaling_config.n_frames
                )

            # Setup and run MLP simulation if requested
            if not args.skip_mlp and simulation_config.n_mlp_steps > 0:
                print("\nRunning MLP simulation...")
                print(f"  MLP: {general_config.mlp_name}")
                print(f"  Device: {simulation_config.mlp_device}")
                print(f"  Steps: {simulation_config.n_mlp_steps}")

                mlp_simulation = self._setup_mlp_simulation(
                    tensor_system,
                    general_config.mlp_name,
                    temperature=simulation_config.temperature,
                    friction_coeff=simulation_config.friction_coeff,
                    timestep=simulation_config.timestep,
                    mlp_device=simulation_config.mlp_device,
                    platform=simulation_config.platform,
                )

                coords, box_vectors = self._run_mlp_simulation(
                    mlp_simulation, coords, box_vectors, simulation_config.n_mlp_steps
                )

            # Save system state
            system_file = self._output_path(
                args.output_dir, f"system_{system.name}.pkl"
            )
            system_state = {
                "tensor_system": tensor_system,
                "tensor_forcefield": tensor_forcefield,
                "coords": coords,
                "box_vectors": box_vectors,
                "nmol": nmol_list,
                "components": [
                    {"smiles": comp.smiles, "nmol": comp.nmol}
                    for comp in system.components
                ],
            }

            save_pickle(system_state, system_file)
            print(f"  System state saved: {system_file}")

            results[system.name] = {
                "trajectory": str(trajectory_path),
                "system_file": str(system_file),
            }

        print(f"\n{'=' * 80}")
        print("MDNode completed successfully")
        print(f"{'=' * 80}")

        return results

    @staticmethod
    def _run_md_simulation(system, force_field, output_path, config):
        """Run OpenMM MD simulation with smee."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute beta
        beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * config.temperature)

        # Equilibration configurations
        equilibrate_config = [
            smee.mm.MinimizationConfig(),
            smee.mm.SimulationConfig(
                temperature=config.temperature,
                pressure=None,
                n_steps=config.n_equilibration_nvt_steps,
                timestep=config.timestep,
            ),
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
        initial_coords, box_vectors = smee.mm.generate_system_coords(
            system, force_field
        )

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

    @staticmethod
    def _load_last_frames(trajectory_path, n_frames=1):
        """Load the last N frames from a trajectory file.

        Parameters
        ----------
        trajectory_path : Path or str
            Path to the trajectory file.
        n_frames : int
            Number of last frames to load. Default is 1.

        Returns
        -------
        coords_quantity : openmm.unit.Quantity
            Coordinates array with shape (n_frames, n_atoms, 3) if n_frames > 1,
            or (n_atoms, 3) if n_frames == 1.
        box_vectors_quantity : openmm.unit.Quantity
            Box vectors array with shape (n_frames, 3, 3) if n_frames > 1,
            or (3, 3) if n_frames == 1.
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
                f"Requested {n_frames} frames but trajectory only has {total_frames} frames. "
                f"Please reduce n_frames in the scaling config."
            )

        # Get the last n_frames
        if n_frames == 1:
            # Return single frame without batch dimension
            last_coords = coords_list[-1]
            last_box_vectors = box_vectors_list[-1]
            coords_quantity = last_coords.detach().cpu().numpy() * openmm.unit.angstrom
            box_vectors_quantity = (
                last_box_vectors.detach().cpu().numpy() * openmm.unit.angstrom
            )
        else:
            # Return multiple frames with batch dimension
            selected_coords = coords_list[-n_frames:]
            selected_box_vectors = box_vectors_list[-n_frames:]

            coords_array = np.stack(
                [c.detach().cpu().numpy() for c in selected_coords], axis=0
            )
            box_vectors_array = np.stack(
                [b.detach().cpu().numpy() for b in selected_box_vectors], axis=0
            )

            coords_quantity = coords_array * openmm.unit.angstrom
            box_vectors_quantity = box_vectors_array * openmm.unit.angstrom

        print(f"  Loaded {n_frames} frame(s) from {total_frames} total frames")
        return coords_quantity, box_vectors_quantity
