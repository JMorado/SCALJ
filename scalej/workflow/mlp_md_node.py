"""MLP MD relaxation node."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import openmm
import openmm.unit
import smee.mm
from tqdm import tqdm

from ..cli.utils import create_configs_from_dict, load_config
from ._utils import load_pickle, save_pickle
from .base_nodes import MLPotentialBaseNode


class MLPMDNode(MLPotentialBaseNode):
    """
    MLP MD node for running ML potential relaxation on trajectory frames.

    Inputs:
    - system_{system}.pkl: System state from SystemSetupNode
    - trajectory_{system}.dcd: Trajectory from MDNode (or external)
    - config: MLP simulation parameters

    Outputs:
    - mlp_coords_{system}.pkl: Relaxed coordinates and box vectors
    """

    @classmethod
    def name(cls) -> str:
        return "mlp_md"

    @classmethod
    def description(cls) -> str:
        return """MLP MD node for running ML potential relaxation on trajectory frames.

Inputs:
- system_{system}.pkl: System state from SystemSetupNode
- trajectory_{system}.dcd: Trajectory from MDNode (or external)
- config: MLP simulation parameters

Outputs:
- mlp_coords_{system}.pkl: Relaxed coordinates and box vectors"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--system-name",
            type=str,
            help="Process only this system (default: all systems in config)",
        )
        parser.add_argument(
            "--system-file",
            type=str,
            help="Path to system pickle file (if not using default naming)",
        )
        parser.add_argument(
            "--trajectory",
            type=str,
            help="Path to trajectory file (overrides config/default)",
        )

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Execute MLP MD relaxation."""
        print("=" * 80)
        print("MLPMDNode: MLP Relaxation")
        print("=" * 80)

        # Load configuration
        config_dict = load_config(args.config)
        general_config, simulation_config, scaling_config, _, _ = (
            create_configs_from_dict(config_dict)
        )

        self._ensure_output_dir(args.output_dir)

        if simulation_config.n_mlp_steps <= 0:
            print("n_mlp_steps is 0, skipping MLP relaxation")
            return {}

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

            # Load system state
            if args.system_file:
                system_file = Path(args.system_file)
            else:
                system_file = self._output_path(
                    args.output_dir, f"system_{system.name}.pkl"
                )

            if not system_file.exists():
                raise FileNotFoundError(
                    f"System file not found: {system_file}. "
                    "Run 'scalej system_setup' first."
                )

            system_state = load_pickle(system_file)
            tensor_system = system_state["tensor_system"]

            # Determine trajectory path
            trajectory_path = self._get_trajectory_path(args, system)
            if not Path(trajectory_path).exists():
                raise FileNotFoundError(f"Trajectory not found: {trajectory_path}")

            print(f"Loading trajectory: {trajectory_path}")
            coords, box_vectors = self._load_last_frames(
                trajectory_path, scaling_config.n_frames
            )

            # Run MLP simulation
            print("\nRunning MLP relaxation...")
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

            coords_relaxed, box_vectors_relaxed = self._run_mlp_simulation(
                mlp_simulation, coords, box_vectors, simulation_config.n_mlp_steps
            )

            # Save relaxed coordinates
            output_file = self._output_path(
                args.output_dir, f"mlp_coords_{system.name}.pkl"
            )
            output_data = {
                "coords": coords_relaxed,
                "box_vectors": box_vectors_relaxed,
                "n_mlp_steps": simulation_config.n_mlp_steps,
                "mlp_name": general_config.mlp_name,
            }

            save_pickle(output_data, output_file)
            print(f"  MLP relaxed coordinates saved: {output_file}")

            results[system.name] = {
                "mlp_coords_file": str(output_file),
            }

        print(f"\n{'=' * 80}")
        print("MLPMDNode completed successfully")
        print(f"{'=' * 80}")

        return results

    def _get_trajectory_path(self, args: argparse.Namespace, system) -> str:
        """Determine trajectory path from args or config."""
        # Priority 1: Command-line override
        if args.trajectory:
            return args.trajectory

        # Priority 2: Config trajectory_path
        if system.trajectory_path:
            return system.trajectory_path

        # Priority 3: Default location from MDNode output
        return str(self._output_path(args.output_dir, f"trajectory_{system.name}.dcd"))

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
                f"Requested {n_frames} frames but trajectory only has "
                f"{total_frames} frames. Please reduce n_frames in scaling config."
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
