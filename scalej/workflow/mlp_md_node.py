"""MLP MD relaxation node."""

import argparse
from pathlib import Path
from typing import Any

from .. import energy, simulation
from ..cli.utils import create_configs_from_dict, load_config
from ..io import load_pickle, save_pickle
from .node import WorkflowNode


class MLPMDNode(WorkflowNode):
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
            frames = simulation.load_trajectory_frames(
                trajectory_path, n_frames=scaling_config.n_frames
            )
            coords = frames.coords
            box_vectors = frames.box_vectors

            # Run MLP simulation using API
            print("\nRunning MLP relaxation...")
            print(f"  MLP: {general_config.mlp_name}")
            print(f"  Device: {simulation_config.mlp_device}")
            print(f"  Steps: {simulation_config.n_mlp_steps}")

            mlp_simulation = energy.setup_mlp_simulation(
                tensor_system,
                general_config.mlp_name,
                temperature=simulation_config.temperature,
                friction_coeff=simulation_config.friction_coeff,
                timestep=simulation_config.timestep,
                mlp_device=simulation_config.mlp_device,
                platform=simulation_config.platform,
            )

            coords_relaxed, box_vectors_relaxed = energy.run_mlp_relaxation(
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
        if args.trajectory:
            return args.trajectory

        if system.trajectory_path:
            return system.trajectory_path

        # Priority 3: Default location from MDNode output
        return str(self._output_path(args.output_dir, f"trajectory_{system.name}.dcd"))
