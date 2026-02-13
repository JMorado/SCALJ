"""Configuration scaling node."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import openmm.unit

from ..cli.utils import create_configs_from_dict, load_config
from ..io import load_pickle, save_pickle

# Import API functions
from ..scaling import create_scaled_configurations, generate_scale_factors
from ..simulation import load_trajectory_frames
from .node import WorkflowNode


class ScalingNode(WorkflowNode):
    """
    Scaling node for generating scaled configurations for LJ parameter fitting.

    Inputs:
    - system_{system}.pkl: System state from SystemSetupNode
    - trajectory_{system}.dcd OR mlp_coords_{system}.pkl: Coordinates source
    - config: Scaling parameters

    Outputs:
    - scaled_{system}.pkl: Scaled coordinates and box vectors
    - scale_factors_{system}.npy: Scale factors array
    """

    @classmethod
    def name(cls) -> str:
        return "scaling"

    @classmethod
    def description(cls) -> str:
        return """Scaling node for generating scaled configurations.

Inputs:
- system_{system}.pkl: System state from SystemSetupNode
- trajectory or mlp_coords file: Coordinates source
- config: Scaling parameters

Outputs:
- scaled_{system}.pkl: Scaled coordinates and box vectors
- scale_factors_{system}.npy: Scale factors array"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--system-name",
            type=str,
            help="Process only this system (default: all systems)",
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
        parser.add_argument(
            "--mlp-coords",
            type=str,
            help="Path to MLP coords file (if using MLP relaxed frames)",
        )

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Execute configuration scaling."""
        print("=" * 80)
        print("ScalingNode: Configuration Scaling")
        print("=" * 80)

        # Load configuration
        config_dict = load_config(args.config)
        general_config, _, scaling_config, _, _ = create_configs_from_dict(config_dict)

        self._ensure_output_dir(args.output_dir)

        # Generate scale factors
        print("\nGenerating scale factors...")
        print(f"  Close range: {scaling_config.close_range}")
        print(f"  Equilibrium range: {scaling_config.equilibrium_range}")
        print(f"  Long range: {scaling_config.long_range}")

        # Use API function
        scale_factors = generate_scale_factors(
            close_range=scaling_config.close_range,
            equilibrium_range=scaling_config.equilibrium_range,
            long_range=scaling_config.long_range,
        )
        print(f"  Total scale factors: {len(scale_factors)}")

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

            # Load system state (topology/forcefield only in new format)
            if args.system_file:
                system_file = Path(args.system_file)
            else:
                system_file = self._output_path(
                    args.output_dir, f"system_{system.name}.pkl"
                )

            system_state = load_pickle(system_file)
            tensor_system = system_state["tensor_system"]

            # Load coordinates from appropriate source
            coords, box_vectors = self._load_coordinates(
                args, system, system_state, scaling_config.n_frames
            )

            # Convert to numpy arrays in angstroms (if they're OpenMM quantities)
            if hasattr(coords, "value_in_unit"):
                coords_np = coords.value_in_unit(openmm.unit.angstrom)
            else:
                coords_np = coords

            if hasattr(box_vectors, "value_in_unit"):
                box_vectors_np = box_vectors.value_in_unit(openmm.unit.angstrom)
            else:
                box_vectors_np = box_vectors

            # Determine number of input frames
            n_input_frames = coords_np.shape[0] if coords_np.ndim == 3 else 1

            # Generate scaled configurations using API function
            print("Creating scaled configurations...")
            print(f"  Input frames: {n_input_frames}")
            print(f"  Scale factors: {len(scale_factors)}")
            print(f"  Total configurations: {n_input_frames * len(scale_factors)}")
            scaling_result = create_scaled_configurations(
                tensor_system, coords_np, box_vectors_np, scale_factors
            )
            coords_scaled = scaling_result.coords
            box_vectors_scaled = scaling_result.box_vectors
            expanded_scale_factors = scaling_result.scale_factors

            print(f"  Generated {len(coords_scaled)} scaled configurations")

            # Save scaled data
            scaled_file = self._output_path(
                args.output_dir, f"scaled_{system.name}.pkl"
            )
            scaled_data = {
                "coords_scaled": coords_scaled,
                "box_vectors_scaled": box_vectors_scaled,
                "scale_factors": expanded_scale_factors,
                "tensor_system": tensor_system,
                "tensor_forcefield": system_state.get("tensor_forcefield"),
                "components": system_state.get("components", []),
            }

            save_pickle(scaled_data, scaled_file)
            print(f"  Scaled data saved: {scaled_file}")

            results[system.name] = {
                "scaled_file": str(scaled_file),
                "n_configurations": len(coords_scaled),
            }

        # Save scale factors
        if args.system_name:
            scale_factors_file = self._output_path(
                args.output_dir, f"scale_factors_{args.system_name}.npy"
            )
        else:
            scale_factors_file = self._output_path(args.output_dir, "scale_factors.npy")

        np.save(scale_factors_file, scale_factors)
        print(f"\nScale factors saved: {scale_factors_file}")

        print(f"\n{'=' * 80}")
        print("ScalingNode completed successfully")
        print(f"{'=' * 80}")

        return {"systems": results, "scale_factors_file": str(scale_factors_file)}

    def _load_coordinates(self, args, system, system_state, n_frames):
        """Load coordinates from appropriate source.

        Priority:
        1. --mlp-coords argument
        2. mlp_coords_{system}.pkl (if exists)
        3. --trajectory argument
        4. system.trajectory_path from config
        5. trajectory_{system}.dcd (default)
        6. coords in system_state (backward compatibility)
        """
        # Check for MLP coords file
        mlp_coords_file = None
        if args.mlp_coords:
            mlp_coords_file = Path(args.mlp_coords)
        else:
            default_mlp = self._output_path(
                args.output_dir, f"mlp_coords_{system.name}.pkl"
            )
            if default_mlp.exists():
                mlp_coords_file = default_mlp

        if mlp_coords_file and mlp_coords_file.exists():
            print(f"Loading MLP relaxed coordinates: {mlp_coords_file}")
            mlp_data = load_pickle(mlp_coords_file)
            return mlp_data["coords"], mlp_data["box_vectors"]

        # Check for trajectory file
        trajectory_path = None
        if args.trajectory:
            trajectory_path = Path(args.trajectory)
        elif system.trajectory_path:
            trajectory_path = Path(system.trajectory_path)
        else:
            default_traj = self._output_path(
                args.output_dir, f"trajectory_{system.name}.dcd"
            )
            if default_traj.exists():
                trajectory_path = default_traj

        if trajectory_path and trajectory_path.exists():
            print(f"Loading trajectory: {trajectory_path}")
            frames = load_trajectory_frames(trajectory_path, n_frames=n_frames)
            return frames.coords, frames.box_vectors

        # Backward compatibility: coords in system_state
        if "coords" in system_state and "box_vectors" in system_state:
            print("Using coordinates from system state (backward compat)")
            return system_state["coords"], system_state["box_vectors"]

        raise FileNotFoundError(
            f"No coordinate source found for system '{system.name}'. "
            "Run MDNode or provide --trajectory or --mlp-coords."
        )
