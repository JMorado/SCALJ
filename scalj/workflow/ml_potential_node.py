"""ML potential energy and force computation node."""

import argparse
from pathlib import Path
from typing import Any

from ..cli.utils import create_configs_from_dict, load_config
from ._utils import load_pickle, save_pickle
from .base_nodes import MLPotentialBaseNode


class MLPotentialNode(MLPotentialBaseNode):
    """
    MLPotential node for computing energies and forces using a machine learning potential.

    Inputs:
    - system_{system}.pkl: System state from MDNode
    - scaled_{system}.pkl: Scaled configurations from ScalingNode
    - config: ML potential name and device settings

    Outputs:
    - energies_forces_{system}.pkl: Computed energies and forces using smee
    """

    @classmethod
    def name(cls) -> str:
        return "ml_potential"

    @classmethod
    def description(cls) -> str:
        return """MLPotential node for computing energies and forces using a machine learning potential.

Inputs:
- system_{system}.pkl: System state from MDNode
- scaled_{system}.pkl: Scaled configurations from ScalingNode
- config: ML potential name and device settings

Outputs:
- energies_forces_{system}.pkl: Computed energies and forces using smee"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--system-name",
            type=str,
            help="Process only this system (default: all systems)",
        )

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Execute ML potential energy/force computation."""
        print("=" * 80)
        print("MLPotentialNode: Energy/Force Computation")
        print("=" * 80)

        # Load configuration
        config_dict = load_config(args.config)
        (
            general_config,
            simulation_config,
            _,
            _,
            _,
        ) = create_configs_from_dict(config_dict)

        self._ensure_output_dir(args.output_dir)

        print(f"ML Potential: {general_config.mlp_name}")
        print(f"Device: {simulation_config.mlp_device}")

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
            system_file = self._output_path(
                args.output_dir, f"system_{system.name}.pkl"
            )
            system_state = load_pickle(system_file)
            tensor_system = system_state["tensor_system"]

            # Load scaled configurations
            scaled_file = self._output_path(
                args.output_dir, f"scaled_{system.name}.pkl"
            )
            scaled_data = load_pickle(scaled_file)

            coords_scaled = scaled_data["coords_scaled"]
            box_vectors_scaled = scaled_data["box_vectors_scaled"]

            print(f"Loaded {len(coords_scaled)} scaled configurations")

            # Setup ML potential simulation with smee
            print("Setting up ML potential...")
            mlp_simulation = self._setup_mlp_simulation(
                tensor_system,
                general_config.mlp_name,
                mlp_device=simulation_config.mlp_device,
                platform=simulation_config.platform,
            )

            # Compute energies and forces using smee (differentiable)
            print("Computing energies and forces...")
            energies, forces = self._compute_energies_forces(
                mlp_simulation, coords_scaled, box_vectors_scaled
            )

            print(f"  Computed energies: {len(energies)} configurations")
            print(f"  Energy shape: {energies.shape}")
            print(f"  Forces shape: {forces.shape}")

            # Save energies and forces
            ef_file = self._output_path(
                args.output_dir, f"energies_forces_{system.name}.pkl"
            )
            ef_data = {
                "energies": energies,
                "forces": forces,
                "coords_scaled": coords_scaled,
                "box_vectors_scaled": box_vectors_scaled,
                "scale_factors": scaled_data.get("scale_factors"),
                "components": scaled_data.get("components", []),
            }

            save_pickle(ef_data, ef_file)
            print(f"  Energies/forces saved: {ef_file}")

            results[system.name] = {
                "energies_forces_file": str(ef_file),
                "n_configurations": len(energies),
            }

        print(f"\n{'=' * 80}")
        print("MLPotentialNode completed successfully")
        print(f"{'=' * 80}")

        return results
