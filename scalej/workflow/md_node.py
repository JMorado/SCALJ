"""Molecular dynamics simulation node."""

import argparse
from pathlib import Path
from typing import Any

from .. import simulation
from ..cli.utils import create_configs_from_dict, load_config
from ..io import load_pickle
from .node import WorkflowNode


class MDNode(WorkflowNode):
    """
    MD node for running classical molecular dynamics simulations.

    Inputs:
    - system_{system}.pkl: System state from SystemSetupNode
    - config: Simulation parameters

    Outputs:
    - trajectory_{system}.dcd: DCD trajectory file
    """

    @classmethod
    def name(cls) -> str:
        return "md"

    @classmethod
    def description(cls) -> str:
        return """MD node for running classical molecular dynamics simulations.

Inputs:
- system_{system}.pkl: System state from SystemSetupNode
- config: Simulation parameters

Outputs:
- trajectory_{system}.dcd: DCD trajectory file"""

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

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Execute MD simulation."""
        print("=" * 80)
        print("MDNode: Classical Molecular Dynamics Simulation")
        print("=" * 80)

        # Load configuration
        config_dict = load_config(args.config)
        general_config, simulation_config, _, _, _ = create_configs_from_dict(
            config_dict
        )

        self._ensure_output_dir(args.output_dir)

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

            # Check if trajectory_path is set - skip MD if so
            if system.trajectory_path:
                print(f"Trajectory path set in config: {system.trajectory_path}")
                print("Skipping MD simulation (use existing trajectory)")
                results[system.name] = {
                    "trajectory": system.trajectory_path,
                    "skipped": True,
                }
                continue

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
            tensor_forcefield = system_state["tensor_forcefield"]

            print("Components:")
            for comp in system_state.get("components", []):
                print(f"  - {comp['smiles']}: {comp['nmol']} molecules")

            # Define trajectory output path
            trajectory_path = self._output_path(
                args.output_dir, f"trajectory_{system.name}.dcd"
            )

            # Run simulation using the public API
            print("\nRunning MD simulation...")
            print(f"  Platform: {simulation_config.platform}")
            print(f"  Temperature: {simulation_config.temperature}")
            print(f"  Pressure: {simulation_config.pressure}")
            print(f"  Production steps: {simulation_config.n_production_steps}")

            simulation.run_md_simulation(
                tensor_system,
                tensor_forcefield,
                trajectory_path,
                temperature=simulation_config.temperature,
                pressure=simulation_config.pressure,
                timestep=simulation_config.timestep,
                n_equilibration_nvt_steps=simulation_config.n_equilibration_nvt_steps,
                n_equilibration_npt_steps=simulation_config.n_equilibration_npt_steps,
                n_production_steps=simulation_config.n_production_steps,
                report_interval=simulation_config.report_interval,
            )
            print(f"  Trajectory saved: {trajectory_path}")

            results[system.name] = {
                "trajectory": str(trajectory_path),
            }

        print(f"\n{'=' * 80}")
        print("MDNode completed successfully")
        print(f"{'=' * 80}")

        return results
