"""Molecular dynamics simulation node."""

import argparse
from pathlib import Path
from typing import Any

import openmm
import openmm.app
import openmm.unit
import smee
import smee.mm

from ..cli.utils import create_configs_from_dict, load_config
from ._utils import load_pickle
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

            # Run simulation
            print("\nRunning MD simulation...")
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

            results[system.name] = {
                "trajectory": str(trajectory_path),
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
        pdb_reporter_file = output_path.parent / f"trajectory_{output_path.stem}.pdb"
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
