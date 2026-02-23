"""System setup node."""

import argparse
from typing import Any

from .. import systems
from ..cli.utils import create_configs_from_dict, load_config
from ..io import save_pickle
from .node import WorkflowNode


class SystemSetupNode(WorkflowNode):
    """
    System setup node for creating system state from configuration.

    Inputs (from config):
    - system definitions (SMILES, composition)
    - force field name

    Outputs:
    - system_{system}.pkl: System state with tensor_system, tensor_forcefield,
      topologies, nmol, components
    """

    @classmethod
    def name(cls) -> str:
        return "system_setup"

    @classmethod
    def description(cls) -> str:
        return """System setup node for creating system state from configuration.

Inputs (from config):
- system definitions (SMILES, composition)
- force field name

Outputs:
- system_{system}.pkl: System state with tensor_system, tensor_forcefield,
  topologies, nmol, components"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--system-name",
            type=str,
            help="Process only this system (default: all systems in config)",
        )

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Execute system setup."""
        print("=" * 80)
        print("SystemSetupNode: Creating System State")
        print("=" * 80)

        # Load configuration
        config_dict = load_config(args.config)
        general_config, *_ = create_configs_from_dict(config_dict)

        self._ensure_output_dir(args.output_dir)

        print(f"Force field: {general_config.force_field_name}")

        # Filter systems if requested
        systems_to_process = general_config.systems
        if args.system_name:
            systems_to_process = [
                s for s in systems_to_process if s.name == args.system_name
            ]
            if not systems_to_process:
                raise ValueError(f"System '{args.system_name}' not found in config")

        results = {}

        for system in systems_to_process:
            print(f"\n{'=' * 80}")
            print(f"Processing system: {system.name}")
            print(f"{'=' * 80}")

            # Create system from SMILES using the public API
            smiles_list = [comp.smiles for comp in system.components]
            nmol_list = [comp.nmol for comp in system.components]

            tensor_system, tensor_forcefield, topologies = (
                systems.create_system_from_smiles(
                    smiles_list,
                    nmol_list,
                    forcefield_name=general_config.force_field_name,
                )
            )

            print("Components:")
            for comp in system.components:
                print(f"  - {comp.smiles}: {comp.nmol} molecules")

            # Save system state (NO coords/box_vectors - just topology/forcefield)
            system_file = self._output_path(
                args.output_dir, f"system_{system.name}.pkl"
            )
            system_state = {
                "tensor_system": tensor_system,
                "tensor_forcefield": tensor_forcefield,
                "topologies": topologies,
                "nmol": nmol_list,
                "components": [
                    {"smiles": comp.smiles, "nmol": comp.nmol}
                    for comp in system.components
                ],
            }

            save_pickle(system_state, system_file)
            print(f"  System state saved: {system_file}")

            results[system.name] = {
                "system_file": str(system_file),
            }

        print(f"\n{'=' * 80}")
        print("SystemSetupNode completed successfully")
        print(f"{'=' * 80}")

        return results
