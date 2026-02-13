"""System setup node for creating system state from configuration."""

import argparse
from typing import Any

import smee
import smee.converters
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule

from ..cli.utils import create_configs_from_dict, load_config
from ._utils import save_pickle
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
        general_config, _, _, _, _ = create_configs_from_dict(config_dict)

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
