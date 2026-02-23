"""Force field export node."""

import argparse
from pathlib import Path
from typing import Any

from openff.toolkit import ForceField

from ..cli.utils import create_configs_from_dict, load_config

# Import API function
from ..io import export_forcefield_to_offxml, load_pickle
from .node import WorkflowNode


class ExportNode(WorkflowNode):
    """
    Export node for saving trained force field parameters.

    Inputs:
    - trained_parameters.pkl: Trained parameters from TrainingNode
    - composite_system.pkl: Composite system from DatasetNode
    - config: Force field configuration

    Outputs:
    - optimized_forcefield.offxml: Force field in OpenFF XML format
    """

    @classmethod
    def name(cls) -> str:
        return "export"

    @classmethod
    def description(cls) -> str:
        return """Export node for saving trained force field parameters.

Inputs:
- trained_parameters.pkl: Trained parameters from TrainingNode
- composite_system.pkl: Composite system from DatasetNode
- config: Force field configuration

Outputs:
- optimized_forcefield.offxml: Force field in OpenFF XML format"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--params-file",
            type=str,
            default="trained_parameters.pkl",
            help="Name of parameters file (default: trained_parameters.pkl)",
        )
        parser.add_argument(
            "--output-name",
            type=str,
            default="optimized_forcefield.offxml",
            help="Name of output force field file (default: optimized_forcefield.offxml)",
        )

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Execute force field export."""
        print("=" * 80)
        print("ExportNode: Force Field Export")
        print("=" * 80)

        # Load configuration
        config_dict = load_config(args.config)
        general_config, *_ = create_configs_from_dict(config_dict)

        self._ensure_output_dir(args.output_dir)

        # Load trained parameters
        params_file = Path(args.params_file)
        params_data = load_pickle(params_file)

        # Handle both initial and final parameter files
        if "final_force_field" in params_data:
            final_force_field = params_data["final_force_field"]
            print(f"Loaded trained parameters from {params_file}")
        elif "initial_force_field" in params_data:
            final_force_field = params_data["initial_force_field"]
            print(f"Loaded initial parameters from {params_file}")
        else:
            raise KeyError(
                "Parameter file must contain either 'final_force_field' or 'initial_force_field' key"
            )
        print("Using force field for export")

        # Load composite system to get original force field
        composite_file = self._output_path(args.output_dir, "composite_system.pkl")
        composite_data = load_pickle(composite_file)
        force_field = composite_data.get("force_field")

        if force_field is None:
            # TODO: maybe remove this fallback and require the composite system to always include the force field
            print(f"Loading base force field: {general_config.force_field_name}")
            force_field = ForceField(general_config.force_field_name, load_plugins=True)

        # Export to OpenFF XML format using API function
        print("\nExporting to OpenFF XML format...")
        off_xml_file = self._output_path(args.output_dir, args.output_name)

        export_forcefield_to_offxml(force_field, final_force_field, off_xml_file)

        print(f"\n{'=' * 80}")
        print(f"Force field exported: {off_xml_file}")
        print(f"{'=' * 80}")

        return {
            "forcefield_file": str(off_xml_file),
        }
