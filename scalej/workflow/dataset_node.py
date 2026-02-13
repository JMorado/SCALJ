"""Dataset preparation and combination node."""

import argparse
from typing import Any

from .. import systems
from ..cli.utils import create_configs_from_dict, load_config
from ..io import load_pickle, save_pickle
from .node import WorkflowNode


class DatasetNode(WorkflowNode):
    """
    Dataset node for preparing combined datasets for training.

    Inputs:
    - energies_forces_{system}.pkl: Energy/force data from MLPotentialNode
    - config: System definitions and force field

    Outputs:
    - combined_dataset.pkl: Combined dataset ready for training
    - composite_system.pkl: Composite tensor system and force field
    """

    @classmethod
    def name(cls) -> str:
        return "dataset"

    @classmethod
    def description(cls) -> str:
        return """Dataset node for preparing combined datasets for training.

Inputs:
- energies_forces_{system}.pkl: Energy/force data from MLPotentialNode
- config: System definitions and force field

Outputs:
- combined_dataset.pkl: Combined dataset ready for training
- composite_system.pkl: Composite tensor system and force field"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--force-rebuild",
            action="store_true",
            help="Force rebuild of composite system even if cached",
        )

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Execute dataset preparation."""
        print("=" * 80)
        print("DatasetNode: Dataset Preparation")
        print("=" * 80)

        # Load configuration
        config_dict = load_config(args.config)
        general_config, _, _, _, _ = create_configs_from_dict(config_dict)

        self._ensure_output_dir(args.output_dir)

        print(f"Force field: {general_config.force_field_name}")

        # Build datasets for each system
        print(f"\n{'=' * 80}")
        print("Loading datasets from individual systems...")
        print(f"{'=' * 80}")

        systems_ds = {}

        for system in general_config.systems:
            print(f"\nSystem: {system.name}")

            # Load energies/forces
            ef_file = self._output_path(
                args.output_dir, f"energies_forces_{system.name}.pkl"
            )
            ef_data = load_pickle(ef_file)

            energies = ef_data["energies"]
            forces = ef_data["forces"]
            coords_scaled = ef_data["coords_scaled"]
            box_vectors_scaled = ef_data["box_vectors_scaled"]

            # Create dataset entry using the public API
            smiles_str = ".".join([comp.smiles for comp in system.components])

            entry = systems.create_dataset_entry(
                mixture_id=system.name,
                smiles=smiles_str,
                coords_list=coords_scaled,
                box_vectors_list=box_vectors_scaled,
                energies=energies,
                forces=forces,
            )

            # Create dataset using the public API
            system_dataset = systems.create_dataset([entry])
            systems_ds[system.name] = system_dataset

            print(f"  Entries: 1")
            print(f"  Dataset size: {len(system_dataset)}")

        # Build composite system and force field using the public API
        print(f"\n{'=' * 80}")
        print("Building composite system for multi-system training...")
        print(f"{'=' * 80}")

        # Convert system configs to dict format for API
        systems_config = [
            {
                "name": system.name,
                "components": [
                    {"smiles": comp.smiles, "nmol": comp.nmol}
                    for comp in system.components
                ],
            }
            for system in general_config.systems
        ]

        (
            composite_tensor_forcefield,
            composite_tensor_system,
            composite_topologies,
            all_tensor_systems,
            force_field,
        ) = systems.create_composite_system(
            systems_config, forcefield_name=general_config.force_field_name
        )

        print("Composite system created:")
        print(f"  Total components: {len(composite_topologies)}")
        print(
            f"  Total molecules: {sum(comp.nmol for system in general_config.systems for comp in system.components)}"
        )

        # Combine datasets using the public API
        print(f"\n{'=' * 80}")
        print("Combining datasets...")
        print(f"{'=' * 80}")

        combined_dataset = systems.combine_datasets(systems_ds)

        print(f"Combined dataset created:")
        print(f"  Total configurations: {len(combined_dataset)}")
        print(f"  Systems: {', '.join(systems_ds.keys())}")

        # Save combined dataset
        dataset_file = self._output_path(args.output_dir, "combined_dataset.pkl")
        save_pickle(combined_dataset, dataset_file)
        print(f"\nCombined dataset saved: {dataset_file}")

        # Save composite system
        composite_file = self._output_path(args.output_dir, "composite_system.pkl")
        composite_data = {
            "composite_tensor_forcefield": composite_tensor_forcefield,
            "composite_tensor_system": composite_tensor_system,
            "composite_topologies": composite_topologies,
            "all_tensor_systems": all_tensor_systems,
            "force_field": force_field,
        }

        save_pickle(composite_data, composite_file)
        print(f"Composite system saved: {composite_file}")

        print(f"\n{'=' * 80}")
        print("DatasetNode completed successfully")
        print(f"{'=' * 80}")

        return {
            "dataset_file": str(dataset_file),
            "composite_file": str(composite_file),
            "n_configurations": len(combined_dataset),
            "n_systems": len(systems_ds),
        }
