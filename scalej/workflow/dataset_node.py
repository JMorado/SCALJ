"""Dataset preparation and combination node."""

import argparse
from typing import Any

import datasets
import datasets.table
import numpy as np
import pyarrow
import smee
import smee.converters
import torch
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule

from ..cli.utils import create_configs_from_dict, load_config
from ._utils import load_pickle, save_pickle
from .node import WorkflowNode

# Define dataset schema
DATA_SCHEMA = pyarrow.schema(
    [
        ("mixture_id", pyarrow.string()),
        ("smiles", pyarrow.string()),
        ("coords", pyarrow.list_(pyarrow.float64())),
        ("box_vectors", pyarrow.list_(pyarrow.float64())),
        ("energy", pyarrow.list_(pyarrow.float64())),
        ("forces", pyarrow.list_(pyarrow.float64())),
    ]
)


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

    @staticmethod
    def _create_entries_from_ml_output(
        mixture_id, smiles, coords_list, box_vectors_list, energies, forces
    ):
        """Convert ML potential outputs into dataset entry dictionaries."""
        all_coords = []
        all_box_vectors = []
        all_energies = []
        all_forces = []

        for coord, box_vec, energy, force in zip(
            coords_list, box_vectors_list, energies, forces
        ):
            # Ensure NumPy arrays
            if not isinstance(coord, np.ndarray):
                coord = np.asarray(coord)
            if not isinstance(box_vec, np.ndarray):
                box_vec = np.asarray(box_vec)

            all_coords.extend(coord.flatten().tolist())
            all_box_vectors.extend(box_vec.flatten().tolist())
            all_energies.append(float(energy))
            all_forces.extend(force.flatten().tolist())

        entry = {
            "mixture_id": mixture_id,
            "smiles": smiles,
            "coords": all_coords,
            "box_vectors": all_box_vectors,
            "energy": all_energies,
            "forces": all_forces,
        }

        return [entry]

    @staticmethod
    def _create_dataset(entries):
        """Create a dataset from a list of entries."""
        table = pyarrow.Table.from_pylist(
            [
                {
                    "mixture_id": entry["mixture_id"],
                    "smiles": entry["smiles"],
                    "coords": entry["coords"],
                    "box_vectors": entry["box_vectors"],
                    "energy": entry["energy"],
                    "forces": entry["forces"],
                }
                for entry in entries
            ],
            schema=DATA_SCHEMA,
        )

        dataset = datasets.Dataset(datasets.table.InMemoryTable(table))
        return dataset

    @staticmethod
    def _combine_datasets(
        mixture_datasets, composite_tensor_system=None, all_tensor_systems=None
    ):
        """Combine multiple mixture datasets into a single dataset."""
        all_entries = []

        for mixture_id, ds in mixture_datasets.items():
            for entry in ds:
                entry_dict = dict(entry)
                if "mixture_id" not in entry_dict or entry_dict["mixture_id"] is None:
                    entry_dict["mixture_id"] = mixture_id
                all_entries.append(entry_dict)

        # Ensure tensors are converted to lists for PyArrow
        for entry in all_entries:
            if isinstance(entry["coords"], torch.Tensor):
                entry["coords"] = entry["coords"].flatten().tolist()
            if isinstance(entry["forces"], torch.Tensor):
                entry["forces"] = entry["forces"].flatten().tolist()

        # Create combined dataset
        table = pyarrow.Table.from_pylist(all_entries, schema=DATA_SCHEMA)
        combined = datasets.Dataset(datasets.table.InMemoryTable(table))
        combined.set_format("torch")

        return combined

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

        # Load force field for composite system creation
        force_field = ForceField(general_config.force_field_name, load_plugins=True)
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

            # Create dataset entries
            smiles_str = ".".join([comp.smiles for comp in system.components])

            entries = self._create_entries_from_ml_output(
                mixture_id=system.name,
                smiles=smiles_str,
                coords_list=coords_scaled,
                box_vectors_list=box_vectors_scaled,
                energies=energies,
                forces=forces,
            )

            # Create dataset
            system_dataset = self._create_dataset(entries)
            systems_ds[system.name] = system_dataset

            print(f"  Entries: {len(entries)}")
            print(f"  Dataset size: {len(system_dataset)}")

        # Build composite system and force field
        print(f"\n{'=' * 80}")
        print("Building composite system for multi-system training...")
        print(f"{'=' * 80}")

        composite_mols = [
            Molecule.from_smiles(comp.smiles)
            for system in general_config.systems
            for comp in system.components
        ]

        composite_interchanges = [
            Interchange.from_smirnoff(force_field, [mol]) for mol in composite_mols
        ]

        composite_tensor_forcefield, composite_topologies = (
            smee.converters.convert_interchange(composite_interchanges)
        )

        composite_tensor_system = smee.TensorSystem(
            composite_topologies,
            [
                comp.nmol
                for system in general_config.systems
                for comp in system.components
            ],
            is_periodic=True,
        )

        print("Composite system created:")
        print(f"  Total components: {len(composite_topologies)}")
        print(
            f"  Total molecules: {sum(comp.nmol for system in general_config.systems for comp in system.components)}"
        )

        # Create individual tensor systems for each mixture
        all_tensor_systems = {}
        idx_counter = 0

        for system in general_config.systems:
            n_comps = len(system.components)
            system_topologies = composite_topologies[
                idx_counter : idx_counter + n_comps
            ]
            idx_counter += n_comps

            all_tensor_systems[system.name] = smee.TensorSystem(
                system_topologies,
                [comp.nmol for comp in system.components],
                is_periodic=True,
            )

        # Combine datasets with padding
        print(f"\n{'=' * 80}")
        print("Combining datasets...")
        print(f"{'=' * 80}")

        combined_dataset = self._combine_datasets(
            systems_ds,
            composite_tensor_system=composite_tensor_system,
            all_tensor_systems=all_tensor_systems,
        )

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
