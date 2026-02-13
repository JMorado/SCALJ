"""System setup functions."""

from typing import TYPE_CHECKING

import datasets
import datasets.table
import numpy as np
import pyarrow
import smee
import smee.converters
import torch
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule

if TYPE_CHECKING:
    pass

# Define dataset schema for energy/force data
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


def create_system_from_smiles(
    smiles_list: list[str],
    nmol_list: list[int],
    forcefield_name: str = "openff-2.0.0.offxml",
) -> tuple[smee.TensorSystem, smee.TensorForceField, list[smee.TensorTopology]]:
    """Create a tensor system from SMILES strings.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings, one per component.
    nmol_list : list[int]
        Number of molecules for each component.
    forcefield_name : str
        Name of the OpenFF force field to use.

    Returns
    -------
    tuple[smee.TensorSystem, smee.TensorForceField, list[smee.TensorTopology]]
        The tensor system, force field, and list of topologies.

    Examples
    --------
    >>> system, ff, topos = create_system_from_smiles(
    ...     ["CCO", "O"],  # Ethanol and water
    ...     [100, 500],    # 100 ethanol, 500 water
    ... )
    """
    force_field = ForceField(forcefield_name, load_plugins=True)

    mols = [Molecule.from_smiles(smiles) for smiles in smiles_list]
    interchanges = [Interchange.from_smirnoff(force_field, [mol]) for mol in mols]

    tensor_forcefield, topologies = smee.converters.convert_interchange(interchanges)
    tensor_system = smee.TensorSystem(topologies, nmol_list, is_periodic=True)

    return tensor_system, tensor_forcefield, topologies


def create_composite_system(
    systems_config: list[dict],
    forcefield_name: str = "openff-2.0.0.offxml",
) -> tuple[
    smee.TensorForceField,
    smee.TensorSystem,
    list[smee.TensorTopology],
    dict[str, smee.TensorSystem],
    ForceField,
]:
    """Create a composite system from multiple system configurations.

    Builds a shared force field and individual tensor systems for
    multi-system training.

    Parameters
    ----------
    systems_config : list[dict]
        List of system configurations, each with:
        - "name": System identifier
        - "components": List of {"smiles": str, "nmol": int}
    forcefield_name : str
        Name of the OpenFF force field.

    Returns
    -------
    tuple
        - composite_tensor_forcefield: Shared force field
        - composite_tensor_system: Combined system
        - composite_topologies: All topologies
        - all_tensor_systems: Dict mapping names to individual systems
        - force_field: Original OpenFF force field

    Examples
    --------
    >>> config = [
    ...     {"name": "ethanol", "components": [{"smiles": "CCO", "nmol": 200}]},
    ...     {"name": "water", "components": [{"smiles": "O", "nmol": 1000}]},
    ... ]
    >>> ff, system, topos, systems, off_ff = create_composite_system(config)
    """
    force_field = ForceField(forcefield_name, load_plugins=True)

    # Collect all molecules
    composite_mols = [
        Molecule.from_smiles(comp["smiles"])
        for system in systems_config
        for comp in system["components"]
    ]

    composite_interchanges = [
        Interchange.from_smirnoff(force_field, [mol]) for mol in composite_mols
    ]

    composite_tensor_forcefield, composite_topologies = (
        smee.converters.convert_interchange(composite_interchanges)
    )

    composite_tensor_system = smee.TensorSystem(
        composite_topologies,
        [comp["nmol"] for system in systems_config for comp in system["components"]],
        is_periodic=True,
    )

    # Create individual tensor systems
    all_tensor_systems = {}
    idx_counter = 0

    for system in systems_config:
        n_comps = len(system["components"])
        system_topologies = composite_topologies[idx_counter : idx_counter + n_comps]
        idx_counter += n_comps

        all_tensor_systems[system["name"]] = smee.TensorSystem(
            system_topologies,
            [comp["nmol"] for comp in system["components"]],
            is_periodic=True,
        )

    return (
        composite_tensor_forcefield,
        composite_tensor_system,
        composite_topologies,
        all_tensor_systems,
        force_field,
    )


def create_dataset_entry(
    mixture_id: str,
    smiles: str,
    coords_list: list[np.ndarray],
    box_vectors_list: list[np.ndarray],
    energies: np.ndarray,
    forces: np.ndarray,
) -> dict:
    """Create a dataset entry from energy/force data.

    Parameters
    ----------
    mixture_id : str
        Unique identifier for this mixture/system.
    smiles : str
        SMILES string (dot-separated for mixtures).
    coords_list : list[np.ndarray]
        List of coordinate arrays.
    box_vectors_list : list[np.ndarray]
        List of box vector arrays.
    energies : np.ndarray
        Energy values in kcal/mol.
    forces : np.ndarray
        Force values in kcal/mol/Å.

    Returns
    -------
    dict
        Dataset entry dictionary.
    """
    all_coords = []
    all_box_vectors = []
    all_energies = []
    all_forces = []

    for coord, box_vec, energy, force in zip(
        coords_list, box_vectors_list, energies, forces
    ):
        if not isinstance(coord, np.ndarray):
            coord = np.asarray(coord)
        if not isinstance(box_vec, np.ndarray):
            box_vec = np.asarray(box_vec)

        all_coords.extend(coord.flatten().tolist())
        all_box_vectors.extend(box_vec.flatten().tolist())
        all_energies.append(float(energy))
        all_forces.extend(force.flatten().tolist())

    return {
        "mixture_id": mixture_id,
        "smiles": smiles,
        "coords": all_coords,
        "box_vectors": all_box_vectors,
        "energy": all_energies,
        "forces": all_forces,
    }


def create_dataset(entries: list[dict]) -> datasets.Dataset:
    """Create a HuggingFace dataset from entry dictionaries.

    Parameters
    ----------
    entries : list[dict]
        List of dataset entries from create_dataset_entry().

    Returns
    -------
    datasets.Dataset
        HuggingFace dataset with torch format.

    Examples
    --------
    >>> entry = create_dataset_entry("ethanol", "CCO", coords, boxes, E, F)
    >>> dataset = create_dataset([entry])
    """
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


def combine_datasets(
    mixture_datasets: dict[str, datasets.Dataset],
) -> datasets.Dataset:
    """Combine multiple mixture datasets into a single dataset.

    Parameters
    ----------
    mixture_datasets : dict[str, datasets.Dataset]
        Dictionary mapping mixture IDs to their datasets.

    Returns
    -------
    datasets.Dataset
        Combined dataset with torch format.

    Examples
    --------
    >>> combined = combine_datasets({
    ...     "ethanol": ethanol_dataset,
    ...     "water": water_dataset,
    ... })
    """
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
