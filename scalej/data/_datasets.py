"""Dataset creation and combination functions."""

import datasets
import datasets.table
import numpy as np
import pyarrow
import torch

DATA_SCHEMA = pyarrow.schema(
    [
        ("id", pyarrow.string()),
        ("smiles", pyarrow.string()),
        ("coords", pyarrow.list_(pyarrow.float64())),
        ("box_vectors", pyarrow.list_(pyarrow.float64())),
        ("energy", pyarrow.list_(pyarrow.float64())),
        ("forces", pyarrow.list_(pyarrow.float64())),
    ]
)


def create_dataset_entry(
    id: str,
    smiles: str,
    coords_list: list[np.ndarray],
    box_vectors_list: list[np.ndarray],
    energies: np.ndarray,
    forces: np.ndarray,
) -> dict:
    """
    Create a dataset entry from energy/force data.

    Parameters
    ----------
    id : str
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
        coords_list, box_vectors_list, energies, forces, strict=True
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
        "id": id,
        "smiles": smiles,
        "coords": all_coords,
        "box_vectors": all_box_vectors,
        "energy": all_energies,
        "forces": all_forces,
    }


def create_dataset(entries: list[dict]) -> datasets.Dataset:
    """
    Create a HuggingFace dataset from entry dictionaries.

    Parameters
    ----------
    entries : list[dict]
        List of dataset entries from create_dataset_entry().

    Returns
    -------
    datasets.Dataset
        HuggingFace dataset with torch format.
    """
    table = pyarrow.Table.from_pylist(
        [
            {
                "id": entry["id"],
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
    dataset.set_format("torch")
    return dataset


def combine_datasets(
    mixture_datasets: dict[str, datasets.Dataset],
) -> datasets.Dataset:
    """
    Combine multiple datasets into a single dataset.

    Parameters
    ----------
    datasets : dict[str, datasets.Dataset]
        Dictionary mapping IDs to their datasets.

    Returns
    -------
    datasets.Dataset
        Combined dataset with torch format.
    """
    # Gather all entries from the individual datasets.
    all_entries = []
    for ds in mixture_datasets.values():
        for entry in ds:
            entry_dict = dict(entry)
            all_entries.append(entry_dict)

    # Ensure tensors are converted to lists for PyArrow.
    for entry in all_entries:
        for key in ("coords", "forces", "box_vectors", "energy"):
            if isinstance(entry[key], torch.Tensor):
                entry[key] = entry[key].flatten().tolist()

    # Create combined dataset.
    table = pyarrow.Table.from_pylist(all_entries, schema=DATA_SCHEMA)
    combined = datasets.Dataset(datasets.table.InMemoryTable(table))
    combined.set_format("torch")

    return combined
