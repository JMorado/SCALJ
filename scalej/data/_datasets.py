"""Dataset creation and combination functions."""

import datasets
import datasets.table
import numpy as np
import pyarrow
import torch

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
