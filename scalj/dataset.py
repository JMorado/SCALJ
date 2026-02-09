"""Dataset creation and management for training."""

import typing

import datasets
import datasets.table
import pyarrow
import torch

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


class Entry(typing.TypedDict):
    """
    Represents a set of reference energies and forces.

    Attributes
    ----------
    mixture_id : str
        Identifier for the mixture/system this entry belongs to.
    smiles : str
        The indexed SMILES description of the molecule the energies and forces
        were computed for.
    coords : torch.Tensor
        The coordinates [Å] the energies and forces were evaluated at with
        shape=(n_confs, n_particles, 3).
    box_vectors : torch.Tensor
        The box vectors [Å] the energies and forces were evaluated at with
        shape=(n_confs, 3, 3).
    energy : torch.Tensor
        The reference energies [kcal/mol] with shape=(n_confs,).
    forces : torch.Tensor
        The reference forces [kcal/mol/Å] with shape=(n_confs, n_particles, 3).
    """

    mixture_id: str
    smiles: str
    coords: torch.Tensor
    box_vectors: torch.Tensor
    energy: torch.Tensor
    forces: torch.Tensor


def create_dataset(entries: list[Entry]) -> datasets.Dataset:
    """
    Create a dataset from a list of existing entries.

    Args:
        entries: The entries to create the dataset from.

    Returns:
        The created dataset.
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


def create_entries_from_ml_output(
    mixture_id: str,
    smiles: str,
    coords_list: list,
    box_vectors_list: list,
    energies,
    forces,
) -> list[dict]:
    """
    Convert ML potential outputs into dataset entry dictionaries.

    Parameters
    ----------
    mixture_id : str
        Identifier for the mixture/system.
    smiles : str
        SMILES string for the molecule(s).
    coords_list : list
        List of coordinate arrays.
    box_vectors_list : list
        List of box vector arrays.
    energies : np.ndarray
        Array of energies from ML potential.
    forces : np.ndarray
        Array of forces from ML potential.

    Returns
    -------
    list[dict]
        List of entry dictionaries ready for create_dataset.
    """
    import numpy as np

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


def combine_datasets(
    mixture_datasets: dict[str, datasets.Dataset],
    composite_tensor_system=None,
    all_tensor_systems: dict[str] | None = None,
) -> datasets.Dataset:
    """
    Combine multiple mixture datasets into a single dataset.

    Each dataset will have its mixture_id field set to the corresponding key.
    If composite_tensor_system and all_tensor_systems are provided, coordinates
    and forces will be padded to match the composite system.

    Args:
        mixture_datasets: Dictionary mapping mixture_id to dataset.
        composite_tensor_system: Optional composite system for padding.
        all_tensor_systems: Optional dict mapping mixture_id to individual systems.

    Returns:
        Combined dataset with all mixtures.
    """
    all_entries = []

    for mixture_id, ds in mixture_datasets.items():
        for entry in ds:
            # Add mixture_id if not already present
            entry_dict = dict(entry)
            if "mixture_id" not in entry_dict or entry_dict["mixture_id"] is None:
                entry_dict["mixture_id"] = mixture_id
            all_entries.append(entry_dict)

    # Pad if composite system is provided
    if composite_tensor_system is not None and all_tensor_systems is not None:
        total_particles = composite_tensor_system.n_particles

        for entry in all_entries:
            mixture_id = entry["mixture_id"]
            system = all_tensor_systems[mixture_id]
            n_particles_in_system = system.n_particles
            n_confs = len(entry["energy"])

            # Convert to tensors if they're lists and reshape
            coords = entry["coords"]
            if isinstance(coords, list):
                coords = torch.tensor(coords, dtype=torch.float64)
            coords = coords.reshape(n_confs, n_particles_in_system, 3)

            forces = entry["forces"]
            if isinstance(forces, list):
                forces = torch.tensor(forces, dtype=torch.float64)
            forces = forces.reshape(n_confs, n_particles_in_system, 3)

            # Pad coordinates
            coords_padded = torch.zeros(
                (n_confs, total_particles, 3), dtype=coords.dtype, device=coords.device
            )
            coords_padded[:, :n_particles_in_system, :] = coords
            entry["coords"] = coords_padded.flatten().tolist()

            # Pad forces
            forces_padded = torch.zeros(
                (n_confs, total_particles, 3), dtype=forces.dtype, device=forces.device
            )
            forces_padded[:, :n_particles_in_system, :] = forces
            entry["forces"] = forces_padded.flatten().tolist()
    else:
        # If not padding, ensure tensors are converted to lists for PyArrow
        for entry in all_entries:
            if isinstance(entry["coords"], torch.Tensor):
                entry["coords"] = entry["coords"].flatten().tolist()
            if isinstance(entry["forces"], torch.Tensor):
                entry["forces"] = entry["forces"].flatten().tolist()

    # Create combined dataset
    table = pyarrow.Table.from_pylist(all_entries, schema=DATA_SCHEMA)
    combined = datasets.Dataset(datasets.table.InMemoryTable(table))

    # Set format to torch after combining the datasets
    combined.set_format("torch")

    return combined


def extract_smiles(dataset: datasets.Dataset) -> list[str]:
    """
    Return a list of unique SMILES strings in the dataset.

    Args:
        dataset: The dataset to extract the SMILES strings from.

    Returns:
        The list of unique SMILES strings.
    """
    return sorted({*dataset.unique("smiles")})


def extract_mixture_ids(dataset: datasets.Dataset) -> list[str]:
    """
    Return a list of unique mixture IDs in the dataset.

    Args:
        dataset: The dataset to extract mixture IDs from.

    Returns:
        The list of unique mixture IDs.
    """
    return sorted({*dataset.unique("mixture_id")})
