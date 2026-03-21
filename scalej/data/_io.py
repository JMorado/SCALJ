"""File I/O utilities."""

import copy
import json
import pickle
from pathlib import Path
from typing import Any

import datasets
import pandas as pd
import smee
import torch
from openff.toolkit import ForceField


def load_object(file_path: Path | str) -> Any:
    """
    Load an object serialised with :func:`save_object` (via ``torch.load``).

    Parameters
    ----------
    file_path : Path | str
        Path to the serialised file (``.pt``).

    Returns
    -------
    Any
        The loaded object.
    """
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")

    return torch.load(file, weights_only=False, map_location="cpu")


def save_object(obj: Any, file_path: Path | str) -> None:
    """
    Save an arbitrary object using ``torch.save``.

    Parameters
    ----------
    obj : Any
        The object to save.
    file_path : Path | str
        Path for the output file (``.pt``).
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, file_path)


def load_pickle(file_path: Path | str) -> Any:
    """Load an object from a standard pickle file."""
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")
    with open(file, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, file_path: Path | str) -> None:
    """Save an object to a standard pickle file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def save_dataset(dataset: datasets.Dataset, path: Path | str) -> None:
    """
    Save a HuggingFace ``Dataset`` to disk as Arrow IPC files.

    Parameters
    ----------
    dataset : datasets.Dataset
        The dataset to persist.
    path : Path | str
        Directory where the Arrow files will be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))


def load_dataset(path: Path | str) -> datasets.Dataset:
    """
    Load a HuggingFace ``Dataset`` previously saved with :func:`save_dataset`.

    Parameters
    ----------
    path : Path | str
        Directory containing the Arrow IPC files.

    Returns
    -------
    datasets.Dataset
    """
    import datasets as _datasets

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    return _datasets.Dataset.load_from_disk(str(path))


def save_parquet(df: pd.DataFrame, file_path: Path | str) -> None:
    """
    Write a pandas DataFrame to a Parquet file.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write.
    file_path : Path | str
        Output ``.parquet`` path.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)


def load_parquet(file_path: Path | str) -> pd.DataFrame:
    """
    Read a Parquet file into a pandas DataFrame.

    Parameters
    ----------
    file_path : Path | str
        Path to the ``.parquet`` file.

    Returns
    -------
    pd.DataFrame
    """
    import pandas as _pd

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    return _pd.read_parquet(file_path)


def save_json(obj: Any, file_path: Path | str) -> None:
    """
    Write a JSON-serialisable object to a file.

    Parameters
    ----------
    obj : Any
        Must be JSON-serialisable.
    file_path : Path | str
        Output ``.json`` path.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(file_path: Path | str) -> Any:
    """
    Read a JSON file.

    Parameters
    ----------
    file_path : Path | str
        Path to the ``.json`` file.

    Returns
    -------
    Any
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    with open(file_path) as f:
        return json.load(f)


def export_forcefield_to_offxml(
    base_forcefield: ForceField,
    tensor_forcefield: smee.TensorForceField,
    output_path: Path | str,
) -> ForceField:
    """
    Export tensor force field parameters to OpenFF XML format.

    Updates a base OpenFF ForceField with parameters from a tensor-based
    force field and saves to an OFFXML file.

    Parameters
    ----------
    base_forcefield : ForceField
        The original OpenFF force field to update.
    tensor_forcefield : smee.TensorForceField
        The tensor-based force field containing new parameters.
    output_path : Path | str
        Path for the output OFFXML file.

    Returns
    -------
    ForceField
        The updated OpenFF force field.
    """
    from openff.units import unit as offunit

    forcefield = copy.deepcopy(base_forcefield)

    # Determine which handler to use
    tag = (
        "vdW"
        if "vdW" in forcefield.registered_parameter_handlers
        else "DoubleExponential"
    )

    potential_vdw = tensor_forcefield.potentials_by_type["vdW"]
    off_potential_vdw = forcefield.get_parameter_handler(tag)

    for i in range(potential_vdw.parameters.shape[1]):
        col = potential_vdw.parameter_cols[i]
        for j in range(potential_vdw.parameters.shape[0]):
            smirk_id = potential_vdw.parameter_keys[j].id
            if "EP" in smirk_id:
                continue
            val = potential_vdw.parameters[j, i]
            unit = (
                offunit.kilocalories_per_mole if col == "epsilon" else offunit.angstrom
            )
            param = off_potential_vdw.get_parameter({"smirks": smirk_id})[0]
            setattr(param, col, val.item() * unit)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forcefield.to_file(str(output_path))

    return forcefield
