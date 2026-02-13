"""I/O utilities for SCALeJ."""

import copy
import pickle as pkl
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import smee
    from openff.toolkit import ForceField


def load_pickle(file_path: Path | str) -> Any:
    """Load an object from a pickle file.

    Parameters
    ----------
    file_path : Path | str
        Path to the pickle file.

    Returns
    -------
    Any
        The loaded object.

    Raises
    ------
    FileNotFoundError
        If the pickle file does not exist.

    Examples
    --------
    >>> data = load_pickle("output/system.pkl")
    """
    file = Path(file_path)
    if not file.exists():
        raise FileNotFoundError(f"Pickle file not found: {file}")

    with open(file_path, "rb") as f:
        return pkl.load(f)


def save_pickle(obj: Any, file_path: Path | str) -> None:
    """Save an object to a pickle file.

    Creates parent directories if they don't exist.

    Parameters
    ----------
    obj : Any
        The object to save.
    file_path : Path | str
        Path for the output pickle file.

    Examples
    --------
    >>> save_pickle({"data": [1, 2, 3]}, "output/data.pkl")
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        pkl.dump(obj, f)


def export_forcefield_to_offxml(
    base_forcefield: "ForceField",
    tensor_forcefield: "smee.TensorForceField",
    output_path: Path | str,
) -> "ForceField":
    """Export tensor force field parameters to OpenFF XML format.

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

    Notes
    -----
    Currently only supports updating vdW parameters (epsilon, sigma)
    for Lennard-Jones or Double Exponential potentials.

    Examples
    --------
    >>> updated_ff = export_forcefield_to_offxml(
    ...     original_ff, trained_tensor_ff, "optimized.offxml"
    ... )
    """
    from openff.units import unit as offunit

    forcefield = copy.deepcopy(base_forcefield)

    # Determine which handler to use
    tag = (
        "vdW"
        if forcefield.get_parameter_handler("vdW").parameters
        else "DoubleExponential"
    )

    potential_vdw = tensor_forcefield.potentials_by_type["vdW"]
    off_potential_vdw = forcefield.get_parameter_handler(tag)

    for i in range(potential_vdw.parameters.shape[1]):
        col = potential_vdw.parameter_cols[i]
        for j in range(potential_vdw.parameters.shape[0]):
            smirk_id = potential_vdw.parameter_keys[j].id
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


def load_forcefield(
    forcefield_name: str,
    load_plugins: bool = True,
) -> "ForceField":
    """Load an OpenFF force field by name.

    Parameters
    ----------
    forcefield_name : str
        Name of the force field (e.g., "openff-2.0.0.offxml").
    load_plugins : bool
        Whether to load force field plugins.

    Returns
    -------
    ForceField
        The loaded OpenFF force field.

    Examples
    --------
    >>> ff = load_forcefield("openff-2.0.0.offxml")
    """
    from openff.toolkit import ForceField

    return ForceField(forcefield_name, load_plugins=load_plugins)
