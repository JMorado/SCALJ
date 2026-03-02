"""System creation functions for tensor systems and force fields."""

import smee
import smee.converters
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule


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
