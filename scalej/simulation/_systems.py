"""System creation functions for tensor systems and forcefields."""

from typing import Optional

import smee
import smee.converters
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule


def create_system_from_smiles(
    smiles_list: list[str],
    nmol_list: list[int],
    forcefield_name: str = "openff-2.0.0.offxml",
    charge_assignment_callback: Optional[callable] = None,
) -> tuple[smee.TensorSystem, smee.TensorForceField, list[smee.TensorTopology]]:
    """
    Create a tensor system from SMILES strings.

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
    """
    force_field = ForceField(forcefield_name, load_plugins=True)
    mols = [Molecule.from_smiles(smiles) for smiles in smiles_list]

    # Optionally assign charges using the provided callback.
    if charge_assignment_callback is not None:
        for mol in mols:
            charge_assignment_callback(mol)

    # Create an Interchange for each molecule, then convert to tensor format.
    interchanges = [
        Interchange.from_smirnoff(
            force_field,
            [mol],
            charge_from_molecules=[mol] if charge_assignment_callback else None,
        )
        for mol in mols
    ]
    tensor_forcefield, topologies = smee.converters.convert_interchange(interchanges)
    tensor_system = smee.TensorSystem(topologies, nmol_list, is_periodic=True)

    return tensor_system, tensor_forcefield, topologies


def create_composite_system(
    systems_config: list[dict],
    forcefield_name: str = "openff-2.0.0.offxml",
    charge_assignment_callback: Optional[callable] = None,
) -> tuple[
    smee.TensorForceField,
    smee.TensorSystem,
    list[smee.TensorTopology],
    dict[str, smee.TensorSystem],
    ForceField,
]:
    """
    Create a composite system from multiple system configurations.

    Notes
    -----
    The forcefield will be created from the combined set of molecules across all systems, so it is shared.

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
    """
    # Get all smiles and nmol values.
    all_smiles = [
        comp["smiles"] for system in systems_config for comp in system["components"]
    ]
    all_nmols = [
        comp["nmol"] for system in systems_config for comp in system["components"]
    ]

    # Create the composite system using the shared forcefield and topologies.
    composite_tensor_system, composite_tensor_forcefield, composite_topologies = (
        create_system_from_smiles(
            all_smiles, all_nmols, forcefield_name, charge_assignment_callback
        )
    )

    # Create the force field instance.
    force_field = ForceField(forcefield_name, load_plugins=True)

    # Create individual tensor systems by slicing the shared topologies.
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
