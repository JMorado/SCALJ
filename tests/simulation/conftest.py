"""Shared fixtures for simulation tests."""

import openmm.unit
import pytest
import smee.mm

from scalej.simulation._mlp import setup_mlp_simulation


@pytest.fixture(scope="module")
def initial_coords_box(water_system):
    """Initial coordinates and box vectors (OpenMM Quantity) for the water system."""
    tensor_system, tensor_forcefield, _ = water_system
    coords, box = smee.mm.generate_system_coords(tensor_system, tensor_forcefield)
    return coords, box


@pytest.fixture(scope="module")
def initial_coords_box_angstrom(initial_coords_box):
    """Initial coords and box stripped to plain numpy arrays in Angstrom."""
    coords_q, box_q = initial_coords_box
    return (
        coords_q.value_in_unit(openmm.unit.angstrom),
        box_q.value_in_unit(openmm.unit.angstrom),
    )


@pytest.fixture(scope="module")
def ani2x_simulation(water_system):
    """ANI-2x simulation for the water system (CPU)."""
    tensor_system, _, _ = water_system
    return setup_mlp_simulation(
        tensor_system, "ani2x", mlp_device="cpu", platform="CPU"
    )
