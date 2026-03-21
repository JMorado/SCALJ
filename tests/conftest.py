"""Shared fixtures and constants for data tests."""

import numpy as np
import pytest

# fmt: off
WATER_DIMER_COORDS = np.array(
    [
        [0.000000, 0.000000,  0.000000],  # O1
        [0.758602, 0.000000,  0.504284],  # H1a
        [0.260455, 0.000000, -0.872893],  # H1b
        [3.000000, 0.500000,  0.000000],  # O2
        [3.758602, 0.500000,  0.504284],  # H2a
        [3.260455, 0.500000, -0.872893],  # H2b
    ]
)
# fmt: on

BOX = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])

N_ATOMS_PER_MOL = 3


@pytest.fixture()
def water_dimer_coords():
    return WATER_DIMER_COORDS.copy()


@pytest.fixture()
def water_dimer_box():
    return BOX.copy()


@pytest.fixture()
def water_dimer_coords_multiframe():
    """Two frames of the water dimer."""
    frame2 = WATER_DIMER_COORDS + np.array([0.00, 0.00, 3.00])
    return np.stack([WATER_DIMER_COORDS, frame2], axis=0)


@pytest.fixture()
def water_dimer_box_multiframe():
    """Two box vectors for a two-frame trajectory."""
    return np.stack([BOX, BOX * 1.01], axis=0)
