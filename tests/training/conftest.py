"""Shared fixtures for training tests."""

import numpy as np
import pytest

# 2 water molecules × 3 atoms per molecule
N_ATOMS = 6
N_CONFORMERS = 4

_RNG = np.random.default_rng(42)

# Near-equilibrum water dimer geometry (angstroms)
_BASE_COORDS = np.array(
    [
        [0.000, 0.000, 0.000],
        [0.960, 0.000, 0.000],
        [-0.240, 0.926, 0.000],
        [5.000, 5.000, 5.000],
        [5.960, 5.000, 5.000],
        [4.760, 5.926, 5.000],
    ]
)

# Slightly perturbed conformers so variance is non-zero
TRAINING_COORDS = _BASE_COORDS + _RNG.standard_normal((N_CONFORMERS, N_ATOMS, 3)) * 0.1
# Box must be > 2 × vdW cutoff (9 Å) → use 25 Å
TRAINING_BOX = np.stack([np.eye(3) * 25.0] * N_CONFORMERS)
# Varying energies so torch.var() is non-zero in conformer weighting
TRAINING_ENERGIES = np.array([-100.0, -99.0, -98.0, -97.0])
TRAINING_FORCES = _RNG.standard_normal((N_CONFORMERS, N_ATOMS, 3)) * 0.5


@pytest.fixture(scope="module")
def water_dataset():
    """Small HuggingFace dataset for a 2-water-molecule system (4 conformers)."""
    from scalej.data import create_dataset, create_dataset_entry

    entry = create_dataset_entry(
        id="water",
        smiles="O.O",
        coords_list=list(TRAINING_COORDS),
        box_vectors_list=list(TRAINING_BOX),
        energies=TRAINING_ENERGIES,
        forces=list(TRAINING_FORCES),
    )
    return create_dataset([entry])


@pytest.fixture(scope="module")
def training_tensor_systems(water_system):
    """Tensor systems dict keyed by dataset entry id."""
    tensor_system, _, _ = water_system
    return {"water": tensor_system}


@pytest.fixture
def water_trainable(water_system):
    """Fresh trainable for each test (function-scoped to avoid parameter bleed)."""
    from scalej.training import create_trainable

    _, tensor_forcefield, _ = water_system
    return create_trainable(tensor_forcefield, parameters_cols=["epsilon", "sigma"])
