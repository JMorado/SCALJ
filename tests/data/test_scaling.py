"""Tests for scaling functions."""

import numpy as np
import pytest

from scalej.data._scaling import (
    compute_molecule_coms,
    generate_scale_factors,
    get_box_center,
    scale_molecule_positions,
)
from ..conftest import BOX, N_ATOMS_PER_MOL, WATER_DIMER_COORDS


class TestGenerateScaleFactors:
    @pytest.fixture
    def scale_factors(self):
        close_range = (0.75, 0.9, 5)
        equilibrium_range = (0.9, 1.1, 15)
        long_range = (1.1, 2.0, 12)
        return generate_scale_factors(
            close_range=close_range,
            equilibrium_range=equilibrium_range,
            long_range=long_range,
        )

    def test_default_output_type(self, scale_factors):
        assert isinstance(scale_factors, np.ndarray)
        assert len(scale_factors) == 30
        assert scale_factors == pytest.approx(
            np.concatenate(
                (
                    np.linspace(0.75, 0.9, 5),
                    np.linspace(0.9, 1.1, 15)[1:],
                    np.linspace(1.1, 2.0, 12)[1:],
                )
            )
        )


class TestComputeMoleculeComs:
    def test_single_frame(self, water_dimer_coords):
        coms = compute_molecule_coms(water_dimer_coords, N_ATOMS_PER_MOL)
        assert coms.shape == (2, 3)
        expected_com1 = water_dimer_coords[:3].mean(axis=0)
        expected_com2 = water_dimer_coords[3:].mean(axis=0)
        assert coms[0] == pytest.approx(expected_com1)
        assert coms[1] == pytest.approx(expected_com2)

    def test_multiframe(self, water_dimer_coords_multiframe):
        coms = compute_molecule_coms(water_dimer_coords_multiframe, N_ATOMS_PER_MOL)
        assert coms.shape == (2, 2, 3)
        expected_com1_frame1 = water_dimer_coords_multiframe[0, :3].mean(axis=0)
        expected_com2_frame1 = water_dimer_coords_multiframe[0, 3:].mean(axis=0)
        expected_com1_frame2 = water_dimer_coords_multiframe[1, :3].mean(axis=0)
        expected_com2_frame2 = water_dimer_coords_multiframe[1, 3:].mean(axis=0)
        assert coms[0, 0] == pytest.approx(expected_com1_frame1)
        assert coms[0, 1] == pytest.approx(expected_com2_frame1)
        assert coms[1, 0] == pytest.approx(expected_com1_frame2)
        assert coms[1, 1] == pytest.approx(expected_com2_frame2)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            compute_molecule_coms(np.zeros((2, 6, 3, 1)), N_ATOMS_PER_MOL)


class TestGetBoxCenter:
    def test_shape_single_frame(self, water_dimer_box):
        center = get_box_center(water_dimer_box)
        assert center.shape == (3,)
        assert center == pytest.approx([5.0, 5.0, 5.0])

    def test_shape_multiframe(self, water_dimer_box_multiframe):
        center = get_box_center(water_dimer_box_multiframe)
        assert center.shape == (2, 3)
        assert center[0] == pytest.approx([5.0, 5.0, 5.0])
        assert center[1] == pytest.approx([5.05, 5.05, 5.05])

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            get_box_center(np.zeros((2, 3, 3, 1)))


class TestScaleMoleculePositions:
    def test_single_frame(self, water_dimer_coords, water_dimer_box):
        scaled_coords, scaled_box = scale_molecule_positions(
            water_dimer_coords, water_dimer_box, N_ATOMS_PER_MOL, 0.9
        )
        assert isinstance(scaled_coords, np.ndarray)
        assert isinstance(scaled_box, np.ndarray)
        assert scaled_coords.shape == water_dimer_coords.shape
        assert scaled_box.shape == water_dimer_box.shape
        assert scaled_box == pytest.approx(water_dimer_box * 0.9)

    def test_multiframe(
        self, water_dimer_coords_multiframe, water_dimer_box_multiframe
    ):
        scaled_coords, scaled_box = scale_molecule_positions(
            water_dimer_coords_multiframe,
            water_dimer_box_multiframe,
            N_ATOMS_PER_MOL,
            0.9,
        )
        assert scaled_coords.shape == water_dimer_coords_multiframe.shape
        assert scaled_box.shape == water_dimer_box_multiframe.shape

    def test_identity_scale(self, water_dimer_coords, water_dimer_box):
        scaled_coords, scaled_box = scale_molecule_positions(
            water_dimer_coords, water_dimer_box, N_ATOMS_PER_MOL, 1.0
        )
        assert scaled_coords == pytest.approx(water_dimer_coords)
        assert scaled_box == pytest.approx(water_dimer_box)

    def test_internal_geometry_preserved(self, water_dimer_coords, water_dimer_box):
        """Internal bond geometry within each molecule must not change."""
        scaled_coords, _ = scale_molecule_positions(
            water_dimer_coords, water_dimer_box, N_ATOMS_PER_MOL, 0.8
        )
        for mol_idx in range(2):
            start = mol_idx * N_ATOMS_PER_MOL
            end = start + N_ATOMS_PER_MOL
            original_mol = water_dimer_coords[start:end]
            scaled_mol = scaled_coords[start:end]
            original_internal = original_mol - original_mol.mean(axis=0)
            scaled_internal = scaled_mol - scaled_mol.mean(axis=0)
            assert scaled_internal == pytest.approx(original_internal)

    def test_scaling_direction(self, water_dimer_coords, water_dimer_box):
        box_center = get_box_center(water_dimer_box)
        original_coms = compute_molecule_coms(water_dimer_coords, N_ATOMS_PER_MOL)

        compressed_coords, _ = scale_molecule_positions(
            water_dimer_coords, water_dimer_box, N_ATOMS_PER_MOL, 0.5
        )
        expanded_coords, _ = scale_molecule_positions(
            water_dimer_coords, water_dimer_box, N_ATOMS_PER_MOL, 2.0
        )
        compressed_coms = compute_molecule_coms(compressed_coords, N_ATOMS_PER_MOL)
        expanded_coms = compute_molecule_coms(expanded_coords, N_ATOMS_PER_MOL)

        for i in range(2):
            original_dist = np.linalg.norm(original_coms[i] - box_center)
            assert np.linalg.norm(compressed_coms[i] - box_center) < original_dist
            assert np.linalg.norm(expanded_coms[i] - box_center) > original_dist
