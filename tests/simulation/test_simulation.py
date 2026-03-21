"""Tests for MD simulation functions."""

import unittest.mock

import numpy as np
import openmm.unit
import pytest

from scalej.simulation._simulation import (
    load_trajectory_frames_smee,
    run_simulation_omm,
    run_simulation_smee,
)
from scalej.types import TrajectoryFrames


class TestRunSimulationOmm:
    def test_returns_arrays(self, ani2x_simulation, initial_coords_box):
        """run_simulation_omm returns coordinate and box-vector arrays."""
        import openmm.unit

        coords_q, box_q = initial_coords_box
        coords = coords_q.value_in_unit(openmm.unit.angstrom)
        box = box_q.value_in_unit(openmm.unit.angstrom)

        final_coords, final_box = run_simulation_omm(
            ani2x_simulation,
            coords * openmm.unit.angstrom,
            box * openmm.unit.angstrom,
            n_steps=10,
        )
        assert hasattr(final_coords, "value_in_unit")
        assert hasattr(final_box, "value_in_unit")
        coords_np = final_coords.value_in_unit(openmm.unit.angstrom)
        box_np = final_box.value_in_unit(openmm.unit.angstrom)
        assert coords_np.shape == coords.shape
        assert box_np.shape == (3, 3)


class TestGenerateInitialCoords:
    def test_shapes_and_types(self, initial_coords_box, water_system):
        coords, box = initial_coords_box
        assert isinstance(coords, openmm.unit.Quantity)
        assert isinstance(box, openmm.unit.Quantity)
        tensor_system, _, _ = water_system
        n_atoms = sum(
            t.n_atoms * n
            for t, n in zip(
                tensor_system.topologies, tensor_system.n_copies, strict=True
            )
        )
        assert coords.value_in_unit(openmm.unit.angstrom).shape == (n_atoms, 3)
        assert box.value_in_unit(openmm.unit.angstrom).shape == (3, 3)


class TestLoadTrajectoryFrames:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            load_trajectory_frames_smee(tmp_path / "missing.dcd")

    def test_too_many_frames_raises(self, tmp_path):
        """Requesting more frames than exist must raise ValueError."""
        import openmm.unit
        import smee.mm

        # We only test the error path by patching smee.mm._reporters.unpack_frames
        # with a mocked single-frame trajectory. Rather than running real MD, we
        # exercise the ValueError branch directly via monkeypatching.
        import torch

        fake_coord = torch.zeros(6, 3)
        fake_box = torch.eye(3) * 10.0

        original_unpack = smee.mm._reporters.unpack_frames

        def _fake_unpack(f):
            yield fake_coord, fake_box, None, None

        smee.mm._reporters.unpack_frames = _fake_unpack
        dummy_path = tmp_path / "dummy.dcd"
        dummy_path.write_bytes(b"")
        try:
            with pytest.raises(ValueError, match="frames"):
                load_trajectory_frames_smee(dummy_path, n_frames=5)
        finally:
            smee.mm._reporters.unpack_frames = original_unpack

    @pytest.mark.parametrize(
        "n_frames, from_end, expected_coords_shape, expected_box_shape",
        [
            (1, True, (6, 3), (3, 3)),
            (3, True, (3, 6, 3), (3, 3, 3)),
            (2, False, (2, 6, 3), (2, 3, 3)),
        ],
    )
    def test_frames_shape(
        self, tmp_path, n_frames, from_end, expected_coords_shape, expected_box_shape
    ):
        """Frames are stacked along axis 0 for n>1; no batch dim for n=1.
        Both from_end=True and from_end=False selection paths are covered.
        """
        import smee.mm
        import torch

        total_frames = 4
        fake_coords = [torch.full((6, 3), float(i)) for i in range(total_frames)]
        fake_box = torch.eye(3) * 10.0
        original_unpack = smee.mm._reporters.unpack_frames

        def _fake_unpack(f):
            for c in fake_coords:
                yield c, fake_box, None, None

        smee.mm._reporters.unpack_frames = _fake_unpack
        dummy_path = tmp_path / "dummy.dcd"
        dummy_path.write_bytes(b"")
        try:
            frames = load_trajectory_frames_smee(
                dummy_path, n_frames=n_frames, from_end=from_end
            )
        finally:
            smee.mm._reporters.unpack_frames = original_unpack

        assert isinstance(frames, TrajectoryFrames)
        assert frames.n_frames == n_frames
        assert frames.coords.shape == expected_coords_shape
        assert frames.box_vectors.shape == expected_box_shape


class TestRunSimulationSmee:
    """Tests for run_simulation_smee, using mocked smee.mm internals."""

    def _make_mocks(self, tmp_path, tensor_system, tensor_forcefield):
        """Return (mock_coords, mock_box) Quantities and the patch context."""
        import openmm
        import openmm.unit

        n_atoms = sum(
            t.n_atoms * n
            for t, n in zip(
                tensor_system.topologies, tensor_system.n_copies, strict=True
            )
        )
        mock_coords = np.zeros((n_atoms, 3)) * openmm.unit.nanometer
        mock_box = np.eye(3) * 2.0 * openmm.unit.nanometer
        return mock_coords, mock_box

    @pytest.mark.parametrize("save_pdb", [False, True])
    def test_returns_coords_and_box(self, water_system, tmp_path, save_pdb):
        """run_simulation_smee delegates to smee.mm and returns coords/box."""
        import openmm.app
        import smee.mm

        tensor_system, tensor_forcefield, _ = water_system
        mock_coords, mock_box = self._make_mocks(
            tmp_path, tensor_system, tensor_forcefield
        )

        fake_reporter = unittest.mock.MagicMock()
        fake_ctx = unittest.mock.MagicMock()
        fake_ctx.__enter__ = unittest.mock.Mock(return_value=fake_reporter)
        fake_ctx.__exit__ = unittest.mock.Mock(return_value=False)

        with (
            unittest.mock.patch.object(
                smee.mm, "generate_system_coords", return_value=(mock_coords, mock_box)
            ),
            unittest.mock.patch.object(
                smee.mm, "tensor_reporter", return_value=fake_ctx
            ),
            unittest.mock.patch.object(smee.mm, "simulate"),
            unittest.mock.patch.object(
                openmm.app, "PDBReporter", return_value=unittest.mock.MagicMock()
            ),
        ):
            result_coords, result_box = run_simulation_smee(
                tensor_system,
                tensor_forcefield,
                tmp_path / "traj.dcd",
                n_equilibration_nvt_steps=0,
                n_equilibration_npt_steps=0,
                n_production_steps=0,
                save_pdb=save_pdb,
            )

        assert result_coords is mock_coords
        assert result_box is mock_box
