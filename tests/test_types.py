"""Tests for scalej.types dataclasses."""

import pytest
import torch

from scalej.types import (
    BenchmarkResult,
    EnergyForceResult,
    EvaluationMetrics,
    PredictionResult,
    ScalingResult,
    TrajectoryFrames,
    TrainingResult,
)


class TestScalingResult:
    def test_fields_stored(self):
        coords = [[0.0] * 9]
        box_vectors = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]
        scale_factors = [0.9, 1.0, 1.1]
        r = ScalingResult(coords=coords, box_vectors=box_vectors, scale_factors=scale_factors)
        assert r.coords is coords
        assert r.box_vectors is box_vectors
        assert r.scale_factors == pytest.approx([0.9, 1.0, 1.1])


class TestEnergyForceResult:
    def test_fields_stored(self):
        import numpy as np
        energies = np.array([-1.0, -2.0])
        forces = np.zeros((2, 3, 3))
        r = EnergyForceResult(energies=energies, forces=forces)
        assert list(r.energies) == pytest.approx([-1.0, -2.0])
        assert r.forces.shape == (2, 3, 3)
        assert r.forces.sum() == pytest.approx(0.0)


class TestPredictionResult:
    def _make(self, n=3):
        return PredictionResult(
            energy_ref=torch.zeros(n),
            energy_pred=torch.ones(n),
            forces_ref=torch.zeros(n, 3),
            forces_pred=torch.ones(n, 3),
            weights_energy=torch.full((n,), 1.0 / n),
            weights_forces=torch.full((n, 1), 1.0 / n),
            mask_idxs=[torch.arange(n)],
        )

    def test_fields_stored(self):
        r = self._make()
        assert r.energy_ref.shape == (3,)
        assert r.energy_pred.shape == (3,)
        assert r.forces_ref.shape == (3, 3)
        assert r.forces_pred.shape == (3, 3)
        assert len(r.mask_idxs) == 1

    def test_weights_sum_to_one(self):
        r = self._make(4)
        assert r.weights_energy.sum().item() == pytest.approx(1.0)


class TestTrainingResult:
    def test_fields_stored(self):
        p0 = torch.tensor([1.0, 2.0])
        p1 = torch.tensor([1.1, 2.1])
        r = TrainingResult(
            initial_parameters=p0,
            trained_parameters=p1,
            energy_losses=[1.0, 0.5],
            force_losses=[0.2, 0.1],
        )
        torch.testing.assert_close(r.initial_parameters, p0)
        torch.testing.assert_close(r.trained_parameters, p1)
        assert r.energy_losses == [1.0, 0.5]
        assert r.force_losses == [0.2, 0.1]

    def test_combined_losses_default_is_none(self):
        r = TrainingResult(
            initial_parameters=torch.zeros(2),
            trained_parameters=torch.zeros(2),
            energy_losses=[],
            force_losses=[],
        )
        assert r.combined_losses is None

    def test_combined_losses_can_be_set(self):
        r = TrainingResult(
            initial_parameters=torch.zeros(2),
            trained_parameters=torch.zeros(2),
            energy_losses=[],
            force_losses=[],
            combined_losses=[0.5, 0.3],
        )
        assert r.combined_losses == [0.5, 0.3]


class TestEvaluationMetrics:
    def _make(self):
        return EvaluationMetrics(
            energy_mae=0.1,
            energy_rmse=0.2,
            energy_r2=0.9,
            forces_mae=0.3,
            forces_rmse=0.4,
            forces_r2=0.8,
        )

    def test_fields_stored(self):
        m = self._make()
        assert m.energy_mae == pytest.approx(0.1)
        assert m.energy_rmse == pytest.approx(0.2)
        assert m.energy_r2 == pytest.approx(0.9)
        assert m.forces_mae == pytest.approx(0.3)
        assert m.forces_rmse == pytest.approx(0.4)
        assert m.forces_r2 == pytest.approx(0.8)

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        assert set(d.keys()) == {"energy", "forces"}
        assert set(d["energy"].keys()) == {"mae", "rmse", "r2"}
        assert set(d["forces"].keys()) == {"mae", "rmse", "r2"}

    def test_to_dict_values(self):
        d = self._make().to_dict()
        assert d["energy"]["mae"] == pytest.approx(0.1)
        assert d["energy"]["rmse"] == pytest.approx(0.2)
        assert d["energy"]["r2"] == pytest.approx(0.9)
        assert d["forces"]["mae"] == pytest.approx(0.3)
        assert d["forces"]["rmse"] == pytest.approx(0.4)
        assert d["forces"]["r2"] == pytest.approx(0.8)


class TestBenchmarkResult:
    def test_all_none_by_default(self):
        r = BenchmarkResult()
        assert r.density_ref is None
        assert r.density_pred is None
        assert r.density_std is None
        assert r.hvap_ref is None
        assert r.hvap_pred is None
        assert r.hvap_std is None

    def test_partial_fields(self):
        r = BenchmarkResult(density_ref=0.997, density_pred=1.001)
        assert r.density_ref == pytest.approx(0.997)
        assert r.density_pred == pytest.approx(1.001)
        assert r.density_std is None

    def test_all_fields(self):
        r = BenchmarkResult(
            density_ref=0.997,
            density_pred=1.001,
            density_std=0.002,
            hvap_ref=10.5,
            hvap_pred=10.3,
            hvap_std=0.1,
        )
        assert r.hvap_ref == pytest.approx(10.5)
        assert r.hvap_std == pytest.approx(0.1)


class TestTrajectoryFrames:
    def test_fields_stored(self):
        import numpy as np
        coords = np.zeros((5, 6, 3))
        box_vectors = np.stack([np.eye(3)] * 5)
        r = TrajectoryFrames(coords=coords, box_vectors=box_vectors, n_frames=5)
        assert r.n_frames == 5
        assert r.coords.shape == (5, 6, 3)
        assert r.coords.sum() == pytest.approx(0.0)
        assert r.box_vectors.shape == (5, 3, 3)
        assert r.box_vectors[0].diagonal().tolist() == pytest.approx([1.0, 1.0, 1.0])
