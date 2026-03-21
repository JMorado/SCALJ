"""Tests for scalej.training._predict (predict_energies_forces)."""

import pytest
import torch

from scalej.types import PredictionResult


def _call(force_field, dataset, tensor_systems, **kwargs):
    from scalej.training._predict import predict_energies_forces

    return predict_energies_forces(
        dataset=dataset,
        force_field=force_field,
        tensor_systems=tensor_systems,
        device="cpu",
        **kwargs,
    )


class TestPredict:
    @pytest.fixture(autouse=True)
    def _setup(self, water_system, water_dataset, training_tensor_systems):
        _, self.force_field, _ = water_system
        self.dataset = water_dataset
        self.systems = training_tensor_systems

    def _call(self, **kwargs):
        return _call(self.force_field, self.dataset, self.systems, **kwargs)

    # --- Basic return-type and shape tests ---
    def test_returns_prediction_result(self):
        assert isinstance(self._call(), PredictionResult)

    def test_energy_shapes_match(self):
        result = self._call()
        assert result.energy_ref.numel() == result.energy_pred.numel()

    def test_forces_shapes_match(self):
        result = self._call()
        assert result.forces_ref.shape == result.forces_pred.shape

    def test_weights_energy_sum_to_one(self):
        assert self._call().weights_energy.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_weights_forces_sum_to_one(self):
        assert self._call().weights_forces.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_mask_idxs_length_matches_dataset(self):
        assert len(self._call().mask_idxs) == len(self.dataset)

    def test_energies_are_finite(self):
        assert torch.all(torch.isfinite(self._call().energy_pred))

    def test_forces_are_finite(self):
        assert torch.all(torch.isfinite(self._call().forces_pred))

    # --- Reference modes ---
    @pytest.mark.parametrize("reference", ["none", "mean", "min"])
    def test_reference_mode_runs(self, reference):
        result = self._call(reference=reference)
        assert isinstance(result, PredictionResult)
        assert torch.all(torch.isfinite(result.energy_pred))

    def test_none_reference_pred_not_offset(self):
        r_none = self._call(reference="none")
        r_mean = self._call(reference="mean")
        assert not torch.allclose(r_none.energy_pred, r_mean.energy_pred)

    def test_invalid_reference_raises(self):
        with pytest.raises(NotImplementedError):
            self._call(reference="invalid_ref")

    # --- Energy cutoff filtering ---
    def test_no_cutoff_keeps_all_conformers(self):
        r_full = self._call()
        r_cut = self._call(energy_cutoff=1e9)
        assert r_full.energy_pred.shape == r_cut.energy_pred.shape

    def test_tight_cutoff_reduces_conformers(self):
        r_full = self._call()
        r_cut = self._call(energy_cutoff=0.001)
        assert r_cut.energy_pred.shape[0] <= r_full.energy_pred.shape[0]

    # --- Weighting ---
    def test_uniform_weights_are_equal(self):
        w = self._call(weighting_method="uniform").weights_energy
        assert torch.allclose(w, w[0].expand_as(w), atol=1e-6)

    def test_boltzmann_weights_run(self):
        result = self._call(weighting_method="boltzmann", weighting_temperature=300.0)
        assert isinstance(result, PredictionResult)
        assert result.weights_energy.sum().item() == pytest.approx(1.0, abs=1e-5)
