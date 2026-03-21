"""Tests for scalej.training._loss (LossConfig, get_losses, to_scalej_closure)."""

import pytest
import torch

from scalej.training import LossConfig, get_losses, to_scalej_closure


class TestLossConfig:
    def test_default_energy_weight(self):
        assert LossConfig().energy_weight == 1.0

    def test_default_force_weight(self):
        assert LossConfig().force_weight == 1.0

    def test_default_reference(self):
        assert LossConfig().reference == "none"

    def test_default_weighting_method(self):
        assert LossConfig().weighting_method == "uniform"

    def test_default_compute_forces(self):
        assert LossConfig().compute_forces is True

    def test_default_energy_cutoff_is_none(self):
        assert LossConfig().energy_cutoff is None

    def test_custom_values_stored(self):
        config = LossConfig(
            energy_weight=2.0,
            force_weight=0.5,
            reference="min",
            weighting_method="boltzmann",
        )
        assert config.energy_weight == 2.0
        assert config.force_weight == 0.5
        assert config.reference == "min"
        assert config.weighting_method == "boltzmann"


class TestGetLosses:
    @pytest.fixture(autouse=True)
    def _params(self, water_trainable):
        self.params = (
            water_trainable.to_values().to("cpu").detach().requires_grad_(True)
        )
        self.trainable = water_trainable

    def test_returns_four_tensors(self, water_dataset, training_tensor_systems):
        result = get_losses(
            self.params,
            self.trainable,
            water_dataset[0],
            training_tensor_systems,
            device="cpu",
        )
        assert len(result) == 4
        assert all(isinstance(t, torch.Tensor) for t in result)

    def test_grad_shape_matches_params(self, water_dataset, training_tensor_systems):
        *_, grad = get_losses(
            self.params,
            self.trainable,
            water_dataset[0],
            training_tensor_systems,
            device="cpu",
        )
        assert grad.shape == self.params.shape

    def test_energy_loss_non_negative(self, water_dataset, training_tensor_systems):
        _, energy_loss, _, _ = get_losses(
            self.params,
            self.trainable,
            water_dataset[0],
            training_tensor_systems,
            device="cpu",
        )
        assert energy_loss.item() >= 0.0

    def test_force_loss_zero_when_disabled(
        self, water_dataset, training_tensor_systems
    ):
        _, _, force_loss, _ = get_losses(
            self.params,
            self.trainable,
            water_dataset[0],
            training_tensor_systems,
            compute_forces=False,
            device="cpu",
        )
        assert force_loss.item() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("reference", ["none", "mean", "min"])
    def test_reference_modes(self, water_dataset, training_tensor_systems, reference):
        total, _, _, grad = get_losses(
            self.params,
            self.trainable,
            water_dataset[0],
            training_tensor_systems,
            reference=reference,
            device="cpu",
        )
        assert total.item() >= 0.0
        assert grad.shape == self.params.shape

    @pytest.mark.parametrize("method", ["uniform", "boltzmann", "mixed"])
    def test_weighting_methods(self, water_dataset, training_tensor_systems, method):
        total, _, _, grad = get_losses(
            self.params,
            self.trainable,
            water_dataset[0],
            training_tensor_systems,
            weighting_method=method,
            device="cpu",
        )
        assert total.item() >= 0.0
        assert grad.shape == self.params.shape

    def test_energy_cutoff_filters_conformers(
        self, water_dataset, training_tensor_systems
    ):
        total, _, _, grad = get_losses(
            self.params,
            self.trainable,
            water_dataset[0],
            training_tensor_systems,
            energy_cutoff=0.5,
            device="cpu",
        )
        assert total.item() >= 0.0
        assert grad.shape == self.params.shape


class TestToScalejClosure:
    def test_returns_callable(
        self, water_trainable, water_dataset, training_tensor_systems
    ):
        closure = to_scalej_closure(
            water_trainable,
            water_dataset,
            training_tensor_systems,
            LossConfig(),
            device="cpu",
        )
        assert callable(closure)

    def test_returns_loss_and_grad(
        self, water_trainable, water_dataset, training_tensor_systems
    ):
        closure = to_scalej_closure(
            water_trainable,
            water_dataset,
            training_tensor_systems,
            LossConfig(),
            device="cpu",
        )
        params = water_trainable.to_values().to("cpu").detach().requires_grad_(True)
        loss, grad, hessian = closure(
            params, compute_gradient=True, compute_hessian=False
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0
        assert grad is not None
        assert grad.shape == params.shape
        assert hessian is None

    def test_no_gradient_when_not_requested(
        self, water_trainable, water_dataset, training_tensor_systems
    ):
        closure = to_scalej_closure(
            water_trainable,
            water_dataset,
            training_tensor_systems,
            LossConfig(),
            device="cpu",
        )
        params = water_trainable.to_values().to("cpu").detach().requires_grad_(True)
        loss, grad, _ = closure(params, compute_gradient=False, compute_hessian=False)
        assert loss.item() >= 0.0
        assert grad is None


class TestConformerWeightsExact:
    """Pin down exact numeric outputs of _compute_conformer_weights.

    All tests use hand-crafted tensors and require **no smee calls**;
    they verify the closed-form weight and variance formulas directly.
    """

    # Deterministic inputs reused across every test method
    ENERGIES = torch.tensor([-5.0, -4.0, -3.0, -2.0])  # 4 conformers
    # Non-trivial forces so variance is non-zero; shape [4, 2, 3]
    FORCES = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [-1.0, 0.5, 0.0]],
            [[0.5, -0.5, 1.0], [0.0, 1.0, -1.0]],
            [[-0.3, 0.8, 0.2], [0.7, -0.2, 0.5]],
            [[0.1, -0.9, 0.4], [-0.6, 0.3, -0.8]],
        ]
    )

    def _weights(self, config):
        from scalej.training._loss import _compute_conformer_weights
        from scalej.training._types import EntryData

        entry_data = EntryData(
            energy_ref=self.ENERGIES,
            forces_ref=self.FORCES,
            coords=None,
            box_vectors=None,
            system=None,
            n_mols=1,
            n_atoms=self.FORCES.shape[1],
        )
        return _compute_conformer_weights(entry_data, config, "cpu")

    def test_uniform_weights_each_equal_one_over_n(self):
        result = self._weights(LossConfig(weighting_method="uniform", reference="none"))
        expected = torch.full((4,), 0.25)
        torch.testing.assert_close(result.weights, expected, atol=1e-6, rtol=0)

    def test_uniform_weights_sum_to_one(self):
        result = self._weights(LossConfig(weighting_method="uniform", reference="none"))
        assert result.weights.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_energy_var_none_reference_matches_torch_var_of_raw_energies(self):
        """reference='none' → energy_ref_0=0, so shifted energies == raw energies."""
        result = self._weights(LossConfig(weighting_method="uniform", reference="none"))
        expected_var = torch.var(self.ENERGIES)
        torch.testing.assert_close(
            result.energy_var, expected_var, atol=1e-6, rtol=1e-5
        )

    def test_energy_var_mean_reference_matches_torch_var_of_shifted_energies(self):
        """reference='mean' → shifted energies = energies - mean(energies)."""
        result = self._weights(LossConfig(weighting_method="uniform", reference="mean"))
        shifted = self.ENERGIES - self.ENERGIES.mean()
        expected_var = torch.var(shifted)
        torch.testing.assert_close(
            result.energy_var, expected_var, atol=1e-6, rtol=1e-5
        )

    def test_forces_var_matches_torch_var_of_all_force_components(self):
        result = self._weights(LossConfig(weighting_method="uniform", reference="none"))
        expected_var = torch.var(self.FORCES)
        torch.testing.assert_close(
            result.forces_var, expected_var, atol=1e-6, rtol=1e-5
        )

    def test_none_reference_energy_ref_0_is_zero(self):
        result = self._weights(LossConfig(weighting_method="uniform", reference="none"))
        assert result.energy_ref_0.item() == pytest.approx(0.0, abs=1e-7)

    def test_mean_reference_energy_ref_0_is_mean_of_energies(self):
        result = self._weights(LossConfig(weighting_method="uniform", reference="mean"))
        # mean([-5, -4, -3, -2]) = -3.5
        assert result.energy_ref_0.item() == pytest.approx(-3.5, abs=1e-5)

    def test_min_reference_energy_ref_0_is_minimum_energy(self):
        result = self._weights(LossConfig(weighting_method="uniform", reference="min"))
        # min([-5, -4, -3, -2]) = -5.0
        assert result.energy_ref_0.item() == pytest.approx(-5.0, abs=1e-5)

    def test_boltzmann_weights_match_exp_formula(self):
        from scalej.training._loss import _compute_kbt

        T = 300.0
        result = self._weights(
            LossConfig(
                weighting_method="boltzmann",
                reference="none",
                weighting_temperature=T,
            )
        )
        kbt = _compute_kbt(T)
        e_rel = self.ENERGIES - self.ENERGIES.min()  # [0, 1, 2, 3]
        raw = torch.exp(-e_rel / kbt)
        expected = raw / raw.sum()
        torch.testing.assert_close(result.weights, expected, atol=1e-6, rtol=1e-5)

    def test_boltzmann_weights_sum_to_one(self):
        result = self._weights(
            LossConfig(
                weighting_method="boltzmann",
                reference="none",
                weighting_temperature=300.0,
            )
        )
        assert result.weights.sum().item() == pytest.approx(1.0, abs=1e-6)

    def test_energy_cutoff_keeps_conformers_below_threshold(self):
        """
        Energies = [-5, -4, -3, -2]; min = -5.
        Relative = [0, 1, 2, 3]; cutoff = 1.5 → only indices 0 and 1 survive.
        """
        result = self._weights(
            LossConfig(weighting_method="uniform", reference="none", energy_cutoff=1.5)
        )
        assert len(result.valid_indices) == 2
        torch.testing.assert_close(
            result.valid_indices, torch.tensor([0, 1]), atol=0, rtol=0
        )

    def test_energy_cutoff_zero_keeps_only_minimum(self):
        result = self._weights(
            LossConfig(weighting_method="uniform", reference="none", energy_cutoff=0.0)
        )
        assert len(result.valid_indices) == 1
        assert result.valid_indices[0].item() == 0  # -5.0 is the minimum


class TestReferenceGradientCorrection:
    """Test the chain-rule correction: grad += alpha * ref_grad.grad."""

    def test_no_correction_when_ref_offset_grad_is_none(self):
        from scalej.training._loss import _apply_reference_gradient_correction

        grad = torch.tensor([1.0, 2.0, 3.0])
        result = _apply_reference_gradient_correction(
            grad, torch.tensor(5.0), ref_offset_grad=None
        )
        torch.testing.assert_close(result, grad)

    def test_correction_with_positive_alpha(self):
        from scalej.training._loss import _apply_reference_gradient_correction
        from scalej.training._types import ReferenceOffsetGradient

        grad = torch.tensor([1.0, 2.0])
        ref = ReferenceOffsetGradient(
            energy_pred_0=torch.tensor(0.0),
            grad=torch.tensor([0.5, 1.0]),
        )
        result = _apply_reference_gradient_correction(grad, torch.tensor(3.0), ref)
        # [1.0 + 3.0*0.5, 2.0 + 3.0*1.0] = [2.5, 5.0]
        torch.testing.assert_close(result, torch.tensor([2.5, 5.0]), atol=1e-6, rtol=0)

    def test_correction_with_negative_alpha(self):
        from scalej.training._loss import _apply_reference_gradient_correction
        from scalej.training._types import ReferenceOffsetGradient

        grad = torch.tensor([1.0, 2.0])
        ref = ReferenceOffsetGradient(
            energy_pred_0=torch.tensor(0.0),
            grad=torch.tensor([0.5, 1.0]),
        )
        result = _apply_reference_gradient_correction(grad, torch.tensor(-0.4), ref)
        # [1.0 + (-0.4)*0.5, 2.0 + (-0.4)*1.0] = [0.8, 1.6]
        torch.testing.assert_close(result, torch.tensor([0.8, 1.6]), atol=1e-6, rtol=0)

    def test_zero_alpha_leaves_grad_unchanged(self):
        from scalej.training._loss import _apply_reference_gradient_correction
        from scalej.training._types import ReferenceOffsetGradient

        grad = torch.tensor([3.0, -1.0, 2.5])
        ref = ReferenceOffsetGradient(
            energy_pred_0=torch.tensor(0.0),
            grad=torch.tensor([100.0, -50.0, 200.0]),
        )
        result = _apply_reference_gradient_correction(grad, torch.tensor(0.0), ref)
        torch.testing.assert_close(result, grad, atol=1e-7, rtol=0)


class TestGradientNumerics:
    """Compare the manually accumulated gradient from get_losses to baselines.

    These tests catch bugs in batch accumulation and the chain-rule correction.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, water_trainable):
        self.params = (
            water_trainable.to_values().to("cpu").detach().requires_grad_(True)
        )
        self.trainable = water_trainable

    @pytest.mark.parametrize("reference", ["none", "mean", "min"])
    def test_gradient_identical_across_batch_sizes(
        self, water_dataset, training_tensor_systems, reference
    ):
        """The accumulated gradient must not change when batch size changes."""
        entry = water_dataset[0]
        shared = {
            "trainable": self.trainable,
            "entry": entry,
            "tensor_systems": training_tensor_systems,
            "reference": reference,
            "compute_forces": False,
            "device": "cpu",
        }
        *_, grad_bs1 = get_losses(self.params, conformer_batch_size=1, **shared)
        *_, grad_bs4 = get_losses(self.params, conformer_batch_size=4, **shared)
        torch.testing.assert_close(grad_bs1, grad_bs4, atol=1e-5, rtol=1e-4)

    def test_gradient_matches_autograd_energy_only(
        self, water_dataset, training_tensor_systems
    ):
        """Manual gradient agrees with single-pass torch.autograd.grad (no forces)."""
        import smee.utils

        from scalej.training._loss import (
            _compute_batch_energies,
            _compute_conformer_weights,
            _prepare_entry_data,
        )

        entry = water_dataset[0]
        config = LossConfig(
            reference="none", weighting_method="uniform", compute_forces=False
        )

        *_, grad_manual = get_losses(
            self.params,
            self.trainable,
            entry,
            training_tensor_systems,
            conformer_batch_size=100,
            reference=config.reference,
            weighting_method=config.weighting_method,
            compute_forces=config.compute_forces,
            device="cpu",
        )

        params_ag = self.params.detach().clone().requires_grad_(True)
        entry_data = _prepare_entry_data(entry, training_tensor_systems, "cpu")
        conf_weights = _compute_conformer_weights(entry_data, config, "cpu")

        force_field = self.trainable.to_force_field(params_ag.abs())
        ff_dtype = force_field.potentials[0].parameters.dtype

        valid = conf_weights.valid_indices
        coords = smee.utils.tensor_like(
            entry_data.coords[valid], force_field.potentials[0].parameters
        )
        boxes = smee.utils.tensor_like(
            entry_data.box_vectors[valid], force_field.potentials[0].parameters
        )
        energy_ref = entry_data.energy_ref[valid].to(ff_dtype)

        energies = (
            _compute_batch_energies(entry_data.system, force_field, coords, boxes)
            / entry_data.n_mols
        )

        w = conf_weights.weights.to(ff_dtype)
        loss_ag = (
            config.energy_weight
            * torch.sum(w * (energies - energy_ref) ** 2)
            / conf_weights.energy_var
        )
        (grad_autograd,) = torch.autograd.grad(loss_ag, params_ag)

        torch.testing.assert_close(grad_manual, grad_autograd, atol=1e-5, rtol=1e-4)

    def test_gradient_matches_autograd_with_forces(
        self, water_dataset, training_tensor_systems
    ):
        """Manual gradient agrees with single-pass torch.autograd.grad (with forces)."""
        import smee.utils

        from scalej.training._loss import (
            _compute_batch_energies,
            _compute_batch_forces,
            _compute_conformer_weights,
            _prepare_entry_data,
        )

        entry = water_dataset[0]
        config = LossConfig(
            reference="none", weighting_method="uniform", compute_forces=True
        )

        *_, grad_manual = get_losses(
            self.params,
            self.trainable,
            entry,
            training_tensor_systems,
            conformer_batch_size=100,
            reference=config.reference,
            weighting_method=config.weighting_method,
            compute_forces=config.compute_forces,
            device="cpu",
        )

        params_ag = self.params.detach().clone().requires_grad_(True)
        entry_data = _prepare_entry_data(entry, training_tensor_systems, "cpu")
        conf_weights = _compute_conformer_weights(entry_data, config, "cpu")

        force_field = self.trainable.to_force_field(params_ag.abs())
        ff_dtype = force_field.potentials[0].parameters.dtype

        valid = conf_weights.valid_indices
        coords = smee.utils.tensor_like(
            entry_data.coords[valid], force_field.potentials[0].parameters
        ).requires_grad_(True)
        boxes = smee.utils.tensor_like(
            entry_data.box_vectors[valid], force_field.potentials[0].parameters
        )
        energy_ref = entry_data.energy_ref[valid].to(ff_dtype)
        forces_ref = entry_data.forces_ref[valid].to(ff_dtype)

        energies = (
            _compute_batch_energies(entry_data.system, force_field, coords, boxes)
            / entry_data.n_mols
        )
        forces_pred = _compute_batch_forces(energies, coords)

        w_e = conf_weights.weights.to(ff_dtype)
        w_f = conf_weights.weights_forces.to(ff_dtype)

        loss_energy = (
            config.energy_weight
            * torch.sum(w_e * (energies - energy_ref) ** 2)
            / conf_weights.energy_var
        )
        loss_forces = (
            config.force_weight
            * torch.sum(w_f * (forces_pred - forces_ref) ** 2)
            / conf_weights.forces_var
        )
        (grad_autograd,) = torch.autograd.grad(loss_energy + loss_forces, params_ag)

        torch.testing.assert_close(grad_manual, grad_autograd, atol=1e-5, rtol=1e-4)

    def test_dloss_d_pred0_matches_finite_difference(
        self, water_dataset, training_tensor_systems
    ):
        """Verify -2 * w_E * sum(w_i * diff_i) / var formula via finite differences."""
        import smee.utils

        from scalej.training._loss import (
            _compute_batch_energies,
            _compute_conformer_weights,
            _prepare_entry_data,
        )

        entry = water_dataset[0]
        config = LossConfig(
            reference="none", weighting_method="uniform", compute_forces=False
        )
        entry_data = _prepare_entry_data(entry, training_tensor_systems, "cpu")
        conf_weights = _compute_conformer_weights(entry_data, config, "cpu")

        force_field = self.trainable.to_force_field(self.params.abs().detach())
        ff_dtype = force_field.potentials[0].parameters.dtype

        valid = conf_weights.valid_indices
        coords = smee.utils.tensor_like(
            entry_data.coords[valid], force_field.potentials[0].parameters
        )
        boxes = smee.utils.tensor_like(
            entry_data.box_vectors[valid], force_field.potentials[0].parameters
        )
        energy_ref = entry_data.energy_ref[valid].to(ff_dtype)

        with torch.no_grad():
            energies = (
                _compute_batch_energies(entry_data.system, force_field, coords, boxes)
                / entry_data.n_mols
            )

        w = conf_weights.weights.to(ff_dtype)

        # L(E0) = w_E * sum_i(w_i * (E_i - E0 - E_ref_i)^2) / var
        def loss_at(e0):
            diff = energies - e0 - energy_ref
            return (
                config.energy_weight * torch.sum(w * diff**2) / conf_weights.energy_var
            )

        # Analytic formula
        diff0 = energies - energy_ref  # at e0 = 0
        dloss_analytic = (
            -2.0 * config.energy_weight * torch.sum(w * diff0) / conf_weights.energy_var
        )

        # Centred finite difference
        eps = torch.tensor(1e-4, dtype=ff_dtype)
        dloss_fd = (loss_at(eps) - loss_at(-eps)) / (2.0 * eps)

        torch.testing.assert_close(dloss_analytic, dloss_fd, atol=1e-4, rtol=1e-3)
