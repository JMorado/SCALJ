"""Tests for scalej.training._train (train_parameters, train_from_closure)."""

import numpy as np
import pytest
import torch

from scalej.training import train_from_closure, train_parameters
from scalej.types import TrainingResult


def _quadratic_closure(x, compute_gradient, compute_hessian):
    """Minimal closure: L(x) = sum(x^2), dL/dx = 2x."""
    loss = (x**2).sum()
    grad = (2 * x).detach() if compute_gradient else None
    return loss.detach(), grad, None


class TestTrainParameters:
    def test_train_parameters(
        self, water_trainable, water_dataset, training_tensor_systems
    ):
        n_epochs = 5
        result = train_parameters(
            water_trainable,
            water_dataset,
            training_tensor_systems,
            n_epochs=n_epochs,
            conformer_batch_size=4,
            device="cpu",
            verbose=False,
        )
        assert isinstance(result, TrainingResult)
        assert len(result.energy_losses) == n_epochs
        assert len(result.force_losses) == n_epochs
        assert isinstance(result.initial_parameters, torch.Tensor)
        assert isinstance(result.trained_parameters, torch.Tensor)
        assert result.initial_parameters.shape == result.trained_parameters.shape
        assert all(torch.isfinite(torch.tensor(v)) for v in result.energy_losses)
        assert all(torch.isfinite(torch.tensor(v)) for v in result.force_losses)
        initial_param = result.initial_parameters.detach().numpy()
        final_param = result.trained_parameters.detach().numpy()
        assert initial_param == pytest.approx(
            np.array([0.1521, 3.1507, 0.0000, 1.0000])
        )
        print(final_param)
        assert final_param == pytest.approx(
            np.array([0.2022, 3.2009, 0.0000, 1.0000]), rel=1e-3
        )


class TestTrainFromClosure:
    def test_train_from_closure(self, water_trainable):
        n_epochs = 5
        result = train_from_closure(
            water_trainable,
            _quadratic_closure,
            n_epochs=n_epochs,
            device="cpu",
            verbose=False,
        )
        assert isinstance(result, TrainingResult)
        assert len(result.force_losses) == 0
        assert len(result.energy_losses) == 0
        assert len(result.combined_losses) == n_epochs
        assert result.combined_losses[-1] < result.combined_losses[0]
        assert result.initial_parameters.shape == result.trained_parameters.shape
        initial_param = result.initial_parameters.detach().numpy()
        final_param = result.trained_parameters.detach().numpy()
        assert initial_param == pytest.approx(
            np.array([0.1521, 3.1507, 0.0000, 1.0000])
        )
        assert final_param == pytest.approx(
            np.array([0.1025, 3.1007, 0.0000, 0.9500]), rel=1e-3
        )
