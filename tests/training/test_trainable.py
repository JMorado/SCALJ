"""Tests for scalej.training._trainable (create_trainable)."""

import descent.train
import pytest
import smee
import torch


class TestCreateTrainable:
    def test_trainable_creation(self, water_trainable):
        assert water_trainable.to_values().numel() > 0
        assert isinstance(water_trainable, descent.train.Trainable)
        values = water_trainable.to_values()
        assert isinstance(values, torch.Tensor)
        assert values.ndim == 1
        ff = water_trainable.to_force_field(values)
        assert isinstance(ff, smee.TensorForceField)

    def test_custom_parameter_columns(self, water_system):
        from scalej.training import create_trainable

        _, tensor_forcefield, _ = water_system
        trainable_eps_only = create_trainable(
            tensor_forcefield,
            parameters_cols=["epsilon"],
        )
        trainable_both = create_trainable(
            tensor_forcefield,
            parameters_cols=["epsilon", "sigma"],
        )
        n_eps = trainable_eps_only.to_values().numel()
        n_both = trainable_both.to_values().numel()
        assert n_both == 2 * n_eps
