"""Fitting module for SCALJ."""

from .training import create_trainable, predict, train_parameters

__all__ = ["create_trainable", "train_parameters", "predict"]
