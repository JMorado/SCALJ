"""Training module — public API re-exports."""

from ._ddp import train_parameters_ddp
from ._loss import get_losses
from ._predict import predict_energies_forces
from ._train import train_parameters
from ._trainable import create_trainable
from ._types import (
    BatchResult,
    ConformerWeights,
    EntryData,
    LossConfig,
    ReferenceMode,
    ReferenceOffsetGradient,
    WeightingMethod,
)

__all__ = [
    # Public functions
    "create_trainable",
    "get_losses",
    "predict_energies_forces",
    "train_parameters",
    "train_parameters_ddp",
    # Types and dataclasses
    "BatchResult",
    "ConformerWeights",
    "EntryData",
    "LossConfig",
    "ReferenceMode",
    "ReferenceOffsetGradient",
    "WeightingMethod",
]
