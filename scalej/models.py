"""Data models for SCALeJ API."""

import dataclasses

import numpy as np
import torch


@dataclasses.dataclass
class ScalingResult:
    """Result from volume scaling computation.

    Attributes
    ----------
    coords : list[np.ndarray]
        List of scaled coordinate arrays, each with shape (n_atoms, 3).
    box_vectors : list[np.ndarray]
        List of scaled box vector arrays, each with shape (3, 3).
    scale_factors : np.ndarray
        Array of scale factors used.
    """

    coords: list[np.ndarray]
    box_vectors: list[np.ndarray]
    scale_factors: np.ndarray


@dataclasses.dataclass
class EnergyForceResult:
    """Result from MLP energy/force computation.

    Attributes
    ----------
    energies : np.ndarray
        Energies in kcal/mol with shape (n_configurations,).
    forces : np.ndarray
        Forces in kcal/mol/Å with shape (n_configurations, n_atoms, 3).
    """

    energies: np.ndarray
    forces: np.ndarray


@dataclasses.dataclass
class PredictionResult:
    """Result from energy/force prediction with force field.

    Attributes
    ----------
    energy_ref : torch.Tensor
        Reference energies [kcal/mol].
    energy_pred : torch.Tensor
        Predicted energies [kcal/mol].
    forces_ref : torch.Tensor
        Reference forces [kcal/mol/Å].
    forces_pred : torch.Tensor
        Predicted forces [kcal/mol/Å].
    weights_energy : torch.Tensor
        Energy weights for loss computation.
    weights_forces : torch.Tensor
        Force weights for loss computation.
    mask_idxs : list[torch.Tensor]
        Indices of conformers kept after filtering.
    """

    energy_ref: torch.Tensor
    energy_pred: torch.Tensor
    forces_ref: torch.Tensor
    forces_pred: torch.Tensor
    weights_energy: torch.Tensor
    weights_forces: torch.Tensor
    mask_idxs: list[torch.Tensor]


@dataclasses.dataclass
class TrainingResult:
    """Result from parameter training optimization.

    Attributes
    ----------
    initial_parameters : torch.Tensor
        Parameters before training.
    trained_parameters : torch.Tensor
        Optimized parameters after training.
    energy_losses : list[float]
        Energy loss history per epoch.
    force_losses : list[float]
        Force loss history per epoch.
    """

    initial_parameters: torch.Tensor
    trained_parameters: torch.Tensor
    energy_losses: list[float]
    force_losses: list[float]


@dataclasses.dataclass
class EvaluationMetrics:
    """Evaluation metrics for energy and force predictions.

    Attributes
    ----------
    energy_mae : float
        Mean absolute error for energies [kcal/mol].
    energy_rmse : float
        Root mean square error for energies [kcal/mol].
    energy_r2 : float
        R² score for energies.
    forces_mae : float
        Mean absolute error for forces [kcal/mol/Å].
    forces_rmse : float
        Root mean square error for forces [kcal/mol/Å].
    forces_r2 : float
        R² score for forces.
    """

    energy_mae: float
    energy_rmse: float
    energy_r2: float
    forces_mae: float
    forces_rmse: float
    forces_r2: float

    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return {
            "energy": {
                "mae": self.energy_mae,
                "rmse": self.energy_rmse,
                "r2": self.energy_r2,
            },
            "forces": {
                "mae": self.forces_mae,
                "rmse": self.forces_rmse,
                "r2": self.forces_r2,
            },
        }


@dataclasses.dataclass
class BenchmarkResult:
    """Result from thermodynamic benchmark.

    Attributes
    ----------
    density_ref : float | None
        Reference density [g/mL].
    density_pred : float | None
        Predicted density [g/mL].
    density_std : float | None
        Predicted density uncertainty [g/mL].
    hvap_ref : float | None
        Reference heat of vaporization [kcal/mol].
    hvap_pred : float | None
        Predicted heat of vaporization [kcal/mol].
    hvap_std : float | None
        Predicted Hvap uncertainty [kcal/mol].
    """

    density_ref: float | None = None
    density_pred: float | None = None
    density_std: float | None = None
    hvap_ref: float | None = None
    hvap_pred: float | None = None
    hvap_std: float | None = None


@dataclasses.dataclass
class LossResult:
    """Loss computation result.

    Attributes
    ----------
    total_loss : torch.Tensor
        Combined weighted loss.
    energy_loss : torch.Tensor
        Energy component of loss.
    force_loss : torch.Tensor
        Force component of loss.
    """

    total_loss: torch.Tensor
    energy_loss: torch.Tensor
    force_loss: torch.Tensor


@dataclasses.dataclass
class TrajectoryFrames:
    """Container for trajectory frame data.

    Attributes
    ----------
    coords : np.ndarray
        Coordinates with shape (n_frames, n_atoms, 3) or (n_atoms, 3).
    box_vectors : np.ndarray
        Box vectors with shape (n_frames, 3, 3) or (3, 3).
    n_frames : int
        Number of frames loaded.
    """

    coords: np.ndarray
    box_vectors: np.ndarray
    n_frames: int
