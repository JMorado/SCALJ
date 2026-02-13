"""Evaluation functions for force field assessment."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .models import BenchmarkResult, EvaluationMetrics, PredictionResult

if TYPE_CHECKING:
    import datasets
    import smee


def compute_metrics(
    prediction: PredictionResult,
) -> EvaluationMetrics:
    """Compute error metrics from prediction results.

    Calculates MAE, RMSE, and R² for both energies and forces.

    Parameters
    ----------
    prediction : PredictionResult
        Prediction results containing reference and predicted values.

    Returns
    -------
    EvaluationMetrics
        Metrics for energy and force predictions.

    Examples
    --------
    >>> prediction = predict_energies_forces(dataset, force_field, systems)
    >>> metrics = compute_metrics(prediction)
    >>> print(f"Energy MAE: {metrics.energy_mae:.4f} kcal/mol")
    """
    energy_ref = prediction.energy_ref.detach().cpu().numpy()
    energy_pred = prediction.energy_pred.detach().cpu().numpy()
    forces_ref = prediction.forces_ref.flatten().detach().cpu().numpy()
    forces_pred = prediction.forces_pred.flatten().detach().cpu().numpy()

    # Energy metrics
    energy_mae = float(np.mean(np.abs(energy_pred - energy_ref)))
    energy_rmse = float(np.sqrt(np.mean((energy_pred - energy_ref) ** 2)))
    energy_r2 = float(
        1
        - np.sum((energy_ref - energy_pred) ** 2)
        / np.sum((energy_ref - np.mean(energy_ref)) ** 2)
    )

    # Force metrics
    forces_mae = float(np.mean(np.abs(forces_pred - forces_ref)))
    forces_rmse = float(np.sqrt(np.mean((forces_pred - forces_ref) ** 2)))
    forces_r2 = float(
        1
        - np.sum((forces_ref - forces_pred) ** 2)
        / np.sum((forces_ref - np.mean(forces_ref)) ** 2)
    )

    return EvaluationMetrics(
        energy_mae=energy_mae,
        energy_rmse=energy_rmse,
        energy_r2=energy_r2,
        forces_mae=forces_mae,
        forces_rmse=forces_rmse,
        forces_r2=forces_r2,
    )


def compute_metrics_from_arrays(
    energy_ref: np.ndarray,
    energy_pred: np.ndarray,
    forces_ref: np.ndarray,
    forces_pred: np.ndarray,
) -> EvaluationMetrics:
    """Compute error metrics from numpy arrays.

    Parameters
    ----------
    energy_ref : np.ndarray
        Reference energies.
    energy_pred : np.ndarray
        Predicted energies.
    forces_ref : np.ndarray
        Reference forces.
    forces_pred : np.ndarray
        Predicted forces.

    Returns
    -------
    EvaluationMetrics
        Metrics for energy and force predictions.
    """
    # Flatten forces if needed
    forces_ref_flat = forces_ref.flatten()
    forces_pred_flat = forces_pred.flatten()

    # Energy metrics
    energy_mae = float(np.mean(np.abs(energy_pred - energy_ref)))
    energy_rmse = float(np.sqrt(np.mean((energy_pred - energy_ref) ** 2)))

    energy_var = np.var(energy_ref)
    if energy_var > 0:
        energy_r2 = float(
            1
            - np.sum((energy_ref - energy_pred) ** 2)
            / np.sum((energy_ref - np.mean(energy_ref)) ** 2)
        )
    else:
        energy_r2 = 0.0

    # Force metrics
    forces_mae = float(np.mean(np.abs(forces_pred_flat - forces_ref_flat)))
    forces_rmse = float(np.sqrt(np.mean((forces_pred_flat - forces_ref_flat) ** 2)))

    forces_var = np.var(forces_ref_flat)
    if forces_var > 0:
        forces_r2 = float(
            1
            - np.sum((forces_ref_flat - forces_pred_flat) ** 2)
            / np.sum((forces_ref_flat - np.mean(forces_ref_flat)) ** 2)
        )
    else:
        forces_r2 = 0.0

    return EvaluationMetrics(
        energy_mae=energy_mae,
        energy_rmse=energy_rmse,
        energy_r2=energy_r2,
        forces_mae=forces_mae,
        forces_rmse=forces_rmse,
        forces_r2=forces_r2,
    )


def run_thermo_benchmark(
    force_field: "smee.TensorForceField",
    topologies: dict[str, Any],
    smiles_a: str,
    smiles_b: str | None = None,
    density_ref: float | None = None,
    hvap_ref: float | None = None,
    temperature: float = 298.15,
    pressure: float = 1.0,
    output_dir: Path | str = Path("./predictions"),
    cache_dir: Path | str | None = Path("./cache"),
) -> BenchmarkResult:
    """Run thermodynamic benchmark for density and heat of vaporization.

    Uses descent.targets.thermo to compute thermodynamic properties
    and compare against reference values.

    Parameters
    ----------
    force_field : smee.TensorForceField
        Force field to use for predictions.
    topologies : dict
        Dictionary of topologies {smiles: topology}.
    smiles_a : str
        SMILES string for component A.
    smiles_b : str, optional
        SMILES string for component B (for mixtures).
    density_ref : float, optional
        Reference density [g/mL].
    hvap_ref : float, optional
        Reference heat of vaporization [kcal/mol].
    temperature : float
        Temperature in K.
    pressure : float
        Pressure in atm.
    output_dir : Path | str
        Directory for output files.
    cache_dir : Path | str, optional
        Directory for cache files.

    Returns
    -------
    BenchmarkResult
        Benchmark results with predicted and reference values.

    Examples
    --------
    >>> result = run_thermo_benchmark(
    ...     force_field, topologies, "CCO",
    ...     density_ref=0.789, hvap_ref=10.1
    ... )
    >>> print(f"Density error: {result.density_pred - result.density_ref:.3f} g/mL")
    """
    import descent.targets.thermo

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    x_a = 1.0
    x_b = None

    entries = []

    # Density entry
    if density_ref is not None:
        entries.append(
            {
                "type": "density",
                "smiles_a": smiles_a,
                "x_a": x_a,
                "smiles_b": smiles_b,
                "x_b": x_b,
                "temperature": temperature,
                "pressure": pressure,
                "value": density_ref,
                "std": 0.0,
                "units": "g/mL",
                "source": "benchmark",
            }
        )

    # Hvap entry
    if hvap_ref is not None:
        entries.append(
            {
                "type": "hvap",
                "smiles_a": smiles_a,
                "x_a": x_a,
                "smiles_b": smiles_b,
                "x_b": x_b,
                "temperature": temperature,
                "pressure": pressure,
                "value": hvap_ref,
                "std": 0.0,
                "units": "kcal/mol",
                "source": "benchmark",
            }
        )

    if not entries:
        return BenchmarkResult()

    # Create dataset and run predictions
    dataset = descent.targets.thermo.create_dataset(*entries)

    results = descent.targets.thermo.predict(
        dataset,
        force_field,
        topologies,
        output_dir,
        cached_dir=cache_dir,
        verbose=True,
    )

    # Parse results
    result = BenchmarkResult()

    if isinstance(results, tuple) and len(results) == 4:
        y_true, y_true_std, y_pred, y_pred_std = results

        idx = 0
        if density_ref is not None:
            result.density_ref = float(y_true[idx])
            result.density_pred = float(y_pred[idx])
            result.density_std = float(y_pred_std[idx])
            idx += 1

        if hvap_ref is not None:
            result.hvap_ref = float(y_true[idx])
            result.hvap_pred = float(y_pred[idx])
            result.hvap_std = float(y_pred_std[idx])

    return result


def evaluate_force_field(
    force_field: "smee.TensorForceField",
    dataset: "datasets.Dataset",
    tensor_systems: dict[str, "smee.TensorSystem"],
    reference: str = "none",
    energy_cutoff: float | None = None,
    device: str = "cpu",
) -> tuple[PredictionResult, EvaluationMetrics]:
    """Evaluate a force field on a dataset and compute metrics.

    Convenience function that runs prediction and computes metrics in one call.

    Parameters
    ----------
    force_field : smee.TensorForceField
        Force field to evaluate.
    dataset : datasets.Dataset
        Dataset with reference energies and forces.
    tensor_systems : dict[str, smee.TensorSystem]
        Dictionary mapping mixture IDs to tensor systems.
    reference : str
        Reference energy mode ("mean", "min", or "none").
    energy_cutoff : float, optional
        Energy cutoff for filtering.
    device : str
        Device for computations.

    Returns
    -------
    tuple[PredictionResult, EvaluationMetrics]
        Prediction results and computed metrics.

    Examples
    --------
    >>> prediction, metrics = evaluate_force_field(
    ...     force_field, dataset, systems, reference="mean"
    ... )
    >>> print(f"Energy R²: {metrics.energy_r2:.4f}")
    """
    from .training import predict_energies_forces

    prediction = predict_energies_forces(
        dataset,
        force_field.to(device),
        tensor_systems=tensor_systems,
        reference=reference,
        normalize=False,
        device=device,
        energy_cutoff=energy_cutoff,
    )

    metrics = compute_metrics(prediction)

    return prediction, metrics
