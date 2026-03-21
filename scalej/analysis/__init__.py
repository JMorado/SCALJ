"""Analysis module — evaluation metrics, benchmarks, and plots."""

from ._evaluation import (
    compute_metrics,
    compute_metrics_from_arrays,
    evaluate_force_field,
    run_thermo_benchmark,
    save_evaluation_parquets,
)
from ._plots import plot_energy_vs_scale, plot_parity, plot_training_losses
from ._summary import ThermodynamicSummary, TrainingSummary

__all__ = [
    # Evaluation
    "compute_metrics",
    "compute_metrics_from_arrays",
    "evaluate_force_field",
    "run_thermo_benchmark",
    "save_evaluation_parquets",
    # Plots
    "plot_energy_vs_scale",
    "plot_parity",
    "plot_training_losses",
    # Summary
    "ThermodynamicSummary",
    "TrainingSummary",
]
