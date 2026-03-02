"""Analysis module — evaluation metrics, benchmarks, and plots."""

from ._evaluation import (
    compute_metrics,
    compute_metrics_from_arrays,
    evaluate_force_field,
    run_thermo_benchmark,
)
from ._plots import plot_energy_vs_scale, plot_parity, plot_training_losses

__all__ = [
    # Evaluation
    "compute_metrics",
    "compute_metrics_from_arrays",
    "evaluate_force_field",
    "run_thermo_benchmark",
    # Plots
    "plot_energy_vs_scale",
    "plot_parity",
    "plot_training_losses",
]
