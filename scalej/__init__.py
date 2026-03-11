"""SCALeJ: Lennard-Jones Parameter Fitting via Condensed-Phase Volume-Scaling.

SCALeJ provides both a workflow-based interface (via Snakemake) and
a programmatic API for fitting Lennard-Jones parameters using ML potentials.

Example API Usage
-----------------
>>> import scalej
>>>
>>> # Create a system from SMILES
>>> system, ff, topos = scalej.create_system_from_smiles(
...     ["CCO"], [200], "openff-2.0.0.offxml"
... )
>>>
>>> # Generate scaled configurations
>>> scales = scalej.generate_scale_factors()
>>> result = scalej.create_scaled_configurations(system, coords, box, scales)
>>>
>>> # Compute MLP energies and forces
>>> ef_result = scalej.compute_mlp_energies_forces_single(
...     system, result.coords, result.box_vectors, mlp_name="ani2x"
... )
>>>
>>> # Train parameters
>>> trainable = scalej.create_trainable(ff)
>>> training_result = scalej.train_parameters(
...     trainable, dataset, tensor_systems, n_epochs=100
... )
"""

import importlib.metadata

# Submodules
from . import analysis, cli, config, data, simulation, workflow  # noqa: F401

# Evaluation and plotting functions
from .analysis import (
    compute_metrics,
    compute_metrics_from_arrays,
    evaluate_force_field,
    plot_energy_vs_scale,
    plot_parity,
    plot_training_losses,
    run_thermo_benchmark,
)

# Data functions
from .data import (
    combine_datasets,
    create_dataset,
    create_dataset_entry,
    create_scaled_configurations,
    export_forcefield_to_offxml,
    generate_scale_factors,
    load_pickle,
    save_pickle,
    scale_molecule_positions,
)

# Data models
from .models import (
    BenchmarkResult,
    EnergyForceResult,
    EvaluationMetrics,
    PredictionResult,
    ScalingResult,
    TrainingResult,
    TrajectoryFrames,
)

# Simulation functions
from .simulation import (
    atoms_template_from_tensor_system,
    compute_ase_energies_forces,
    compute_mlp_energies_forces,
    compute_mlp_energies_forces_single,
    create_composite_system,
    create_system_from_smiles,
    generate_initial_coords,
    load_trajectory_frames,
    relax_with_mlp,
    run_md_simulation,
    run_mlp_relaxation,
    setup_mlp_simulation,
)

# Training functions
from .training import (
    create_trainable,
    get_losses,
    predict_energies_forces,
    train_parameters,
    train_parameters_ddp,
)

try:
    __version__ = importlib.metadata.version("scalej")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Submodules
    "analysis",
    "cli",
    "simulation",
    "config",
    "data",
    "workflow",
    # Data models
    "BenchmarkResult",
    "EnergyForceResult",
    "EvaluationMetrics",
    "PredictionResult",
    "ScalingResult",
    "TrainingResult",
    "TrajectoryFrames",
    # Scaling
    "generate_scale_factors",
    "scale_molecule_positions",
    "create_scaled_configurations",
    # Energy
    "setup_mlp_simulation",
    "run_mlp_relaxation",
    "compute_mlp_energies_forces",
    "compute_mlp_energies_forces_single"
    "compute_ase_energies_forces",
    # Simulation
    "atoms_template_from_tensor_system",
    "run_md_simulation",
    "load_trajectory_frames",
    "generate_initial_coords",
    "relax_with_mlp",
    # Training
    "create_trainable",
    "predict_energies_forces",
    "get_losses",
    "train_parameters",
    "train_parameters_ddp",
    # Evaluation and plotting
    "compute_metrics",
    "compute_metrics_from_arrays",
    "run_thermo_benchmark",
    "evaluate_force_field",
    "plot_energy_vs_scale",
    "plot_parity",
    "plot_training_losses",
    # Systems
    "create_system_from_smiles",
    "create_composite_system",
    "create_dataset_entry",
    "create_dataset",
    "combine_datasets",
    # I/O
    "load_pickle",
    "save_pickle",
    "export_forcefield_to_offxml",
]
