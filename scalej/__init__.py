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
from . import cli, config, workflow

# Energy/force computation
from .energy import (
    compute_mlp_energies_forces,
    compute_mlp_energies_forces_single,
    run_mlp_relaxation,
    setup_mlp_simulation,
)

# Evaluation functions
from .evaluation import (
    compute_metrics,
    compute_metrics_from_arrays,
    evaluate_force_field,
    run_thermo_benchmark,
)

# I/O functions
from .io import (
    export_forcefield_to_offxml,
    load_forcefield,
    load_pickle,
    save_pickle,
)

# Data models
from .models import (
    BenchmarkResult,
    EnergyForceResult,
    EvaluationMetrics,
    LossResult,
    PredictionResult,
    ScalingResult,
    TrainingResult,
    TrajectoryFrames,
)

# Scaling functions
from .scaling import (
    compute_molecule_coms,
    create_scaled_configurations,
    generate_scale_factors,
    get_box_center,
    scale_molecule_positions,
)

# Simulation functions
from .simulation import (
    generate_initial_coords,
    load_trajectory_frames,
    relax_with_mlp,
    run_md_simulation,
)

# System setup functions
from .systems import (
    combine_datasets,
    create_composite_system,
    create_dataset,
    create_dataset_entry,
    create_system_from_smiles,
)

# Training functions
from .training import (
    compute_loss,
    create_trainable,
    predict_energies_forces,
    train_parameters,
)

try:
    __version__ = importlib.metadata.version("scalej")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Submodules
    "cli",
    "config",
    "workflow",
    # Data models
    "BenchmarkResult",
    "EnergyForceResult",
    "EvaluationMetrics",
    "LossResult",
    "PredictionResult",
    "ScalingResult",
    "TrainingResult",
    "TrajectoryFrames",
    # Scaling
    "generate_scale_factors",
    "compute_molecule_coms",
    "get_box_center",
    "scale_molecule_positions",
    "create_scaled_configurations",
    # Energy
    "setup_mlp_simulation",
    "run_mlp_relaxation",
    "compute_mlp_energies_forces",
    "compute_mlp_energies_forces_single",
    # Simulation
    "run_md_simulation",
    "load_trajectory_frames",
    "generate_initial_coords",
    "relax_with_mlp",
    # Training
    "create_trainable",
    "predict_energies_forces",
    "compute_loss",
    "train_parameters",
    # Evaluation
    "compute_metrics",
    "compute_metrics_from_arrays",
    "run_thermo_benchmark",
    "evaluate_force_field",
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
    "load_forcefield",
]
