"""Data module — datasets, scaling, and I/O utilities."""

from ._datasets import (
    combine_datasets,
    create_dataset,
    create_dataset_entry,
)
from ._io import (
    export_forcefield_to_offxml,
    load_dataset,
    load_json,
    load_object,
    load_parquet,
    load_pickle,
    save_dataset,
    save_json,
    save_object,
    save_parquet,
    save_pickle,
)
from ._scaling import (
    create_scaled_configurations,
    generate_scale_factors,
    scale_molecule_positions,
)

__all__ = [
    # Datasets
    "combine_datasets",
    "create_dataset",
    "create_dataset_entry",
    # I/O — objects (torch.save)
    "load_object",
    "save_object",
    "load_pickle",
    "save_pickle",
    # I/O — HuggingFace datasets (Arrow IPC)
    "load_dataset",
    "save_dataset",
    # I/O — Parquet
    "load_parquet",
    "save_parquet",
    # I/O — JSON
    "load_json",
    "save_json",
    # I/O — force field export
    "export_forcefield_to_offxml",
    # Scaling
    "create_scaled_configurations",
    "generate_scale_factors",
    "scale_molecule_positions",
]
