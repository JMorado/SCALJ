from pathlib import Path
from typing import Any

import descent.targets.thermo
import torch


def run_thermo_benchmark(
    smiles: str,
    trainable,
    topologies: list,
    parameters: torch.Tensor | None = None,
    n_replicas: int = 1,
    output_dir: Path = Path("./predictions"),
    cache_dir: Path = Path("./cache"),
    density_ref: float = 0.79,
    hvap_ref: float = 38.30,
) -> list[Any]:
    """
    Run thermodynamic benchmark (Density, Hvap) for a given system.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    trainable : descent.train.Trainable
        Trainable object containing the force field.
    topologies : list
        List of topologies (used to map SMILES to topology).
    parameters : torch.Tensor, optional
        Optimized parameters to apply to the force field. If None, uses trainable defaults.
    n_replicas : int, optional
        Number of replicas to run, by default 1.
    output_dir : Path, optional
        Directory to save predictions, by default ./predictions.
    cache_dir : Path, optional
        Directory for caching, by default ./cache.
    density_ref : float, optional
        Reference density value (g/mL), by default 0.79.
    hvap_ref : float, optional
        Reference Hvap value (kcal/mol), by default 38.30.

    Returns
    -------
    list[Any]
        List of results from descent.targets.thermo.predict.
    """
    results_final = []

    # Prepare force field
    if parameters is not None:
        force_field = trainable.to_force_field(parameters).to("cpu")
    else:
        # If parameters not provided, use default from trainable
        # Assuming to_force_field can be called without args or we prefer to_force_field(trainable.to_values())
        # descent.train.Trainable.to_force_field typically takes parameters.
        # If we want current params:
        force_field = trainable.to_force_field(trainable.to_values()).to("cpu")

    for n in range(n_replicas):
        density_pure = {
            "type": "density",
            "smiles_a": smiles,
            "x_a": 1.0,
            "smiles_b": None,
            "x_b": None,
            "temperature": 300.0,
            "pressure": 1.0,
            "value": density_ref,
            "std": 0.001,
            "units": "g/mL",
            "source": None,
        }

        hvap = {
            "type": "hvap",
            "smiles_a": smiles,
            "x_a": 1.0,
            "smiles_b": None,
            "x_b": None,
            "temperature": 300.0,
            "pressure": 1.0,
            "value": hvap_ref,
            "std": 0.001,
            "units": "kcal/mol",
            "source": None,
        }

        dataset = descent.targets.thermo.create_dataset(density_pure, hvap)

        # Ensure output directory is unique per replica or handled by predict?
        # The snippet used one dir. We'll use the passed dir.

        results = descent.targets.thermo.predict(
            dataset,
            force_field,
            {smiles: topologies[0]},
            output_dir=output_dir,
            cached_dir=cache_dir,
            verbose=True,
        )
        results_final.append(results)

    return results_final
