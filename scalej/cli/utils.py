"""Utility functions for CLI."""

from pathlib import Path
from typing import Optional

import yaml

from .. import config as cfg


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str or None, optional
        Path to YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    if config_path is None:
        return {}

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def create_configs_from_dict(config_dict: dict) -> tuple:
    """
    Create configuration objects from dictionary.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary.

    Returns
    -------
    tuple
        Tuple of (GeneralConfig, SimulationConfig, ScalingConfig, TrainingConfig,
        ParameterConfig).
    """
    general_config = cfg.GeneralConfig(**config_dict.get("general", {}))
    simulation_config = cfg.SimulationConfig(**config_dict.get("simulation", {}))
    scaling_config = cfg.ScalingConfig(**config_dict.get("scaling", {}))
    training_config = cfg.TrainingConfig(**config_dict.get("training", {}))
    parameter_config = cfg.ParameterConfig(**config_dict.get("parameters", {}))
    attribute_config = cfg.AttributeConfig(**config_dict.get("attributes", {}))

    return (
        general_config,
        simulation_config,
        scaling_config,
        training_config,
        parameter_config,
        attribute_config,
    )


def generate_config(args):
    """
    Generate an example configuration file.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing output path.
    """
    # Create example configurations with all options
    example_general = cfg.GeneralConfig(
        force_field_name="de-force-1.0.3.offxml",
        mlp_name="mace-off24-medium",
        output_dir="output",
        systems=[
            cfg.SystemConfig(
                name="methanol",
                weight=1.0,
                trajectory_path=None,
                components=[
                    cfg.MoleculeComponent(
                        smiles="[C:1]([H:3])([H:4])([H:5])[O:2][H:6]",
                        nmol=350,
                    )
                ],
            )
        ],
    )

    example_simulation = cfg.SimulationConfig(
        temperature="300 K",
        pressure="1.0 atm",
        timestep="1.0 fs",
        n_minimization_steps=0,
        n_nvt_steps=50000,
        n_npt_equilibration_steps=50000,
        n_production_steps=1000000,
        n_mlp_steps=0,
        report_interval=2000,
    )

    example_scaling = cfg.ScalingConfig(
        close_range=[0.75, 0.9, 5],
        equilibrium_range=[0.9, 1.1, 15],
        long_range=[1.1, 2.0, 12],
        subsample_frequency=20,
    )

    example_training = cfg.TrainingConfig(
        learning_rate=0.01,
        n_epochs=100,
        device="cuda",
        energy_weight=1.0,
        force_weight=1.0,
        reference="min",
        normalize=True,
        energy_cutoff=20.0,
        weighting_method="boltzmann",
        weighting_temperature="2000 K",
    )

    example_parameters = cfg.ParameterConfig(
        cols=["epsilon", "r_min"],
        scales={"epsilon": 10.0, "r_min": 1.0},
        limits={"epsilon": [None, None], "r_min": [0.0, None]},
    )

    example_attributes = cfg.AttributeConfig(
        cols=["alpha", "beta"],
        scales={"alpha": 10.0, "beta": 1.0},
        limits={"alpha": [None, None], "beta": [None, None]},
    )

    # Convert to dictionary, excluding runtime-populated fields
    config_dict = {
        "general": example_general.model_dump(
            mode="python",
            exclude_none=True,
            exclude={
                "systems": {
                    "__all__": {"tensor_forcefield", "tensor_system", "topologies"}
                }
            },
        ),
        "simulation": example_simulation.model_dump(mode="python"),
        "scaling": example_scaling.model_dump(mode="python"),
        "training": example_training.model_dump(mode="python"),
        "parameters": example_parameters.model_dump(mode="python"),
        "attributes": example_attributes.model_dump(mode="python"),
    }

    # Convert tuples to lists recursively to avoid !!python/tuple tags in YAML
    def convert_tuples_to_lists(obj):
        if isinstance(obj, dict):
            return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_tuples_to_lists(item) for item in obj]
        else:
            return obj

    config_dict = convert_tuples_to_lists(config_dict)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Example configuration saved to: {output_path}")
    print("\nYou can now edit this file and run:")
    print(f"  scalej run --config {output_path}")
