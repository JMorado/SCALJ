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

    return (
        general_config,
        simulation_config,
        scaling_config,
        training_config,
        parameter_config,
    )
