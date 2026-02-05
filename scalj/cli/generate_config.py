"""Generate example configuration files."""

from pathlib import Path

import yaml

from .. import config as cfg


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
                        nmol=323,
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
        reference="none",
        normalize=True,
    )

    example_parameters = cfg.ParameterConfig(
        cols=["epsilon", "r_min"],
        scales={"epsilon": 10.0, "r_min": 1.0},
        limits={"epsilon": [None, None], "r_min": [0.0, None]},
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
    print(f"  scalj run --config {output_path}")
