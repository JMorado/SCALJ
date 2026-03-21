"""Trainable object creation for parameter optimization."""

import descent.train
import smee


def create_trainable(
    force_field: smee.TensorForceField,
    parameters_cols: list[str] = ["epsilon", "sigma"],
    parameters_scales: dict[str, float] | None = None,
    parameters_limits: dict[str, tuple[float, float]] | None = None,
    attributes_cols: list[str] = [],
    attributes_scales: dict[str, float] | None = None,
    attributes_limits: dict[str, tuple[float, float]] | None = None,
    device: str = "cpu",
) -> descent.train.Trainable:
    """
    Create a trainable object for parameter optimization.

    Parameters
    ----------
    force_field : smee.TensorForceField
        The force field with parameters to train.
    parameters_cols : list[str]
        Parameter columns to optimize (e.g., ["epsilon", "sigma"]).
    parameters_scales : dict[str, float], optional
        Scaling factors for each parameter type.
    parameters_limits : dict[str, tuple[float, float]], optional
        Min/max limits for each parameter type.
    attributes_cols : list[str]
        Attribute columns to optimize (e.g., ["charge"]).
    attributes_scales : dict[str, float], optional
        Scaling factors for each attribute type.
    attributes_limits : dict[str, tuple[float, float]], optional
        Min/max limits for each attribute type.
    device : str
        Device to use for training.

    Returns
    -------
    descent.train.Trainable
        Trainable object for optimization.
    """
    parameters_scales = parameters_scales or {}
    parameters_limits = parameters_limits or {}
    attributes_scales = attributes_scales or {}
    attributes_limits = attributes_limits or {}

    # Create trainable parameter config for vdW parameters.
    vdw_parameter_config = descent.train.ParameterConfig(
        cols=parameters_cols,
        scales=parameters_scales,
        limits=parameters_limits,
    )

    # Create trainable attribute config for vdW attributes.
    vdw_attribute_config = descent.train.AttributeConfig(
        cols=attributes_cols,
        scales=attributes_scales,
        limits=attributes_limits,
    )

    # Ensure vdW parameters require gradients.
    vdw_potential = force_field.potentials_by_type["vdW"]
    vdw_potential.parameters.requires_grad = True

    # Create trainable object.
    trainable = descent.train.Trainable(
        force_field=force_field.to(device),
        parameters={"vdW": vdw_parameter_config},
        attributes={"vdW": vdw_attribute_config},
    )

    return trainable
