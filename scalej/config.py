"""Configuration models."""

from typing import Literal

import openmm.unit
import pydantic
from pydantic_units import OpenMMQuantity, quantity_serializer
from smee import TensorForceField, TensorSystem, TensorTopology

# Unit type aliases
_KELVIN = openmm.unit.kelvin
_ATMOSPHERE = openmm.unit.atmosphere
_FEMTOSECOND = openmm.unit.femtosecond
_INV_PICOSECOND = 1.0 / openmm.unit.picoseconds


if pydantic.__version__.startswith("1."):

    class BaseModel(pydantic.BaseModel):
        class Config:
            json_encoders = {openmm.unit.Quantity: quantity_serializer}
            arbitrary_types_allowed = True

else:

    class BaseModel(pydantic.BaseModel):
        class Config:
            arbitrary_types_allowed = True


class SimulationConfig(BaseModel):
    """
    Configuration for molecular dynamics simulations.

    Attributes
    ----------
    temperature : openmm.unit.Quantity
        Simulation temperature.
    pressure : openmm.unit.Quantity
        Simulation pressure.
    timestep : openmm.unit.Quantity
        Integration timestep.
    friction_coeff : openmm.unit.Quantity
        Langevin friction coefficient.
    n_minimization_steps : int
        Number of minimization steps (0 uses MinimizationConfig).
    n_nvt_steps : int
        Number of NVT equilibration steps.
    n_npt_equilibration_steps : int
        Number of NPT equilibration steps.
    n_production_steps : int
        Number of production MD steps.
    report_interval : int
        Interval for saving trajectory frames.
    """

    temperature: OpenMMQuantity[_KELVIN] = pydantic.Field(
        300 * _KELVIN,
        description="Simulation temperature with units compatible with kelvin.",
    )
    pressure: OpenMMQuantity[_ATMOSPHERE] = pydantic.Field(
        1.0 * _ATMOSPHERE,
        description="Simulation pressure with units compatible with atmosphere.",
    )
    timestep: OpenMMQuantity[_FEMTOSECOND] = pydantic.Field(
        1.0 * _FEMTOSECOND,
        description="Integration timestep with units compatible with femtosecond.",
    )

    friction_coeff: OpenMMQuantity[_INV_PICOSECOND] = pydantic.Field(
        1.0 * _INV_PICOSECOND,
        description="Langevin friction coefficient with units compatible with 1/picosecond.",
    )

    n_minimization_steps: int = pydantic.Field(
        0, description="Number of minimization steps (0 uses MinimizationConfig)."
    )
    n_equilibration_nvt_steps: int = pydantic.Field(
        50_000, description="Number of NVT equilibration steps."
    )
    n_equilibration_npt_steps: int = pydantic.Field(
        50_000, description="Number of NPT equilibration steps."
    )
    n_production_steps: int = pydantic.Field(
        1_000_000, description="Number of production MD steps."
    )
    n_mlp_steps: int = pydantic.Field(
        100, description="Number of MLP steps to run after the production MD steps."
    )
    mlp_device: str = pydantic.Field(
        "cpu", description="Device to use for MLP ('cuda' or 'cpu')."
    )
    platform: str = pydantic.Field(
        "CPU", description="Platform to use for OpenMM ('CPU', 'CUDA', etc.)."
    )
    report_interval: int = pydantic.Field(
        1000, description="Interval for saving trajectory frames."
    )


class ScalingConfig(BaseModel):
    """
    Configuration for molecular position scaling.

    Attributes
    ----------
    close_range : tuple[float, float, int]
        (start, end, n_points) for close-range scaling.
    equilibrium_range : tuple[float, float, int]
        (start, end, n_points) for equilibrium-range scaling.
    long_range : tuple[float, float, int]
        (start, end, n_points) for long-range scaling.
    subsample_frequency : int
        Frequency for subsampling trajectory frames.
    n_frames : int
        Number of last frames to load from trajectory for scaling.
    """

    close_range: tuple[float, float, int] = pydantic.Field(
        (0.75, 0.9, 5), description="(start, end, n_points) for close-range scaling."
    )
    equilibrium_range: tuple[float, float, int] = pydantic.Field(
        (0.9, 1.1, 15),
        description="(start, end, n_points) for equilibrium-range scaling.",
    )
    long_range: tuple[float, float, int] = pydantic.Field(
        (1.1, 2.0, 12), description="(start, end, n_points) for long-range scaling."
    )
    subsample_frequency: int = pydantic.Field(
        20, description="Frequency for subsampling trajectory frames."
    )
    n_frames: int = pydantic.Field(
        1, description="Number of last frames to load from trajectory for scaling."
    )


class TrainingConfig(BaseModel):
    """
    Configuration for parameter training.

    Attributes
    ----------
    learning_rate : float
        Learning rate for optimizer.
    n_epochs : int
        Number of training epochs.
    device : str
        Device to use for training ('cuda' or 'cpu').
    energy_weight : float
        Weight for energy loss term.
    force_weight : float
        Weight for force loss term.
    reference : Literal["mean", "min", "none"]
        Reference energy mode for relative energies.
    normalize : bool
        Whether to normalize losses by number of conformers/atoms.
    """

    learning_rate: float = pydantic.Field(
        0.01, description="Learning rate for optimizer."
    )
    n_epochs: int = pydantic.Field(100, description="Number of training epochs.")
    device: str = pydantic.Field(
        "cuda", description="Device to use for training ('cuda' or 'cpu')."
    )
    energy_weight: float = pydantic.Field(
        1.0, description="Weight for energy loss term."
    )
    force_weight: float = pydantic.Field(1.0, description="Weight for force loss term.")
    reference: Literal["mean", "min", "none"] = pydantic.Field(
        "none", description="Reference energy mode for relative energies."
    )
    normalize: bool = pydantic.Field(
        True, description="Whether to normalize losses by number of conformers/atoms."
    )
    energy_cutoff: float | None = pydantic.Field(
        None, description="Energy cutoff in kcal/mol to filter high-energy conformers."
    )
    weighting_method: Literal["uniform", "boltzmann"] = pydantic.Field(
        "uniform", description="Method to weight conformers in loss function."
    )
    weighting_temperature: OpenMMQuantity[_KELVIN] = pydantic.Field(
        300.0 * _KELVIN,
        description="Temperature in Kelvin for Boltzmann weighting.",
    )


class MoleculeComponent(BaseModel):
    """
    Single molecular component in a mixture.

    Attributes
    ----------
    smiles : str
        Indexed SMILES string for the molecule.
    nmol : int
        Number of molecules of this component.
    """

    smiles: str = pydantic.Field(
        ..., description="Indexed SMILES string for the molecule."
    )
    nmol: int = pydantic.Field(
        ..., description="Number of molecules of this component."
    )


class SystemConfig(BaseModel):
    """
    Configuration for a single system.

    Attributes
    ----------
    name : str
        Identifier for this system.
    components : list[MoleculeComponent]
        List of molecular components in this mixture.
    trajectory_path : Optional[str]
        Path to existing trajectory file for this mixture.
    weight : float
        Weight of this mixture in the loss function (default: 1.0).
    """

    name: str = pydantic.Field(..., description="Identifier for this mixture.")
    components: list[MoleculeComponent] = pydantic.Field(
        ..., description="List of molecular components in this mixture."
    )
    trajectory_path: str | None = pydantic.Field(
        None, description="Path to existing trajectory file for this mixture."
    )
    weight: float = pydantic.Field(
        1.0, description="Weight of this mixture in the loss function."
    )
    tensor_forcefield: TensorForceField | None = pydantic.Field(
        None, description="Tensor force field (populated during workflow)."
    )
    tensor_system: TensorSystem | None = pydantic.Field(
        None, description="Tensor system (populated during workflow)."
    )
    topologies: list[TensorTopology] | None = pydantic.Field(
        None, description="Molecular topologies (populated during workflow)."
    )
    nmol: list[int] | None = pydantic.Field(
        None,
        description="Number of molecules for each component (populated during workflow).",
    )


class GeneralConfig(BaseModel):
    """
    Configuration for molecular system setup.

    Attributes
    ----------
    systems : list[SystemConfig]
        List of mixtures/systems to simulate and optimize.
    force_field_name : str
        Force field file name (shared across all systems).
    ml_potential_name : str
        ML potential model name (shared across all systems).
    output_dir : str
        Directory for output files.
    """

    systems: list[SystemConfig] = pydantic.Field(
        ..., description="List of mixtures/systems to simulate and optimize."
    )
    force_field_name: str = pydantic.Field(
        "de-force-1.0.3.offxml", description="Force field file name."
    )
    mlp_name: str = pydantic.Field(
        "mace-off24-medium", description="ML potential model name."
    )
    output_dir: str = pydantic.Field(
        "output", description="Directory for output files."
    )
    trajectory_path: str | None = pydantic.Field(
        None, description="Global path to existing trajectory file (fallback)."
    )


class ParameterConfig(BaseModel):
    """
    Configuration for trainable parameters.

    Attributes
    ----------
    cols : list of str
        Parameter column names to train.
    scales : dict of {str: float}
        Scaling factors for each parameter.
    limits : dict of {str: tuple of (float or None, float or None)}
        (min, max) limits for each parameter.
    """

    cols: list[str] = pydantic.Field(
        default_factory=lambda: ["epsilon", "r_min"],
        description="Parameter column names to train.",
    )
    scales: dict[str, float] = pydantic.Field(
        default_factory=lambda: {"epsilon": 10.0, "r_min": 1.0},
        description="Scaling factors for each parameter.",
    )
    limits: dict[str, tuple[float | None, float | None]] = pydantic.Field(
        default_factory=lambda: {"epsilon": (None, None), "r_min": (0.0, None)},
        description="(min, max) limits for each parameter.",
    )


class AttributeConfig(BaseModel):
    """
    Configuration for trainable attributes.

    Attributes
    ----------
    cols : list of str
        Attribute column names to train.
    scales : dict of {str: float}
        Scaling factors for each attribute.
    limits : dict of {str: tuple of (float or None, float or None)}
        (min, max) limits for each attribute.
    """

    cols: list[str] = pydantic.Field(
        default_factory=list, description="Attribute column names to train."
    )
    scales: dict[str, float] = pydantic.Field(
        default_factory=dict, description="Scaling factors for each attribute."
    )
    limits: dict[str, tuple[float | None, float | None]] = pydantic.Field(
        default_factory=dict, description="(min, max) limits for each attribute."
    )