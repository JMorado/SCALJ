"""Parameter training node."""

import argparse
from typing import Any

import descent.utils.reporting

from .. import plots, training
from ..cli.utils import create_configs_from_dict, load_config
from ..io import load_pickle, save_pickle
from .node import WorkflowNode


class TrainingNode(WorkflowNode):
    """
    Training node for optimizing LJ parameters.

    Inputs:
    - combined_dataset.pkl: Combined dataset from DatasetNode
    - composite_system.pkl: Composite system from DatasetNode
    - config: Training parameters (learning_rate, n_epochs, etc.)

    Outputs:
    - initial_parameters.pkl: Parameters before training (for initial evaluation)
    - trained_parameters.pkl: Optimized parameters after training
    - training_losses.png: Training loss curves
    """

    @classmethod
    def name(cls) -> str:
        return "training"

    @classmethod
    def description(cls) -> str:
        return """Training node for optimizing LJ parameters.

Inputs:
- combined_dataset.pkl: Combined dataset from DatasetNode
- composite_system.pkl: Composite system from DatasetNode
- config: Training parameters (learning_rate, n_epochs, etc.)

Outputs:
- initial_parameters.pkl: Parameters before training (for initial evaluation)
- trained_parameters.pkl: Optimized parameters after training
- training_losses.png: Training loss curves"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--learning-rate", type=float, help="Override learning rate from config"
        )
        parser.add_argument(
            "--n-epochs", type=int, help="Override number of epochs from config"
        )
        parser.add_argument(
            "--device",
            type=str,
            choices=["cpu", "cuda"],
            help="Override device from config",
        )

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Execute parameter training."""
        print("=" * 80)
        print("TrainingNode: Parameter Optimization")
        print("=" * 80)

        # Load configuration
        config_dict = load_config(args.config)
        _, _, _, training_config, parameter_config, attribute_config = create_configs_from_dict(
            config_dict
        )

        # Override training parameters if specified
        if args.learning_rate:
            training_config.learning_rate = args.learning_rate
        if args.n_epochs:
            training_config.n_epochs = args.n_epochs
        if args.device:
            training_config.device = args.device

        self._ensure_output_dir(args.output_dir)

        print("Training configuration:")
        print(f"  Learning rate: {training_config.learning_rate}")
        print(f"  Epochs: {training_config.n_epochs}")
        print(f"  Device: {training_config.device}")
        print(f"  Parameters: {parameter_config.cols}")
        print(f"  Attributes: {attribute_config.cols}")
        print(f"  Energy weight: {training_config.energy_weight}")
        print(f"  Force weight: {training_config.force_weight}")

        # Load dataset
        dataset_file = self._output_path(args.output_dir, "combined_dataset.pkl")
        combined_dataset = load_pickle(dataset_file)
        print(f"\nLoaded dataset: {len(combined_dataset)} configurations")

        # Load composite system
        composite_file = self._output_path(args.output_dir, "composite_system.pkl")
        composite_data = load_pickle(composite_file)
        all_tensor_systems = composite_data["all_tensor_systems"]
        composite_tensor_forcefield = composite_data["composite_tensor_forcefield"]
        print(f"Loaded composite system with {len(all_tensor_systems)} systems")

        # Create trainable force field using the public API
        print(f"\n{'=' * 80}")
        print("Creating trainable force field...")
        print(f"{'=' * 80}")

        composite_trainable = training.create_trainable(
            composite_tensor_forcefield,
            parameters_cols=parameter_config.cols,
            parameters_scales=parameter_config.scales,
            parameters_limits=parameter_config.limits,
            attributes_cols=attribute_config.cols,
            attributes_scales=attribute_config.scales,
            attributes_limits=attribute_config.limits,
            device=training_config.device,
        )

        print("Initial parameters:")
        descent.utils.reporting.print_potential_summary(
            composite_tensor_forcefield.potentials_by_type["vdW"]
        )

        # Save initial parameters for initial evaluation
        initial_params_file = self._output_path(
            args.output_dir, "initial_parameters.pkl"
        )
        initial_params_data = {
            "initial_params": composite_trainable.to_values(),
            "initial_force_field": composite_tensor_forcefield,
            "training_config": training_config,
            "parameter_config": parameter_config,
            "attribute_config": attribute_config,
            "composite_trainable": composite_trainable,
        }

        save_pickle(initial_params_data, initial_params_file)
        print(f"\nInitial parameters saved: {initial_params_file}")

        # Train parameters using the public API
        print(f"\n{'=' * 80}")
        print("Training parameters...")
        print(f"{'=' * 80}")

        training_result = training.train_parameters(
            composite_trainable,
            combined_dataset,
            all_tensor_systems,
            n_epochs=training_config.n_epochs,
            learning_rate=training_config.learning_rate,
            energy_weight=training_config.energy_weight,
            force_weight=training_config.force_weight,
            reference=training_config.reference,
            normalize=training_config.normalize,
            energy_cutoff=training_config.energy_cutoff,
            weighting_method=training_config.weighting_method,
            weighting_temperature=training_config.weighting_temperature,
            device=training_config.device,
            verbose=True,
        )

        print("\nTraining completed!")

        # Plot training losses
        loss_plot_path = self._output_path(args.output_dir, "training_losses.png")
        plots.plot_training_losses(
            training_result.energy_losses, training_result.force_losses, loss_plot_path
        )
        print(f"Training losses saved: {loss_plot_path}")

        # Display final parameters
        print(f"\n{'=' * 80}")
        print("Final parameters:")
        print(f"{'=' * 80}")

        final_force_field = composite_trainable.to_force_field(
            training_result.trained_parameters
        )
        descent.utils.reporting.print_potential_summary(
            final_force_field.potentials_by_type["vdW"]
        )

        # Save trained parameters
        params_file = self._output_path(args.output_dir, "trained_parameters.pkl")
        params_data = {
            "final_params": training_result.trained_parameters,
            "final_force_field": final_force_field,
            "energy_losses": training_result.energy_losses,
            "force_losses": training_result.force_losses,
            "training_config": training_config,
            "parameter_config": parameter_config,
            "attribute_config": attribute_config,
            "composite_trainable": composite_trainable,
        }

        save_pickle(params_data, params_file)
        print(f"\nTrained parameters saved: {params_file}")

        print(f"\n{'=' * 80}")
        print("TrainingNode completed successfully")
        print(f"{'=' * 80}")

        return {
            "initial_params_file": str(initial_params_file),
            "params_file": str(params_file),
            "loss_plot": str(loss_plot_path),
            "final_energy_loss": (
                float(training_result.energy_losses[-1])
                if training_result.energy_losses
                else None
            ),
            "final_force_loss": (
                float(training_result.force_losses[-1])
                if training_result.force_losses
                else None
            ),
        }
