"""Evaluation node for predictions and plotting."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .. import plots
from ..cli.utils import create_configs_from_dict, load_config
from ._utils import load_pickle
from .base_nodes import PredictionBaseNode


class EvaluationNode(PredictionBaseNode):
    """
    Evaluation node for generating predictions and plots.

    Inputs:
    - trained_parameters.pkl: Trained parameters (from TrainingNode or elsewhere)
    - combined_dataset.pkl: Combined dataset from DatasetNode
    - composite_system.pkl: Composite system from DatasetNode
    - scale_factors.npy: (optional) Scale factors for energy vs scale plots
    - config.yaml: Evaluation settings (energy cutoff, device, etc.)

    Outputs:
    - parity_energy_*.png: Energy parity plots
    - parity_forces_*.png: Force parity plots
    - energy_vs_scale_*.png: (optional) Energy vs scale factor plots per system
    """

    @classmethod
    def name(cls) -> str:
        return "evaluation"

    @classmethod
    def description(cls) -> str:
        return """Evaluation node for generating parity plots and evaluation metrics.

Inputs:
- trained_parameters.pkl (or initial_parameters.pkl): Parameters to evaluate
- combined_dataset.pkl: Dataset from DatasetNode
- composite_system.pkl: Composite system from DatasetNode
- config: Evaluation settings

Outputs:
- parity_energy_*.png: Energy parity plots
- parity_forces_*.png: Force parity plots
- energy_vs_scale_*.png: (optional) Energy vs scale factor plots per system
- metrics_*.json: Evaluation metrics (MAE, RMSE, R²)"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--params-file",
            type=str,
            default="trained_parameters.pkl",
            help="Name of parameters file (default: trained_parameters.pkl)",
        )
        parser.add_argument(
            "--plot-prefix",
            type=str,
            default="",
            help='Prefix for plot filenames (e.g., "initial_" or "final_")',
        )
        parser.add_argument(
            "--system-name",
            type=str,
            help="Process only this system (default: all systems)",
        )

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """
        Run the evaluation node.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed command-line arguments.

        Returns
        -------
        dict[str, Any]
            Dictionary containing paths to generated plots and any relevant metrics.
        """
        print("=" * 80)
        print("EvaluationNode: Prediction and Plotting")
        print("=" * 80)

        # Load configuration
        config_dict = load_config(args.config)
        _, _, _, training_config, _ = create_configs_from_dict(config_dict)

        self._ensure_output_dir(args.output_dir)

        # Load trained parameters
        # Handle both relative filenames and full paths
        params_path = Path(args.params_file)
        if params_path.is_absolute() or params_path.parts[0] == args.output_dir:
            params_file = params_path
        else:
            params_file = self._output_path(args.output_dir, args.params_file)
        params_data = load_pickle(params_file)
        print(f"Loaded parameters from {params_file}")

        # Check for force field in parameters
        force_field = params_data.get("final_force_field") or params_data.get(
            "initial_force_field"
        )
        if not force_field:
            raise KeyError(
                "Parameter file must contain 'final_force_field' or 'initial_force_field'"
            )

        # Load dataset
        dataset_file = self._output_path(args.output_dir, "combined_dataset.pkl")
        combined_dataset = load_pickle(dataset_file)
        print(f"Loaded dataset: {len(combined_dataset)} configurations")

        # Load composite system
        composite_file = self._output_path(args.output_dir, "composite_system.pkl")
        composite_data = load_pickle(composite_file)
        all_tensor_systems = composite_data["all_tensor_systems"]
        print(f"Loaded composite system with {len(all_tensor_systems)} systems")

        # Filter systems if requested
        if args.system_name:
            if args.system_name not in all_tensor_systems:
                raise ValueError(
                    f"System '{args.system_name}' not found. "
                    f"Available: {list(all_tensor_systems.keys())}"
                )
            print(f"Filtering to system: {args.system_name}")
            all_tensor_systems = {
                args.system_name: all_tensor_systems[args.system_name]
            }

        # Generate predictions
        print(f"\n{'=' * 80}")
        print("Calculating energy and force predictions with settings:")
        print(f"{'=' * 80}")
        print(f"  Energy cutoff: {training_config.energy_cutoff} kcal/mol")
        print(f"  Device: {training_config.device}")

        energy_ref, energy_pred, forces_ref, forces_pred, _, _, all_mask_idxs = (
            self._predict(
                combined_dataset,
                force_field.to(training_config.device),
                all_tensor_systems=all_tensor_systems,
                reference=training_config.reference,
                normalize=False,
                device=training_config.device,
                energy_cutoff=training_config.energy_cutoff,
            )
        )

        prefix = args.plot_prefix if args.plot_prefix else ""
        results = {"per_system_plots": []}

        # Calculate overall error metrics
        energy_ref_np = energy_ref.detach().cpu().numpy()
        energy_pred_np = energy_pred.detach().cpu().numpy()
        forces_ref_np = forces_ref.flatten().detach().cpu().numpy()
        forces_pred_np = forces_pred.flatten().detach().cpu().numpy()

        energy_mae = float(np.mean(np.abs(energy_pred_np - energy_ref_np)))
        energy_rmse = float(np.sqrt(np.mean((energy_pred_np - energy_ref_np) ** 2)))
        energy_r2 = float(
            1
            - np.sum((energy_ref_np - energy_pred_np) ** 2)
            / np.sum((energy_ref_np - np.mean(energy_ref_np)) ** 2)
        )

        forces_mae = float(np.mean(np.abs(forces_pred_np - forces_ref_np)))
        forces_rmse = float(np.sqrt(np.mean((forces_pred_np - forces_ref_np) ** 2)))
        forces_r2 = float(
            1
            - np.sum((forces_ref_np - forces_pred_np) ** 2)
            / np.sum((forces_ref_np - np.mean(forces_ref_np)) ** 2)
        )

        # Save metrics to file
        metrics_data = {
            "energy": {
                "mae": energy_mae,
                "rmse": energy_rmse,
                "r2": energy_r2,
            },
            "forces": {
                "mae": forces_mae,
                "rmse": forces_rmse,
                "r2": forces_r2,
            },
        }
        metrics_file = self._output_path(
            args.output_dir, f"metrics_{prefix if prefix else 'evaluation'}.json"
        )
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
        print(
            f"  Energy - MAE: {energy_mae:.4f}, RMSE: {energy_rmse:.4f}, R²: {energy_r2:.4f}"
        )
        print(
            f"  Forces - MAE: {forces_mae:.4f}, RMSE: {forces_rmse:.4f}, R²: {forces_r2:.4f}"
        )

        results["metrics"] = metrics_data
        results["metrics_file"] = str(metrics_file)

        if not args.system_name:
            # Generate overall plots when processing all systems
            print("\nGenerating overall parity plots...")
            energy_parity_file = self._output_path(
                args.output_dir, f"{prefix}parity_energy.png"
            )
            forces_parity_file = self._output_path(
                args.output_dir, f"{prefix}parity_forces.png"
            )

            plots.plot_parity(
                energy_ref.detach().cpu().numpy(),
                energy_pred.detach().cpu().numpy(),
                "Energy",
                "kcal/mol",
                energy_parity_file,
            )

            plots.plot_parity(
                forces_ref.flatten().detach().cpu().numpy(),
                forces_pred.flatten().detach().cpu().numpy(),
                "Forces",
                "kcal/mol/Å",
                forces_parity_file,
            )

            print(f"  Energy parity: {energy_parity_file}")
            print(f"  Forces parity: {forces_parity_file}")

            results["energy_parity"] = str(energy_parity_file)
            results["forces_parity"] = str(forces_parity_file)

        # Generate per-system plots (either for all systems or just the specified one)
        print("\nGenerating per-system plots...")
        offset = 0
        for i, system_name in enumerate(all_tensor_systems.keys()):
            mask = all_mask_idxs[i].detach().cpu().numpy()
            n_points = len(mask)

            e_ref_sys = energy_ref.detach().cpu().numpy()[offset : offset + n_points]
            e_pred_sys = energy_pred.detach().cpu().numpy()[offset : offset + n_points]

            # Per-system energy parity plot
            parity_energy_file = self._output_path(
                args.output_dir, f"parity_energy_{prefix}{system_name}.png"
            )
            plots.plot_parity(
                e_ref_sys,
                e_pred_sys,
                f"Energy ({system_name})",
                "kcal/mol",
                parity_energy_file,
            )

            # Per-system force parity plot
            # TODO: change this to use only the forces for the current system
            parity_forces_file = self._output_path(
                args.output_dir, f"parity_forces_{prefix}{system_name}.png"
            )
            plots.plot_parity(
                forces_ref.flatten().detach().cpu().numpy(),
                forces_pred.flatten().detach().cpu().numpy(),
                f"Forces ({system_name})",
                "kcal/mol/Å",
                parity_forces_file,
            )

            # Check if scale_factors.npy exists for energy vs scale plot
            scale_factors_file = self._output_path(args.output_dir, "scale_factors.npy")
            if scale_factors_file.exists():
                scale_factors = np.load(scale_factors_file)

                # Per-system energy vs scale plot
                energy_vs_scale_file = self._output_path(
                    args.output_dir, f"energy_vs_scale_{prefix}{system_name}.png"
                )
                plots.plot_energy_vs_scale(
                    scale_factors[mask],
                    [e_ref_sys, e_pred_sys],
                    energy_vs_scale_file,
                    labels=["Reference", "Predicted"],
                    lims=(0, 30),
                )
            else:
                energy_vs_scale_file = None
                print(
                    f"  scale_factors.npy not found, skipping "
                    f"energy vs scale plot for {system_name}."
                )

            results["per_system_plots"].append(
                {
                    "system": system_name,
                    "parity_energy": str(parity_energy_file),
                    "parity_forces": str(parity_forces_file),
                    "energy_vs_scale": str(energy_vs_scale_file),
                }
            )

            print(f"  {system_name}: plots saved")
            offset += n_points

        return results
