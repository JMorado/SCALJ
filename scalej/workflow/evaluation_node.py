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
    - config: Evaluation settings

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
        # Remove trailing underscore from prefix for metrics filename
        metrics_prefix = prefix.rstrip("_") if prefix else "evaluation"
        metrics_file = self._output_path(
            args.output_dir, f"metrics_{metrics_prefix}.json"
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

        # Collect all data for overall energy vs scale plot
        if not args.system_name:
            print("\nPreparing data for frame-based energy vs scale plots...")
            all_frame_data = []  # List of (scale_factors, e_ref, e_pred, system, frame)

            temp_offset = 0
            for i, system_name in enumerate(all_tensor_systems.keys()):
                temp_mask = all_mask_idxs[i].detach().cpu().numpy()
                temp_n_points = len(temp_mask)

                temp_e_ref = (
                    energy_ref.detach()
                    .cpu()
                    .numpy()[temp_offset : temp_offset + temp_n_points]
                )
                temp_e_pred = (
                    energy_pred.detach()
                    .cpu()
                    .numpy()[temp_offset : temp_offset + temp_n_points]
                )

                # Load scale factors for this system
                ef_file = self._output_path(
                    args.output_dir, f"energies_forces_{system_name}.pkl"
                )
                if ef_file.exists():
                    try:
                        ef_data = load_pickle(ef_file)
                        system_scale_factors = ef_data.get("scale_factors")

                        if system_scale_factors is not None:
                            # Apply mask to scale factors
                            filtered_scale_factors = system_scale_factors[temp_mask]

                            # Determine unique scale factors and number of frames
                            unique_scales = np.unique(filtered_scale_factors)
                            n_unique = len(unique_scales)
                            n_configs = len(filtered_scale_factors)
                            n_frames = n_configs // n_unique

                            print(
                                f"  {system_name}: {n_frames} frames × "
                                f"{n_unique} scale factors = {n_configs} configs"
                            )

                            # Extract data for each frame
                            for frame_idx in range(n_frames):
                                # Get indices for this frame across all scales
                                frame_indices = np.arange(
                                    frame_idx, n_configs, n_frames
                                )
                                frame_scales = filtered_scale_factors[frame_indices]
                                frame_e_ref = temp_e_ref[frame_indices]
                                frame_e_pred = temp_e_pred[frame_indices]

                                all_frame_data.append(
                                    {
                                        "scale_factors": frame_scales,
                                        "e_ref": frame_e_ref,
                                        "e_pred": frame_e_pred,
                                        "system": system_name,
                                        "frame": frame_idx,
                                    }
                                )
                    except Exception as e:
                        print(
                            f"  Warning: Could not process scale factors "
                            f"for {system_name}: {e}"
                        )

                temp_offset += temp_n_points

            # Generate per-frame plots (reference + predicted)
            if all_frame_data:
                print("\nGenerating per-frame energy vs scale plots...")
                for frame_info in all_frame_data:
                    frame_plot_file = self._output_path(
                        args.output_dir,
                        f"{prefix}energy_vs_scale_{frame_info['system']}_"
                        f"frame{frame_info['frame']}.png",
                    )
                    plots.plot_energy_vs_scale(
                        frame_info["scale_factors"],
                        [frame_info["e_ref"], frame_info["e_pred"]],
                        frame_plot_file,
                        labels=["Reference", "Predicted"],
                        lims=(0, 30),
                    )
                print(f"  Generated {len(all_frame_data)} per-frame plots\n")

                # Generate overall plot (reference + predicted)
                print("Generating overall energy vs scale plot...")
                all_scales = np.concatenate(
                    [f["scale_factors"] for f in all_frame_data]
                )
                all_e_ref = np.concatenate([f["e_ref"] for f in all_frame_data])
                all_e_pred = np.concatenate([f["e_pred"] for f in all_frame_data])

                overall_plot_file = self._output_path(
                    args.output_dir, f"{prefix}energy_vs_scale_total.png"
                )
                plots.plot_energy_vs_scale(
                    all_scales,
                    [all_e_ref, all_e_pred],
                    overall_plot_file,
                    labels=["Reference", "Predicted"],
                    lims=(0, 30),
                )
                print(f"  Overall plot: {overall_plot_file}")
                results["overall_energy_vs_scale"] = str(overall_plot_file)

        # Generate per-system plots (either for all systems or just the specified one)
        print("\nGenerating per-system parity plots...")
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

            results["per_system_plots"].append(
                {
                    "system": system_name,
                    "parity_energy": str(parity_energy_file),
                    "parity_forces": str(parity_forces_file),
                }
            )

            print(f"  {system_name}: plots saved")
            offset += n_points

        return results
