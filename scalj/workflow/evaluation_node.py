"""Final evaluation and benchmarking node - fully self-contained."""

import argparse
import json
from typing import Any

import numpy as np

from .. import plots
from ..cli.utils import create_configs_from_dict, load_config
from ._utils import load_pickle
from .base_nodes import PredictionBaseNode


class EvaluationNode(PredictionBaseNode):
    """
    Evaluation node for predictions and final plots.

    Inputs:
    - trained_parameters.pkl: Trained parameters from TrainingNode
    - combined_dataset.pkl: Combined dataset from DatasetNode
    - composite_system.pkl: Composite system from DatasetNode
    - config: Evaluation settings

    Outputs:
    - parity_final_*.png: Final model predictions
    - energy_vs_scale_final_*.png: Energy curves vs scale factors
    """

    @classmethod
    def name(cls) -> str:
        return "evaluation"

    @classmethod
    def description(cls) -> str:
        return """Evaluation node for predictions and final plots.

Inputs:
- trained_parameters.pkl: Trained parameters from TrainingNode
- combined_dataset.pkl: Combined dataset from DatasetNode
- composite_system.pkl: Composite system from DatasetNode
- config: Evaluation settings

Outputs:
- parity_final_*.png: Final model predictions
- energy_vs_scale_final_*.png: Energy curves vs scale factors"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        pass

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Execute final evaluation."""
        print("=" * 80)
        print("EvaluationNode: Final Evaluation")
        print("=" * 80)

        # Load configuration
        config_dict = load_config(args.config)
        _, _, _, training_config, _ = create_configs_from_dict(config_dict)

        self._ensure_output_dir(args.output_dir)
        params_file = self._output_path(args.output_dir, "trained_parameters.pkl")
        params_data = load_pickle(params_file)

        final_force_field = params_data["final_force_field"]

        print("Loaded trained parameters")

        # Load dataset
        dataset_file = self._output_path(args.output_dir, "combined_dataset.pkl")
        combined_dataset = load_pickle(dataset_file)
        print(f"Loaded dataset: {len(combined_dataset)} configurations")

        # Load composite system
        composite_file = self._output_path(args.output_dir, "composite_system.pkl")
        composite_data = load_pickle(composite_file)
        all_tensor_systems = composite_data["all_tensor_systems"]
        composite_trainable = params_data.get("composite_trainable")

        # If composite_trainable not in params, need to recreate it
        if composite_trainable is None:
            # Import TrainingNode to access _create_trainable
            from .training_node import TrainingNode

            composite_trainable = TrainingNode._create_trainable(
                composite_data["composite_tensor_forcefield"],
                params_data["parameter_config"],
                training_config,
            )

        # Generate final predictions
        print(f"\n{'=' * 80}")
        print("Generating final predictions...")
        print(f"{'=' * 80}")
        print(f"  Energy cutoff: {training_config.energy_cutoff} kcal/mol")
        print(f"  Device: {training_config.device}")

        energy_ref, energy_pred, forces_ref, forces_pred, _, _, all_mask_idxs = (
            self._predict(
                combined_dataset,
                final_force_field.to(training_config.device),
                all_tensor_systems=all_tensor_systems,
                reference=training_config.reference,
                normalize=False,
                device=training_config.device,
                energy_cutoff=training_config.energy_cutoff,
            )
        )

        # Plot final parity plots
        print("\nGenerating parity plots...")

        plots.plot_parity(
            energy_ref.detach().cpu().numpy(),
            energy_pred.detach().cpu().numpy(),
            "Energy",
            "kcal/mol",
            self._output_path(args.output_dir, "parity_energy_final.png"),
        )

        plots.plot_parity(
            forces_ref.flatten().detach().cpu().numpy(),
            forces_pred.flatten().detach().cpu().numpy(),
            "Forces",
            "kcal/mol/Å",
            self._output_path(args.output_dir, "parity_forces_final.png"),
        )

        print("  Overall parity plots saved")

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
        metrics_file = self._output_path(args.output_dir, "metrics_final.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")

        # Load scale factors for energy vs scale plots
        scale_factors_file = self._output_path(args.output_dir, "scale_factors.npy")

        # Initialize results dictionary
        results = {
            "final_plots": {
                "energy_parity": str(
                    self._output_path(args.output_dir, "parity_energy_final.png")
                ),
                "forces_parity": str(
                    self._output_path(args.output_dir, "parity_forces_final.png")
                ),
            },
            "per_system_plots": [],
            "metrics": {
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
            },
            "trained_params_file": str(params_file),
            "metrics_file": str(metrics_file),
        }

        if scale_factors_file.exists():
            scale_factors = np.load(scale_factors_file)

            # Generate per-system final plots
            print("\nGenerating per-system plots...")
            offset = 0
            for i, system_name in enumerate(all_tensor_systems.keys()):
                mask = all_mask_idxs[i].detach().cpu().numpy()
                n_points = len(mask)

                e_ref_sys = (
                    energy_ref.detach().cpu().numpy()[offset : offset + n_points]
                )
                e_pred_sys = (
                    energy_pred.detach().cpu().numpy()[offset : offset + n_points]
                )

                parity_energy_file = self._output_path(
                    args.output_dir, f"parity_energy_final_{system_name}.png"
                )
                plots.plot_parity(
                    e_ref_sys,
                    e_pred_sys,
                    f"Energy ({system_name})",
                    "kcal/mol",
                    parity_energy_file,
                )

                energy_vs_scale_file = self._output_path(
                    args.output_dir, f"energy_vs_scale_final_{system_name}.png"
                )
                plots.plot_energy_vs_scale(
                    scale_factors[mask],
                    [e_ref_sys, e_pred_sys],
                    energy_vs_scale_file,
                    labels=["Reference", "Optimized"],
                    lims=(0, 30),
                )

                results["per_system_plots"].append(
                    {
                        "system": system_name,
                        "parity_energy": str(parity_energy_file),
                        "energy_vs_scale": str(energy_vs_scale_file),
                    }
                )

                print(f"  {system_name}: plots saved")
                offset += n_points
        else:
            print("\nNote: scale_factors.npy not found, skipping per-system plots")

        # Export optimized force field
        print(f"\n{'=' * 80}")
        print("Optimized force field available in trained_parameters.pkl")
        print(f"{'=' * 80}")

        print(f"\n{'=' * 80}")
        print("EvaluationNode completed successfully")
        print(f"{'=' * 80}")

        return results
