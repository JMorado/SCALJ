"""Thermodynamic benchmark node."""

import argparse
import os
from pathlib import Path
from typing import Any

import descent.targets.thermo
import torch

from ..cli.utils import create_configs_from_dict, load_config
from ..io import load_pickle
from .node import WorkflowNode


class BenchmarkNode(WorkflowNode):
    """
    Benchmark node for calculating thermodynamic properties (density, Hvap, etc.).

    Inputs:
    - trained_parameters.pkl: Trained parameters from TrainingNode
    - composite_system.pkl: Composite system from DatasetNode
    - config: System definitions and benchmark settings

    Outputs:
    - benchmark_results.txt: Thermodynamic properties (density, Hvap) with uncertainties
    """

    @classmethod
    def name(cls) -> str:
        return "benchmark"

    @classmethod
    def description(cls) -> str:
        return """Benchmark node for calculating thermodynamic properties (density, Hvap, etc.).

Inputs:
- trained_parameters.pkl: Trained parameters from TrainingNode
- composite_system.pkl: Composite system from DatasetNode
- config: System definitions and benchmark settings

Outputs:
- benchmark_results.txt: Thermodynamic properties (density, Hvap) with uncertainties"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--params-file",
            type=str,
            default="trained_parameters.pkl",
            help="Name of parameters file (default: trained_parameters.pkl)",
        )
        parser.add_argument(
            "--system-name",
            type=str,
            help="Run benchmark for only this system (default: all systems)",
        )
        parser.add_argument(
            "--n-replicas",
            type=int,
            default=3,
            help="Number of replicas for uncertainty estimation (default: 3)",
        )

    @staticmethod
    def _run_thermo_benchmark(
        force_field: Any,
        topologies: dict[str, Any],
        smiles_a: str,
        smiles_b: str | None = None,
        n_replicas: int = 3,
        output_dir: Path = Path("./predictions"),
        cache_dir: Path = Path("./cache"),
        density_ref: float | None = None,
        hvap_ref: float | None = None,
        temperature: float = 298.15,
        pressure: float = 1.0,
    ) -> dict[str, Any]:
        """
        Run thermodynamic benchmark (Density, Hvap) for a given system.

        Parameters
        ----------
        force_field : smee.TensorForceField
            Force field to use for predictions
        topologies : dict
            Dictionary of topologies {smiles: topology}
        smiles_a : str
            SMILES string for component A
        smiles_b : str, optional
            SMILES string for component B (for mixtures)
        n_replicas : int
            Number of replicas for uncertainty estimation (not used in new API)
        output_dir : Path
            Directory for output files
        cache_dir : Path
            Directory for cache files
        density_ref : float, optional
            Reference density value [g/mL]
        hvap_ref : float, optional
            Reference Hvap value [kcal/mol]
        temperature : float
            Temperature in K
        pressure : float
            Pressure in atm

        Returns
        -------
        dict
            Dictionary with predicted values and errors
        """
        # Create data entries for thermodynamic properties
        x_a = 1.0  # Pure component or major component
        x_b = None

        entries = []

        # Density entry
        if density_ref is not None:
            entries.append(
                {
                    "type": "density",
                    "smiles_a": smiles_a,
                    "x_a": x_a,
                    "smiles_b": smiles_b,
                    "x_b": x_b,
                    "temperature": temperature,
                    "pressure": pressure,
                    "value": density_ref,
                    "std": 0.0,
                    "units": "g/mL",
                    "source": "benchmark",
                }
            )

        # Hvap entry
        if hvap_ref is not None:
            entries.append(
                {
                    "type": "hvap",
                    "smiles_a": smiles_a,
                    "x_a": x_a,
                    "smiles_b": smiles_b,
                    "x_b": x_b,
                    "temperature": temperature,
                    "pressure": pressure,
                    "value": hvap_ref,
                    "std": 0.0,
                    "units": "kcal/mol",
                    "source": "benchmark",
                }
            )

        if not entries:
            print("  Warning: No reference values provided, skipping benchmark")
            return {}

        # Create dataset
        dataset = descent.targets.thermo.create_dataset(*entries)

        # Run predictions
        print(f"  Running thermodynamic predictions...")
        output_dir.mkdir(parents=True, exist_ok=True)
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

        results = descent.targets.thermo.predict(
            dataset,
            force_field,
            topologies,
            output_dir,
            cached_dir=cache_dir,
            verbose=True,
        )

        # Parse tuple results: (y_true, y_true_std, y_pred, y_pred_std)
        if isinstance(results, tuple) and len(results) == 4:
            y_true, y_true_std, y_pred, y_pred_std = results

            # Convert tensors to floats
            result_dict = {}

            # Assuming order is [density, hvap] based on entries order
            idx = 0
            if density_ref is not None:
                result_dict["density_ref"] = float(y_true[idx])
                result_dict["density_pred"] = float(y_pred[idx])
                result_dict["density_pred_std"] = float(y_pred_std[idx])
                idx += 1

            if hvap_ref is not None:
                result_dict["hvap_ref"] = float(y_true[idx])
                result_dict["hvap_pred"] = float(y_pred[idx])
                result_dict["hvap_pred_std"] = float(y_pred_std[idx])

            return result_dict

        # Fallback for dict results (if API changes back)
        return results

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Execute thermodynamic benchmarks."""
        print("=" * 80)
        print("BenchmarkNode: Thermodynamic Property Calculation")
        print("=" * 80)

        # Load configuration
        config_dict = load_config(args.config)
        general_config, _, _, training_config, *_ = create_configs_from_dict(config_dict)

        self._ensure_output_dir(args.output_dir)

        # Load trained parameters - handle both relative and absolute paths
        params_file = Path(args.params_file)
        params_data = load_pickle(params_file)

        # Handle both initial and final parameter files
        if "final_params" in params_data:
            final_params = params_data["final_params"]
            print(f"Loaded trained parameters from {params_file}")
        elif "initial_params" in params_data:
            final_params = params_data["initial_params"]
            print(f"Loaded initial parameters from {params_file}")
        else:
            raise KeyError(
                f"Parameter file must contain either 'final_params' or 'initial_params' key"
            )
        print(f"Using parameters for benchmark")

        # Load composite system
        composite_file = self._output_path(args.output_dir, "composite_system.pkl")
        composite_data = load_pickle(composite_file)
        all_tensor_systems = composite_data["all_tensor_systems"]

        # Get or recreate trainable
        composite_trainable = params_data.get("composite_trainable")
        if composite_trainable is None:
            print("Recreating trainable object...")
            from .training_node import TrainingNode

            composite_trainable = TrainingNode._create_trainable(
                composite_data["composite_tensor_forcefield"],
                **params_data["parameter_config"],
                **params_data["attribute_config"],
                device=training_config.device,
            )

        # Enable PME for benchmarks
        print("\nEnabling PME for accurate electrostatics...")
        os.environ["SMEE_USE_PME"] = "1"

        # Filter systems if requested
        systems = general_config.systems
        if args.system_name:
            systems = [s for s in systems if s.name == args.system_name]
            if not systems:
                raise ValueError(f"System '{args.system_name}' not found in config")

        # Run benchmarks
        print(f"\n{'=' * 80}")
        print("Running thermodynamic benchmarks...")
        print(f"{'=' * 80}")
        print(f"  Number of replicas: {args.n_replicas}")
        print(f"  Systems: {len(systems)}")

        benchmark_results = {}

        for system in systems:
            print(f"\n{'=' * 40}")
            print(f"System: {system.name}")
            print(f"{'=' * 40}")

            smiles_a = system.components[0].smiles
            smiles_b = (
                system.components[1].smiles if len(system.components) > 1 else None
            )

            try:
                # Get force field with trained parameters
                force_field = composite_trainable.to_force_field(final_params)

                # Create topologies dict for descent API
                tensor_system = all_tensor_systems[system.name]
                topologies_dict = {smiles_a: tensor_system.topologies[0]}
                if smiles_b and len(tensor_system.topologies) > 1:
                    topologies_dict[smiles_b] = tensor_system.topologies[1]

                thermo_results = self._run_thermo_benchmark(
                    force_field.to(torch.device("cpu")),
                    topologies_dict,
                    smiles_a,
                    smiles_b,
                    n_replicas=args.n_replicas,
                    output_dir=self._output_path(args.output_dir, "benchmark")
                    / system.name,
                    cache_dir=self._output_path(args.output_dir, "cache"),
                    density_ref=0.7914
                    if system.name == "methanol"
                    else None,  # TODO: Get from config
                    hvap_ref=8.94
                    if system.name == "methanol"
                    else None,  # TODO: Get from config
                    temperature=298.15,
                    pressure=1.0,
                )

                print(f"  Benchmark results for {system.name}:")
                print(thermo_results)

                # Process results
                if "error" not in thermo_results and thermo_results:
                    benchmark_results[system.name] = {
                        "density": {
                            "mean": thermo_results.get("density_pred", 0.0),
                            "std": thermo_results.get("density_pred_std", 0.0),
                            "ref": thermo_results.get("density_ref", 0.0),
                        },
                        "hvap": {
                            "mean": thermo_results.get("hvap_pred", 0.0),
                            "std": thermo_results.get("hvap_pred_std", 0.0),
                            "ref": thermo_results.get("hvap_ref", 0.0),
                        },
                    }
                    print(f"  ✓ Benchmark completed for {system.name}")
                else:
                    benchmark_results[system.name] = {"error": thermo_results["error"]}

            except Exception as e:
                print(f"\n  Warning: Benchmark failed for {system.name}:")
                print(f"  {e}")
                benchmark_results[system.name] = {"error": str(e)}

        # Save benchmark results
        benchmark_file = self._output_path(args.output_dir, "benchmark_results.txt")
        with open(benchmark_file, "w", encoding="utf-8") as f:
            f.write("Thermodynamic Benchmark Results\n")
            f.write("=" * 80 + "\n\n")
            for system_name, data in benchmark_results.items():
                f.write(f"System: {system_name}\n")
                if "error" in data:
                    f.write(f"  Error: {data['error']}\n")
                else:
                    density_ref = data["density"].get("ref", 0.0)
                    hvap_ref = data["hvap"].get("ref", 0.0)
                    f.write(
                        f"  Density: {data['density']['mean']:.4f} ± {data['density']['std']:.4f} g/mL "
                        f"(ref: {density_ref:.4f})\n"
                    )
                    f.write(
                        f"  Hvap: {data['hvap']['mean']:.4f} ± {data['hvap']['std']:.4f} kcal/mol "
                        f"(ref: {hvap_ref:.4f})\n"
                    )
                f.write("\n")

        print(f"\n{'=' * 80}")
        print(f"Benchmark results saved: {benchmark_file}")
        print(f"{'=' * 80}")

        return {
            "benchmark_file": str(benchmark_file),
            "benchmark_results": benchmark_results,
        }
