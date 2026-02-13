"""Main CLI entry point."""

import argparse
import sys

from .utils import generate_config


def main():
    """
    Main CLI entry point.

    Parses command-line arguments and executes the appropriate subcommand.
    """
    parser = argparse.ArgumentParser(
        description="SCALeJ - LJ Parameter Fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate example configuration
  scalej config --output config.yaml
  
  # Run individual workflow nodes
  scalej md --config config.yaml --output-dir output
  scalej scaling --config config.yaml --output-dir output
  scalej ml_potential --config config.yaml --output-dir output
  scalej dataset --config config.yaml --output-dir output
  scalej training --config config.yaml --output-dir output
  scalej evaluation --config config.yaml --output-dir output
  scalej export --config config.yaml --output-dir output
  scalej benchmark --config config.yaml --output-dir output
  
  # Run with Snakemake (recommended for parallel execution)
  snakemake --cores 1 --configfile config.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Config command
    config_parser = subparsers.add_parser(
        "config", help="Generate example configuration file"
    )
    config_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="config.yaml",
        help="Output path for configuration file (default: config.yaml)",
    )
    config_parser.set_defaults(func=generate_config)

    # Add workflow node subcommands
    _add_node_subcommands(subparsers)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


def _add_node_subcommands(subparsers):
    """Add subcommands for each workflow node."""
    from ..workflow.benchmark_node import BenchmarkNode
    from ..workflow.dataset_node import DatasetNode
    from ..workflow.evaluation_node import EvaluationNode
    from ..workflow.export_node import ExportNode
    from ..workflow.md_node import MDNode
    from ..workflow.ml_potential_node import MLPotentialNode
    from ..workflow.mlp_md_node import MLPMDNode
    from ..workflow.scaling_node import ScalingNode
    from ..workflow.system_setup_node import SystemSetupNode
    from ..workflow.training_node import TrainingNode

    # Define all workflow nodes
    nodes = [
        SystemSetupNode,
        MDNode,
        MLPMDNode,
        ScalingNode,
        MLPotentialNode,
        DatasetNode,
        TrainingNode,
        EvaluationNode,
        ExportNode,
        BenchmarkNode,
    ]

    # Create a subcommand for each node
    for node_class in nodes:
        node_parser = subparsers.add_parser(
            node_class.name(),
            help=node_class.description().split("\n")[0],  # First line as help
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=node_class.description(),
        )

        # Add common node arguments
        node_parser.add_argument(
            "--config", type=str, required=True, help="Path to YAML configuration file"
        )

        node_parser.add_argument(
            "--output-dir", type=str, required=True, help="Directory for output files"
        )

        node_parser.add_argument(
            "--log-file", type=str, help="Path to log file (optional)"
        )

        # Add node-specific arguments
        node_class.add_arguments(node_parser)

        # Set the execution function
        node_parser.set_defaults(func=lambda args, nc=node_class: _run_node(nc, args))


def _run_node(node_class, args):
    """Execute a workflow node."""
    node = node_class()
    result = node.run(args)

    # Optionally save result to log file
    if args.log_file:
        import json
        from pathlib import Path

        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
