"""Main CLI entry point."""

import argparse
import sys

from .utils import generate_config
from .run import run_workflow


def main():
    """
    Main CLI entry point.

    Parses command-line arguments and executes the appropriate subcommand.
    """
    parser = argparse.ArgumentParser(
        description="SCALJ - LJ Parameter Fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate example configuration
  scalj config --output config.yaml
  
  # Run workflow with configuration file
  scalj run --config config.yaml
  
  # Run with command-line overrides
  scalj run --config config.yaml --n-epochs 200 --device cpu
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the SCALJ workflow")
    run_parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    run_parser.add_argument("--force-field", type=str, help="Force field file name")
    run_parser.add_argument("--ml-potential", type=str, help="ML potential model name")
    run_parser.add_argument("--output-dir", type=str, help="Output directory")
    run_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    run_parser.add_argument("--n-epochs", type=int, help="Number of training epochs")
    run_parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], help="Device for training"
    )
    run_parser.set_defaults(func=run_workflow)

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

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
