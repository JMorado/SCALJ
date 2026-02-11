"""Base workflow node class for modular execution."""

import abc
import argparse
import json
from pathlib import Path
from typing import Any


class WorkflowNode(abc.ABC):
    """
    Base class for workflow nodes.

    Each node represents a discrete step in the SCALeJ workflow that:
    - Takes specific input file paths
    - Performs a well-defined computation
    - Writes output files to disk
    - Can be executed independently via CLI

    This design enables Snakemake orchestration and modular testing.
    """

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """Return the name of the workflow node (used as CLI subcommand)."""
        pass

    @classmethod
    @abc.abstractmethod
    def description(cls) -> str:
        """Return a description of what this node does."""
        pass

    @classmethod
    @abc.abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """
        Add node-specific arguments to the argument parser.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The argument parser to add arguments to.
        """
        pass

    @abc.abstractmethod
    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """
        Execute the node's computation.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed command-line arguments.

        Returns
        -------
        dict[str, Any]
            A dictionary containing information about outputs created,
            useful for logging and validation.
        """
        pass

    @classmethod
    def execute(cls, argv: list[str] | None = None) -> dict[str, Any]:
        """
        Execute this node from the command line.

        Parameters
        ----------
        argv : list[str] | None
            Command-line arguments. If None, uses sys.argv.

        Returns
        -------
        dict[str, Any]
            Result dictionary from the run() method.
        """
        parser = argparse.ArgumentParser(
            description=cls.description(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Add common arguments
        parser.add_argument(
            "--config", type=str, required=True, help="Path to YAML configuration file"
        )

        parser.add_argument(
            "--output-dir", type=str, required=True, help="Directory for output files"
        )

        parser.add_argument("--log-file", type=str, help="Path to log file (optional)")

        # Add node-specific arguments
        cls.add_arguments(parser)

        # Parse arguments
        args = parser.parse_args(argv)

        # Create output directory if needed
        cls._ensure_output_dir(args.output_dir)

        # Execute the node
        node = cls()
        result = node.run(args)

        # Log result if requested
        if args.log_file:
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

        return result

    @staticmethod
    def _ensure_path(path: str) -> Path:
        """Convert string to Path and verify it exists."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        return p

    @staticmethod
    def _ensure_output_dir(output_dir: str) -> Path:
        """
        Convert output_dir string to Path and ensure directory exists.

        Parameters
        ----------
        output_dir : str
            Output directory path

        Returns
        -------
        Path
            Path object with directory created
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

    @staticmethod
    def _output_path(base_dir: str, filename: str) -> Path:
        """Create an output file path."""
        return Path(base_dir) / filename
