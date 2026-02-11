"""Configuration scaling node."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import openmm
import openmm.unit

from ..cli.utils import create_configs_from_dict, load_config
from ._utils import load_pickle, save_pickle
from .node import WorkflowNode


class ScalingNode(WorkflowNode):
    """
    Scaling node for generating scaled configurations for LJ parameter fitting.

    Inputs:
    - system_{system}.pkl: System state from MDNode
    - config: Scaling parameters

    Outputs:
    - scaled_{system}.pkl: Scaled coordinates and box vectors
    - scale_factors.npy: Scale factors array
    """

    @classmethod
    def name(cls) -> str:
        return "scaling"

    @classmethod
    def description(cls) -> str:
        return """Scaling node for generating scaled configurations for LJ parameter fitting.

Inputs:
- system_{system}.pkl: System state from MDNode
- config: Scaling parameters

Outputs:
- scaled_{system}.pkl: Scaled coordinates and box vectors
- scale_factors.npy: Scale factors array"""

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--system-name",
            type=str,
            help="Process only this system (default: all systems)",
        )
        parser.add_argument(
            "--system-file",
            type=str,
            help="Path to system pickle file (if not using default naming)",
        )

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        """Execute configuration scaling."""
        print("=" * 80)
        print("ScalingNode: Configuration Scaling")
        print("=" * 80)

        # Load configuration
        config_dict = load_config(args.config)
        general_config, _, scaling_config, _, _ = create_configs_from_dict(config_dict)

        self._ensure_output_dir(args.output_dir)

        # Generate scale factors
        print("\nGenerating scale factors...")
        print(f"  Close range: {scaling_config.close_range}")
        print(f"  Equilibrium range: {scaling_config.equilibrium_range}")
        print(f"  Long range: {scaling_config.long_range}")

        scale_factors = self._generate_scale_factors(scaling_config)
        print(f"  Total scale factors: {len(scale_factors)}")

        # Filter systems if requested
        systems = general_config.systems
        if args.system_name:
            systems = [s for s in systems if s.name == args.system_name]
            if not systems:
                raise ValueError(f"System '{args.system_name}' not found in config")

        results = {}

        for system in systems:
            print(f"\n{'=' * 80}")
            print(f"Processing system: {system.name}")
            print(f"{'=' * 80}")

            # Load system state
            if args.system_file:
                system_file = Path(args.system_file)
            else:
                system_file = self._output_path(
                    args.output_dir, f"system_{system.name}.pkl"
                )

            # Load system state from pickle
            system_state = load_pickle(system_file)
            tensor_system = system_state["tensor_system"]
            coords = system_state["coords"]
            box_vectors = system_state["box_vectors"]

            # Convert to numpy arrays in angstroms (if they're OpenMM quantities)
            if hasattr(coords, "value_in_unit"):
                coords_np = coords.value_in_unit(openmm.unit.angstrom)
            else:
                coords_np = coords

            if hasattr(box_vectors, "value_in_unit"):
                box_vectors_np = box_vectors.value_in_unit(openmm.unit.angstrom)
            else:
                box_vectors_np = box_vectors

            # Determine number of input frames
            n_input_frames = coords_np.shape[0] if coords_np.ndim == 3 else 1

            # Generate scaled configurations
            print("Creating scaled configurations...")
            print(f"  Input frames: {n_input_frames}")
            print(f"  Scale factors: {len(scale_factors)}")
            print(
                f"  Total configurations to generate: {n_input_frames * len(scale_factors)}"
            )
            coords_scaled, box_vectors_scaled = self._create_scaled_dataset(
                tensor_system, coords_np, box_vectors_np, scale_factors
            )

            print(f"  Generated {len(coords_scaled)} scaled configurations")

            # Create expanded scale factors array that matches the number of configurations
            # When we have N input frames and M scale factors, we generate N×M configurations
            # We need to repeat each scale factor N times to maintain the correspondence
            if n_input_frames > 1:
                expanded_scale_factors = np.repeat(scale_factors, n_input_frames)
                print(
                    f"  Expanded scale factors from {len(scale_factors)} to {len(expanded_scale_factors)} to match configurations"
                )
            else:
                expanded_scale_factors = scale_factors

            # Save scaled data
            scaled_file = self._output_path(
                args.output_dir, f"scaled_{system.name}.pkl"
            )
            scaled_data = {
                "coords_scaled": coords_scaled,
                "box_vectors_scaled": box_vectors_scaled,
                "scale_factors": expanded_scale_factors,  # Save expanded scale factors
                "tensor_system": tensor_system,
                "components": system_state.get("components", []),
            }

            save_pickle(scaled_data, scaled_file)
            print(f"  Scaled data saved: {scaled_file}")

            results[system.name] = {
                "scaled_file": str(scaled_file),
                "n_configurations": len(coords_scaled),
            }

        # Save scale factors (use unique values for the global file)
        if args.system_name:
            scale_factors_file = self._output_path(
                args.output_dir, f"scale_factors_{args.system_name}.npy"
            )
        else:
            scale_factors_file = self._output_path(args.output_dir, "scale_factors.npy")

        # Save unique scale factors for reference (not expanded)
        np.save(scale_factors_file, scale_factors)
        print(f"\nScale factors saved: {scale_factors_file}")

        print(f"\n{'=' * 80}")
        print("ScalingNode completed successfully")
        print(f"{'=' * 80}")

        return {"systems": results, "scale_factors_file": str(scale_factors_file)}

    @staticmethod
    def _generate_scale_factors(scaling_config):
        """Generate scale factors for density variation."""
        close = np.linspace(*scaling_config.close_range)
        equilibrium = np.linspace(*scaling_config.equilibrium_range)
        long = np.linspace(*scaling_config.long_range)
        scale_factors = np.concatenate((close, equilibrium[1:], long[1:]))
        return scale_factors

    @staticmethod
    def _compute_molecule_coms(coords, n_atoms_per_mol):
        """Compute center of mass for each molecule."""
        if coords.ndim == 2:
            n_atoms = coords.shape[0]
            n_molecules = n_atoms // n_atoms_per_mol
            coords_reshaped = coords.reshape(n_molecules, n_atoms_per_mol, 3)
            coms = coords_reshaped.mean(axis=1)
        elif coords.ndim == 3:
            n_frames, n_atoms = coords.shape[:2]
            n_molecules = n_atoms // n_atoms_per_mol
            coords_reshaped = coords.reshape(n_frames, n_molecules, n_atoms_per_mol, 3)
            coms = coords_reshaped.mean(axis=2)
        else:
            raise ValueError(f"coords must be 2D or 3D, got shape {coords.shape}")
        return coms

    @staticmethod
    def _get_box_center(box_vectors):
        """Compute the center of the simulation box."""
        if box_vectors.ndim == 2:
            center = 0.5 * np.diag(box_vectors)
        elif box_vectors.ndim == 3:
            center = 0.5 * np.diagonal(box_vectors, axis1=1, axis2=2)
        else:
            raise ValueError(
                f"box_vectors must be 2D or 3D, got shape {box_vectors.shape}"
            )
        return center

    @staticmethod
    def _scale_molecule_positions(coords, box_vectors, n_atoms_per_mol, scale_factor):
        """Scale molecular positions rigidly around the box center."""
        single_frame = coords.ndim == 2
        if single_frame:
            coords = np.expand_dims(coords, axis=0)
            box_vectors = np.expand_dims(box_vectors, axis=0)

        n_frames, n_atoms = coords.shape[:2]
        n_molecules = n_atoms // n_atoms_per_mol

        # Compute molecular centers of mass
        coms = ScalingNode._compute_molecule_coms(coords, n_atoms_per_mol)

        # Get box centers
        box_centers = ScalingNode._get_box_center(box_vectors)

        # Compute displacement vectors from box center to each COM
        displacements = coms - box_centers[:, np.newaxis, :]

        # Scale displacements
        scaled_displacements = displacements * scale_factor

        # Compute new COMs
        new_coms = box_centers[:, np.newaxis, :] + scaled_displacements

        # Compute translation vector for each molecule
        translations = new_coms - coms

        # Apply translations to all atoms in each molecule
        coords_reshaped = coords.reshape(n_frames, n_molecules, n_atoms_per_mol, 3)
        translations_expanded = translations[:, :, np.newaxis, :]

        # Apply translation
        scaled_coords = coords_reshaped + translations_expanded

        # Reshape back to original shape
        scaled_coords = scaled_coords.reshape(n_frames, n_atoms, 3)

        # Scale box vectors
        scaled_box_vectors = box_vectors * scale_factor

        # Remove batch dimension if input was single frame
        if single_frame:
            scaled_coords = np.squeeze(scaled_coords, axis=0)
            scaled_box_vectors = np.squeeze(scaled_box_vectors, axis=0)

        return scaled_coords, scaled_box_vectors

    @staticmethod
    def _create_scaled_dataset(tensor_system, coords, box_vectors, scale_factors):
        """Create a dataset with multiple scaled versions of the input configurations."""
        all_coords = []
        all_box_vectors = []

        for scale in scale_factors:
            scaled_slices = []
            current_idx = 0
            final_scaled_box_vecs = None

            for topology, n_copy in zip(
                tensor_system.topologies, tensor_system.n_copies
            ):
                n_atoms_per_mol = len(topology.atomic_nums)
                n_atoms_total_species = n_atoms_per_mol * n_copy

                # Slice coords for this species
                if coords.ndim == 2:
                    species_coords = coords[
                        current_idx : current_idx + n_atoms_total_species
                    ]
                else:
                    species_coords = coords[
                        :, current_idx : current_idx + n_atoms_total_species, :
                    ]

                # Scale this block of molecules
                scaled_species_coords, scaled_box_vecs = (
                    ScalingNode._scale_molecule_positions(
                        species_coords, box_vectors, n_atoms_per_mol, float(scale)
                    )
                )

                scaled_slices.append(scaled_species_coords)
                current_idx += n_atoms_total_species

                # Keep the scaled box vectors (they are the same for all species)
                final_scaled_box_vecs = scaled_box_vecs

            # Concatenate all scaled species coordinates
            if coords.ndim == 2:
                full_scaled_coords = np.concatenate(scaled_slices, axis=0)
                all_coords.append(full_scaled_coords)
                all_box_vectors.append(final_scaled_box_vecs)
            else:
                # Multiple frames: concatenate species and flatten frames into list
                full_scaled_coords = np.concatenate(scaled_slices, axis=1)
                # full_scaled_coords has shape (n_frames, n_atoms, 3)
                # Unpack each frame into a separate list element
                for frame_idx in range(full_scaled_coords.shape[0]):
                    all_coords.append(full_scaled_coords[frame_idx])
                    all_box_vectors.append(final_scaled_box_vecs[frame_idx])

        return all_coords, all_box_vectors
