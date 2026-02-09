"""Molecular position scaling utilities for density variation."""

import numpy as np

from ..config import ScalingConfig


def _compute_molecule_coms(coords: np.ndarray, n_atoms_per_mol: int) -> np.ndarray:
    """
    Compute center of mass for each molecule (assuming equal atomic masses).

    Parameters
    ----------
    coords : np.ndarray
        Coordinates with shape (n_atoms, 3) or (n_frames, n_atoms, 3).
    n_atoms_per_mol : int
        Number of atoms per molecule.

    Returns
    -------
    np.ndarray
        Center of mass positions with shape (n_molecules, 3) or
        (n_frames, n_molecules, 3).
    """
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


def _get_box_center(box_vectors: np.ndarray) -> np.ndarray:
    """
    Compute the center of the simulation box.

    Parameters
    ----------
    box_vectors : np.ndarray
        Box vectors with shape (3, 3) or (n_frames, 3, 3).

    Returns
    -------
    np.ndarray
        Box center with shape (3,) or (n_frames, 3).
    """
    if box_vectors.ndim == 2:
        center = 0.5 * np.diag(box_vectors)
    elif box_vectors.ndim == 3:
        center = 0.5 * np.diagonal(box_vectors, axis1=1, axis2=2)
    else:
        raise ValueError(f"box_vectors must be 2D or 3D, got shape {box_vectors.shape}")

    return center


def _scale_molecule_positions(
    coords: np.ndarray,
    box_vectors: np.ndarray,
    n_atoms_per_mol: int,
    scale_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scale molecular positions rigidly around the box center.

    Parameters
    ----------
    coords : np.ndarray
        Atomic coordinates with shape (n_atoms, 3) or (n_frames, n_atoms, 3).
    box_vectors : np.ndarray
        Box vectors with shape (3, 3) or (n_frames, 3, 3).
    n_atoms_per_mol : int
        Number of atoms per molecule.
    scale_factor : float
        Scaling factor (>1 expands, <1 contracts, 1.0 = no change).

    Returns
    -------
    tuple of (scaled_coords, scaled_box_vectors) with same shapes as inputs
    """
    single_frame = coords.ndim == 2
    if single_frame:
        coords = np.expand_dims(coords, axis=0)
        box_vectors = np.expand_dims(box_vectors, axis=0)

    n_frames, n_atoms = coords.shape[:2]
    n_molecules = n_atoms // n_atoms_per_mol

    # Compute molecular centers of mass
    coms = _compute_molecule_coms(coords, n_atoms_per_mol)

    # Get box centers
    box_centers = _get_box_center(box_vectors)

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


def create_scaled_dataset(
    tensor_system,
    coords: np.ndarray,
    box_vectors: np.ndarray,
    scale_factors: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Create a dataset with multiple scaled versions of the input configurations.

    Parameters
    ----------
    tensor_system : smee.TensorSystem
        The molecular system to simulate
    coords : np.ndarray
        Atomic coordinates with shape (n_atoms, 3) or (n_frames, n_atoms, 3).
    box_vectors : np.ndarray
        Box vectors with shape (3, 3) or (n_frames, 3, 3).
    scale_factors : np.ndarray
        Array of scaling factors to apply.

    Returns
    -------
    tuple of (all_coords, all_box_vectors)
        all_coords : list of np.ndarray
            List of scaled coordinates with shape (n_frames, n_atoms, 3).
        all_box_vectors : list of np.ndarray
            List of scaled box vectors with shape (n_frames, 3, 3).
    """
    all_coords = []
    all_box_vectors = []

    for scale in scale_factors:
        scaled_slices = []
        current_idx = 0
        final_scaled_box_vecs = None

        for topology, n_copy in zip(tensor_system.topologies, tensor_system.n_copies):
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
            scaled_species_coords, scaled_box_vecs = _scale_molecule_positions(
                species_coords, box_vectors, n_atoms_per_mol, float(scale)
            )

            scaled_slices.append(scaled_species_coords)
            current_idx += n_atoms_total_species

            # Keep the scaled box vectors (they are the same for all species)
            final_scaled_box_vecs = scaled_box_vecs

        # Concatenate all scaled species coordinates
        if coords.ndim == 2:
            full_scaled_coords = np.concatenate(scaled_slices, axis=0)
        else:
            full_scaled_coords = np.concatenate(scaled_slices, axis=1)

        all_coords.append(full_scaled_coords)
        all_box_vectors.append(final_scaled_box_vecs)

    return all_coords, all_box_vectors


def generate_scale_factors(scaling_config: ScalingConfig) -> np.ndarray:
    """
    Generate scale factors for creating configurations at different densities.

    Parameters
    ----------
    scaling_config : ScalingConfig
        Scaling configuration object containing the scaling ranges.

    Returns
    -------
    scale_factors : np.ndarray
        Array of scale factors.
    """
    close = np.linspace(*scaling_config.close_range)
    equilibrium = np.linspace(*scaling_config.equilibrium_range)
    long = np.linspace(*scaling_config.long_range)
    scale_factors = np.concatenate((close, equilibrium[1:], long[1:]))
    return scale_factors
