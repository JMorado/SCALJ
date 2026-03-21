"""Volume scaling functions."""

import numpy as np
import smee

from ..types import ScalingResult


def generate_scale_factors(
    close_range: tuple[float, float, int] = (0.75, 0.9, 5),
    equilibrium_range: tuple[float, float, int] = (0.9, 1.1, 15),
    long_range: tuple[float, float, int] = (1.1, 2.0, 12),
) -> np.ndarray:
    """
    Generate scale factors for density variation.

    Parameters
    ----------
    close_range : tuple[float, float, int]
        (start, end, n_points) for close-range scaling (compressed).
    equilibrium_range : tuple[float, float, int]
        (start, end, n_points) for equilibrium-range scaling.
    long_range : tuple[float, float, int]
        (start, end, n_points) for long-range scaling (expanded).

    Returns
    -------
    np.ndarray
        Array of scale factors spanning all regions.
    """
    close = np.linspace(*close_range)
    equilibrium = np.linspace(*equilibrium_range)
    long = np.linspace(*long_range)
    scale_factors = np.concatenate((close, equilibrium[1:], long[1:]))
    return scale_factors


def compute_molecule_coms(
    coords: np.ndarray,
    n_atoms_per_mol: int,
) -> np.ndarray:
    """
    Compute center of mass for each molecule assuming uniform atomic masses.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates with shape (n_atoms, 3) or (n_frames, n_atoms, 3).
    n_atoms_per_mol : int
        Number of atoms per molecule.

    Returns
    -------
    np.ndarray
        Centers of mass with shape (n_molecules, 3) or (n_frames, n_molecules, 3).
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


def get_box_center(box_vectors: np.ndarray) -> np.ndarray:
    """
    Compute the center of the simulation box.

    Notes
    -----
    Assumes the box is orthorhombic (i.e., box vectors are diagonal).

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


def scale_molecule_positions(
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
        Coordinates in Å with shape (n_atoms, 3) or (n_frames, n_atoms, 3).
    box_vectors : np.ndarray
        Box vectors in Å with shape (3, 3) or (n_frames, 3, 3).
    n_atoms_per_mol : int
        Number of atoms per molecule.
    scale_factor : float
        Scaling factor (< 1 compresses, > 1 expands).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (scaled_coords, scaled_box_vectors).
    """
    single_frame = coords.ndim == 2
    if single_frame:
        coords = np.expand_dims(coords, axis=0)
        box_vectors = np.expand_dims(box_vectors, axis=0)

    n_frames, n_atoms = coords.shape[:2]
    n_molecules = n_atoms // n_atoms_per_mol

    # Compute COMs and box centers.
    coms = compute_molecule_coms(coords, n_atoms_per_mol)
    box_centers = get_box_center(box_vectors)

    # Compute displacement vectors from box center to each COM.
    displacements = coms - box_centers[:, np.newaxis, :]

    # Scale displacements.
    scaled_displacements = displacements * scale_factor

    # Compute new COMs.
    new_coms = box_centers[:, np.newaxis, :] + scaled_displacements

    # Compute translation vector for each molecule.
    translations = new_coms - coms

    # Apply translations to all atoms in each molecule.
    coords_reshaped = coords.reshape(n_frames, n_molecules, n_atoms_per_mol, 3)
    translations_expanded = translations[:, :, np.newaxis, :]

    # Apply translation.
    scaled_coords = coords_reshaped + translations_expanded

    # Reshape back to original shape.
    scaled_coords = scaled_coords.reshape(n_frames, n_atoms, 3)

    # Scale box vectors.
    scaled_box_vectors = box_vectors * scale_factor

    # Remove batch dimension if input was single frame.
    if single_frame:
        scaled_coords = np.squeeze(scaled_coords, axis=0)
        scaled_box_vectors = np.squeeze(scaled_box_vectors, axis=0)

    return scaled_coords, scaled_box_vectors


def create_scaled_configurations(
    tensor_system: smee.TensorSystem,
    coords: np.ndarray,
    box_vectors: np.ndarray,
    scale_factors: np.ndarray,
) -> ScalingResult:
    """
    Create a dataset with multiple scaled versions of input configurations.

    Parameters
    ----------
    tensor_system : smee.TensorSystem
        System topology containing molecule definitions.
    coords : np.ndarray
        Coordinates in Å with shape (n_atoms, 3) or (n_frames, n_atoms, 3).
    box_vectors : np.ndarray
        Box vectors in Å with shape (3, 3) or (n_frames, 3, 3).
    scale_factors : np.ndarray
        Array of scale factors to apply.

    Returns
    -------
    ScalingResult
        Result containing scaled coordinates, box vectors, and scale factors.
    """
    all_coords = []
    all_box_vectors = []

    for scale in scale_factors:
        scaled_slices = []
        current_idx = 0

        for topology, n_copy in zip(
            tensor_system.topologies, tensor_system.n_copies, strict=True
        ):
            # We assume that the copies of each molecule are contiguous in the coords array,
            # so we can slice them out directly.
            # TODO: Check if this is always guaranteed.
            n_atoms_per_mol = len(topology.atomic_nums)
            total_n_atoms = n_atoms_per_mol * n_copy

            if coords.ndim == 2:
                species_coords = coords[current_idx : current_idx + total_n_atoms]
            else:
                species_coords = coords[:, current_idx : current_idx + total_n_atoms, :]

            # Scale this block of molecules.
            scaled_species_coords, scaled_box_vecs = scale_molecule_positions(
                species_coords, box_vectors, n_atoms_per_mol, float(scale)
            )

            scaled_slices.append(scaled_species_coords)
            current_idx += total_n_atoms

        # Concatenate all scaled species coordinates.
        if coords.ndim == 2:
            full_scaled_coords = np.concatenate(scaled_slices, axis=0)
            all_coords.append(full_scaled_coords)
            all_box_vectors.append(scaled_box_vecs)
        else:
            # Multiple frames: concatenate species and flatten frames into list.
            full_scaled_coords = np.concatenate(scaled_slices, axis=1)
            # Unpack each frame into a separate list element.
            for frame_idx in range(full_scaled_coords.shape[0]):
                all_coords.append(full_scaled_coords[frame_idx])
                all_box_vectors.append(scaled_box_vecs[frame_idx])

    # Create expanded scale factors for the result.
    if coords.ndim == 2:
        expanded_scale_factors = scale_factors
    else:
        n_frames = coords.shape[0]
        expanded_scale_factors = np.repeat(scale_factors, n_frames)

    return ScalingResult(
        coords=all_coords,
        box_vectors=all_box_vectors,
        scale_factors=expanded_scale_factors,
    )
