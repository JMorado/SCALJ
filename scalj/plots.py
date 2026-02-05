"""Plotting utilities for SCALJ."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_energy_vs_scale(
    scale_factors: list[float],
    energies: np.ndarray,
    n_molecules: int,
    output_path: Path,
):
    """
    Plot ML potential energies vs scale factor.

    Parameters
    ----------
    scale_factors : list[float]
        List of scale factors used.
    energies : np.ndarray
        Array of potential energies (kcal/mol).
    n_molecules : int
        Number of molecules for normalization.
    output_path : Path
        Path to save the plot.
    """
    plt.figure(figsize=(8, 6))

    # Normalize energies by subtracting min and dividing by n_molecules
    # This matches the notebook implementation
    normalized_energies = (energies - energies.min()) / n_molecules

    plt.plot(
        scale_factors, normalized_energies, marker="o", linestyle="-", color="blue"
    )

    plt.title("ML Potential Energies vs. Scale Factor")
    plt.xlabel("Scale Factor")
    plt.ylabel("Relative Potential Energy (kcal/mol/mol)")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Plot saved to: {output_path}")


def plot_training_losses(
    energy_losses: list[float],
    force_losses: list[float],
    output_path: Path,
):
    """
    Plot training losses over epochs.

    Parameters
    ----------
    energy_losses : list[float]
        List of energy loss values per epoch.
    force_losses : list[float]
        List of force loss values per epoch.
    output_path : Path
        Path to save the plot.
    """
    epochs = range(1, len(energy_losses) + 1)

    plt.figure(figsize=(10, 5))

    # Plot energy loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, energy_losses, label="Energy Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Energy Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    # Plot force loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, force_losses, label="Force Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Force Loss")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Plot saved to: {output_path}")


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    units: str,
    output_path: Path,
):
    """
    Plot parity plot of predicted vs reference values.

    Parameters
    ----------
    y_true : np.ndarray
        Reference values.
    y_pred : np.ndarray
        Predicted values.
    label : str
        Label for the data (e.g. "Energy", "Forces").
    units : str
        Units of the data (e.g. "kcal/mol", "kcal/mol/Å").
    output_path : Path
        Path to save the plot.
    """
    plt.figure(figsize=(6, 6))

    # Calculate statistics
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    # Plot data
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)

    # Plot diagonal line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    buffer = (max_val - min_val) * 0.05
    plt.plot(
        [min_val - buffer, max_val + buffer],
        [min_val - buffer, max_val + buffer],
        "k--",
        lw=1,
    )

    plt.xlabel(f"Reference {label} [{units}]")
    plt.ylabel(f"Predicted {label} [{units}]")
    plt.title(f"{label} Parity Plot\nMAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axis("equal")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   Plot saved to: {output_path}")
