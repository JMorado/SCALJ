"""Plotting utilities for SCALJ."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_energy_vs_scale(
    scale_factors: list[float],
    energies_list: list[np.ndarray],
    output_path: Path,
    labels: list[str] | None = None,
    lims: tuple[float, float] | None = None,
):
    """
    Plot ML potential energies vs scale factor.

    Parameters
    ----------
    scale_factors : list[float]
        List of scale factors used.
    energies_list : list[np.ndarray]
        List of arrays of potential energies (kcal/mol).
    output_path : Path
        Path to save the plot.
    labels : list[str] | None
        List of labels for the legend.
    lims : tuple[float, float] | None
        Limits for the y-axis.
    """
    plt.figure(figsize=(8, 6))

    # Define colors for multiple plots
    colors = ["blue", "red", "green", "orange", "purple", "brown"]

    if labels is None:
        labels = [f"Set {i + 1}" for i in range(len(energies_list))]

    ref_normalized = None
    if len(energies_list) > 0:
        # Assuming the first one is reference
        ref_normalized = energies_list[0] - energies_list[0].min()

    for i, energies in enumerate(energies_list):
        # Subtract minimum energy to make relative energies
        energies = energies - energies.min()

        rmse_str = ""
        if ref_normalized is not None:
            rmse = np.sqrt(np.mean((energies - ref_normalized) ** 2))
            rmse_str = f" (RMSE: {rmse:.4f})"

        color = colors[i % len(colors)]
        label = f"{labels[i]}{rmse_str}"

        plt.plot(
            scale_factors,
            energies,
            marker="o",
            linestyle="-",
            color=color,
            label=label,
        )

    plt.xlabel("Scale Factor")
    plt.ylabel(r"Potential Energy [kcal.mol$^{-1}$]")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

    if lims is not None:
        plt.ylim(lims)

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
    plt.plot(epochs, energy_losses, color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Energy Loss [unitless]")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Plot force loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, force_losses, color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Force Loss [unitless]")
    plt.grid(True, linestyle="--", alpha=0.7)

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
