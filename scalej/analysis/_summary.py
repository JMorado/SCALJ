"""Interactive HTML summary tables for thermodynamic and training results."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from openff.toolkit import ForceField

    from ..models import PredictionResult


def _smiles_to_labeled_image(
    smiles: str,
    ff: ForceField,
    handler_key: str = "vdW",
    img_size: tuple[int, int] = (400, 300),
) -> str:
    """
    Render a molecule as a PNG image with force field atom-type labels.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    ff : ForceField
        OpenFF force field used to label atoms.
    handler_key : str
        The parameter handler key to use for labeling (e.g. ``"vdW"``).
    img_size : tuple[int, int]
        Width and height of the output image in pixels.

    Returns
    -------
    str
        An HTML ``<img>`` tag containing the base64-encoded PNG.
    """
    from openff.toolkit import Molecule as OFFMolecule
    from rdkit.Chem import AllChem, Draw

    off_mol = OFFMolecule.from_smiles(smiles, allow_undefined_stereo=True)
    labels = ff.label_molecules(off_mol.to_topology())[0]

    atom_labels: dict[int, str] = {}
    if handler_key in labels:
        for atom_indices, param in labels[handler_key].items():
            for idx in atom_indices:
                atom_labels[idx] = param.id

    rdmol = off_mol.to_rdkit()
    AllChem.Compute2DCoords(rdmol)

    for atom in rdmol.GetAtoms():
        label = atom_labels.get(atom.GetIdx(), "")
        if label:
            atom.SetProp("atomNote", label)

    drawer = Draw.MolDraw2DCairo(img_size[0], img_size[1])
    drawer.drawOptions().annotationFontScale = 0.6
    drawer.DrawMolecule(rdmol)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" width="{img_size[0]}" height="{img_size[1]}">'


class ThermodynamicSummary:
    """
    Build and export interactive HTML summary tables for thermodynamic benchmarks.


    Parameters
    ----------
    df_main : pd.DataFrame
        Reference dataset with columns ``"Id"``, ``"Component 1"``,
        ``"Component 2"``, ``"Mole Fraction 1"``, ``"Mole Fraction 2"``,
        ``"Density Value (g / ml)"``, and
        ``"EnthalpyOfMixing Value (kJ / mol)"``.
    df_density : pd.DataFrame
        Predicted densities with columns ``"run"`` and ``"density"`` (g/mL).
    df_hmix : pd.DataFrame
        Predicted enthalpies of mixing with columns ``"run"`` and ``"hvap"``
        (kcal/mol — converted internally to kJ/mol).

    Examples
    --------
    >>> summary = ThermodynamicSummary(df, df_density, df_hmix)
    >>> summary_df = summary.build_summary_df("openff-2.3.0.offxml")
    >>> summary.save_summary_html(summary_df, "summary.html")
    """

    def __init__(
        self,
        df_main: pd.DataFrame,
        df_density: pd.DataFrame,
        df_hmix: pd.DataFrame,
    ) -> None:
        self._df_main = df_main
        self._df_density = df_density
        self._df_hmix = df_hmix

    def _prepare_density(
        self,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Align and filter density predictions against reference data.

        Returns
        -------
        run_indices, y_true, y_pred : pd.Series
        """
        run_idx = self._df_density["run"].astype(int)
        y_pred = self._df_density["density"].reset_index(drop=True)
        y_true = self._df_main.loc[run_idx, "Density Value (g / ml)"].reset_index(
            drop=True
        )

        mask = ~y_true.isna() & ~y_pred.isna()
        run_idx = run_idx.reset_index(drop=True)[mask].reset_index(drop=True)
        y_true = y_true[mask].reset_index(drop=True)
        y_pred = y_pred[mask].reset_index(drop=True)
        return run_idx, y_true, y_pred

    def _prepare_hmix(
        self,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Align and filter enthalpy of mixing predictions against reference data.

        Returns
        -------
        run_indices, y_true, y_pred : pd.Series
            Predicted values are converted from kcal/mol to kJ/mol.
        """
        run_idx = self._df_hmix["run"].astype(int)
        y_pred = self._df_hmix["hvap"].reset_index(drop=True) * 4.184  # kcal → kJ
        y_true = self._df_main.loc[
            run_idx, "EnthalpyOfMixing Value (kJ / mol)"
        ].reset_index(drop=True)

        mask = ~y_true.isna() & ~y_pred.isna()
        run_idx = run_idx.reset_index(drop=True)[mask].reset_index(drop=True)
        y_true = y_true[mask].reset_index(drop=True)
        y_pred = y_pred[mask].reset_index(drop=True)
        return run_idx, y_true, y_pred

    def build_summary_df(
        self,
        forcefield_path: str = "openff-2.2.1.offxml",
        handler_key: str = "vdW",
        img_size: tuple[int, int] = (400, 300),
    ) -> pd.DataFrame:
        """
        Build a summary DataFrame with molecule images and property data.

        Parameters
        ----------
        forcefield_path : str
            Path or name of the OpenFF force field used for atom-type labeling.
        handler_key : str
            The parameter handler key used for atom labels (e.g.  ``"vdW"``).
        img_size : tuple[int, int]
            Width and height of molecule images in pixels.

        Returns
        -------
        pd.DataFrame
            Summary table with columns: Run, ID, Mole Fractions, Molecule,
            Density Pred/Exp/Error, and Hmix Pred/Exp/Error.
        """
        from openff.toolkit import ForceField

        ff = ForceField(forcefield_path, load_plugins=True)

        run_idx_d, y_true_d, y_pred_d = self._prepare_density()
        run_idx_h, y_true_h, y_pred_h = self._prepare_hmix()

        all_runs = sorted(set(run_idx_d.tolist()) | set(run_idx_h.tolist()))

        rows: list[dict] = []
        for run in all_runs:
            entry = self._df_main.iloc[run]
            entry_id = entry.get("Id", run)
            comp1 = entry.get("Component 1", "")
            comp2 = entry.get("Component 2", "")
            x1 = entry.get("Mole Fraction 1", None)
            x2 = entry.get("Mole Fraction 2", None)

            # Build molecule images with mole fraction labels
            components = [
                (comp1, x1, "Comp 1"),
                (comp2, x2, "Comp 2"),
            ]
            images: list[str] = []
            for smi, xfrac, label in components:
                if pd.notna(smi) and smi:
                    xfrac_str = f" (x={xfrac:.4f})" if pd.notna(xfrac) else ""
                    img_tag = _smiles_to_labeled_image(
                        str(smi), ff, handler_key=handler_key, img_size=img_size
                    )
                    images.append(f"<b>{label}{xfrac_str}</b><br>{img_tag}")
            mol_html = "<br>".join(images)

            # Composition string
            comp_parts: list[str] = []
            if pd.notna(x1) and pd.notna(comp1) and comp1:
                comp_parts.append(f"{x1:.4f}")
            if pd.notna(x2) and pd.notna(comp2) and comp2:
                comp_parts.append(f"{x2:.4f}")
            composition = " / ".join(comp_parts)

            # Density
            d_mask = run_idx_d == run
            dens_pred = float(y_pred_d[d_mask].iloc[0]) if d_mask.any() else None
            dens_true = float(y_true_d[d_mask].iloc[0]) if d_mask.any() else None
            dens_err = (
                abs(dens_pred - dens_true)
                if dens_pred is not None and dens_true is not None
                else None
            )

            # Hmix
            h_mask = run_idx_h == run
            hmix_pred = float(y_pred_h[h_mask].iloc[0]) if h_mask.any() else None
            hmix_true = float(y_true_h[h_mask].iloc[0]) if h_mask.any() else None
            hmix_err = (
                abs(hmix_pred - hmix_true)
                if hmix_pred is not None and hmix_true is not None
                else None
            )

            rows.append(
                {
                    "Run": f"run_{run:04d}",
                    "ID": entry_id,
                    "Mole Fractions": composition,
                    "Molecule": mol_html,
                    "Density Pred (g/mL)": dens_pred,
                    "Density Exp (g/mL)": dens_true,
                    "Density Error (g/mL)": dens_err,
                    "Hmix Pred (kJ/mol)": hmix_pred,
                    "Hmix Exp (kJ/mol)": hmix_true,
                    "Hmix Error (kJ/mol)": hmix_err,
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def save_summary_html(
        summary_df: pd.DataFrame,
        output_file: str | Path = "summary_table.html",
        page_size: int = 25,
    ) -> None:
        """Save a summary DataFrame as an interactive HTML table.

        Parameters
        ----------
        summary_df : pd.DataFrame
            DataFrame produced by :meth:`build_summary_df`.
        output_file : str | Path
            Path of the HTML file to write.
        page_size : int | None
            Number of rows per page.  If *None* the page size is
            estimated automatically from the row height.
        """
        import bokeh.models.widgets.tables
        import panel

        number_fmt = bokeh.models.widgets.tables.NumberFormatter(format="0.0000")
        formatters: dict = {}
        for col in summary_df.columns:
            if col == "Molecule":
                formatters[col] = "html"
            elif col not in ("Run", "ID", "Mole Fractions"):
                formatters[col] = number_fmt

        # Scale row height based on the maximum number of molecule images
        max_imgs = summary_df["Molecule"].apply(lambda h: h.count("<img")).max()
        row_height = max(320, 340 * max_imgs)
        n_rows = page_size if page_size is not None else max(3, 1200 // row_height)

        tabulator = panel.widgets.Tabulator(
            summary_df,
            show_index=False,
            selectable=False,
            disabled=True,
            formatters=formatters,
            configuration={"rowHeight": row_height},
            sizing_mode="stretch_width",
            frozen_columns=["Run", "ID", "Mole Fractions", "Molecule"],
            page_size=n_rows,
            pagination="local",
        )

        css = (
            ".tabulator-cell { overflow: visible !important; }"
            ".tabulator-row { overflow: visible !important; }"
            ".tabulator { overflow-x: auto !important; }"
        )

        layout = panel.Column(
            panel.pane.HTML(f"<style>{css}</style>"),
            panel.pane.Markdown("# Mixture Summary"),
            tabulator,
        )
        layout.save(str(output_file), title="Mixture Summary", embed=True)
        print(f"Summary saved to {output_file}")


def _plot_to_base64_img(
    fig: "matplotlib.figure.Figure",
    width: int = 600,
    height: int = 350,
) -> str:
    """Render a matplotlib figure as a base64 ``<img>`` tag and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to render.
    width : int
        Display width in the HTML table.
    height : int
        Display height in the HTML table.

    Returns
    -------
    str
        An HTML ``<img>`` tag with the PNG data.
    """
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" width="{width}" height="{height}">'


class TrainingSummary:
    """
    Build and export interactive HTML summary tables for training results.


    Parameters
    ----------
    predictions : PredictionResult
        Predictions from the **trained** force field.
    initial_predictions : PredictionResult
        Predictions from the initial (perturbed) force field.
    initial_predictions_openff : PredictionResult
        Predictions from the unperturbed OpenFF force field.
    scale_factors : np.ndarray
        Scale factors with shape ``(num_samples, num_points)``.
    df : pd.DataFrame
        Reference dataset with ``"Id"``, ``"Component 1"``,
        ``"Component 2"``, ``"Mole Fraction 1"``, and
        ``"Mole Fraction 2"`` columns.
    labels : tuple[str, str, str] | None
        Display labels for (reference, trained, initial_perturbed,
        initial_openff) curves. Defaults are provided if *None*.

    Examples
    --------
    >>> ts = TrainingSummary(
    ...     predictions, initial_predictions, initial_predictions_openff,
    ...     scale_factors, df,
    ... )
    >>> summary_df = ts.build_summary_df()
    >>> ts.save_summary_html(summary_df, "training_summary.html")
    """

    def __init__(
        self,
        predictions: PredictionResult,
        initial_predictions: PredictionResult,
        initial_predictions_openff: PredictionResult,
        scale_factors: np.ndarray,
        df: pd.DataFrame,
        labels: tuple[str, str, str, str] | None = None,
    ) -> None:
        self._predictions = predictions
        self._initial_predictions = initial_predictions
        self._initial_predictions_openff = initial_predictions_openff
        self._scale_factors = scale_factors
        self._df = df
        self._labels = labels or (
            "Reference (MLP)",
            "Trained",
            "Initial (perturbed)",
            "Initial (unperturbed)",
        )

    def _extract_sample(
        self, i: int, offset_ptr: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Extract energy curves for sample *i*.

        Returns
        -------
        x, y_ref, y_pred, y_init, y_init_openff, new_offset_ptr
        """
        mask = self._predictions.mask_idxs[i].detach().cpu().numpy()
        n = len(mask)

        x = self._scale_factors[i, mask]
        sl = slice(offset_ptr, offset_ptr + n)

        y_ref = self._predictions.energy_ref[sl].detach().cpu().numpy()
        y_pred = self._predictions.energy_pred[sl, 0].detach().cpu().numpy()
        y_init = self._initial_predictions.energy_pred[sl, 0].detach().cpu().numpy()
        y_init_off = (
            self._initial_predictions_openff.energy_pred[sl, 0].detach().cpu().numpy()
        )

        return x, y_ref, y_pred, y_init, y_init_off, offset_ptr + n

    @staticmethod
    def _compute_sample_metrics(
        y_ref: np.ndarray,
        y_pred: np.ndarray,
        y_init: np.ndarray,
        y_init_openff: np.ndarray,
    ) -> dict[str, float]:
        """Compute RMSE, MAE, and offset for a single sample."""
        return {
            "RMSE Pred": float(np.sqrt(np.mean((y_pred - y_ref) ** 2))),
            "RMSE Init": float(np.sqrt(np.mean((y_init - y_ref) ** 2))),
            "RMSE OpenFF": float(np.sqrt(np.mean((y_init_openff - y_ref) ** 2))),
            "MAE Pred": float(np.mean(np.abs(y_pred - y_ref))),
            "MAE Init": float(np.mean(np.abs(y_init - y_ref))),
            "MAE OpenFF": float(np.mean(np.abs(y_init_openff - y_ref))),
            "Offset Pred": float(np.mean(y_pred - y_ref)),
            "Offset Init": float(np.mean(y_init - y_ref)),
            "Offset OpenFF": float(np.mean(y_init_openff - y_ref)),
        }

    def _render_plot(
        self,
        x: np.ndarray,
        y_ref: np.ndarray,
        y_pred: np.ndarray,
        y_init: np.ndarray,
        y_init_openff: np.ndarray,
        plot_size: tuple[int, int],
        y_lim: tuple[float, float] | None,
    ) -> str:
        """
        Render an energy-vs-scale-factor plot as a base64 ``<img>`` tag."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lbl_ref, lbl_pred, lbl_init, lbl_off = self._labels

        fig, ax = plt.subplots(figsize=(plot_size[0] / 100, plot_size[1] / 100))
        ax.plot(x, y_ref, label=lbl_ref, color="black", linewidth=1.5, zorder=2)
        ax.plot(x, y_pred, label=lbl_pred, color="red", linewidth=1.5, zorder=3)
        ax.plot(
            x,
            y_init,
            label=lbl_init,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            zorder=1,
        )
        ax.plot(
            x,
            y_init_openff,
            label=lbl_off,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            zorder=1,
        )
        ax.set_ylabel("Energy [kcal/mol]")
        ax.set_xlabel("Scale Factor")
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.legend(fontsize="small")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        return _plot_to_base64_img(fig, width=plot_size[0], height=plot_size[1])

    def build_summary_df(
        self,
        forcefield_path: str | None = None,
        handler_key: str = "vdW",
        mol_img_size: tuple[int, int] = (400, 300),
        plot_size: tuple[int, int] = (600, 350),
        y_lim: tuple[float, float] | None = (-30, 30),
    ) -> pd.DataFrame:
        """
        Build a summary DataFrame with per-mixture plots and metrics.

        Parameters
        ----------
        forcefield_path : str | None
            If given, renders labeled molecule images using the specified
            OpenFF force field. Set to *None* to skip molecule rendering.
        handler_key : str
            The parameter handler key for atom labels (e.g. ``"vdW"``).
        mol_img_size : tuple[int, int]
            Width and height of molecule images in pixels.
        plot_size : tuple[int, int]
            Width and height of the inline energy plot in pixels.
        y_lim : tuple[float, float] | None
            Y-axis limits for energy plots.  *None* for auto-scaling.

        Returns
        -------
        pd.DataFrame
            Summary table with columns for mixture info, molecule image,
            energy plot, and error metrics.
        """
        ff = None
        if forcefield_path is not None:
            from openff.toolkit import ForceField

            ff = ForceField(forcefield_path, load_plugins=True)

        num_samples = len(self._predictions.mask_idxs)
        offset_ptr = 0
        rows: list[dict] = []

        for i in range(num_samples):
            x, y_ref, y_pred, y_init, y_init_off, offset_ptr = self._extract_sample(
                i, offset_ptr
            )
            metrics = self._compute_sample_metrics(y_ref, y_pred, y_init, y_init_off)

            entry = self._df.iloc[i]
            entry_id = entry.get("Id", i)
            comp1 = entry.get("Component 1", "")
            comp2 = entry.get("Component 2", "")
            x1 = entry.get("Mole Fraction 1", None)
            x2 = entry.get("Mole Fraction 2", None)

            # Molecule images
            mol_html = ""
            if ff is not None:
                components = [(comp1, x1, "Comp 1"), (comp2, x2, "Comp 2")]
                images: list[str] = []
                for smi, xfrac, label in components:
                    if pd.notna(smi) and smi:
                        xfrac_str = f" (x={xfrac:.4f})" if pd.notna(xfrac) else ""
                        img_tag = _smiles_to_labeled_image(
                            str(smi),
                            ff,
                            handler_key=handler_key,
                            img_size=mol_img_size,
                        )
                        images.append(f"<b>{label}{xfrac_str}</b><br>{img_tag}")
                mol_html = "<br>".join(images)

            # Mole fractions string
            frac_parts: list[str] = []
            if pd.notna(x1):
                frac_parts.append(f"{x1:.4f}")
            if pd.notna(x2):
                frac_parts.append(f"{x2:.4f}")
            fractions = " / ".join(frac_parts)

            # Inline plot
            plot_html = self._render_plot(
                x,
                y_ref,
                y_pred,
                y_init,
                y_init_off,
                plot_size=plot_size,
                y_lim=y_lim,
            )

            row: dict = {
                "Run": f"run_{i:04d}",
                "ID": entry_id,
                "Mole Fractions": fractions,
                "Molecule": mol_html,
                "Energy Plot": plot_html,
            }
            row.update(metrics)
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def save_summary_html(
        summary_df: pd.DataFrame,
        output_file: str | Path = "training_summary.html",
        title: str = "Training Summary",
        page_size: int | None = 25,
    ) -> None:
        """
        Save a training summary DataFrame as an interactive HTML table.

        Parameters
        ----------
        summary_df : pd.DataFrame
            DataFrame produced by :meth:`build_summary_df`.
        output_file : str | Path
            Path of the HTML file to write.
        title : str
            Title shown in the HTML page.
        page_size : int | None
            Number of rows per page.  If *None* the page size is
            estimated automatically from the row height.
        """
        import bokeh.models.widgets.tables
        import panel

        number_fmt = bokeh.models.widgets.tables.NumberFormatter(format="0.0000")
        html_cols = {"Molecule", "Energy Plot"}
        string_cols = {"Run", "ID", "Mole Fractions"}

        formatters: dict = {}
        for col in summary_df.columns:
            if col in html_cols:
                formatters[col] = "html"
            elif col not in string_cols:
                formatters[col] = number_fmt

        # Row height: accommodate molecule images + plot
        max_imgs = 0
        if "Molecule" in summary_df.columns:
            max_imgs = (
                summary_df["Molecule"]
                .apply(lambda h: h.count("<img") if isinstance(h, str) else 0)
                .max()
            )
        row_height = max(380, 340 * max(1, max_imgs))
        n_rows = page_size if page_size is not None else max(3, 1500 // row_height)

        tabulator = panel.widgets.Tabulator(
            summary_df,
            show_index=False,
            selectable=False,
            disabled=True,
            formatters=formatters,
            configuration={"rowHeight": row_height},
            sizing_mode="stretch_width",
            frozen_columns=["Run", "ID", "Mole Fractions"],
            page_size=n_rows,
            pagination="local",
        )

        css = (
            ".tabulator-cell { overflow: visible !important; }"
            ".tabulator-row { overflow: visible !important; }"
            ".tabulator { overflow-x: auto !important; }"
        )

        layout = panel.Column(
            panel.pane.HTML(f"<style>{css}</style>"),
            panel.pane.Markdown(f"# {title}"),
            tabulator,
        )
        layout.save(str(output_file), title=title, embed=True)
        print(f"Summary saved to {output_file}")
