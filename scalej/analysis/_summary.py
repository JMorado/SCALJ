"""Interactive HTML summary tables for thermodynamic and training results."""

import base64
import io
from pathlib import Path

import matplotlib.figure
import numpy as np
import pandas as pd
from openff.toolkit import ForceField


def _smiles_to_labeled_image(
    smiles: str,
    ff: ForceField | None = None,
    handler_key: str = "vdW",
    img_size: tuple[int, int] = (400, 300),
) -> str:
    """
    Render a molecule as a PNG image, optionally with force-field atom-type labels.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    ff : ForceField | None
        OpenFF force field used to label atoms.  When *None* the molecule
        is rendered without labels.
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

    atom_labels: dict[int, str] = {}
    if ff is not None:
        labels = ff.label_molecules(off_mol.to_topology())[0]
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

        ncols = len(summary_df.columns)
        table_width = max(1800, 300 * ncols)

        tabulator = panel.widgets.Tabulator(
            summary_df,
            show_index=False,
            selectable=False,
            disabled=True,
            formatters=formatters,
            configuration={"rowHeight": row_height},
            sizing_mode="fixed",
            width=table_width,
            frozen_columns=["Run", "ID", "Mole Fractions", "Molecule"],
            page_size=n_rows,
            pagination="local",
        )

        css = (
            ".tabulator-cell { overflow: visible !important; }"
            ".tabulator-row { overflow: visible !important; }"
        )

        layout = panel.Column(
            panel.pane.HTML(f"<style>{css}</style>"),
            panel.pane.Markdown("# Mixture Summary"),
            tabulator,
            sizing_mode="fixed",
            width=table_width,
        )

        out = str(output_file)
        layout.save(out, title="Mixture Summary", embed=True)

        _patch_html_scroll(Path(out))
        print(f"Summary saved to {output_file}")


def _plot_to_base64_img(
    fig: matplotlib.figure.Figure,
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


def _patch_html_scroll(path: Path) -> None:
    """Post-process a Panel-saved HTML file to enable horizontal scrolling.

    BokehJS hard-codes ``overflow: hidden`` on ``.bk`` elements and sets the
    root div to a fixed pixel width.  We override every ``overflow:hidden``
    occurrence that BokehJS inlines in ``<style>`` blocks, and force the root
    and body to allow horizontal overflow.
    """
    import re

    html = path.read_text(encoding="utf-8")

    # 1. Remove every `overflow: hidden` / `overflow:hidden` that BokehJS injects
    #    inside <style> blocks (not inside attribute values).
    html = re.sub(r"overflow\s*:\s*hidden\s*(!important)?", "overflow:visible", html)

    # 2. Inject our own rules at the end of <head> to ensure the page scrolls.
    patch = (
        "<style>"
        "html,body{overflow-x:auto!important;margin:0;padding:8px;}"
        ".bk-root,.bk{overflow:visible!important;}"
        "</style>"
    )
    html = html.replace("</head>", patch + "</head>", 1)

    path.write_text(html, encoding="utf-8")


class TrainingSummary:
    """
    Build and export interactive HTML summary tables for training results.

    Parameters
    ----------
    pre_perturbation_parquet : str | Path
        Path to ``pre_perturbation_evaluations.parquet``.
    perturbed_parquet : str | Path
        Path to ``perturbed_evaluations.parquet``.
    final_parquet : str | Path
        Path to ``final_evaluations.parquet``.
    df : pd.DataFrame | None
        Optional DataFrame with per-mixture component info.  Must contain a
        ``"mixture_id"`` column (or use ``mixture_id`` as the index) plus
        ``"Component 1"``, ``"Component 2"``, ``"Mole Fraction 1"``, and
        ``"Mole Fraction 2"`` columns.
    meta_json : str | Path | None
        Path to a ``combined_dataset_meta.json`` file.  When given the
        component names and mole fractions are extracted automatically and
        *df* is ignored.
    labels : tuple[str, str, str] | None
        Display labels for the (pre_perturbation, perturbed, final) stages.
        Defaults to ``("Pre-perturbation", "Perturbed", "Final")`` if *None*.

    Examples
    --------
    >>> ts = TrainingSummary(
    ...     "pre_perturbation_evaluations.parquet",
    ...     "perturbed_evaluations.parquet",
    ...     "final_evaluations.parquet",
    ...     meta_json="combined_dataset_meta.json",
    ... )
    >>> summary_df = ts.build_summary_df()
    >>> ts.save_summary_html(summary_df, "training_summary.html")
    """

    @staticmethod
    def _meta_json_to_df(path: str | Path) -> pd.DataFrame:
        """Parse a ``combined_dataset_meta.json`` into a metadata DataFrame."""
        import json as _json

        with open(path) as fh:
            meta = _json.load(fh)

        rows: list[dict] = []
        for run in meta["runs"]:
            comps = run["components"]
            row: dict = {"mixture_id": run["name"]}
            for i, comp in enumerate(comps, start=1):
                row[f"Component {i}"] = comp.get("smiles", "")
                row[f"Mole Fraction {i}"] = comp.get("mole_fraction", 1.0)
            rows.append(row)
        return pd.DataFrame(rows)

    def __init__(
        self,
        pre_perturbation_parquet: str | Path,
        perturbed_parquet: str | Path,
        final_parquet: str | Path,
        df: pd.DataFrame | None = None,
        meta_json: str | Path | None = None,
        labels: tuple[str, str, str] | None = None,
    ) -> None:
        self._df_pre = pd.read_parquet(pre_perturbation_parquet)
        self._df_pert = pd.read_parquet(perturbed_parquet)
        self._df_final = pd.read_parquet(final_parquet)
        if meta_json is not None:
            self._meta_df = self._meta_json_to_df(meta_json)
        else:
            self._meta_df = df
        self._labels = labels or ("Pre-perturbation", "Perturbed", "Final")

    def _get_mixture_ids(self) -> list[str]:
        """Return sorted unique mixture IDs from the final-stage parquet."""
        return sorted(self._df_final["mixture_id"].unique().tolist())

    def _extract_sample(self, mix_id: str) -> dict[str, np.ndarray]:
        """Extract energy and force arrays for *mix_id* from all three stages."""

        def _sub(df: pd.DataFrame) -> pd.DataFrame:
            return df[df["mixture_id"] == mix_id].sort_values("scale_factor")

        def _flat_forces(series: pd.Series) -> np.ndarray:
            return np.concatenate(
                [np.asarray(f, dtype=float).reshape(-1) for f in series]
            )

        sub_pre = _sub(self._df_pre)
        sub_pert = _sub(self._df_pert)
        sub_final = _sub(self._df_final)

        return {
            "scale_factor": sub_final["scale_factor"].to_numpy(dtype=float),
            "energy_ref": sub_final["energy_ref"].to_numpy(dtype=float),
            "energy_pred_pre": sub_pre["energy_pred"].to_numpy(dtype=float),
            "energy_pred_pert": sub_pert["energy_pred"].to_numpy(dtype=float),
            "energy_pred_final": sub_final["energy_pred"].to_numpy(dtype=float),
            "forces_ref": _flat_forces(sub_final["forces_ref"]),
            "forces_pred_pre": _flat_forces(sub_pre["forces_pred"]),
            "forces_pred_pert": _flat_forces(sub_pert["forces_pred"]),
            "forces_pred_final": _flat_forces(sub_final["forces_pred"]),
        }

    @staticmethod
    def _compute_metrics(data: dict[str, np.ndarray]) -> dict[str, float]:
        """Compute energy and force RMSE/MAE for all three stages."""
        metrics: dict[str, float] = {}
        e_ref = data["energy_ref"]
        f_ref = data["forces_ref"]
        for stage_key, label in [
            ("pre", "PrePert"),
            ("pert", "Pert"),
            ("final", "Final"),
        ]:
            e_pred = data[f"energy_pred_{stage_key}"]
            f_pred = data[f"forces_pred_{stage_key}"]
            metrics[f"E RMSE {label}"] = float(np.sqrt(np.mean((e_pred - e_ref) ** 2)))
            metrics[f"E MAE {label}"] = float(np.mean(np.abs(e_pred - e_ref)))
            metrics[f"F RMSE {label}"] = float(np.sqrt(np.mean((f_pred - f_ref) ** 2)))
            metrics[f"F MAE {label}"] = float(np.mean(np.abs(f_pred - f_ref)))
        return metrics

    def _render_energy_plot(
        self,
        data: dict[str, np.ndarray],
        plot_size: tuple[int, int],
        y_lim: tuple[float, float] | None,
    ) -> str:
        """Render an energy-vs-scale-factor plot as a base64 ``<img>`` tag."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lbl_pre, lbl_pert, lbl_final = self._labels
        x = data["scale_factor"]
        e_ref = data["energy_ref"]

        def _e_rmse(key: str) -> str:
            return f"{np.sqrt(np.mean((data[key] - e_ref) ** 2)):.2e}"

        fig, ax = plt.subplots(figsize=(plot_size[0] / 100, plot_size[1] / 100))
        ax.plot(
            x, e_ref, label="Reference (MLP)", color="black", linewidth=1.5, zorder=3
        )
        ax.plot(
            x,
            data["energy_pred_pre"],
            label=f"{lbl_pre} (RMSE={_e_rmse('energy_pred_pre')})",
            color="blue",
            linestyle="--",
            linewidth=1.5,
            zorder=1,
        )
        ax.plot(
            x,
            data["energy_pred_pert"],
            label=f"{lbl_pert} (RMSE={_e_rmse('energy_pred_pert')})",
            color="gray",
            linestyle="--",
            linewidth=1.5,
            zorder=1,
        )
        ax.plot(
            x,
            data["energy_pred_final"],
            label=f"{lbl_final} (RMSE={_e_rmse('energy_pred_final')})",
            color="red",
            linewidth=1.5,
            zorder=2,
        )
        ax.set_ylabel("Energy [kcal/mol]")
        ax.set_xlabel("Scale Factor")
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.legend(fontsize="small", loc="upper left")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        return _plot_to_base64_img(fig, width=plot_size[0], height=plot_size[1])

    def _render_force_scatter(
        self,
        data: dict[str, np.ndarray],
        plot_size: tuple[int, int],
    ) -> str:
        """Render an atomic force scatter plot as a base64 ``<img>`` tag."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lbl_pre, lbl_pert, lbl_final = self._labels
        f_ref = data["forces_ref"]
        alpha = min(0.3, max(0.05, 500 / max(len(f_ref), 1)))

        def _f_rmse(key: str) -> str:
            return f"{np.sqrt(np.mean((data[key] - f_ref) ** 2)):.2e}"

        fig, ax = plt.subplots(figsize=(plot_size[0] / 100, plot_size[1] / 100))
        ax.scatter(
            f_ref,
            data["forces_pred_pre"],
            label=f"{lbl_pre} (RMSE={_f_rmse('forces_pred_pre')})",
            color="blue",
            s=2,
            alpha=alpha,
            rasterized=True,
        )
        ax.scatter(
            f_ref,
            data["forces_pred_pert"],
            label=f"{lbl_pert} (RMSE={_f_rmse('forces_pred_pert')})",
            color="gray",
            s=2,
            alpha=alpha,
            rasterized=True,
        )
        ax.scatter(
            f_ref,
            data["forces_pred_final"],
            label=f"{lbl_final} (RMSE={_f_rmse('forces_pred_final')})",
            color="red",
            s=2,
            alpha=alpha,
            rasterized=True,
        )
        lim = float(np.percentile(np.abs(f_ref), 99)) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8, label="y = x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Force Ref [kcal/mol/\u00c5]")
        ax.set_ylabel("Force Pred [kcal/mol/\u00c5]")
        ax.legend(fontsize="small", loc="upper left")
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
        Build a summary DataFrame with per-mixture energy and force plots.

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
            Width and height of the inline plots in pixels.
        y_lim : tuple[float, float] | None
            Y-axis limits for energy plots.  *None* for auto-scaling.

        Returns
        -------
        pd.DataFrame
            Summary table with columns for mixture info, molecule image,
            energy plot, force scatter plot, and error metrics.
        """
        ff = None
        if forcefield_path is not None:
            from openff.toolkit import ForceField

            ff = ForceField(forcefield_path, load_plugins=True)

        # Build metadata lookup keyed by mixture_id
        meta_lookup: dict = {}
        if self._meta_df is not None:
            meta_df = self._meta_df.copy()
            if "mixture_id" in meta_df.columns:
                meta_df = meta_df.drop_duplicates(subset="mixture_id").set_index(
                    "mixture_id"
                )
            elif meta_df.index.duplicated().any():
                meta_df = meta_df[~meta_df.index.duplicated(keep="first")]
            meta_lookup = meta_df.to_dict("index")

        rows: list[dict] = []
        for mix_id in self._get_mixture_ids():
            data = self._extract_sample(mix_id)
            metrics = self._compute_metrics(data)

            meta = meta_lookup.get(mix_id, {})
            comp1 = meta.get("Component 1", "")
            comp2 = meta.get("Component 2", "")
            x1 = meta.get("Mole Fraction 1", None)
            x2 = meta.get("Mole Fraction 2", None)
            entry_id = meta.get("Id", mix_id)

            # Molecule column: image + SMILES + mole fraction per component
            mol_html = ""
            if comp1 or comp2:
                components = [(comp1, x1, "Comp 1"), (comp2, x2, "Comp 2")]
                parts: list[str] = []
                for smi, xfrac, label in components:
                    if pd.notna(smi) and smi:
                        xfrac_str = f"x = {xfrac:.4f}" if pd.notna(xfrac) else ""
                        img_tag = _smiles_to_labeled_image(
                            str(smi),
                            ff=ff,
                            handler_key=handler_key,
                            img_size=mol_img_size,
                        )
                        header = f"<b>{label}</b>"
                        if xfrac_str:
                            header += f" ({xfrac_str})"
                        parts.append(f"{header}<br><code>{smi}</code><br>{img_tag}")
                mol_html = "<br>".join(parts)

            row: dict = {
                "Run": mix_id,
                "ID": entry_id,
                "Molecule": mol_html,
                "Energy Plot": self._render_energy_plot(data, plot_size, y_lim),
                "Force Plot": self._render_force_scatter(data, plot_size),
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
        html_cols = {"Molecule", "Energy Plot", "Force Plot"}
        string_cols = {"Run", "ID"}

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

        ncols = len(summary_df.columns)
        table_width = max(1800, 300 * ncols)

        tabulator = panel.widgets.Tabulator(
            summary_df,
            show_index=False,
            selectable=False,
            disabled=True,
            formatters=formatters,
            configuration={"rowHeight": row_height},
            sizing_mode="fixed",
            width=table_width,
            frozen_columns=["Run", "ID"],
            page_size=n_rows,
            pagination="local",
        )

        css = (
            ".tabulator-cell { overflow: visible !important; }"
            ".tabulator-row { overflow: visible !important; }"
        )

        layout = panel.Column(
            panel.pane.HTML(f"<style>{css}</style>"),
            panel.pane.Markdown(f"# {title}"),
            tabulator,
            sizing_mode="fixed",
            width=table_width,
        )

        out = str(output_file)
        layout.save(out, title=title, embed=True)

        _patch_html_scroll(Path(out))
        print(f"Summary saved to {output_file}")
