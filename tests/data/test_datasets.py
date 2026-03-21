"""Tests for datasets."""

import datasets
import numpy as np
import pytest
import torch

from scalej.data._datasets import combine_datasets, create_dataset, create_dataset_entry


@pytest.fixture()
def single_frame_entry(water_dimer_coords, water_dimer_box):
    """A single-frame entry for a water dimer (6 atoms)."""
    forces = np.array(
        [
            [-0.10, -0.20, -0.30],
            [0.05, 0.10, 0.15],
            [0.05, 0.10, 0.15],
            [0.08, -0.12, 0.04],
            [-0.04, 0.06, -0.02],
            [-0.04, 0.06, -0.02],
        ]
    )
    return create_dataset_entry(
        id="water_dimer",
        smiles="O.O",
        coords_list=[water_dimer_coords],
        box_vectors_list=[water_dimer_box],
        energies=np.array([-9.42]),
        forces=[forces],
    )


@pytest.fixture()
def two_frame_entry(water_dimer_coords_multiframe, water_dimer_box_multiframe):
    """A two-frame entry for a water dimer (6 atoms, 2 frames)."""
    forces_frame1 = np.array(
        [
            [-0.10, -0.20, -0.30],
            [0.05, 0.10, 0.15],
            [0.05, 0.10, 0.15],
            [0.08, -0.12, 0.04],
            [-0.04, 0.06, -0.02],
            [-0.04, 0.06, -0.02],
        ]
    )
    forces_frame2 = np.array(
        [
            [-0.12, -0.22, -0.32],
            [0.06, 0.11, 0.16],
            [0.06, 0.11, 0.16],
            [0.09, -0.13, 0.05],
            [-0.05, 0.07, -0.03],
            [-0.04, 0.06, -0.02],
        ]
    )
    return create_dataset_entry(
        id="water_dimer",
        smiles="O.O",
        coords_list=[
            water_dimer_coords_multiframe[0],
            water_dimer_coords_multiframe[1],
        ],
        box_vectors_list=[water_dimer_box_multiframe[0], water_dimer_box_multiframe[1]],
        energies=np.array([-9.42, -9.38]),
        forces=[forces_frame1, forces_frame2],
    )


class TestCreateDatasetEntry:
    def test_single_frame_entry(self, single_frame_entry):
        assert set(single_frame_entry.keys()) == {
            "id",
            "smiles",
            "coords",
            "box_vectors",
            "energy",
            "forces",
        }
        assert isinstance(single_frame_entry["id"], str)
        assert isinstance(single_frame_entry["smiles"], str)
        assert isinstance(single_frame_entry["coords"], list)
        assert isinstance(single_frame_entry["box_vectors"], list)
        assert isinstance(single_frame_entry["energy"], list)
        assert isinstance(single_frame_entry["forces"], list)
        assert single_frame_entry["id"] == "water_dimer"
        assert single_frame_entry["smiles"] == "O.O"
        assert len(single_frame_entry["coords"]) == 18
        assert len(single_frame_entry["box_vectors"]) == 9
        assert len(single_frame_entry["energy"]) == 1
        assert len(single_frame_entry["forces"]) == 18
        assert single_frame_entry["energy"] == pytest.approx([-9.42])

    def test_two_frame_entry(self, two_frame_entry):
        assert set(two_frame_entry.keys()) == {
            "id",
            "smiles",
            "coords",
            "box_vectors",
            "energy",
            "forces",
        }
        assert isinstance(two_frame_entry["id"], str)
        assert isinstance(two_frame_entry["smiles"], str)
        assert isinstance(two_frame_entry["coords"], list)
        assert isinstance(two_frame_entry["box_vectors"], list)
        assert isinstance(two_frame_entry["energy"], list)
        assert isinstance(two_frame_entry["forces"], list)
        assert two_frame_entry["id"] == "water_dimer"
        assert two_frame_entry["smiles"] == "O.O"
        assert len(two_frame_entry["coords"]) == 36
        assert len(two_frame_entry["box_vectors"]) == 18
        assert len(two_frame_entry["energy"]) == 2
        assert len(two_frame_entry["forces"]) == 36
        assert two_frame_entry["energy"] == pytest.approx([-9.42, -9.38])

    def test_accepts_plain_lists(self):
        entry = create_dataset_entry(
            id="test",
            smiles="C",
            coords_list=[[[0.0, 0.0, 0.0]]],
            box_vectors_list=[[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]],
            energies=np.array([0.5]),
            forces=[np.array([[0.1, 0.2, 0.3]])],
        )
        assert len(entry["coords"]) == 3

    def test_mismatched_lengths_raises(self):
        with pytest.raises((ValueError, TypeError)):
            create_dataset_entry(
                id="bad",
                smiles="O",
                coords_list=[np.zeros((3, 3))],
                box_vectors_list=[],  # mismatched
                energies=np.array([-1.0]),
                forces=np.array([[-0.1, -0.2, -0.3]] * 3),
            )


class TestCreateDataset:
    def test_single_entry(self, single_frame_entry):
        ds = create_dataset([single_frame_entry])
        assert isinstance(ds, datasets.Dataset)
        assert len(ds) == 1
        assert set(ds.column_names) == {
            "id",
            "smiles",
            "coords",
            "box_vectors",
            "energy",
            "forces",
        }
        row = ds[0]
        assert isinstance(row["id"], str)
        assert isinstance(row["smiles"], str)
        assert isinstance(row["coords"], torch.Tensor)
        assert isinstance(row["box_vectors"], torch.Tensor)
        assert isinstance(row["energy"], torch.Tensor)
        assert isinstance(row["forces"], torch.Tensor)

        assert row["id"] == "water_dimer"
        assert row["smiles"] == "O.O"
        assert row["coords"].shape == (18,)
        assert row["box_vectors"].shape == (9,)
        assert row["forces"].shape == (18,)
        assert torch.allclose(row["energy"].float(), torch.tensor([-9.42]))

    def test_multiple_entries(self, single_frame_entry, two_frame_entry):
        ds = create_dataset([single_frame_entry, two_frame_entry])
        assert isinstance(ds, datasets.Dataset)
        assert len(ds) == 2
        assert set(ds.column_names) == {
            "id",
            "smiles",
            "coords",
            "box_vectors",
            "energy",
            "forces",
        }

        row0 = ds[0]
        assert isinstance(row0["id"], str)
        assert isinstance(row0["smiles"], str)
        assert isinstance(row0["coords"], torch.Tensor)
        assert isinstance(row0["box_vectors"], torch.Tensor)
        assert isinstance(row0["energy"], torch.Tensor)
        assert isinstance(row0["forces"], torch.Tensor)
        assert row0["id"] == "water_dimer"
        assert row0["smiles"] == "O.O"
        assert row0["coords"].shape == (18,)
        assert row0["box_vectors"].shape == (9,)
        assert row0["forces"].shape == (18,)
        assert torch.allclose(row0["energy"].float(), torch.tensor([-9.42]))

        row1 = ds[1]
        assert isinstance(row1["id"], str)
        assert isinstance(row1["smiles"], str)
        assert isinstance(row1["coords"], torch.Tensor)
        assert isinstance(row1["box_vectors"], torch.Tensor)
        assert isinstance(row1["energy"], torch.Tensor)
        assert isinstance(row1["forces"], torch.Tensor)
        assert row1["id"] == "water_dimer"
        assert row1["smiles"] == "O.O"
        assert row1["coords"].shape == (36,)
        assert row1["box_vectors"].shape == (18,)
        assert row1["forces"].shape == (36,)
        assert torch.allclose(row1["energy"].float(), torch.tensor([-9.42, -9.38]))

    def test_empty_entries(self):
        ds = create_dataset([])
        assert len(ds) == 0


class TestCombineDatasets:
    def test_combine_single_dataset(self, single_frame_entry):
        ds = create_dataset([single_frame_entry])
        combined = combine_datasets({"water": ds})
        assert isinstance(combined, datasets.Dataset)
        assert len(combined) == 1
        assert set(combined.column_names) == {
            "id",
            "smiles",
            "coords",
            "box_vectors",
            "energy",
            "forces",
        }
        row = combined[0]
        assert isinstance(row["id"], str)
        assert isinstance(row["smiles"], str)
        assert isinstance(row["coords"], torch.Tensor)
        assert isinstance(row["box_vectors"], torch.Tensor)
        assert isinstance(row["energy"], torch.Tensor)
        assert isinstance(row["forces"], torch.Tensor)
        assert row["id"] == "water_dimer"
        assert row["smiles"] == "O.O"
        assert row["coords"].shape == (18,)
        assert row["box_vectors"].shape == (9,)
        assert row["forces"].shape == (18,)
        assert torch.allclose(row["energy"], ds[0]["energy"])
        assert torch.allclose(row["coords"], ds[0]["coords"])

    def test_combine_two_datasets(self, single_frame_entry, two_frame_entry):
        ds1 = create_dataset([single_frame_entry])
        ds2 = create_dataset([two_frame_entry])
        combined = combine_datasets({"water": ds1, "dimer": ds2})
        assert isinstance(combined, datasets.Dataset)
        assert len(combined) == 2
        assert set(combined.column_names) == {
            "id",
            "smiles",
            "coords",
            "box_vectors",
            "energy",
            "forces",
        }

        row0 = combined[0]
        assert isinstance(row0["coords"], torch.Tensor)
        assert isinstance(row0["energy"], torch.Tensor)
        assert row0["id"] == "water_dimer"
        assert row0["smiles"] == "O.O"
        assert row0["coords"].shape == (18,)
        assert row0["box_vectors"].shape == (9,)
        assert row0["forces"].shape == (18,)
        assert torch.allclose(row0["energy"].float(), torch.tensor([-9.42]))

        row1 = combined[1]
        assert isinstance(row1["coords"], torch.Tensor)
        assert isinstance(row1["energy"], torch.Tensor)
        assert row1["id"] == "water_dimer"
        assert row1["smiles"] == "O.O"
        assert row1["coords"].shape == (36,)
        assert row1["box_vectors"].shape == (18,)
        assert row1["forces"].shape == (36,)
        assert torch.allclose(row1["energy"].float(), torch.tensor([-9.42, -9.38]))

    def test_combine_empty_dict(self):
        combined = combine_datasets({})
        assert isinstance(combined, datasets.Dataset)
        assert len(combined) == 0
