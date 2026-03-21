"""Tests for ML potential functions."""

import numpy as np
import openmm.app
import pytest

from scalej.simulation._mlp import (
    ase_atoms_from_tensor_system,
    compute_mlp_energies_forces,
    setup_mlp_simulation,
)
from scalej.types import EnergyForceResult


class TestAseAtomsFromTensorSystem:
    def test_single_component(self, water_system):
        import ase

        tensor_system, _, _ = water_system
        atoms = ase_atoms_from_tensor_system(tensor_system)
        assert isinstance(atoms, ase.Atoms)
        assert len(atoms) == 6
        assert atoms.get_atomic_numbers().tolist() == [8, 1, 1, 8, 1, 1]
        assert atoms.positions.shape == (6, 3)

    def test_multi_component(self, water_methane_system):
        tensor_system, _, _ = water_methane_system
        atoms = ase_atoms_from_tensor_system(tensor_system)
        assert len(atoms) == 21
        # fmt: off
        assert atoms.get_atomic_numbers().tolist() == (
            [8, 1, 1] * 2          # water
            + [6, 1, 1, 1, 1] * 3  # methane
        )
        # fmt: on
        assert atoms.positions.shape == (21, 3)


class TestSetupMlpSimulation:
    def test_setup_simulation(self, ani2x_simulation, water_system):
        assert isinstance(ani2x_simulation, openmm.app.Simulation)
        tensor_system, _, _ = water_system
        n_atoms = sum(
            t.n_atoms * n
            for t, n in zip(
                tensor_system.topologies, tensor_system.n_copies, strict=True
            )
        )
        assert ani2x_simulation.system.getNumParticles() == n_atoms


class TestComputeMlpEnergiesForces:
    def test_single_frame(self, ani2x_simulation, initial_coords_box_angstrom):
        coords, box = initial_coords_box_angstrom
        result = compute_mlp_energies_forces(
            ani2x_simulation, [coords], [box], show_progress=False
        )
        assert isinstance(result, EnergyForceResult)
        assert isinstance(result.energies, np.ndarray)
        assert isinstance(result.forces, np.ndarray)
        assert result.energies.shape == (1,)
        assert result.forces.shape == (1, 6, 3)

        assert result.energies == pytest.approx(-95868.51804423)
        assert result.forces == pytest.approx(
            np.array(
                [
                    [
                        [4.78449105, -14.91768073, -0.68526833],
                        [-17.67679072, -13.94740555, 0.99580602],
                        [12.13724344, 32.19470285, -1.88714259],
                        [16.36595863, 6.31348517, 1.2666467],
                        [12.30891795, 10.03587697, 18.79967713],
                        [-27.91983493, -19.67903268, -18.48972094],
                    ]
                ]
            ),
        )

    def test_multi_frame(self, ani2x_simulation, initial_coords_box_angstrom):
        coords, box = initial_coords_box_angstrom
        result = compute_mlp_energies_forces(
            ani2x_simulation, [coords, coords], [box, box], show_progress=False
        )
        assert result.energies.shape == (2,)
        assert result.forces.shape == (2, 6, 3)
        assert result.energies[0] == pytest.approx(result.energies[1])
        assert result.forces[0] == pytest.approx(result.forces[1])


class TestComputeAseEnergiesForces:
    """Tests for compute_ase_energies_forces."""

    @pytest.fixture(scope="class")
    def mock_calculator(self):
        """Minimal ASE calculator that returns fixed energies and forces."""
        import ase.calculators.calculator
        import numpy as np

        class _FixedCalc(ase.calculators.calculator.Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=None, system_changes=None):
                n = len(atoms)
                self.results = {
                    "energy": 1.0,
                    "forces": np.zeros((n, 3)),
                }

        return _FixedCalc()

    def test_single_frame_no_progress(
        self, water_system, initial_coords_box_angstrom, mock_calculator
    ):
        from scalej.constants import EV_TO_KCAL_MOL
        from scalej.simulation._mlp import compute_ase_energies_forces

        tensor_system, _, _ = water_system
        coords, box = initial_coords_box_angstrom
        result = compute_ase_energies_forces(
            tensor_system, mock_calculator, [coords], [box], show_progress=False
        )
        assert result.energies.shape == (1,)
        assert result.forces.shape == (1, 6, 3)
        assert result.energies[0] == pytest.approx(1.0 * EV_TO_KCAL_MOL)
        assert np.all(result.forces == 0.0)

    def test_multi_frame(
        self, water_system, initial_coords_box_angstrom, mock_calculator
    ):
        from scalej.simulation._mlp import compute_ase_energies_forces

        tensor_system, _, _ = water_system
        coords, box = initial_coords_box_angstrom
        result = compute_ase_energies_forces(
            tensor_system,
            mock_calculator,
            [coords, coords],
            [box, box],
            show_progress=False,
        )
        assert result.energies.shape == (2,)
        assert result.forces.shape == (2, 6, 3)


class TestIntegrationComputeMlpEnergiesForcesSingle:
    def test_setup_and_compute_integration(
        self, water_system, initial_coords_box_angstrom
    ):
        tensor_system, _, _ = water_system
        coords, box = initial_coords_box_angstrom
        sim = setup_mlp_simulation(
            tensor_system, "ani2x", mlp_device="cpu", platform="CPU"
        )
        result = compute_mlp_energies_forces(sim, [coords], [box], show_progress=False)
        assert isinstance(result, EnergyForceResult)
        assert result.energies.shape == (1,)
        assert result.forces.shape == (1, 6, 3)
        assert result.energies[0] == pytest.approx(-95868.51804423)
        assert result.forces[0] == pytest.approx(
            np.array(
                [
                    [4.78449105, -14.91768073, -0.68526833],
                    [-17.67679072, -13.94740555, 0.99580602],
                    [12.13724344, 32.19470285, -1.88714259],
                    [16.36595863, 6.31348517, 1.2666467],
                    [12.30891795, 10.03587697, 18.79967713],
                    [-27.91983493, -19.67903268, -18.48972094],
                ]
            ),
        )
