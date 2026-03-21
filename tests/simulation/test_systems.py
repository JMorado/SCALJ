"""Tests for system creation functions."""

import smee
from openff.toolkit import ForceField


class TestCreateSystemFromSmiles:
    def test_return_types(self, water_system):
        tensor_system, tensor_forcefield, topologies = water_system
        assert isinstance(tensor_system, smee.TensorSystem)
        assert isinstance(tensor_forcefield, smee.TensorForceField)
        assert isinstance(topologies, list)
        assert all(isinstance(t, smee.TensorTopology) for t in topologies)

    def test_single_component(self, water_system):
        tensor_system, _, topologies = water_system
        assert len(topologies) == 1
        assert topologies[0].n_atoms == 3
        assert tensor_system.n_copies == [2]
        assert tensor_system.is_periodic is True

    def test_multi_component(self, water_methane_system):
        tensor_system, _, topologies = water_methane_system
        assert len(topologies) == 2
        assert topologies[0].n_atoms == 3  # water
        assert topologies[1].n_atoms == 5  # methane
        assert tensor_system.n_copies == [2, 3]

    def test_forcefield_has_standard_potentials(self, water_system):
        _, tensor_forcefield, _ = water_system
        potentials = tensor_forcefield.potentials_by_type
        assert "Bonds" in potentials
        assert "Angles" in potentials
        assert "vdW" in potentials
        assert "Electrostatics" in potentials


class TestCreateCompositeSystem:
    def test_return_types(self, composite_system):
        ctf, cts, ctops, systems, off_ff = composite_system
        assert isinstance(ctf, smee.TensorForceField)
        assert isinstance(cts, smee.TensorSystem)
        assert isinstance(ctops, list)
        assert isinstance(systems, dict)
        assert isinstance(off_ff, ForceField)

    def test_composite_topologies(self, composite_system):
        _, _, ctops, _, _ = composite_system
        assert len(ctops) == 2
        assert ctops[0].n_atoms == 3  # water
        assert ctops[1].n_atoms == 5  # methane

    def test_individual_systems(self, composite_system):
        _, _, _, systems, _ = composite_system
        assert set(systems.keys()) == {"water", "methane"}
        assert systems["water"].n_copies == [2]
        assert systems["methane"].n_copies == [3]
        assert systems["water"].is_periodic is True
        assert systems["methane"].is_periodic is True

    def test_shared_forcefield(self, composite_system):
        ctf, _, _, _, _ = composite_system
        assert "Bonds" in ctf.potentials_by_type
        assert "Angles" in ctf.potentials_by_type
        assert "vdW" in ctf.potentials_by_type
        assert "Electrostatics" in ctf.potentials_by_type
