"""Run the SCALJ workflow."""

from pathlib import Path

import descent.utils.reporting
import openmm
import smee
import smee.converters
import smee.mm
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule, Topology

from .. import dataset as ds
from .. import plots
from ..engine import (
    compute_energies_forces,
    create_scaled_dataset,
    generate_scale_factors,
    load_last_frame,
    run_mlp_simulation,
    run_simulation,
    run_thermo_benchmark,
    setup_mlp_simulation,
)
from ..fitting import (
    create_trainable,
    predict,
    train_parameters,
)
from .utils import create_configs_from_dict, load_config


def run_workflow(args):
    """
    Run the complete SCALJ workflow.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing configuration and overrides.
    """

    print("=" * 80)
    print("scalj - LJ Parameter Fitting Workflow")
    print("=" * 80)

    config_dict = load_config(args.config)

    # Override with command-line arguments
    if args.force_field:
        config_dict.setdefault("system", {})["force_field_name"] = args.force_field
    if args.ml_potential:
        config_dict.setdefault("system", {})["mlp_name"] = args.ml_potential
    if args.output_dir:
        config_dict.setdefault("system", {})["output_dir"] = args.output_dir
    if args.learning_rate:
        config_dict.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.n_epochs:
        config_dict.setdefault("training", {})["n_epochs"] = args.n_epochs
    if args.device:
        config_dict.setdefault("training", {})["device"] = args.device

    # Create configuration objects
    (
        general_config,
        simulation_config,
        scaling_config,
        training_config,
        parameter_config,
    ) = create_configs_from_dict(config_dict)

    # Assert that all systems have different names
    system_names = [system.name for system in general_config.systems]
    if len(system_names) != len(set(system_names)):
        raise ValueError("All systems must have different names.")

    # Create output directory
    output_dir = Path(general_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Force field: {general_config.force_field_name}")
    print(f"ML potential: {general_config.mlp_name}")

    # Process each mixture/system
    if general_config.systems:
        # Load force field
        force_field = ForceField(general_config.force_field_name, load_plugins=True)
        print(f"\nNumber of systems/mixtures: {len(general_config.systems)}")
        for system in general_config.systems:
            print(f"\n   System: {system.name} (weight: {system.weight})")

            # Create the Molecule and Interchange objects for this system
            mols = [Molecule.from_smiles(comp.smiles) for comp in system.components]
            interchanges = [
                Interchange.from_smirnoff(force_field, [mol]) for mol in mols
            ]

            # Create tensor forcefield and topologies
            tensor_forcefield, topologies = smee.converters.convert_interchange(
                interchanges
            )

            system.tensor_forcefield = tensor_forcefield
            system.topologies = topologies
            system.nmol = [comp.nmol for comp in system.components]

            # Create tensor system
            system.tensor_system = smee.TensorSystem(
                topologies, system.nmol, is_periodic=True
            )
            for comp in system.components:
                print(f"      - {comp.smiles}: {comp.nmol} molecules")

    # print(system.tensor_system.n_atoms)
    # exit()
    # Process each mixture/system
    systems_ds = {}

    for system in general_config.systems:
        print("\n" + "=" * 80)
        print(f"Processing system: {system.name}")
        print("=" * 80)

        # Step 1: Load or run simulation for this mixture
        input_trajectory_path = system.trajectory_path or general_config.trajectory_path

        if input_trajectory_path:
            print(f"Loading existing trajectory for {system.name}...")
            trajectory_path = Path(input_trajectory_path)
            if not trajectory_path.exists():
                raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

            coords, box_vectors = load_last_frame(trajectory_path)
            print(f"   Loaded trajectory from: {trajectory_path}")

            print("Setting up MLP simulation...")
            print(f"   MLP name: {general_config.mlp_name}")
            print(f"   MLP device: {simulation_config.mlp_device}")
            mlp_simulation = setup_mlp_simulation(
                system.tensor_system, general_config.mlp_name, simulation_config
            )
        else:
            print(f"Running molecular dynamics simulation for {system.name}...")
            print(f"   Platform: {simulation_config.platform}")
            print(f"   Temperature: {simulation_config.temperature}")
            print(f"   Pressure: {simulation_config.pressure}")
            print(f"   Timestep: {simulation_config.timestep}")
            print(
                f"   Equilibration NVT steps: {simulation_config.n_equilibration_nvt_steps}"
            )
            print(
                f"   Equilibration NPT steps: {simulation_config.n_equilibration_npt_steps}"
            )
            print(f"   Production steps: {simulation_config.n_production_steps}")

            # Run simulation
            trajectory_path = output_dir / f"trajectory_{system.name}.dcd"
            run_simulation(
                system.tensor_system,
                system.tensor_forcefield,
                trajectory_path,
                simulation_config,
            )
            print(f"   Trajectory saved to: {trajectory_path}")
            # Load the last frame to use as starting point for MLP simulation
            coords, box_vectors = load_last_frame(trajectory_path)

        print("Creating MLP simulation...")
        print(f"   MLP name: {general_config.mlp_name}")
        print(f"   MLP device: {simulation_config.mlp_device}")
        mlp_simulation = setup_mlp_simulation(
            system.tensor_system, general_config.mlp_name, simulation_config
        )

        if simulation_config.n_mlp_steps > 0:
            print("Running MLP simulation...")
            print(f"   Steps: {simulation_config.n_mlp_steps}")
            coords, box_vectors, mlp_simulation = run_mlp_simulation(
                mlp_simulation, coords, box_vectors, simulation_config
            )
            # TODO: also save the MLP trajectory

        # Step 2: Generate scaled configurations for this system
        print(f"\nGenerating scaled configurations for {system.name}...")
        print(f"   Close range: {scaling_config.close_range}")
        print(f"   Equilibrium range: {scaling_config.equilibrium_range}")
        print(f"   Long range: {scaling_config.long_range}")
        scale_factors = generate_scale_factors(scaling_config)

        # Convert OpenMM quantities to NumPy arrays in angstroms
        coords_np = coords.value_in_unit(openmm.unit.angstrom)
        box_vectors_np = box_vectors.value_in_unit(openmm.unit.angstrom)

        coords_scaled, box_vectors_scaled = create_scaled_dataset(
            system.tensor_system, coords_np, box_vectors_np, scale_factors
        )

        print(f"   Generated {len(coords_scaled)} scaled configurations")

        # Step 3: Compute ML potential energies and forces for this system
        print(f"\nComputing ML potential energies and forces for {system.name}...")
        energies, forces = compute_energies_forces(
            mlp_simulation, coords_scaled, box_vectors_scaled
        )

        # Create dataset entries using helper function
        smiles_str = ".".join([comp.smiles for comp in system.components])

        entries = ds.create_entries_from_ml_output(
            mixture_id=system.name,
            smiles=smiles_str,
            coords_list=coords_scaled,
            box_vectors_list=box_vectors_scaled,
            energies=energies,
            forces=forces,
        )
        # Create dataset from entries
        system_dataset = ds.create_dataset(entries)
        systems_ds[system.name] = system_dataset
        print(f"   Computed energies and forces for {len(entries)} configurations")

    # Step 4: Build composite system and force field
    print("\n" + "=" * 80)
    print(
        "Generating composite objects for simultaneous training across all mixtures..."
    )
    print("=" * 80)
    composite_mols = [
        Molecule.from_smiles(comp.smiles)
        for system in general_config.systems
        for comp in system.components
    ]
    composite_interchanges = [
        Interchange.from_smirnoff(force_field, [mol]) for mol in composite_mols
    ]
    composite_tensor_forcefield, composite_topologies = (
        smee.converters.convert_interchange(composite_interchanges)
    )
    composite_tensor_system = smee.TensorSystem(
        composite_topologies,
        [comp.nmol for system in general_config.systems for comp in system.components],
        is_periodic=True,
    )

    # Step 5: Combine datasets from all mixtures with padding
    print("\n" + "=" * 80)
    print("Combining datasets from all mixtures...")
    print("=" * 80)
    all_tensor_systems = {}
    idx_counter = 0
    for system in general_config.systems:
        n_comps = len(system.components)
        system_topologies = composite_topologies[idx_counter : idx_counter + n_comps]
        idx_counter += n_comps
        print(idx_counter, idx_counter + n_comps)
        all_tensor_systems[system.name] = smee.TensorSystem(
            system_topologies,
            [comp.nmol for comp in system.components],
            is_periodic=True,
        )

    combined_dataset = ds.combine_datasets(
        systems_ds,
        composite_tensor_system=composite_tensor_system,
        all_tensor_systems=all_tensor_systems,
    )
    print(f"   Combined dataset size: {len(combined_dataset)} configurations")
    print(f"   Mixtures: {', '.join(systems_ds.keys())}")

    composite_trainable = create_trainable(
        composite_tensor_forcefield, parameter_config, training_config
    )
    energy_ref, energy_pred, forces_ref, forces_pred, _, _, mask_idx = predict(
        combined_dataset,
        composite_trainable.to_force_field(composite_trainable.to_values()),
        all_tensor_systems=all_tensor_systems,
        reference=training_config.reference,
        normalize=False,
        device=training_config.device,
        energy_cutoff=training_config.energy_cutoff,
    )
    print(energy_pred, energy_ref)
    # Plot parity
    plots.plot_parity(
        energy_ref.detach().cpu().numpy(),
        energy_pred.detach().cpu().numpy(),
        "Energy",
        "kcal/mol",
        output_dir / "parity_energy_initial.png",
    )

    plots.plot_parity(
        forces_ref.flatten().detach().cpu().numpy(),
        forces_pred.flatten().detach().cpu().numpy(),
        "Forces",
        "kcal/mol/Å",
        output_dir / "parity_forces_initial.png",
    )
    for i in range(len(all_tensor_systems)):
        len_scale = len(scale_factors[mask_idx.detach().cpu().numpy()])
        plots.plot_energy_vs_scale(
            scale_factors[mask_idx.detach().cpu().numpy()],
            [energy_ref.detach().cpu().numpy()[i*len_scale:(i+1)*len_scale], energy_pred.detach().cpu().numpy()[i*len_scale:(i+1)*len_scale]],
            output_dir / f"energy_vs_scale_initial_{list(all_tensor_systems.keys())[i]}.png",
            labels=["Reference", "Optimized"],
            lims=(0, 30),
        )
    exit()
    print("=" * 80)

    # Step 6: Train LJ parameters on combined dataset
    print("\n" + "=" * 80)
    print("Training Lennard-Jones parameters across all mixtures...")
    print("=" * 80)
    print(f"   Learning rate: {training_config.learning_rate}")
    print(f"   Epochs: {training_config.n_epochs}")
    print(f"   Device: {training_config.device}")
    print(f"   Parameters: {parameter_config.cols}")
    final_params, energy_losses, force_losses = train_parameters(
        composite_trainable,
        combined_dataset,
        all_tensor_systems,
        training_config,
    )

    # Plot training losses
    loss_plot_path = output_dir / "training_losses.png"
    plots.plot_training_losses(energy_losses, force_losses, loss_plot_path)

    print("\n" + "=" * 80)
    print("Final parameters...")
    print("=" * 80)
    final_force_field = composite_trainable.to_force_field(final_params)
    descent.utils.reporting.print_potential_summary(
        final_force_field.potentials_by_type["vdW"]
    )

    # Final evaluation on combined dataset
    print("\n" + "=" * 80)
    print("Evaluating final parameters...")
    print("=" * 80)

    energy_ref, energy_pred, forces_ref, forces_pred, _, _, mask_idx = predict(
        combined_dataset,
        final_force_field.to(training_config.device),
        all_tensor_systems=all_tensor_systems,
        reference=training_config.reference,
        normalize=False,
        device=training_config.device,
        energy_cutoff=training_config.energy_cutoff,
    )

    # Plot parity
    plots.plot_parity(
        energy_ref.detach().cpu().numpy(),
        energy_pred.detach().cpu().numpy(),
        "Energy",
        "kcal/mol",
        output_dir / "parity_energy_final.png",
    )

    plots.plot_parity(
        forces_ref.flatten().detach().cpu().numpy(),
        forces_pred.flatten().detach().cpu().numpy(),
        "Forces",
        "kcal/mol/Å",
        output_dir / "parity_forces_final.png",
    )

    plots.plot_energy_vs_scale(
        scale_factors[mask_idx.detach().cpu().numpy()],
        [energy_ref.detach().cpu().numpy(), energy_pred.detach().cpu().numpy()],
        output_dir / "energy_vs_scale_final.png",
        labels=["Reference", "Predicted"],
        lims=(0, 30),
    )

    # Run thermodynamic benchmark
    print("\n" + "=" * 80)
    print("Running thermodynamic benchmark...")
    print("=" * 80)
    for system in general_config.systems:
        results_final = run_thermo_benchmark(
            system.name,
            composite_trainable,
            all_tensor_systems[system.name].topologies,
            final_params,
        )

        import numpy as np

        dens_list = []
        hvap_list = []
        for result in results_final:
            dens_list.append(result[2][0].detach().numpy())
            hvap_list.append(result[2][1].detach().numpy() * 4.184)
        avg_density = np.mean(dens_list)
        avg_hvap = np.mean(hvap_list)
        std_density = np.std(dens_list)
        std_hvap = np.std(hvap_list)

        print(f"Density: {avg_density} ± {std_density} g/mL")
        print(f"Hvap: {avg_hvap} ± {std_hvap} kcal/mol")

    # Save results
    # results_path = output_dir / "results.yaml"
    # print(f"\nSaving results to: {results_path}")

    from ..utils import create_off_forcefield_from_tensor

    print("\n" + "=" * 80)
    off_xml_file = output_dir / "optimized_forcefield.offxml"
    print(f"Saving optimized force field to {off_xml_file}...")
    print("=" * 80)
    updated_forcefield = create_off_forcefield_from_tensor(
        force_field, final_force_field
    )
    updated_forcefield.to_file(off_xml_file)

    print("\n" + "=" * 80)
    print("Workflow completed successfully!")
    print("=" * 80)
