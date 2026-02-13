"""
SCALeJ Snakemake Workflow


Usage:
    # Full workflow
    snakemake -s Snakefile --cores 4 --configfile examples/config.yaml
"""

import yaml

# Default config file - use workflow.configfiles if provided via --configfile
if workflow.configfiles:
    CONFIG_FILE = workflow.configfiles[0]
else:
    CONFIG_FILE = config.get("__config__", "config.yaml")

# Load configuration to extract settings
with open(CONFIG_FILE, 'r') as f:
    config_data = yaml.safe_load(f)

# Set output directory
OUTPUT_DIR = config_data.get("general", {}).get("output_dir", "output")

# Get system names from config
SYSTEMS = [system["name"] for system in config_data.get("general", {}).get("systems", [])]

# Check if MLP relaxation is enabled
USE_MLP_RELAXATION = config_data.get("simulation", {}).get("n_mlp_steps", 0) > 0


# Defines the final targets
rule all:
    input:
        expand(f"{OUTPUT_DIR}/parity_energy_initial_{{system}}.png", system=SYSTEMS),
        expand(f"{OUTPUT_DIR}/parity_energy_final_{{system}}.png", system=SYSTEMS),
        f"{OUTPUT_DIR}/initial_energy_vs_scale_total.png",
        f"{OUTPUT_DIR}/final_energy_vs_scale_total.png",
        f"{OUTPUT_DIR}/optimized_forcefield.offxml",
        f"{OUTPUT_DIR}/benchmark_results.txt"


# ============================================================================
# Step 1: System Setup - Create system topology/forcefield
# ============================================================================
rule system_setup:
    input:
        config=CONFIG_FILE
    output:
        system=f"{OUTPUT_DIR}/system_{{system}}.pkl"
    params:
        output_dir=OUTPUT_DIR,
        system_name="{system}"
    log:
        f"{OUTPUT_DIR}/logs/system_setup_{{system}}.log"
    threads: 1
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalej system_setup \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --system-name {params.system_name} \
            --log-file {log} 2>&1 | tee -a {log}
        """


# ============================================================================
# Step 2: MD Simulation - Run classical MD 
# ============================================================================
rule md_system:
    input:
        config=CONFIG_FILE,
        system=f"{OUTPUT_DIR}/system_{{system}}.pkl"
    output:
        trajectory=f"{OUTPUT_DIR}/trajectory_{{system}}.dcd"
    params:
        output_dir=OUTPUT_DIR,
        system_name="{system}"
    log:
        f"{OUTPUT_DIR}/logs/md_{{system}}.log"
    threads: 1
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalej md \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --system-name {params.system_name} \
            --log-file {log} 2>&1 | tee -a {log}
        """


# ============================================================================
# Step 3: MLP MD Relaxation 
# ============================================================================
rule mlp_md_system:
    input:
        config=CONFIG_FILE,
        system=f"{OUTPUT_DIR}/system_{{system}}.pkl",
        trajectory=lambda wildcards: (
            config_data["general"]["systems"][
                [s["name"] for s in config_data["general"]["systems"]].index(wildcards.system)
            ].get("trajectory_path")
            or f"{OUTPUT_DIR}/trajectory_{wildcards.system}.dcd"
        )
    output:
        mlp_coords=f"{OUTPUT_DIR}/mlp_coords_{{system}}.pkl"
    params:
        output_dir=OUTPUT_DIR,
        system_name="{system}"
    log:
        f"{OUTPUT_DIR}/logs/mlp_md_{{system}}.log"
    threads: 1
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalej mlp_md \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --system-name {params.system_name} \
            --log-file {log} 2>&1 | tee -a {log}
        """


# ============================================================================
# Step 4: Scaling - Generate scaled configurations
# ============================================================================
def get_scaling_inputs(wildcards):
    """Get appropriate input files for scaling based on config."""
    inputs = {
        "config": CONFIG_FILE,
        "system": f"{OUTPUT_DIR}/system_{wildcards.system}.pkl",
    }
    # Check if system has existing trajectory
    system_config = None
    for s in config_data.get("general", {}).get("systems", []):
        if s["name"] == wildcards.system:
            system_config = s
            break

    if USE_MLP_RELAXATION:
        inputs["mlp_coords"] = f"{OUTPUT_DIR}/mlp_coords_{wildcards.system}.pkl"
    elif system_config and system_config.get("trajectory_path"):
        # Use existing trajectory from config (no dependency needed)
        pass
    else:
        inputs["trajectory"] = f"{OUTPUT_DIR}/trajectory_{wildcards.system}.dcd"

    return inputs


rule scaling_system:
    input:
        unpack(get_scaling_inputs)
    output:
        scaled=f"{OUTPUT_DIR}/scaled_{{system}}.pkl",
        factors=f"{OUTPUT_DIR}/scale_factors_{{system}}.npy"
    params:
        output_dir=OUTPUT_DIR,
        system_name="{system}"
    log:
        f"{OUTPUT_DIR}/logs/scaling_{{system}}.log"
    threads: 1
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalej scaling \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --system-name {params.system_name} \
            --log-file {log} 2>&1 | tee -a {log}
        """


# Merge scale factors from all systems
rule merge_scale_factors:
    input:
        expand(f"{OUTPUT_DIR}/scale_factors_{{system}}.npy", system=SYSTEMS)
    output:
        f"{OUTPUT_DIR}/scale_factors.npy"
    run:
        import numpy as np
        # Load all scale factors and concatenate
        all_factors = []
        for factor_file in input:
            factors = np.load(factor_file)
            all_factors.append(factors)
        combined = np.concatenate(all_factors)
        np.save(output[0], combined)


# ============================================================================
# Step 5: ML Potential - Compute MLP energies/forces
# ============================================================================
rule ml_potential_system:
    input:
        config=CONFIG_FILE,
        system=f"{OUTPUT_DIR}/system_{{system}}.pkl",
        scaled=f"{OUTPUT_DIR}/scaled_{{system}}.pkl"
    output:
        f"{OUTPUT_DIR}/energies_forces_{{system}}.pkl"
    params:
        output_dir=OUTPUT_DIR,
        system_name="{system}"
    log:
        f"{OUTPUT_DIR}/logs/ml_potential_{{system}}.log"
    threads: 1
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalej ml_potential \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --system-name {params.system_name} \
            --log-file {log} 2>&1 | tee -a {log}
        """


# ============================================================================
# Step 6: Dataset - Combine datasets from all systems
# ============================================================================
rule dataset:
    input:
        config=CONFIG_FILE,
        energies_forces=expand(f"{OUTPUT_DIR}/energies_forces_{{system}}.pkl", system=SYSTEMS),
        scale_factors=f"{OUTPUT_DIR}/scale_factors.npy"
    output:
        f"{OUTPUT_DIR}/combined_dataset.pkl",
        f"{OUTPUT_DIR}/composite_system.pkl"
    params:
        output_dir=OUTPUT_DIR
    log:
        f"{OUTPUT_DIR}/logs/dataset.log"
    threads: 1
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalej dataset \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --log-file {log} 2>&1 | tee -a {log}
        """


# ============================================================================
# Step 7: Training - Train LJ parameters
# ============================================================================
rule training:
    input:
        config=CONFIG_FILE,
        dataset=f"{OUTPUT_DIR}/combined_dataset.pkl",
        composite=f"{OUTPUT_DIR}/composite_system.pkl"
    output:
        f"{OUTPUT_DIR}/initial_parameters.pkl",
        f"{OUTPUT_DIR}/trained_parameters.pkl",
        f"{OUTPUT_DIR}/training_losses.png"
    params:
        output_dir=OUTPUT_DIR,
    threads: 4
    log:
        f"{OUTPUT_DIR}/logs/training.log"
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalej training \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --log-file {log} 2>&1 | tee -a {log}
        """


# ============================================================================
# Step 8: Evaluation - Evaluate initial parameters (before training)
# ============================================================================
rule evaluation_initial:
    input:
        config=CONFIG_FILE,
        params=f"{OUTPUT_DIR}/initial_parameters.pkl",
        dataset=f"{OUTPUT_DIR}/combined_dataset.pkl",
        composite=f"{OUTPUT_DIR}/composite_system.pkl"
    output:
        f"{OUTPUT_DIR}/metrics_initial.json",
        expand(f"{OUTPUT_DIR}/parity_energy_initial_{{system}}.png", system=SYSTEMS),
        expand(f"{OUTPUT_DIR}/parity_forces_initial_{{system}}.png", system=SYSTEMS),
        f"{OUTPUT_DIR}/initial_energy_vs_scale_total.png"
    params:
        output_dir=OUTPUT_DIR
    threads: 1
    log:
        f"{OUTPUT_DIR}/logs/evaluation_initial.log"
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalej evaluation \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --params-file {input.params} \
            --plot-prefix initial_ \
            --log-file {log} 2>&1 | tee -a {log}
        """


# Evaluate final parameters (after training)
rule evaluation_final:
    input:
        config=CONFIG_FILE,
        params=f"{OUTPUT_DIR}/trained_parameters.pkl",
        dataset=f"{OUTPUT_DIR}/combined_dataset.pkl",
        composite=f"{OUTPUT_DIR}/composite_system.pkl"
    output:
        f"{OUTPUT_DIR}/metrics_final.json",
        expand(f"{OUTPUT_DIR}/parity_energy_final_{{system}}.png", system=SYSTEMS),
        expand(f"{OUTPUT_DIR}/parity_forces_final_{{system}}.png", system=SYSTEMS),
        f"{OUTPUT_DIR}/final_energy_vs_scale_total.png"
    params:
        output_dir=OUTPUT_DIR
    threads: 1
    log:
        f"{OUTPUT_DIR}/logs/evaluation_final.log"
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalej evaluation \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --params-file {input.params} \
            --plot-prefix final_ \
            --log-file {log} 2>&1 | tee -a {log}
        """


# ============================================================================
# Step 9: Benchmark - Run benchmark for all systems
# ============================================================================
rule benchmark:
    input:
        config=CONFIG_FILE,
        params=f"{OUTPUT_DIR}/trained_parameters.pkl",
        composite=f"{OUTPUT_DIR}/composite_system.pkl"
    output:
        f"{OUTPUT_DIR}/benchmark_results.txt"
    params:
        output_dir=OUTPUT_DIR
    threads: 1
    log:
        f"{OUTPUT_DIR}/logs/benchmark.log"
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalej benchmark \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --params-file {input.params} \
            --log-file {log} 2>&1 | tee -a {log}
        """


# ============================================================================
# Step 10: Export - Export optimized force field
# ============================================================================
rule export:
    input:
        config=CONFIG_FILE,
        params=f"{OUTPUT_DIR}/trained_parameters.pkl",
        composite=f"{OUTPUT_DIR}/composite_system.pkl"
    output:
        f"{OUTPUT_DIR}/optimized_forcefield.offxml"
    params:
        output_dir=OUTPUT_DIR
    threads: 1
    log:
        f"{OUTPUT_DIR}/logs/export.log"
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalej export \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --params-file {input.params} \
            --log-file {log} 2>&1 | tee -a {log}
        """


# ============================================================================
# Convenience rules
# ============================================================================

# Run everything up to training
rule trained_parameters:
    input:
        f"{OUTPUT_DIR}/trained_parameters.pkl"


# Run only data generation
rule generate_data:
    input:
        expand(f"{OUTPUT_DIR}/energies_forces_{{system}}.pkl", system=SYSTEMS)


# Clean outputs
rule clean:
    shell:
        f"""
        rm -rf {OUTPUT_DIR}
        """


# Print workflow DAG
rule dag:
    shell:
        """
        snakemake -s Snakefile --dag --configfile {CONFIG_FILE} | dot -Tpng > workflow_dag.png
        echo "Workflow DAG saved to workflow_dag.png"
        """