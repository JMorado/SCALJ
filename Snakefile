"""
SCALeJ Snakemake Workflow

This version enables true parallel execution using --system-name flags.

Usage:
    # Run with multiple cores to parallelize data generation for multiple systems
    snakemake -s Snakefile --cores 4 --configfile examples/config.yaml
"""

import yaml
from pathlib import Path

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

# Defines the final targets
rule all:
    input:
        expand(f"{OUTPUT_DIR}/parity_energy_initial_{{system}}.png", system=SYSTEMS),
        expand(f"{OUTPUT_DIR}/parity_energy_final_{{system}}.png", system=SYSTEMS),
        f"{OUTPUT_DIR}/optimized_forcefield.offxml",
        f"{OUTPUT_DIR}/benchmark_results.txt"


# Run MD simulation for a single system
rule md_system:
    input:
        config=CONFIG_FILE
    output:
        system=f"{OUTPUT_DIR}/system_{{system}}.pkl",
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
        scalj md \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --system-name {params.system_name} \
            --log-file {log} 2>&1 | tee -a {log}
        """


# Generate scaled configurations for a single system
rule scaling_system:
    input:
        config=CONFIG_FILE,
        system=f"{OUTPUT_DIR}/system_{{system}}.pkl"
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
        scalj scaling \
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


# Compute ML potential for a single system
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
    threads: 1  # Each ML potential computation uses 1 thread
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalj ml_potential \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --system-name {params.system_name} \
            --log-file {log} 2>&1 | tee -a {log}
        """


# Combine datasets from all systems
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
        scalj dataset \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --log-file {log} 2>&1 | tee -a {log}
        """


# Train LJ parameters
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
        scalj training \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --log-file {log} 2>&1 | tee -a {log}
        """


# Evaluate initial parameters (before training)
rule evaluation_initial:
    input:
        config=CONFIG_FILE,
        params=f"{OUTPUT_DIR}/initial_parameters.pkl",
        dataset=f"{OUTPUT_DIR}/combined_dataset.pkl",
        composite=f"{OUTPUT_DIR}/composite_system.pkl"
    output:
        expand(f"{OUTPUT_DIR}/parity_energy_initial_{{system}}.png", system=SYSTEMS),
        expand(f"{OUTPUT_DIR}/parity_forces_initial_{{system}}.png", system=SYSTEMS),
        expand(f"{OUTPUT_DIR}/energy_vs_scale_initial_{{system}}.png", system=SYSTEMS)
    params:
        output_dir=OUTPUT_DIR
    threads: 1
    log:
        f"{OUTPUT_DIR}/logs/evaluation_initial.log"
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalj evaluation \
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
        expand(f"{OUTPUT_DIR}/parity_energy_final_{{system}}.png", system=SYSTEMS),
        expand(f"{OUTPUT_DIR}/parity_forces_final_{{system}}.png", system=SYSTEMS),
        expand(f"{OUTPUT_DIR}/energy_vs_scale_final_{{system}}.png", system=SYSTEMS)
    params:
        output_dir=OUTPUT_DIR
    threads: 1
    log:
        f"{OUTPUT_DIR}/logs/evaluation_final.log"
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalj evaluation \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --params-file {input.params} \
            --plot-prefix final_ \
            --log-file {log} 2>&1 | tee -a {log}
        """


# Run benchmark for a single system 
rule benchmark_system:
    input:
        config=CONFIG_FILE,
        params=f"{OUTPUT_DIR}/trained_parameters.pkl",
        composite=f"{OUTPUT_DIR}/composite_system.pkl"
    output:
        temp(f"{OUTPUT_DIR}/benchmark_{{system}}.txt")  # Temporary per-system result
    params:
        output_dir=OUTPUT_DIR,
        system_name="{system}"
    threads: 1  # Each system runs independently
    log:
        f"{OUTPUT_DIR}/logs/benchmark_{{system}}.log"
    shell:
        """
        mkdir -p {params.output_dir}/logs
        scalj benchmark \
            --config {input.config} \
            --output-dir {params.output_dir} \
            --params-file {input.params} \
            --system-name {params.system_name} \
            --log-file {log} 2>&1 | tee -a {log}
        
        # Create a simple marker file
        echo "Benchmark completed for {params.system_name}" > {output}
        """

# Rule: Export force field
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
        scalj export \
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
        snakemake -s Snakefile.parallel --dag --configfile {CONFIG_FILE} | dot -Tpng > workflow_dag_parallel.png
        echo "Workflow DAG saved to workflow_dag_parallel.png"
        """