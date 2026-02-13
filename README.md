# SCALeJ

Lennard-Jones parameters fitting via condensed-phase volume scaling.

## Installation

Clone the repository:

```bash
git clone https://github.com/JMorado/SCALeJ.git
cd SCALeJ
```

Create a conda environment with all the required dependencies:

```bash
conda env create -f environment.yaml
conda activate scalej
```

Install SCALeJ in development mode:

```bash
pip install -e .
```

## CLI Commands

After installation, the `scalej` command provides access to all workflow nodes:

```bash
# View all available commands
scalej --help

# Available nodes:
#   system_setup  - Create system topology and force field
#   md            - Run molecular dynamics simulations
#   mlp_md        - Run MLP-based MD relaxation (optional)
#   scaling       - Generate scaled configurations
#   ml_potential  - Compute ML potential energies and forces
#   dataset       - Prepare combined datasets
#   training      - Train LJ parameters
#   evaluation    - Evaluate parameters and generate plots
#   export        - Export optimized force field
#   benchmark     - Calculate thermodynamic properties

# Get help for any specific node
scalej md --help
scalej training --help
```

## Quick Start

Generate a configuration file using the CLI:

```bash
scalej config
```

Run the workflow with `Snakefile`:

```bash
snakemake -s Snakefile --cores 1 --configfile config.yaml
```

For faster execution with parallel processing:

```bash
# Use 4 cores to parallelize per-system tasks
snakemake -s Snakefile --cores 4 --configfile config.yaml
```

This executes all workflow steps:
1. System setup (parallelized per system)
2. MD simulations (parallelized per system)
3. MLP MD relaxation (optional, parallelized per system)
4. Configuration scaling (parallelized per system)
5. Scale factor merging across all systems
6. ML potential energy/force calculations (parallelized per system)
7. Dataset preparation and combination
8. Parameter training on combined dataset
9. Evaluation of initial and optimized parameters
10. Force field export
11. Benchmarking (parallelized per system)