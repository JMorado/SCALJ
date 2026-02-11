# SCALeJ

Lennard-Jones Parameter Fitting via Condensed-Phase Volume-Scaling.

## Installation

First, clone the repository:

```bash
git clone https://github.com/JMorado/SCALeJ.git
cd SCALeJ
```

Then, create a conda environment with all the required dependencies:

```bash
conda env create -f environment.yaml
conda activate scalej
```

Finally, install SCALeJ in development mode:

```bash
pip install -e .
```

## CLI Commands

After installation, the `scalej` command provides access to all workflow nodes:

```bash
# View all available commands
scalej --help

# Available nodes:
#   md            - Run molecular dynamics simulations
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
1. MD simulations (parallelized per system when using multiple cores)
2. Configuration scaling (parallelized per system)
3. Scale factor merging across all systems
4. ML potential energy/force calculations (parallelized per system)
5. Dataset preparation and combination
6. Parameter training on combined dataset
7. Evaluation of initial and optimized parameters
8. Force field export
9. Benchmarking (parallelized per system)