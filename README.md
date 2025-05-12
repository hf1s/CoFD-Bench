# CoFD-Bench

Covalent docking benchmark system, containing structural data, analysis tools, and visualization scripts.

## Project Overview

CoFD-Bench is a benchmark platform for evaluating covalent docking methods, containing information about ligands, warheads, Apo, Holo structures for multiple systems, as well as RMSD data for all docking results. This project provides a series of analysis tools for evaluating and comparing the performance of different docking methods.

## Data Description

- `pdb_benchmark.csv`: Contains SuCOS-related data
- `pdb_benchmark_RMSD.csv`: Contains RMSD data
- `plifs_recovery_df.csv`: Contains PLIF (Protein-Ligand Interaction Fingerprints) recovery rate data

## Analysis Features

This project implements various analysis and visualization functions:

- **RMSD Statistical Analysis**: Use the `analyze_rmsd_statistics()` function to generate boxplots for different docking methods
- **PLIF Recovery Analysis**: Use the `plot_combined_plif_analysis()` function to evaluate protein-ligand interaction fingerprint recovery rates
- **SuCOS Group Analysis**: Use the `plot_sucos_group_PLIF()` function to compare docking results based on SuCOS_pocket_qcov grouping
- **Success Rate Analysis**: Use the `plot_SuCOS_pocket_all()` function to analyze success rates of different methods at different SuCOS values
- **AF3 Warhead Type Analysis**: Use `analyze_af3_success_rate()` and `plot_af3_warhead_pie()` functions to analyze AlphaFold3 performance on different warhead types

## Example Jupyter Notebooks

- `RMSD_plot.ipynb`: RMSD-related analysis examples
- `SuCOS_pocket_qcov.ipynb`: SuCOS pocket coverage analysis
- `PLIF_plot.ipynb`: PLIF recovery rate analysis examples

## Environment Setup

Create the project environment using conda:

```bash
conda env create -f environment.yml
conda activate CoFD-Bench
```

## Usage Instructions

1. Activate the environment

```bash
conda activate CoFD-Bench
```

2. Run analysis scripts or Jupyter notebooks

```bash
jupyter notebook RMSD_plot.ipynb
```

3. Use functions from `plot.py` for custom analysis

```python
import plot

# RMSD analysis
plot.analyze_rmsd_statistics(
    csv_file='pdb_benchmark_RMSD.csv', 
    output_dir='./rmsd_statistics'
)

# SuCOS and PLIF analysis
plot.plot_sucos_group_PLIF(
    rmsd_csv='pdb_benchmark_RMSD.csv',
    sucos_csv='pdb_benchmark.csv',
    plif_csv='plifs_recovery_df.csv',
    output_dir='./SuCOS_pocket_qcov'
)