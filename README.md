# CoFD-Bench

Covalent docking benchmark system, containing structural data, analysis tools, and visualization scripts.

## Project Overview

CoFD-Bench is a benchmark platform for evaluating covalent docking methods, containing information about ligands, warheads, Apo, Holo structures for multiple systems, as well as RMSD data for all docking results. This project provides a series of analysis tools for evaluating and comparing the performance of different docking methods.

## Data Description

- `pdb_benchmark.csv`: Contains SuCOS-related data and distance information
- `pdb_benchmark_RMSD.csv`: Contains RMSD data for various docking methods
- `plifs_recovery_df.csv`: Contains PLIF (Protein-Ligand Interaction Fingerprints) recovery rate data

## Analysis Features

This project implements various analysis and visualization functions:

- **`analyze_rmsd_statistics()`**: Comprehensive RMSD analysis with boxplots for different docking methods
- **`plot_combined_af3_analysis()`**: Integrates three AF3 analysis plots into one large figure
- **`plot_SuCOS_pocket_all()`**: Multi-method success rate analysis in 2x2 grid layout
- **`plot_combined_plif_analysis()`**: Combined bar and violin plots for PLIF recovery
- **`plot_sucos_group_PLIF()`**: Grouped stacked bar plots comparing docking outcomes by SuCOS groups


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

# Basic RMSD analysis
plot.analyze_rmsd_statistics(
    csv_file='pdb_benchmark_RMSD.csv',
    output_dir='./results'
)

```

## Output Formats

All analysis functions generate high-quality figures in multiple formats:
- **PNG**: High-resolution raster images (600 DPI)
- **PDF**: Vector graphics for publications
- **SVG**: Scalable vector graphics for web use