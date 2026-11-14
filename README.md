# CoFD-Bench

Covalent docking benchmark system, containing structural data, analysis tools, and visualization scripts.

## Project Overview

CoFD-Bench is a benchmark platform for evaluating covalent docking methods, containing information about ligands, warheads, Apo, Holo structures for multiple systems, as well as RMSD data for all docking results. This project provides a series of analysis tools for evaluating and comparing the performance of different docking methods.

## Data Description

- `pdb_benchmark.csv`: Contains SuCOS-related data and distance information
- `add_bond_columns.py`: Regenerates the `site_correct` and `bond_correct` columns of `pdb_benchmark.csv` from `Top-1-correct`, `Top-1-distance` and `real_distance`

### Correctness columns

`pdb_benchmark.csv` reports the Top-1 result through three columns. They answer
different questions and should not be read interchangeably:

| Column | Meaning |
|---|---|
| `Top-1-correct` | The original correctness call, kept unchanged for backward compatibility. Identical to `site_correct`. |
| `site_correct` | The predicted pose places the ligand next to the deposited nucleophile: the closest residue of the matching type in the Top-1 pose is the one the covalent bond is deposited to. This is a **site** prediction and says nothing about bond geometry. |
| `bond_correct` | `site_correct` **and** the predicted warhead-to-nucleophile distance (`Top-1-distance`) is within the range of a deposited covalent bond in this benchmark. |

The threshold used for `bond_correct` is the longest deposited bond in the set,
`max(real_distance) = 2.23 Å`; deposited distances here span 1.29-2.23 Å.

Of the 191 entries with `site_correct = True`, 113 are also `bond_correct`. The
remaining 78 identify the right nucleophile with the warhead outside bonding
range, which is a useful starting point for constrained refinement but is not a
modelled covalent bond. The gap is largest off cysteine and serine.

| Nucleophile | Entries | `site_correct` | `bond_correct` |
|---|---|---|---|
| CYS | 160 | 134 | 72 |
| SER | 43 | 43 | 39 |
| LYS | 9 | 8 | 2 |
| TYR | 3 | 3 | 0 |
| ASP | 2 | 2 | 0 |
| HIS | 1 | 1 | 0 |

One entry (`8PO4`) has no `Top-1-correct` call upstream; `site_correct` and
`bond_correct` are left blank for it rather than assigned a value.

We thank [Michael Varman (@MTVarman)](https://github.com/MTVarman) for proposing and implementing the `site_correct`/`bond_correct` split and its documentation in [#2](https://github.com/hf1s/CoFD-Bench/pull/2).

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