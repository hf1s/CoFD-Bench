import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy import stats
import seaborn as sns
from scipy.stats import kruskal


def analyze_rmsd_statistics(
    csv_file, 
    methods=None, 
    output_dir='./',
    rmsd_threshold=2, 
    fig_size=(24, 10),  # wider aspect ratio
    outward_ticks=True,
    title='', 
    dpi=300,
    ylim=16,
    condition_column=None,
    condition_threshold=None,
    condition_operator='<',
    significance_analysis=False,
    method_comparison=None
):
    """
    Analyze RMSD data and plot boxplots for different docking methods, 
    while printing the mean RMSD and success rate for each method.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing RMSD data. Each column should correspond to a method and pose type.
    methods : dict, optional
        Dictionary mapping method names to a list of two column names: [Top-1 column, Top-5 column].
        If None, a default set of methods and columns will be used.
    output_dir : str, optional
        Directory to save the output plots. Will be created if it does not exist.
    rmsd_threshold : float, optional
        The RMSD threshold (in Å) below which a docking is considered successful.
    fig_size : tuple, optional
        Figure size for the plot (width, height).
    outward_ticks : bool, optional
        If True, y-axis ticks will point outward.
    title : str, optional
        Title for the plot. If empty, no title will be shown.
    dpi : int, optional
        Dots per inch for the figure.
    ylim : float, optional
        Y-axis upper limit.
    condition_column : str, optional
        Column name for conditional filtering (e.g., 'AF3_1pose_RMSD').
    condition_threshold : float, optional
        Threshold value for conditional filtering (e.g., 2.0).
    condition_operator : str, optional
        Comparison operator for conditional filtering ('>' or '<'). Default is '<'.
    significance_analysis : bool, optional
        If True, perform significance analysis between Top-1 and Top-5 for each method.
    method_comparison : list, optional
        List of two method names to compare (e.g., ['AF3', 'Chai-1']).

    Returns
    -------
    None
        The function saves the boxplot as PNG, PDF, and SVG files in the output directory,
        and prints the mean RMSD and success rate for each method and pose type to the console.

    Notes
    -----
    - The function will perform a Mann-Whitney U test between Top-1 and Top-5 for each method,
      and annotate the significance on the plot.
    - The color scheme distinguishes Top-1 (orange) and Top-5 (blue).
    - The function will also draw a horizontal dashed line at the RMSD threshold.
    - Method names and pose types are labeled below the boxplots.
    - If condition_column and condition_threshold are provided, only samples meeting 
      the condition will be analyzed.
    - condition_operator can be '<' or '>' to specify the filtering direction.
    - If significance_analysis is True, statistical significance between Top-1 and Top-5 will be shown.
    - If method_comparison is provided, statistical comparison between specified methods will be performed.
    - The function does not return any value.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set default methods if not provided
    if methods is None:
        methods = {
            "AF3": ['af3_1pose_RMSD', 'af3_5pose_RMSD'],
            "Chai-1": ['chai1_1pose_RMSD', 'chai1_5pose_RMSD'],
            "Chai-1(MSA)": ['chai1_msa_1pose_RMSD', 'chai1_msa_5pose_RMSD'],
            "Boltz-1x": ['boltz_1pose_RMSD', 'boltz_5pose_RMSD'],
            "CovDock-AF3": ['Schrodinger_1pose_RMSD', 'Schrodinger_5pose_RMSD'],
            "Autodock-GPU-AF3": ['Autodock_1pose_RMSD', 'Autodock_5pose_RMSD'],
            "GNINA-AF3": ['Gnina_1pose_RMSD', 'Gnina_5pose_RMSD']
        }
    
    data = pd.read_csv(csv_file)
    
    # Apply conditional filtering if specified
    if condition_column is not None and condition_threshold is not None:
        if condition_column in data.columns:
            original_count = len(data)
            
            # Apply different operators
            if condition_operator == '<':
                condition_mask = (data[condition_column].notna()) & (data[condition_column] < condition_threshold)
            elif condition_operator == '>':
                condition_mask = (data[condition_column].notna()) & (data[condition_column] > condition_threshold)
            else:
                print(f"Warning: Unsupported operator '{condition_operator}', using '<' as default")
                condition_mask = (data[condition_column].notna()) & (data[condition_column] < condition_threshold)
                condition_operator = '<'
            
            data = data[condition_mask]
            filtered_count = len(data)
            
            print(f"Conditional filtering results:")
            print(f"Original sample count: {original_count}")
            print(f"Samples meeting condition {condition_column} {condition_operator} {condition_threshold}: {filtered_count}")
            print(f"Filtering ratio: {filtered_count/original_count*100:.1f}%")
            print("-" * 50)
        else:
            print(f"Warning: Condition column '{condition_column}' not found in data, using all data")
    
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi * 2
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    # Store statistics for each method and pose type
    stats_summary = []

    # Color scheme for Top-1 and Top-5
    pose_colors = {
        "Top-5": (0.2, 0.4, 0.8, 0.7),  # Blue
        "Top-1": (1.0, 0.5, 0.0, 0.9)   # Orange
    }

    # Prepare data for plotting
    plot_data = {}
    grouped_data = []
    grouped_positions = []
    grouped_labels = []
    grouped_colors = []
    grouped_types = []
    method_pose_data = {}
    pose_gap = 0.35
    method_gap = 0.5
    current_pos = 1

    for method_name in methods.keys():
        key1 = f"{method_name}_1pose"
        key5 = f"{method_name}_5pose"
        method_pose_data[method_name] = {}
        
        # Top-1
        if methods[method_name][0] in data.columns:
            values = data[methods[method_name][0]].dropna().values
            if len(values) > 0:
                grouped_positions.append(current_pos)
                grouped_data.append(values)
                grouped_labels.append(method_name)
                grouped_colors.append(pose_colors["Top-1"])
                grouped_types.append("Top-1")
                method_pose_data[method_name]['Top-1'] = {'values': values, 'position': current_pos}
                
                # Calculate success rate and mean RMSD
                success_rate = np.sum(values < rmsd_threshold) / len(values) * 100
                mean_rmsd = np.mean(values)
                
                # Add to statistics summary
                stats_summary.append({
                    'Method': method_name,
                    'Pose': '1pose',
                    'Success_Rate': success_rate,
                    'Mean_RMSD': mean_rmsd
                })
        
        # Top-5
        if methods[method_name][1] in data.columns:
            values = data[methods[method_name][1]].dropna().values
            if len(values) > 0:
                grouped_positions.append(current_pos + pose_gap)
                grouped_data.append(values)
                grouped_labels.append(method_name)
                grouped_colors.append(pose_colors["Top-5"])
                grouped_types.append("Top-5")
                method_pose_data[method_name]['Top-5'] = {'values': values, 'position': current_pos + pose_gap}
                
                # Calculate success rate and mean RMSD
                success_rate = np.sum(values < rmsd_threshold) / len(values) * 100
                mean_rmsd = np.mean(values)
                
                # Add to statistics summary
                stats_summary.append({
                    'Method': method_name,
                    'Pose': '5poses',
                    'Success_Rate': success_rate,
                    'Mean_RMSD': mean_rmsd
                })
        
        current_pos += pose_gap + method_gap

    # Create the boxplot figure
    plt.figure(figsize=fig_size)
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(True, axis='y', linestyle='-', alpha=0.2)

    black_semi = (0, 0, 0, 0.8)
    boxprops = {'linewidth': 4.0, 'color': black_semi}
    whiskerprops = {'linewidth': 4.0, 'color': black_semi}
    medianprops = {'color': black_semi, 'linewidth': 4.0}
    capprops = {'linewidth': 4.0, 'color': black_semi}

    bp = ax.boxplot(grouped_data, positions=grouped_positions, widths=0.3,
                   patch_artist=True, showfliers=False,
                   boxprops=boxprops, whiskerprops=whiskerprops,
                   medianprops=medianprops, capprops=capprops, zorder=3)

    # Set box color and style
    for patch, color in zip(bp['boxes'], grouped_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(black_semi)
        patch.set_linewidth(4.5)
        patch.set_alpha(0.7)

    # Plot mean value as a diamond marker
    for values, pos in zip(grouped_data, grouped_positions):
        mean_val = np.mean(values)
        plt.plot(pos, mean_val, marker='D', color=black_semi, markersize=9, zorder=4)

    # Significance analysis between Top-1 and Top-5 for each method
    if significance_analysis:
        for method_name in methods.keys():
            if 'Top-1' in method_pose_data[method_name] and 'Top-5' in method_pose_data[method_name]:
                data_1pose = method_pose_data[method_name]['Top-1']['values']
                data_5pose = method_pose_data[method_name]['Top-5']['values']
                pos1 = method_pose_data[method_name]['Top-1']['position']
                pos5 = method_pose_data[method_name]['Top-5']['position']
                
                # Perform Mann-Whitney U test
                try:
                    stat, p_value = stats.mannwhitneyu(data_1pose, data_5pose)
                    if p_value < 0.001:
                        significance = "***"
                    elif p_value < 0.01:
                        significance = "**"
                    elif p_value < 0.05:
                        significance = "*"
                    else:
                        significance = "ns"
                except Exception as e:
                    significance = "ns"
                    
                # Set significance marker height by method - unified height for co-folding methods
                if method_name in ["AF3", "Chai-1", "Chai-1(MSA)", "Boltz-1x", "AF3-Cov"]:
                    sig_y = ylim * 0.7  # adaptive height based on ylim
                else:
                    sig_y = ylim * 0.85  # slightly higher for traditional docking methods
                
                # Draw significance line
                plt.plot([pos1, pos5], [sig_y, sig_y], '-', color='black', linewidth=3.5)
                
                # Add significance text above the line
                plt.text((pos1 + pos5) / 2, sig_y, significance,
                        ha='center', va='bottom', fontsize=24, fontweight='bold')

    # Method comparison analysis
    if method_comparison is not None and len(method_comparison) == 2:
        print(f"Method comparison: {method_comparison}")
        method1, method2 = method_comparison
        if method1 in method_pose_data and method2 in method_pose_data:
            # Compare Top-1 data between two methods
            if 'Top-1' in method_pose_data[method1] and 'Top-1' in method_pose_data[method2]:
                data1 = method_pose_data[method1]['Top-1']['values']
                data2 = method_pose_data[method2]['Top-1']['values']
                pos1 = method_pose_data[method1]['Top-1']['position']
                pos2 = method_pose_data[method2]['Top-1']['position']
                
                try:
                    stat, p_value = stats.mannwhitneyu(data1, data2)
                    if p_value < 0.001:
                        significance = "***"
                    elif p_value < 0.01:
                        significance = "**"
                    elif p_value < 0.05:
                        significance = "*"
                    else:
                        significance = "ns"
                except Exception as e:
                    significance = "ns"
                
                # Draw comparison line at higher position
                comp_y = ylim * 0.9
                plt.plot([pos1, pos2], [comp_y, comp_y], '-', color='black', linewidth=3.5)
                plt.text((pos1 + pos2) / 2, comp_y, significance,
                        ha='center', va='bottom', fontsize=24, fontweight='bold', color='black')
                
                print(f"Method comparison ({method1} vs {method2} Top-1): p-value = {p_value:.6f}, significance = {significance}")

    # Draw the RMSD threshold line
    plt.axhline(y=rmsd_threshold, color='r', linestyle='--', linewidth=4.5, zorder=1)

    # Calculate method label positions
    method_positions = {}
    for method_name in methods.keys():
        positions = [pos for pos, label in zip(grouped_positions, grouped_labels) if label == method_name]
        if positions:
            method_positions[method_name] = sum(positions) / len(positions)
    
    # Add method names below the boxplots
    for method_name, pos in method_positions.items():
        plt.text(pos, -0.6, method_name,  # adjust position to reduce crowding
                ha='center', va='top', fontsize=22, fontweight='bold')

    # Remove x-axis ticks
    plt.xticks([])
    
    # Create custom legend for Top-1 and Top-5
    legend_elements = [
        Patch(facecolor=pose_colors["Top-1"], edgecolor='black', label='Top-1'),
        Patch(facecolor=pose_colors["Top-5"], edgecolor='black', label='Top-5'),
        plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2.5, label='')
    ]
    
    # Add legend to the upper left
    leg = ax.legend(handles=legend_elements[:2], loc='upper left',
                   frameon=True, edgecolor='gray', framealpha=0.9, 
                   fontsize=22, ncol=1)
    
    # Update title to include condition info if applicable
    final_title = title
    if condition_column is not None and condition_threshold is not None:
        condition_display = condition_column.replace('_', ' ')
        if final_title:
            final_title = f"{final_title}\n(Conditional on {condition_display} {condition_operator} {condition_threshold})"
        else:
            final_title = f"RMSD Distribution (Conditional on {condition_display} {condition_operator} {condition_threshold})"
    
    if final_title:
        plt.title(final_title, fontsize=18, fontweight='bold')
    plt.ylabel('L-RMSD (Å)', fontsize=26, fontweight='bold', labelpad=12)
    
    # Set y-axis and x-axis limits - reduce y-axis whitespace
    plt.ylim(0, ylim)
    plt.yticks(range(0, ylim, 5))
    plt.xlim(min(grouped_positions) - 0.5, max(grouped_positions) + 0.5)
    
    # Set y-axis tick style
    if outward_ticks:
        ax.tick_params(axis='y', direction='out', length=8, width=2.5,
                      pad=10, labelsize=25, grid_alpha=0.3)
    else:
        ax.tick_params(axis='y', labelsize=25, grid_alpha=0.3)
    
    # Set border thickness and color
    for spine in ax.spines.values():
        spine.set_linewidth(4)
        spine.set_color(black_semi)
    
    # Remove extra whitespace
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.92, wspace=0.1, hspace=0.22)

    # Save the plot in multiple formats
    # Add condition info to filename if applicable
    if condition_column is not None and condition_threshold is not None:
        condition_suffix = f"_conditional_{condition_column.replace('_', '')}_{condition_operator.replace('>', 'gt').replace('<', 'lt')}_{condition_threshold}"
    else:
        condition_suffix = ""
    
    boxplot_file_png = os.path.join(output_dir, f'rmsd_boxplot{condition_suffix}.png')
    boxplot_file_pdf = os.path.join(output_dir, f'rmsd_boxplot{condition_suffix}.pdf')
    boxplot_file_svg = os.path.join(output_dir, f'rmsd_boxplot{condition_suffix}.svg')
    plt.savefig(boxplot_file_png, dpi=600, bbox_inches='tight')
    plt.savefig(boxplot_file_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(boxplot_file_svg, format='svg', bbox_inches='tight')
    plt.show()
    
    # Print statistics summary to console
    print(f"\nStatistics Summary:")
    print("-" * 80)
    for stat in stats_summary:
        print(f"{stat['Method']:<20} {stat['Pose']:<10} Success Rate: {stat['Success_Rate']:.1f}% Mean: {stat['Mean_RMSD']:.2f}")
        
    print(f"\nFigures saved to: {output_dir}")


def analyze_rmsd_statistics_(
    csv_file, methods=None, output_dir='./',
    rmsd_threshold=2, fig_size=(18, 12),
    title='', dpi=300, ylim=16
):
    """
    Plot RMSD boxplots for different docking methods (Top-1 and Top-5) and save the figure.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing RMSD data.
    methods : dict, optional
        Dictionary mapping method names to [Top-1 column, Top-5 column].
    output_dir : str, optional
        Directory to save the output plots.
    rmsd_threshold : float, optional
        RMSD threshold for success (red dashed line).
    fig_size : tuple, optional
        Figure size (width, height).
    title : str, optional
        Title for the plot.
    dpi : int, optional
        Figure DPI.

    Returns
    -------
    None
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Default methods if not provided
    if methods is None:
        methods = {
        "AF3": ['AutoDock_1pose_RMSD_AlphaFold', 'AutoDock_5pose_RMSD_AlphaFold'],
        "ESMfold": ['AutoDock_1pose_RMSD_ESMFold', 'AutoDock_5pose_RMSD_ESMFold'],
        "Apo": ['AutoDock_1pose_RMSD_Apo', 'AutoDock_5pose_RMSD_Apo'],
        "Holo*": ['AutoDock_1pose_RMSD_Holo', 'AutoDock_5pose_RMSD_Holo'],
        "Holo": ['AutoDock_1pose_RMSD', 'AutoDock_5pose_RMSD'],
    }
    data = pd.read_csv(csv_file)
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi * 2
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'

    # Color scheme for Top-1 and Top-5
    pose_colors = {
        "Top-5": (0.2, 0.4, 0.8, 0.7),  # Blue
        "Top-1": (1.0, 0.5, 0.0, 0.9)   # Orange
    }

    # Prepare data for plotting
    grouped_data = []
    grouped_positions = []
    grouped_labels = []
    grouped_colors = []
    grouped_types = []
    method_pose_data = {}
    pose_gap = 0.15
    method_gap = 0.35
    current_pos = 1

    for method_name, (col1, col5) in methods.items():
        method_pose_data[method_name] = {}
        
        # Top-1
        if col1 in data.columns:
            values = data[col1].dropna().values
            if len(values) > 0:
                grouped_positions.append(current_pos)
                grouped_data.append(values)
                grouped_labels.append(method_name)
                grouped_colors.append(pose_colors["Top-1"])
                grouped_types.append("Top-1")
                method_pose_data[method_name]['Top-1'] = {'values': values, 'position': current_pos}
        
        # Top-5
        if col5 in data.columns:
            values = data[col5].dropna().values
            if len(values) > 0:
                grouped_positions.append(current_pos + pose_gap)
                grouped_data.append(values)
                grouped_labels.append(method_name)
                grouped_colors.append(pose_colors["Top-5"])
                grouped_types.append("Top-5")
                method_pose_data[method_name]['Top-5'] = {'values': values, 'position': current_pos + pose_gap}
        
        current_pos += pose_gap + method_gap

    # Create boxplot
    plt.figure(figsize=fig_size)
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(True, axis='y', linestyle='-', alpha=0.2)

    black_semi = (0, 0, 0, 0.8)
    boxprops = {'linewidth': 4.0, 'color': black_semi}
    whiskerprops = {'linewidth': 4.0, 'color': black_semi}
    medianprops = {'color': black_semi, 'linewidth': 4.0}
    capprops = {'linewidth': 4.0, 'color': black_semi}

    bp = ax.boxplot(
        grouped_data, positions=grouped_positions, widths=0.13,
        patch_artist=True, showfliers=False,
        boxprops=boxprops, whiskerprops=whiskerprops,
        medianprops=medianprops, capprops=capprops, zorder=3
    )

    # Set box color and style
    for patch, color in zip(bp['boxes'], grouped_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(black_semi)
        patch.set_linewidth(4.5)
        patch.set_alpha(0.7)

    # Plot mean value as a diamond marker
    for values, pos in zip(grouped_data, grouped_positions):
        mean_val = np.mean(values)
        plt.plot(pos, mean_val, marker='D', color=black_semi, markersize=9, zorder=4)

    # Draw the RMSD threshold line
    plt.axhline(y=rmsd_threshold, color='r', linestyle='--', linewidth=4.5, zorder=1)

    # Calculate method label positions
    method_positions = {}
    for method_name in methods.keys():
        positions = [pos for pos, label in zip(grouped_positions, grouped_labels) if label == method_name]
        if positions:
            method_positions[method_name] = sum(positions) / len(positions)

    # Add method names below the boxplots
    for method_name, pos in method_positions.items():
        plt.text(pos, -0.8, method_name,
                 ha='center', va='top', fontsize=24, fontweight='bold')

    # Remove x-axis ticks
    plt.xticks([])

    if title == 'AutoDock-GPU':
        plt.ylabel(f'{title}, L-RMSD (Å)', fontsize=23, fontweight='bold', labelpad=12)
    else:
        plt.ylabel(f'{title}, L-RMSD (Å)', fontsize=26, fontweight='bold', labelpad=12)

    # Set y-axis and x-axis limits
    plt.ylim(0, ylim)
    plt.yticks(range(0, ylim+1, 5))
    plt.xlim(min(grouped_positions) - 0.2, max(grouped_positions) + 0.2)

    # Set y-axis tick style
    ax.tick_params(axis='y', direction='out', length=8, width=3.5,
                   pad=10, labelsize=24, grid_alpha=0.3)

    # Set border thickness and color
    for spine in ax.spines.values():
        spine.set_linewidth(4.0)
        spine.set_color(black_semi)

    # Remove extra whitespace
    plt.tight_layout()
    plt.subplots_adjust(left=0.07, right=0.95, bottom=0.2, top=0.92, wspace=0.1, hspace=0.22)

    # Save the plot in multiple formats
    boxplot_file_png = os.path.join(output_dir, 'rmsd_boxplot.png')
    boxplot_file_pdf = os.path.join(output_dir, 'rmsd_boxplot.pdf')
    boxplot_file_svg = os.path.join(output_dir, 'rmsd_boxplot.svg')
    plt.savefig(boxplot_file_png, dpi=600, bbox_inches='tight')
    plt.savefig(boxplot_file_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(boxplot_file_svg, format='svg', bbox_inches='tight')
    plt.show()
    print(f"Boxplot saved to {output_dir}")


def plot_combined_af3_analysis(
    csv_file='/data/thzhang/CoFD-Bench/pdb_benchmark_RMSD.csv',
    distance_csv='/data/thzhang/CoFD-Bench/pdb_benchmark.csv',
    output_dir='./',
    output_prefix='combined_af3_analysis',
    rmsd_threshold=2.0,
    dpi=300,
    fig_size=(40, 10),
    axis_limit='auto',
    percentile_cutoff=95,
    x_limit=None,
    y_limit=None
):
    """
    Integrate three AF3 analysis plots into one large figure, maintaining consistent size for each subplot
    """
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set unified matplotlib parameters
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi * 2
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans']
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    
    # Unified color settings
    black_semi = (0, 0, 0, 0.8)
    scatter_color = (0.2, 0.4, 0.8, 0.6)
    outlier_color = (0.8, 0.2, 0.2, 0.9)
    diagonal_color = (0.5, 0.5, 0.5, 0.8)
    threshold_color = 'red'
    correct_color = (0.2, 0.7, 0.3, 0.7)
    incorrect_color = (0.8, 0.3, 0.2, 0.7)
    
    # Create 1x3 subplot layout, three subplots with consistent size, slightly lower height
    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], 
                          wspace=0.15, left=0.08, right=0.98, top=0.85, bottom=0.18)
    
    # Read data
    data = pd.read_csv(csv_file)
    distance_data = pd.read_csv(distance_csv)
    merged_data = pd.merge(distance_data, data, on='pdb_id', how='inner')
    
    # ==================== Left: AF3 covalent vs non-covalent comparison ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('white')
    ax1.grid(True, linestyle='-', alpha=0.2)
    
    # Filter valid data
    af3_cov_col = 'AF3_1pose_RMSD'
    af3_noncov_col = 'AF3_noncov_1pose_RMSD'
    valid_data = data[[af3_cov_col, af3_noncov_col]].dropna()
    
    # Dynamically determine axis limits
    if axis_limit == 'auto':
        cov_percentile = valid_data[af3_cov_col].quantile(percentile_cutoff/100)
        noncov_percentile = valid_data[af3_noncov_col].quantile(percentile_cutoff/100)
        axis_limit = max(cov_percentile, noncov_percentile)
        axis_limit = np.ceil(axis_limit * 1.1)
    
    # Separate normal data points and outliers
    normal_mask = (valid_data[af3_cov_col] <= axis_limit) & (valid_data[af3_noncov_col] <= axis_limit)
    normal_data = valid_data[normal_mask]
    outlier_data = valid_data[~normal_mask]
    
    # Draw normal data points
    ax1.scatter(normal_data[af3_noncov_col], normal_data[af3_cov_col], 
               color=scatter_color, s=45, alpha=0.7, 
               edgecolors=black_semi, linewidths=0.8, zorder=3)
    
    # Handle outlier display
    if len(outlier_data) > 0:
        # x-axis outliers
        x_outliers = outlier_data[(outlier_data[af3_noncov_col] > axis_limit) & 
                                 (outlier_data[af3_cov_col] <= axis_limit)]
        if len(x_outliers) > 0:
            ax1.scatter([axis_limit * 0.98] * len(x_outliers), x_outliers[af3_cov_col], 
                       color=outlier_color, s=80, alpha=0.9, 
                       edgecolors='black', linewidths=1.5, zorder=5, marker='>')
        
        # y-axis outliers
        y_outliers = outlier_data[(outlier_data[af3_cov_col] > axis_limit) & 
                                 (outlier_data[af3_noncov_col] <= axis_limit)]
        if len(y_outliers) > 0:
            ax1.scatter(y_outliers[af3_noncov_col], [axis_limit * 0.98] * len(y_outliers), 
                       color=outlier_color, s=80, alpha=0.9, 
                       edgecolors='black', linewidths=1.5, zorder=5, marker='^')
        
        # Both axis outliers
        both_outliers = outlier_data[(outlier_data[af3_cov_col] > axis_limit) & 
                                    (outlier_data[af3_noncov_col] > axis_limit)]
        if len(both_outliers) > 0:
            n_both = len(both_outliers)
            if n_both == 1:
                ax1.scatter([axis_limit * 0.96], [axis_limit * 0.96], 
                           color=outlier_color, s=100, alpha=0.9, 
                           edgecolors='black', linewidths=1.5, zorder=5, marker='s')
            else:
                x_positions = np.linspace(axis_limit * 0.94, axis_limit * 0.98, n_both)
                y_positions = np.linspace(axis_limit * 0.94, axis_limit * 0.98, n_both)
                ax1.scatter(x_positions, y_positions, 
                           color=outlier_color, s=100, alpha=0.9, 
                           edgecolors='black', linewidths=1.5, zorder=5, marker='s')
    
    # Draw diagonal and threshold lines
    ax1.plot([0, axis_limit], [0, axis_limit], 
            color=diagonal_color, linestyle='-', linewidth=3.5, 
            alpha=0.8, zorder=2)
    ax1.axhline(y=rmsd_threshold, color=threshold_color, 
               linestyle='--', linewidth=3.5, alpha=0.8, zorder=2)
    ax1.axvline(x=rmsd_threshold, color=threshold_color, 
               linestyle='--', linewidth=3.5, alpha=0.8, zorder=2)
    
    # Set axis labels and ticks
    ax1.set_xlim(0, axis_limit)
    ax1.set_ylim(0, axis_limit)
    ax1.set_xlabel('AF3 (Non-covalent), L-RMSD (Å)', 
                  fontsize=27, fontweight='bold', labelpad=12)  # 22+1=23
    ax1.set_ylabel('AF3 (Covalent), L-RMSD (Å)', 
                  fontsize=27, fontweight='bold', labelpad=12)  # 22+1=23
    
    # Dynamically set ticks
    if axis_limit <= 10:
        tick_interval = 1
    elif axis_limit <= 20:
        tick_interval = 2
    else:
        tick_interval = 5
    
    ticks = np.arange(0, axis_limit + tick_interval, tick_interval)
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.tick_params(axis='both', direction='out', length=6, width=2,
                   pad=8, labelsize=18)
    
    # Set borders
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color(black_semi)
    
    # Add (a) title to upper left corner
    ax1.text(-0.1, 1.07, '(a)', transform=ax1.transAxes, 
            fontsize=28, fontweight='bold', va='top', ha='left')
    
    # Add legend to lower right corner, slightly to the left
    legend_elements_a = [
        plt.scatter([], [], color=scatter_color, s=80,
                edgecolors=black_semi, linewidths=1.5, 
                label='AF3 Predictions'),
        plt.scatter([], [], color=outlier_color, s=80,
                marker='^', edgecolors='black', 
                linewidths=1.5, label='Outliers'),
        Line2D([0], [0], color=diagonal_color, linestyle='-', lw=2.5, label='Diagonal Line'),
        Line2D([0], [0], color=threshold_color, linestyle='--', lw=2.5, label='RMSD=2Å'),
    ]
    
    ax1.legend(handles=legend_elements_a, 
              loc='lower right',
              bbox_to_anchor=(0.95, 0.02),
              fontsize=14,
              frameon=True, 
              framealpha=0.9,
              fancybox=True, 
              edgecolor=black_semi)
    
    # ==================== Middle: AF3 SuCOS comparison ====================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('white')
    ax2.grid(True, axis='y', linestyle='-', alpha=0.2)
    
    # Define AF3 methods
    af3_methods = {
        "AF3 (Covalent)": ['AF3_1pose_RMSD', 'AF3_5pose_RMSD'],
        "AF3 (Non-covalent)": ['AF3_noncov_1pose_RMSD', 'AF3_noncov_5pose_RMSD']
    }
    
    # SuCOS grouping
    bins = [-1, 20, 40, 60, 80, 100]
    labels = ['1-20', '20-40', '40-60', '60-80', '80-100']
    
    # Clean data and create SuCOS groups
    data_clean = merged_data.dropna(subset=['SuCOS_pocket_qcov']).copy()
    data_clean['SuCOS_group'] = pd.cut(data_clean['SuCOS_pocket_qcov'], bins=bins, labels=labels, right=True)
    
    # Calculate success rates
    success_rates = {rmsd_col: [] for rmsd_col in ['AF3_1pose_RMSD', 'AF3_5pose_RMSD', 
                                                  'AF3_noncov_1pose_RMSD', 'AF3_noncov_5pose_RMSD'] 
                   if rmsd_col in data_clean.columns}
    valid_groups = []
    
    for group in labels:
        group_df = data_clean[data_clean['SuCOS_group'] == group]
        if len(group_df) > 0:
            valid_groups.append(group)
            
            for rmsd_col in success_rates.keys():
                valid_rows = group_df.dropna(subset=[rmsd_col])
                if len(valid_rows) > 0:
                    success_count = sum(valid_rows[rmsd_col] < rmsd_threshold)
                    success_rate = success_count / len(valid_rows) * 100
                    success_rates[rmsd_col].append(success_rate)
                else:
                    success_rates[rmsd_col].append(0)
    
    # Default colors
    default_colors = {
        "AF3 (Covalent)": "#2E86AB",
        "AF3 (Non-covalent)": "#F18F01"
    }
    
    x_range = range(len(valid_groups))
    
    # Draw lines for each AF3 method
    for method_name, cols in af3_methods.items():
        if len(cols) == 2 and cols[0] in data_clean.columns and cols[1] in data_clean.columns:
            pose1_col = next((rmsd_col for rmsd_col in cols if "1pose" in rmsd_col), None)
            pose5_col = next((rmsd_col for rmsd_col in cols if "5pose" in rmsd_col), None)
            
            if pose1_col is None or pose5_col is None:
                continue
                
            if pose1_col in success_rates and pose5_col in success_rates:
                rates_1pose = success_rates[pose1_col]
                rates_5pose = success_rates[pose5_col]
                color = default_colors.get(method_name, plt.cm.tab10(0))
                
                # Draw Top-1 line
                ax2.plot(x_range, rates_1pose, 
                       marker='o', markersize=10, linewidth=3.0,
                       linestyle='-', color=color, zorder=3)
                
                # Draw Top-5 line
                ax2.plot(x_range, rates_5pose, 
                       marker='s', markersize=10, linewidth=3.0,
                       linestyle='--', color=color, zorder=3)
                
                # Fill area
                ax2.fill_between(x_range, rates_1pose, rates_5pose, 
                               color=color, alpha=0.15, zorder=1)
    
    # Set labels
    ax2.set_ylabel('Success Rate (%)', fontsize=27, labelpad=-4, fontweight='bold')  # 22+1=23
    ax2.set_xlabel('SuCOS-pocket similarity', fontsize=27, labelpad=12, fontweight='bold')  # 22+1=23
    
    # Set ticks and limits
    ax2.set_xticks(x_range)
    ax2.set_xticklabels(valid_groups, fontsize=18)
    ax2.set_ylim(0, 100)
    
    # Set tick parameters
    ax2.tick_params(axis='both', direction='out', length=6, width=2,
                   pad=8, labelsize=18)
    
    # Set borders
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color(black_semi)
    
    # Add (b) title to upper left corner
    ax2.text(-0.1, 1.07, '(b)', transform=ax2.transAxes, 
            fontsize=28, fontweight='bold', va='top', ha='left')
    
    # Add legend to lower right corner, slightly to the left
    legend_elements_b = [
        Line2D([0], [0], color="#2E86AB", lw=3.0, label='AF3 (Covalent)'),
        Line2D([0], [0], color="#F18F01", lw=3.0, label='AF3 (Non-covalent)'),
        Line2D([0], [0], color='black', linestyle='-', lw=2.5, label='Top-1'),
        Line2D([0], [0], color='black', linestyle='--', lw=2.5, label='Top-5'),
    ]
    
    ax2.legend(handles=legend_elements_b, 
              loc='lower right',
              bbox_to_anchor=(0.98, 0.02),
              fontsize=14,
              frameon=True, 
              framealpha=0.9,
              fancybox=True, 
              edgecolor=black_semi)
    
    # ==================== Right: Distance analysis ====================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('white')
    ax3.grid(True, linestyle='-', alpha=0.3)
    
    # Filter valid data and calculate delta distance
    required_cols = ['AF3_noncov_1pose_RMSD', 'Top-1-distance', 'real_distance', 'Top-1-correct']
    valid_distance_data = merged_data[required_cols + ['pdb_id']].dropna()
    valid_distance_data = valid_distance_data.copy()
    valid_distance_data['delta_distance'] = valid_distance_data['Top-1-distance'] - valid_distance_data['real_distance']
    
    # Separate data by correctness
    correct_data = valid_distance_data[valid_distance_data['Top-1-correct'] == True]
    incorrect_data = valid_distance_data[valid_distance_data['Top-1-correct'] == False]
    
    # Draw incorrect predictions (behind)
    if len(incorrect_data) > 0:
        ax3.scatter(incorrect_data['AF3_noncov_1pose_RMSD'], incorrect_data['delta_distance'],
                  color=incorrect_color, s=50, alpha=0.7,
                  edgecolors=black_semi, linewidths=0.8, zorder=3)
    
    # Draw correct predictions
    if len(correct_data) > 0:
        ax3.scatter(correct_data['AF3_noncov_1pose_RMSD'], correct_data['delta_distance'],
                  color=correct_color, s=50, alpha=0.7,
                  edgecolors=black_semi, linewidths=0.8, zorder=4)
    
    # Set coordinate axes to logarithmic scale
    ax3.set_xscale('log')  # x-axis uses logarithmic scale
    ax3.set_yscale('symlog', linthresh=1.0)  # y-axis uses symmetric logarithmic scale, linthresh=1.0 means linear scale within ±1 range
    
    # Add horizontal and vertical lines
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=2.5, alpha=0.8, zorder=2)  # Δd = 0
    ax3.axvline(x=rmsd_threshold, color='red', linestyle='--', linewidth=2.5, alpha=0.8, zorder=2)
    
    # Set axis limits
    if x_limit is None:
        x_min, x_max = valid_distance_data['AF3_noncov_1pose_RMSD'].min(), valid_distance_data['AF3_noncov_1pose_RMSD'].max()
        # For log scale, use multiplication/division instead of addition/subtraction to set margins
        x_limit = (max(x_min * 0.8, 0.1), x_max * 1.2)  # Ensure lower limit is not too small
    
    if y_limit is None:
        # For symlog-scale, set reasonable y-axis range
        y_min = valid_distance_data['delta_distance'].min()
        y_max = valid_distance_data['delta_distance'].max()
        y_padding = max(abs(y_min), abs(y_max)) * 0.1  # Add 10% margin
        y_limit = (y_min - y_padding, y_max + y_padding)
    
    ax3.set_xlim(x_limit)
    ax3.set_ylim(y_limit)
    
    # Set labels
    ax3.set_xlabel('AF3 Non-covalent, L-RMSD (Å)', fontsize=27, fontweight='bold', labelpad=12)  # 22+1=23
    ax3.set_ylabel('Δd = d₁ - d₂ (Å)', fontsize=27, fontweight='bold', labelpad=-10)  # 22+1=23, symlog-scale axis label
    # Set tick parameters
    ax3.tick_params(axis='both', direction='out', length=6, width=2,
                   pad=8, labelsize=18)
    
    # Set borders
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_color(black_semi)
    
    # Add (c) title to upper left corner
    ax3.text(-0.1, 1.07, '(c)', transform=ax3.transAxes, 
            fontsize=28, fontweight='bold', va='top', ha='left')
    
    # Add legend to lower right corner, slightly to the left
    legend_elements_c = [
        plt.scatter([], [], color=correct_color, s=80,
                edgecolors=black_semi, linewidths=1.5, 
                label='Residue Match'),
        plt.scatter([], [], color=incorrect_color, s=80,
                edgecolors=black_semi, linewidths=1.5, 
                label='Residue Mismatch'),
        Line2D([0], [0], color='black', linestyle='-', lw=2.5, label='Δd = 0'),
        Line2D([0], [0], color='red', linestyle='--', lw=2.5, label='RMSD=2Å'),
    ]
    
    ax3.legend(handles=legend_elements_c, 
              loc='lower right',
              bbox_to_anchor=(0.98, 0.02),
              fontsize=14,
              frameon=True, 
              framealpha=0.9,
              fancybox=True, 
              edgecolor=black_semi)
    
    # Save images
    png_file = f"{output_dir}/{output_prefix}.png"
    pdf_file = f"{output_dir}/{output_prefix}.pdf"
    svg_file = f"{output_dir}/{output_prefix}.svg"
    
    plt.savefig(png_file, dpi=600, bbox_inches='tight')
    plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
    plt.savefig(svg_file, format='svg', bbox_inches='tight')
    plt.show()
    
    return {
        'axis_limit': axis_limit,
        'outliers_count': len(outlier_data),
        'valid_samples': len(valid_data),
        'distance_stats': {
            'total_samples': len(valid_distance_data),
            'correct_count': len(correct_data),
            'incorrect_count': len(incorrect_data),
            'correct_percentage': len(correct_data)/len(valid_distance_data)*100
        },
        'files_saved': [png_file, pdf_file, svg_file]
    }


def plot_SuCOS_pocket_all(
    data,
    methods_dict,
    rmsd_threshold=2,
    fig_size=(24, 20),
    custom_colors=None,
    title_font_size=18,
    dpi=300,
    output_dir='.',
    output_prefix='success_rates_plot'
):
    """
    Plot success rates of multiple docking methods in a 2x2 grid, and save the figure.

    Parameters
    ----------
    data : DataFrame
        DataFrame containing SuCOS_pocket_qcov and RMSD columns.
    methods_dict : dict
        Dictionary of method groups, e.g. {"Schrodinger": methods_Schrodinger, ...}
    rmsd_threshold : float
        RMSD threshold for success.
    fig_size : tuple
        Figure size.
    custom_colors : dict or None
        Custom color mapping for methods.
    title_font_size : int
        Title font size.
    dpi : int
        Figure DPI.
    output_dir : str
        Directory to save the output images.
    output_prefix : str
        Prefix for the output image files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi * 2

    fig = plt.figure(figsize=fig_size)
    y_min, y_max = 0, 100
    bins = [-1, 20, 40, 60, 80, 100]
    labels = ['1-20', '20-40', '40-60', '60-80', '80-100']
    x_ticks = range(len(labels))
    # Increase subplot spacing to avoid overlap
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.15, hspace=0.35)
    black_semi = (0, 0, 0, 0.8)
    method_name_map = {
        "alphafold3": "AF3",
        "AlphaFold3": "AF3",
        "af3": "AF3",
        "chai1": "Chai-1",
        "chai1_msa": "Chai-1 (MSA)",
        "boltz": "Boltz-1x",
        "Autodock": "AutoDock-GPU",
        "AutoDock": "AutoDock-GPU",
    }
    pose_type_labels = ["Top-1", "Top-5"]

    for i, (method_group_name, methods) in enumerate(methods_dict.items()):
        if i >= 4:
            break
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor('white')
        ax.grid(True, axis='y', linestyle='-', alpha=0.2)
        rmsd_columns = []
        for method_cols in methods.values():
            rmsd_columns.extend(method_cols)
        # Key: add .copy() to avoid SettingWithCopyWarning
        data_clean = data.dropna(subset=['SuCOS_pocket_qcov']).copy()
        data_clean['SuCOS_group'] = pd.cut(data_clean['SuCOS_pocket_qcov'], bins=bins, labels=labels, right=True)
        success_rates = {rmsd_col: [] for rmsd_col in rmsd_columns if rmsd_col in data_clean.columns}
        sample_counts = {group: 0 for group in labels}
        valid_groups = []
        for group in labels:
            group_df = data_clean[data_clean['SuCOS_group'] == group]
            valid_groups.append(group)
            sample_counts[group] = len(group_df)
            for rmsd_col in rmsd_columns:
                if rmsd_col in data_clean.columns:
                    valid_rows = group_df.dropna(subset=[rmsd_col])
                    if len(valid_rows) > 0:
                        success_count = sum(valid_rows[rmsd_col] < rmsd_threshold)
                        success_rate = success_count / len(valid_rows) * 100
                        success_rates[rmsd_col].append(success_rate)
                    else:
                        success_rates[rmsd_col].append(0)
        default_colors = {
            list(methods.keys())[0] if len(methods) > 0 else "dummy0": plt.cm.tab10(0),
            list(methods.keys())[1] if len(methods) > 1 else "dummy1": plt.cm.tab10(1),
            list(methods.keys())[2] if len(methods) > 2 else "dummy2": plt.cm.tab10(2),
            list(methods.keys())[3] if len(methods) > 3 else "dummy3": plt.cm.tab10(3),
            list(methods.keys())[4] if len(methods) > 4 else "dummy4": plt.cm.tab10(4),
            list(methods.keys())[5] if len(methods) > 5 else "dummy5": plt.cm.tab10(5),
            list(methods.keys())[6] if len(methods) > 6 else "dummy6": plt.cm.tab10(6),
            list(methods.keys())[7] if len(methods) > 7 else "dummy7": plt.cm.tab10(7),
            list(methods.keys())[8] if len(methods) > 8 else "dummy8": plt.cm.tab10(8),
            list(methods.keys())[9] if len(methods) > 9 else "dummy9": plt.cm.tab10(9),
        }
        method_colors = default_colors.copy()
        if custom_colors is not None:
            for method_name, color in custom_colors.items():
                if method_name in method_colors:
                    method_colors[method_name] = color
        pose_markers = {
            "1pose": "o",
            "5pose": "s"
        }
        pose_linestyles = {
            "1pose": "-",
            "5pose": "--"
        }
        x_range = range(len(valid_groups))
        if method_group_name.lower() == '(a)co-folding' or i == 0:
            for j, group in enumerate(valid_groups):
                n = sample_counts[group]
                offset_y = 5
                ax.annotate(f"n={n}", 
                          xy=(j, 5), 
                          xytext=(j, offset_y),
                          fontsize=16,  # Increase font size
                          ha='center', 
                          va='bottom',
                          color='black',
                          bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
        for method_name, cols in methods.items():
            if len(cols) == 2 and cols[0] in data_clean.columns and cols[1] in data_clean.columns:
                pose1_col = next((rmsd_col for rmsd_col in cols if "1pose" in rmsd_col), None)
                pose5_col = next((rmsd_col for rmsd_col in cols if "5pose" in rmsd_col), None)
                if pose1_col is None or pose5_col is None:
                    continue
                if pose1_col in success_rates and pose5_col in success_rates:
                    rates_1pose = success_rates[pose1_col]
                    rates_5pose = success_rates[pose5_col]
                    color = method_colors.get(method_name, plt.cm.tab10(0))
                    ax.plot(x_range, rates_1pose, 
                            marker=pose_markers["1pose"],
                            markersize=10,
                            linewidth=3.5,
                            linestyle=pose_linestyles["1pose"],
                            color=color,
                            zorder=3)
                    ax.plot(x_range, rates_5pose, 
                            marker=pose_markers["5pose"],
                            markersize=10,
                            linewidth=3.5,
                            linestyle=pose_linestyles["5pose"],
                            color=color,
                            zorder=3)
                    ax.fill_between(x_range, rates_1pose, rates_5pose, 
                                   color=color, alpha=0.15, zorder=1)
        legend_elements = []
        for method_name in methods.keys():
            if method_name in method_colors and any(rmsd_col in data_clean.columns for rmsd_col in methods[method_name]):
                display_name = method_name_map.get(method_name, method_name.replace('_', ' '))
                legend_elements.append(Line2D([0], [0], color=method_colors[method_name], lw=3.5, label=display_name))
        legend_elements.append(Line2D([0], [0], color='black', linestyle='-', lw=3.0, label=pose_type_labels[0]))
        legend_elements.append(Line2D([0], [0], color='black', linestyle='--', lw=3.0, label=pose_type_labels[1]))
        
        # Shorten title, reduce space usage, increase font size
        if method_group_name.strip() != "":
            ax.set_title(method_group_name, fontsize=32, fontweight='bold', pad=20)
        
        # Ensure y-axis labels are displayed for left columns, increase font size
        # print(f"Processing subplot - Row: {row}, Col: {col}, Method group: {method_group_name}")
        if col == 0:
            # print(f"Setting ylabel for left column (row={row}, col={col})")
            ylabel = ax.set_ylabel('Success Rate (%)', fontsize=26, labelpad=4, fontweight='bold', color='black')
            # Force set label position and visibility
            ax.yaxis.set_label_coords(-0.08, 0.5)
            ax.yaxis.label.set_visible(True)
            # print(f"Y-label set successfully: '{ylabel.get_text()}'")

        
        # Only display x-axis labels for bottom row, increase font size
        if row == 1:
            ax.set_xlabel('SuCOS-pocket similarity', fontsize=26, labelpad=15, fontweight='bold')
        
        ax.set_xticks(x_range)
        ax.set_xticklabels(valid_groups, fontsize=22)  # Increase x-axis tick label font size
        ax.set_ylim(y_min, y_max)
        
        # Increase tick label font size, add padding to avoid overlap
        ax.tick_params(axis='x', direction='out', length=8, width=3.5, pad=18, 
                      labelsize=22, grid_alpha=0.3)
        ax.tick_params(axis='y', direction='out', length=8, width=3.5, pad=10, 
                      labelsize=22, grid_alpha=0.3)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3.5)
            spine.set_color(black_semi)
        
        if legend_elements:
            # Adjust legend font and position
            ax.legend(handles=legend_elements, loc='upper left', fontsize=20, 
                     frameon=True, framealpha=0.9, fancybox=True, ncol=1)
        
        # Add subplot labels
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        ax.text(-0.12, 1.11, subplot_labels[i], transform=ax.transAxes, fontsize=26, fontweight='bold', 
                va='top', ha='left', color='black')
    
    # Adjust subplot layout, leave more space for y-axis labels, increase top and bottom spacing
    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.88)
    
    # Ensure all y-axis labels are displayed
    for ax in fig.get_axes():
        if ax.get_position().x0 < 0.5:  # Left subplots
            ax.yaxis.label.set_visible(True)

    fig.canvas.draw()
    
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.svg'), format='svg', bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_combined_plif_analysis(
    rmsd_csv='pdb_benchmark.csv',
    plif_csv='plifs_recovery_df_v1.csv',
    output_prefix='combined_plif_analysis',
    output_dir='.',
    methods_info=None,
    display_names=None
):
    """
    Generate a combined bar and violin plot for PLIF recovery analysis.

    Parameters
    ----------
    benchmark_csv : str
        Path to the benchmark CSV file.
    plif_csv : str
        Path to the PLIF recovery CSV file.
    output_prefix : str
        Prefix for the output image files.
    output_dir : str
        Directory to save the output images.
    methods_info : dict or None
        Mapping from method name to RMSD column name. If None, use default.
    display_names : dict or None
        Mapping from method name to display name. If None, use default.
    """
    # Font settings
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    black_semi = (0, 0, 0, 0.8)

    # Default method and display name mappings
    if methods_info is None:
        methods_info = {
            'AF3': 'AF3_5pose_RMSD',
            'Chai-1': 'Chai-1_5pose_RMSD',
            'Chai-1_MSA': 'Chai-1_MSA_5pose_RMSD',
            'Boltz-1': 'Boltz-1_5pose_RMSD',
            'AutoDock': 'AutoDock_5pose_RMSD',
            'CovDock': 'CovDock_5pose_RMSD',
            'GNINA': 'GNINA_5pose_RMSD'
        }
    if display_names is None:
        display_names = {
            'AF3': 'AF3',
            'Chai-1': 'Chai-1',
            'Chai-1_MSA': 'Chai-1(MSA)',
            'Boltz-1': 'Boltz-1x',
            'AutoDock': 'AutoDock-GPU',
            'CovDock': 'CovDock',
            'GNINA': 'GNINA'
        }

    ordered_categories = ['Perfect Match (1.0)', 'Good Match (0.5-1.0)', 'Poor Match (< 0.5)']

    # Read data
    df_benchmark = pd.read_csv(rmsd_csv)
    df_plifs = pd.read_csv(plif_csv, header=None, 
                           names=['method', 'target', 'count_recovery', 'reference_plifs', 'plifs'])
    df_plifs['count_recovery'] = pd.to_numeric(df_plifs['count_recovery'], errors='coerce')

    # Prepare bar plot data
    methods_results = {}
    for method_name, rmsd_column in methods_info.items():
        if rmsd_column not in df_benchmark.columns:
            continue
        valid_pdbs = df_benchmark[df_benchmark[rmsd_column].notna()]['pdb_id'].unique()
        if len(valid_pdbs) == 0:
            continue
        successful_pdb_ids = df_benchmark[
            (df_benchmark[rmsd_column].notna()) & 
            (df_benchmark[rmsd_column] < 2)
        ]['pdb_id'].unique()
        if len(successful_pdb_ids) == 0:
            continue
        filtered_plifs = df_plifs[
            (df_plifs['target'].isin(successful_pdb_ids)) & 
            (df_plifs['method'] == method_name)
        ]
        if len(filtered_plifs) == 0:
            continue
        filtered_plifs = filtered_plifs.dropna(subset=['count_recovery'])
        def categorize_recovery(value):
            if value == 1.0:
                return 'Perfect Match (1.0)'
            elif 0.5 <= value < 1.0:
                return 'Good Match (0.5-1.0)'
            else:
                return 'Poor Match (< 0.5)'
        filtered_plifs['recovery_category'] = filtered_plifs['count_recovery'].apply(categorize_recovery)
        plif_present_pdbs = filtered_plifs['target'].unique()
        category_counts = filtered_plifs['recovery_category'].value_counts()
        category_counts = category_counts.reindex(ordered_categories).fillna(0).astype(int)
        category_percentages = {}
        for category, count in category_counts.items():
            category_percentages[category] = round(count / len(valid_pdbs) * 100, 1)
        failed_with_rmsd = len(valid_pdbs) - len(successful_pdb_ids)
        failed_percentage = round(failed_with_rmsd / len(valid_pdbs) * 100, 1)
        display_name = display_names.get(method_name, method_name)
        methods_results[display_name] = {
            'total_pdbs': len(valid_pdbs),
            'successful_pdbs': len(successful_pdb_ids),
            'plif_present_pdbs': len(plif_present_pdbs),
            'category_counts': category_counts,
            'category_percentages': category_percentages,
            'failed_count': failed_with_rmsd,
            'failed_percentage': failed_percentage
        }

    # Prepare violin plot data
    all_method_data = []
    for method_name, rmsd_column in methods_info.items():
        if rmsd_column not in df_benchmark.columns:
            continue
        valid_pdbs = df_benchmark[df_benchmark[rmsd_column].notna()]['pdb_id'].unique()
        if len(valid_pdbs) == 0:
            continue
        successful_pdb_ids = set(df_benchmark[
            (df_benchmark[rmsd_column].notna()) & 
            (df_benchmark[rmsd_column] < 2)
        ]['pdb_id'].unique())
        failed_pdb_ids = set(df_benchmark[
            (df_benchmark[rmsd_column].notna()) & 
            (df_benchmark[rmsd_column] >= 2)
        ]['pdb_id'].unique())
        method_plifs = df_plifs[df_plifs['method'] == method_name].copy()
        if len(method_plifs) == 0:
            continue
        method_plifs = method_plifs.dropna(subset=['count_recovery'])
        method_plifs['display_name'] = display_names.get(method_name, method_name)
        method_plifs['docking_result'] = method_plifs['target'].apply(
            lambda x: 'Success' if x in successful_pdb_ids else 
                     ('Failure' if x in failed_pdb_ids else 'Unknown')
        )
        method_plifs['perfect_match'] = method_plifs['count_recovery'] == 1.0
        all_method_data.append(method_plifs[['display_name', 'target', 'count_recovery', 'docking_result', 'perfect_match']])
    all_plifs_df = pd.concat(all_method_data)

    # Plotting
    method_colors = {}
    for i, method_name in enumerate(methods_results.keys()):
        method_colors[method_name] = plt.cm.tab10(i % 10)

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.2)
    method_names = list(methods_results.keys())
    index = np.arange(len(method_names))

    # Bar plot
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('white')
    ax1.grid(True, axis='y', linestyle='-', alpha=0.2)
    bar_width = 0.55
    success_colors = ['#2ecc71', '#3498db', '#e74c3c']
    failed_color = '#7f8c8d'
    for i, method_name in enumerate(method_names):
        results = methods_results[method_name]
        bottom = 0
        success_percentage = sum(results['category_percentages'].values())
        for j, category in enumerate(ordered_categories):
            if category in results['category_counts']:
                percentage = results['category_percentages'].get(category, 0)
                ax1.bar(
                    i, 
                    percentage, 
                    bar_width, 
                    bottom=bottom, 
                    color=success_colors[j],
                    edgecolor=black_semi,
                    linewidth=2.0,
                    alpha=0.8,
                    zorder=3
                )
                if percentage > 3:
                    ax1.text(
                        i, 
                        bottom + percentage/2, 
                        f'{percentage:.1f}%',
                        ha='center', 
                        va='center', 
                        color='white', 
                        fontweight='bold',
                        fontsize=14
                    )
                bottom += percentage
        remaining_percentage = 100.0 - success_percentage
        ax1.bar(
            i, 
            remaining_percentage, 
            bar_width, 
            bottom=bottom, 
            color=failed_color,
            edgecolor=black_semi,
            linewidth=2.0,
            alpha=0.8,
            zorder=3
        )
        if remaining_percentage > 3:
            ax1.text(
                i, 
                bottom + remaining_percentage/2, 
                f'{results["failed_percentage"]:.1f}%',
                ha='center', 
                va='center', 
                color='white', 
                fontweight='bold',
                fontsize=14
            )
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Percentage (%)', fontsize=20, labelpad=10, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks(np.arange(0, 101, 20))
    ax1.set_yticklabels([f"{int(x)}" for x in np.arange(0, 101, 20)], fontsize=16)
    ax1.tick_params(axis='y', direction='out', length=6, width=1.5, pad=8, grid_alpha=0.3)
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color(black_semi)
    legend_elements1 = []
    for j, category in enumerate(ordered_categories):
        legend_elements1.append(Patch(facecolor=success_colors[j], edgecolor=black_semi, linewidth=1.5, alpha=0.8, label=category))
    legend_elements1.append(Patch(facecolor=failed_color, edgecolor=black_semi, linewidth=1.5, alpha=0.8, label='Failed Cases (RMSD ≥ 2)'))
    ax1.legend(handles=legend_elements1, loc='upper center', 
               bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=16, framealpha=0.9)

    # Violin plot
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('white')
    ax2.grid(True, axis='y', linestyle='-', alpha=0.2)
    filtered_violin_data = all_plifs_df[all_plifs_df['display_name'].isin(method_names)]
    filtered_violin_data['display_name'] = pd.Categorical(
        filtered_violin_data['display_name'], 
        categories=method_names, 
        ordered=True
    )
    filtered_violin_data = filtered_violin_data.sort_values('display_name')
    custom_palette = {method: method_colors[method] for method in method_names}
    sns.violinplot(x='display_name', y='count_recovery', hue='display_name', 
                   data=filtered_violin_data, palette=custom_palette, 
                   inner=None, cut=0, legend=False, ax=ax2,
                   alpha=0.2)
    sns.boxplot(x='display_name', y='count_recovery', hue='display_name', 
               data=filtered_violin_data, fliersize=0, width=0.3, palette=custom_palette, 
               boxprops=dict(alpha=0.7, linewidth=3.0, edgecolor=black_semi),
               medianprops=dict(linewidth=2.5, color=black_semi),
               whiskerprops=dict(linewidth=2.5, color=black_semi),
               capprops=dict(linewidth=2.5, color=black_semi),
               ax=ax2, legend=False, zorder=3)
    successful_data = filtered_violin_data[filtered_violin_data['docking_result'] == 'Success']
    failed_data = filtered_violin_data[filtered_violin_data['docking_result'] == 'Failure']
    partial_matches = successful_data[successful_data['perfect_match'] == False]
    perfect_matches = successful_data[successful_data['perfect_match'] == True]
    if len(failed_data) > 0:
        sns.stripplot(x='display_name', y='count_recovery', data=failed_data, 
                     size=5, color="red", linewidth=0, alpha=0.4, jitter=True, ax=ax2, zorder=5)
    if len(partial_matches) > 0:
        sns.stripplot(x='display_name', y='count_recovery', data=partial_matches, 
                     size=6, color="blue", linewidth=0, alpha=0.4, jitter=True, ax=ax2, zorder=5)
    if len(perfect_matches) > 0:
        sns.stripplot(x='display_name', y='count_recovery', data=perfect_matches, 
                     size=7, color="green", linewidth=0, alpha=0.5, jitter=True, ax=ax2, zorder=5)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=2.0, zorder=2)
    ax2.text(len(method_names)-0.7, 0.45, 'Recovery = 0.5', color='gray', fontsize=16, fontweight='bold')
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, linewidth=2.0, zorder=2)
    ax2.text(len(method_names)-0.7, 0.95, 'Perfect Match', color='green', fontsize=16, fontweight='bold')
    ax2.set_xlabel('', fontsize=16)
    ax2.set_ylabel('PLIF Recovery', fontsize=21, labelpad=15, fontweight='bold')
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.set_yticklabels([f"{x:.1f}" for x in np.arange(0, 1.1, 0.1)], fontsize=16)
    ax2.set_xticks(index)
    ax2.set_xticklabels(method_names, fontsize=16, rotation=15, ha='right')
    ax2.tick_params(axis='y', direction='out', length=6, width=1.5, pad=8, grid_alpha=0.3)
    ax2.tick_params(axis='x', direction='out', length=6, width=1.5, pad=8, grid_alpha=0.3)
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color(black_semi)
    legend_elements2 = [
        Patch(facecolor='green', edgecolor=black_semi, alpha=0.85, label='Success with perfect interactions (Recovery=1.0)', linewidth=1.5),
        Patch(facecolor='blue', edgecolor=black_semi, alpha=0.75, label='Success with partial interactions (Recovery<1.0)', linewidth=1.5),
        Patch(facecolor='red', edgecolor=black_semi, alpha=0.75, label='Failed docking (RMSD≥2)', linewidth=1.5)
    ]
    ax2.legend(handles=legend_elements2, loc='upper center', 
             bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=16, frameon=True, framealpha=0.9)
    
    # Add subplot labels
    ax1.text(-0.08, 1.05, '(a)', transform=ax1.transAxes, fontsize=24, fontweight='bold', 
             va='top', ha='left', color='black')
    ax2.text(-0.08, 1.05, '(b)', transform=ax2.transAxes, fontsize=24, fontweight='bold', 
             va='top', ha='left', color='black')
    
    plt.tight_layout()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.svg'), format='svg', bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_sucos_group_PLIF(
    rmsd_csv='pdb_benchmark_RMSD.csv',
    sucos_csv='pdb_benchmark.csv',
    plif_csv='plifs_recovery_df.csv',
    output_prefix='sucos_group_docking_stacked_bar',
    output_dir='.',
    sucos_col='SuCOS_pocket_qcov',
    sucos_threshold=60,
    methods_info=None,
    display_names=None
):
    """
    Generate a grouped stacked bar plot comparing docking outcomes by SuCOS_pocket_qcov group.

    Parameters
    ----------
    benchmark_csv : str
        Path to the benchmark CSV file.
    plif_csv : str
        Path to the PLIF recovery CSV file.
    output_prefix : str
        Prefix for the output image files.
    output_dir : str
        Directory to save the output images.
    sucos_col : str
        Column name for SuCOS pocket coverage.
    sucos_threshold : float
        Threshold to split SuCOS groups.
    methods_info : dict or None
        Mapping from method name to RMSD column name. If None, use default.
    display_names : dict or None
        Mapping from method name to display name. If None, use default.
    """
    if methods_info is None:
        methods_info = {
            'AF3': 'AF3_5pose_RMSD',
            'Chai-1': 'Chai-1_5pose_RMSD',
            'Chai-1_MSA': 'Chai-1_MSA_5pose_RMSD',
            'Boltz-1': 'Boltz-1_5pose_RMSD',
            'AutoDock': 'AutoDock_5pose_RMSD',
            'CovDock': 'CovDock_5pose_RMSD',
            'GNINA': 'GNINA_5pose_RMSD'
        }
    if display_names is None:
        display_names = {
            'AF3': 'AF3',
            'Chai-1': 'Chai-1',
            'Chai-1_MSA': 'Chai-1(MSA)',
            'Boltz-1': 'Boltz-1x',
            'AutoDock': 'AutoDock-GPU',
            'CovDock': 'CovDock',
            'GNINA': 'GNINA'
        }

    # Read data
    df_benchmark = pd.merge(pd.read_csv(rmsd_csv), pd.read_csv(sucos_csv), on='pdb_id')
    df_plifs = pd.read_csv(plif_csv, header=None, 
        names=['method', 'target', 'count_recovery', 'reference_plifs', 'plifs'])
    df_plifs['count_recovery'] = pd.to_numeric(df_plifs['count_recovery'], errors='coerce')

    # Grouping
    group_labels = [
        f'{sucos_col} ≥ {sucos_threshold}',
        f'{sucos_col} < {sucos_threshold}'
    ]
    df_benchmark['SuCOS_group'] = np.where(df_benchmark[sucos_col] >= sucos_threshold, group_labels[0], group_labels[1])
    groups = group_labels

    # Color map and categories
    color_map = {
        'Perfect Match (1.0)': '#2ecc71',   # green
        'Good Match (0.5-1.0)': '#3498db',  # blue
        'Poor Match (< 0.5)': '#e74c3c',    # red
        'Fail Case': '#7f8c8d'              # gray
    }
    ordered_categories = ['Perfect Match (1.0)', 'Good Match (0.5-1.0)', 'Poor Match (< 0.5)', 'Fail Case']

    # Statistics
    results = {g: {} for g in groups}
    for group in groups:
        group_df = df_benchmark[df_benchmark['SuCOS_group'] == group]
        for method, rmsd_col in methods_info.items():
            if rmsd_col not in group_df.columns:
                continue
            valid_pdbs = group_df[group_df[rmsd_col].notna()]['pdb_id'].unique()
            if len(valid_pdbs) == 0:
                continue
            successful_pdb_ids = group_df[(group_df[rmsd_col].notna()) & (group_df[rmsd_col] < 2)]['pdb_id'].unique()
            fail_count = len(valid_pdbs) - len(successful_pdb_ids)
            fail_pct = round(fail_count / len(valid_pdbs) * 100, 1)
            filtered_plifs = df_plifs[(df_plifs['target'].isin(successful_pdb_ids)) & (df_plifs['method'] == method)]
            filtered_plifs = filtered_plifs.dropna(subset=['count_recovery'])
            def categorize_recovery(value):
                if value == 1.0:
                    return 'Perfect Match (1.0)'
                elif 0.5 <= value < 1.0:
                    return 'Good Match (0.5-1.0)'
                else:
                    return 'Poor Match (< 0.5)'
            filtered_plifs['recovery_category'] = filtered_plifs['count_recovery'].apply(categorize_recovery)
            category_counts = filtered_plifs['recovery_category'].value_counts().reindex(ordered_categories[:-1]).fillna(0).astype(int)
            category_percentages = {cat: round(category_counts[cat] / len(valid_pdbs) * 100, 1) for cat in ordered_categories[:-1]}
            category_percentages['Fail Case'] = fail_pct
            # Ensure sum is 100%
            total = sum(category_percentages.values())
            if abs(total - 100.0) > 0.1:
                category_percentages['Fail Case'] = round(100.0 - sum(val for key, val in category_percentages.items() if key != 'Fail Case'), 1)
            results[group][display_names[method]] = category_percentages

    # Plotting
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig, ax = plt.subplots(figsize=(20, 8))
    bar_width = 0.35
    method_names = list(display_names.values())
    x = np.arange(len(method_names))
    for i, group in enumerate(groups):
        bottoms = np.zeros(len(method_names))
        for cat in ordered_categories:
            values = [results[group][m][cat] if m in results[group] and cat in results[group][m] else 0 for m in method_names]
            ax.bar(x + (i-0.5)*bar_width, values, bar_width, bottom=bottoms, color=color_map[cat], 
                   label=cat if i==0 else "", alpha=0.85, edgecolor='black', linewidth=1.5)
            for j, v in enumerate(values):
                if v > 3:
                    ax.text(x[j] + (i-0.5)*bar_width, bottoms[j] + v/2, f'{v:.1f}%', 
                          ha='center', va='center', color='white', fontsize=14, fontweight='bold')
            bottoms += values
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_ylabel('Percentage (%)', fontsize=20, labelpad=10)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', direction='out', length=8, width=3.5,
                   pad=10, labelsize=24, grid_alpha=0.3)
    for i, method in enumerate(method_names):
        ax.text(i, -5, method, ha='center', va='center', fontsize=18, fontweight='bold')
    for i in range(len(method_names)):
        ax.text(i-0.175, -10, f"≥ {sucos_threshold}", fontsize=14, ha='center')
        ax.text(i+0.175, -10, f"< {sucos_threshold}", fontsize=14, ha='center')
    legend_elements = [
        Patch(facecolor=color_map['Perfect Match (1.0)'], edgecolor='black', label='Perfect Match (1.0)'),
        Patch(facecolor=color_map['Good Match (0.5-1.0)'], edgecolor='black', label='Good Match (0.5-1.0)'),
        Patch(facecolor=color_map['Poor Match (< 0.5)'], edgecolor='black', label='Poor Match (< 0.5)'),
        Patch(facecolor=color_map['Fail Case'], edgecolor='black', label='Fail Case (RMSD ≥ 2)')
    ]
    ax.legend(handles=legend_elements, fontsize=16, loc='upper center', 
              bbox_to_anchor=(0.5, -0.13), ncol=4)
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.2, top=0.92, wspace=0.1, hspace=0.22)
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.svg'), format='svg', bbox_inches='tight')
    plt.show()
    plt.close(fig)