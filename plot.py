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


def analyze_rmsd_statistics(
    csv_file, 
    methods=None, 
    output_dir='./',
    rmsd_threshold=2, 
    fig_size=(18, 12),
    outward_ticks=True,
    title='', 
    dpi=300
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
    - The function does not return any value.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set default methods if not provided
    if methods is None:
        methods = {
            "AlphaFold3": ['af3_1pose_RMSD', 'af3_5pose_RMSD'],
            "Chai-1": ['chai1_1pose_RMSD', 'chai1_5pose_RMSD'],
            "Chai-1-MSA": ['chai1_msa_1pose_RMSD', 'chai1_msa_5pose_RMSD'],
            "Boltz-1": ['boltz_1pose_RMSD', 'boltz_5pose_RMSD'],
            "CovDock": ['Schrodinger_1pose_RMSD', 'Schrodinger_5pose_RMSD'],
            "Autodock-GPU": ['Autodock_1pose_RMSD', 'Autodock_5pose_RMSD'],
            "GNINA": ['Gnina_1pose_RMSD', 'Gnina_5pose_RMSD']
        }
    
    data = pd.read_csv(csv_file)
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
    method_gap = 1.0
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
    boxprops = {'linewidth': 2.0, 'color': black_semi}
    whiskerprops = {'linewidth': 2.5, 'color': black_semi}
    medianprops = {'color': black_semi, 'linewidth': 2.5}
    capprops = {'linewidth': 3.0, 'color': black_semi}

    bp = ax.boxplot(grouped_data, positions=grouped_positions, widths=0.3,
                   patch_artist=True, showfliers=False,
                   boxprops=boxprops, whiskerprops=whiskerprops,
                   medianprops=medianprops, capprops=capprops, zorder=3)

    # Set box color and style
    for patch, color in zip(bp['boxes'], grouped_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(black_semi)
        patch.set_linewidth(2.0)
        patch.set_alpha(0.7)

    # Plot mean value as a diamond marker
    for values, pos in zip(grouped_data, grouped_positions):
        mean_val = np.mean(values)
        plt.plot(pos, mean_val, marker='D', color=black_semi, markersize=9, zorder=4)

    # Significance analysis between Top-1 and Top-5 for each method
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
                else:
                    significance = "ns"
            except Exception as e:
                significance = "ns"
                
            # Set significance marker height by method
            if method_name in ["AlphaFold3", "Chai-1", "Chai-1-MSA"]:
                sig_y = 8.0
            elif method_name in ["Boltz-1"]:
                sig_y = 9.0
            else:
                sig_y = 12.3
            
            # Draw significance line
            plt.plot([pos1, pos5], [sig_y, sig_y], '-', color='black', linewidth=3.0)
            
            # Add significance text above the line
            plt.text((pos1 + pos5) / 2, sig_y, significance,
                    ha='center', va='bottom', fontsize=24, fontweight='bold')

    # Draw the RMSD threshold line
    plt.axhline(y=rmsd_threshold, color='r', linestyle='--', linewidth=2.5, zorder=1)

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
    
    # Add Top-1 and Top-5 labels below the boxplots
    for i, (pos, pose_type) in enumerate(zip(grouped_positions, grouped_types)):
        color = pose_colors[pose_type]
        if pose_type == "Top-1":
            label_pos_x = pos - 0.1
        else:  # Top-5
            label_pos_x = pos + 0.1

        plt.text(label_pos_x, -0.2, pose_type,
            ha='center', va='top', fontsize=20,
            color=color, fontweight='bold')

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
    
    if title:
        plt.title(title, fontsize=18, fontweight='bold')
    plt.ylabel('L-RMSD (Å)', fontsize=26, fontweight='bold', labelpad=12)
    
    # Set y-axis and x-axis limits
    plt.ylim(0, 15)
    plt.xlim(min(grouped_positions) - 0.5, max(grouped_positions) + 0.5)
    
    # Set y-axis tick style
    if outward_ticks:
        ax.tick_params(axis='y', direction='out', length=8, width=2.0,
                      pad=10, labelsize=20, grid_alpha=0.3)
    else:
        ax.tick_params(axis='y', labelsize=20, grid_alpha=0.3)
    
    # Set border thickness and color
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color(black_semi)
    
    # Remove extra whitespace
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.95)

    # Save the plot in multiple formats
    boxplot_file_png = os.path.join(output_dir, 'rmsd_boxplot.png')
    boxplot_file_pdf = os.path.join(output_dir, 'rmsd_boxplot.pdf')
    boxplot_file_svg = os.path.join(output_dir, 'rmsd_boxplot.svg')
    plt.savefig(boxplot_file_png, dpi=600, bbox_inches='tight')
    plt.savefig(boxplot_file_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(boxplot_file_svg, format='svg', bbox_inches='tight')
    plt.show()
    
    # Print statistics summary to console
    for stat in stats_summary:
        print(f"{stat['Method']:<20} {stat['Pose']:<10} {stat['Success_Rate']:.2f}% {stat['Mean_RMSD']:.2f}")


def analyze_rmsd_statistics_(
    csv_file, methods=None, output_dir='./',
    rmsd_threshold=2, fig_size=(18, 12),
    title='', dpi=300
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
            "AlphaFold3": ['af3_1pose_RMSD', 'af3_5pose_RMSD'],
            "Chai-1": ['chai1_1pose_RMSD', 'chai1_5pose_RMSD'],
            "Chai-1-MSA": ['chai1_msa_1pose_RMSD', 'chai1_msa_5pose_RMSD'],
            "Boltz-1": ['boltz_1pose_RMSD', 'boltz_5pose_RMSD'],
            "CovDock": ['Schrodinger_1pose_RMSD', 'Schrodinger_5pose_RMSD'],
            "Autodock-GPU": ['Autodock_1pose_RMSD', 'Autodock_5pose_RMSD'],
            "GNINA": ['Gnina_1pose_RMSD', 'Gnina_5pose_RMSD']
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
    pose_gap = 0.35
    method_gap = 1.0
    current_pos = 1

    for method_name, (col1, col5) in methods.items():
        # Top-1
        if col1 in data.columns:
            values = data[col1].dropna().values
            if len(values) > 0:
                grouped_positions.append(current_pos)
                grouped_data.append(values)
                grouped_labels.append(method_name)
                grouped_colors.append(pose_colors["Top-1"])
                grouped_types.append("Top-1")
        # Top-5
        if col5 in data.columns:
            values = data[col5].dropna().values
            if len(values) > 0:
                grouped_positions.append(current_pos + pose_gap)
                grouped_data.append(values)
                grouped_labels.append(method_name)
                grouped_colors.append(pose_colors["Top-5"])
                grouped_types.append("Top-5")
        current_pos += pose_gap + method_gap

    # Create boxplot
    plt.figure(figsize=fig_size)
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(True, axis='y', linestyle='-', alpha=0.2)

    black_semi = (0, 0, 0, 0.8)
    boxprops = {'linewidth': 2.0, 'color': black_semi}
    whiskerprops = {'linewidth': 2.5, 'color': black_semi}
    medianprops = {'color': black_semi, 'linewidth': 2.5}
    capprops = {'linewidth': 3.0, 'color': black_semi}

    bp = ax.boxplot(
        grouped_data, positions=grouped_positions, widths=0.3,
        patch_artist=True, showfliers=False,
        boxprops=boxprops, whiskerprops=whiskerprops,
        medianprops=medianprops, capprops=capprops, zorder=3
    )

    # Set box color and style
    for patch, color in zip(bp['boxes'], grouped_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(black_semi)
        patch.set_linewidth(2.0)
        patch.set_alpha(0.7)

    # Plot mean value as a diamond marker
    for values, pos in zip(grouped_data, grouped_positions):
        mean_val = np.mean(values)
        plt.plot(pos, mean_val, marker='D', color=black_semi, markersize=9, zorder=4)

    # Draw the RMSD threshold line
    plt.axhline(y=rmsd_threshold, color='r', linestyle='--', linewidth=2.5, zorder=1)

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

    # Create custom legend for Top-1 and Top-5
    legend_elements = [
        Patch(facecolor=pose_colors["Top-1"], edgecolor='black', label='Top-1'),
        Patch(facecolor=pose_colors["Top-5"], edgecolor='black', label='Top-5'),
        plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2.5, label='')
    ]
    leg = ax.legend(handles=legend_elements[:2], loc='upper left',
                    frameon=True, edgecolor='gray', framealpha=0.9,
                    fontsize=22, ncol=1)

    if title:
        plt.title(title, fontsize=18, fontweight='bold')
    plt.ylabel('L-RMSD (Å)', fontsize=26, fontweight='bold', labelpad=12)

    # Set y-axis and x-axis limits
    plt.ylim(0, 12)
    plt.xlim(min(grouped_positions) - 0.5, max(grouped_positions) + 0.5)

    # Set y-axis tick style
    ax.tick_params(axis='y', direction='out', length=8, width=2.0,
                   pad=10, labelsize=20, grid_alpha=0.3)

    # Set border thickness and color
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color(black_semi)

    # Remove extra whitespace
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.95)

    # Save the plot in multiple formats
    boxplot_file_png = os.path.join(output_dir, 'rmsd_boxplot.png')
    plt.savefig(boxplot_file_png, dpi=600, bbox_inches='tight')
    plt.show()
    print(f"Boxplot saved to {output_dir}")


def analyze_af3_success_rate(data, rmsd_col, rmsd_threshold=2.0, min_count=10):
    """
    Analyze AlphaFold3 success rate by warhead type.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing at least 'warhead' and the specified RMSD column.
    rmsd_col : str
        Column name for RMSD values.
    rmsd_threshold : float, optional
        RMSD threshold for success.
    min_count : int, optional
        Minimum sample count for a warhead to be shown separately.

    Returns
    -------
    pd.DataFrame
        DataFrame with warhead, sample count, success rate, average RMSD, etc.
    """
    if 'warhead' not in data.columns or rmsd_col not in data.columns:
        print(f"Missing required columns: warhead or {rmsd_col}")
        return None

    warhead_counts = data['warhead'].value_counts()
    data['warhead_group'] = data['warhead'].apply(
        lambda x: x if warhead_counts[x] >= min_count else 'other'
    )

    results = []
    for warhead, group_data in data.groupby('warhead_group'):
        total = len(group_data)
        success = sum((group_data[rmsd_col].notna()) & (group_data[rmsd_col] < rmsd_threshold))
        success_rate = (success / total) * 100
        valid_rmsd = group_data[rmsd_col].dropna()
        avg_rmsd = valid_rmsd.mean() if len(valid_rmsd) > 0 else np.nan

        results.append({
            'warhead': warhead.replace('_', ' '),
            'total_samples': total,
            'success_count': success,
            'success_rate': success_rate,
            'avg_rmsd': avg_rmsd,
            'valid_predictions': len(valid_rmsd),
            'missing_rate': (total - len(valid_rmsd)) / total * 100
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('success_rate', ascending=False)
    return result_df

def plot_af3_warhead_pie(result_df, output_dir):
    """
    Plot a donut pie chart of warhead sample distribution, colored by success rate.

    Parameters
    ----------
    result_df : pd.DataFrame
        DataFrame from analyze_af3_success_rate.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    None
    """


    os.makedirs(output_dir, exist_ok=True)
    pie_data = result_df.sort_values('total_samples', ascending=False)

    # Normalize success rates for color mapping
    success_rates = pie_data['success_rate'].values
    if max(success_rates) > min(success_rates):
        normalized_rates = (success_rates - min(success_rates)) / (max(success_rates) - min(success_rates))
    else:
        normalized_rates = np.ones_like(success_rates) * 0.5

    colors = plt.cm.Blues(normalized_rates)

    # Print warhead names and success rates for manual annotation
    print("\nWarhead names for manual annotation:")
    print("-" * 40)
    for i, (_, row) in enumerate(pie_data.iterrows()):
        print(f"Segment {i+1}: {row['warhead']} - Success rate: {row['success_rate']:.1f}%")

    # Plot donut pie chart
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_axes([0.05, 0.1, 0.8, 0.8])
    wedges, _, autotexts = ax.pie(
        pie_data['total_samples'],
        labels=None,
        autopct='%1.1f%%',
        pctdistance=0.75,
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.5, edgecolor='w', linewidth=1.5),
        textprops={'fontsize': 9}
    )
    centre_circle = plt.Circle((0, 0), 0.25, fc='white')
    ax.add_patch(centre_circle)
    ax.set_aspect('equal')

    # Add colorbar for success rate
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    norm = plt.Normalize(vmin=min(pie_data['success_rate']), vmax=max(pie_data['success_rate']))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Success Rate (%)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Save the plot
    plt.savefig(f"{output_dir}/af3_warhead_distribution_pie.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/af3_warhead_distribution_pie.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{output_dir}/af3_warhead_distribution_pie.svg", format='svg', bbox_inches='tight')
    plt.show()
    plt.close()



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
            'Boltz-1': 'Boltz-1',
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
    ax2.text(len(method_names)-0.9, 0.45, 'Recovery = 0.5', color='gray', fontsize=16, fontweight='bold')
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, linewidth=2.0, zorder=2)
    ax2.text(len(method_names)-0.9, 0.95, 'Perfect Match', color='green', fontsize=16, fontweight='bold')
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
            'Boltz-1': 'Boltz-1',
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
    ax.tick_params(axis='y', labelsize=16)
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
    plt.figtext(0.25, 0.03, f'{sucos_col} ≥ {sucos_threshold}', ha='center', fontsize=18, fontweight='bold')
    plt.figtext(0.75, 0.03, f'{sucos_col} < {sucos_threshold}', ha='center', fontsize=18, fontweight='bold')
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.svg'), format='svg', bbox_inches='tight')
    plt.show()
    plt.close(fig)



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
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.2, hspace=0.3)
    black_semi = (0, 0, 0, 0.8)
    method_name_map = {
        "alphafold3": "AF3",
        "AlphaFold3": "AF3",
        "af3": "AF3",
        "chai1": "Chai-1",
        "chai1_msa": "Chai-1 (MSA)",
        "boltz": "Boltz-1",
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
        # 关键：加 .copy()，避免 SettingWithCopyWarning
        data_clean = data.dropna(subset=['SuCOS_pocket_qcov']).copy()
        data_clean['SuCOS_group'] = pd.cut(data_clean['SuCOS_pocket_qcov'], bins=bins, labels=labels, right=True)
        success_rates = {col: [] for col in rmsd_columns if col in data_clean.columns}
        sample_counts = {group: 0 for group in labels}
        valid_groups = []
        for group in labels:
            group_df = data_clean[data_clean['SuCOS_group'] == group]
            valid_groups.append(group)
            sample_counts[group] = len(group_df)
            for col in rmsd_columns:
                if col in data_clean.columns:
                    valid_rows = group_df.dropna(subset=[col])
                    if len(valid_rows) > 0:
                        success_count = sum(valid_rows[col] < rmsd_threshold)
                        success_rate = success_count / len(valid_rows) * 100
                        success_rates[col].append(success_rate)
                    else:
                        success_rates[col].append(0)
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
                          fontsize=12, 
                          ha='center', 
                          va='bottom',
                          color='black',
                          bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
        for method_name, cols in methods.items():
            if len(cols) == 2 and cols[0] in data_clean.columns and cols[1] in data_clean.columns:
                pose1_col = next((col for col in cols if "1pose" in col), None)
                pose5_col = next((col for col in cols if "5pose" in col), None)
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
            if method_name in method_colors and any(col in data_clean.columns for col in methods[method_name]):
                display_name = method_name_map.get(method_name, method_name.replace('_', ' '))
                legend_elements.append(Line2D([0], [0], color=method_colors[method_name], lw=3.5, label=display_name))
        legend_elements.append(Line2D([0], [0], color='black', linestyle='-', lw=3.0, label=pose_type_labels[0]))
        legend_elements.append(Line2D([0], [0], color='black', linestyle='--', lw=3.0, label=pose_type_labels[1]))
        if method_group_name.strip() != "":
            ax.set_title(method_group_name, fontsize=title_font_size+2, fontweight='bold', pad=15)
        ax.set_ylabel('Success Rate (%)', fontsize=22, labelpad=12, fontweight='bold')
        ax.set_xlabel('SuCOS_pocket', fontsize=20, labelpad=10)
        ax.set_xticks(x_range)
        ax.set_xticklabels(valid_groups, fontsize=18)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis='x', direction='out', length=8, width=2.0, pad=15, 
                      labelsize=18, grid_alpha=0.3)
        ax.tick_params(axis='y', direction='out', length=8, width=2.0, pad=8, 
                      labelsize=18, grid_alpha=0.3)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)
            spine.set_color(black_semi)
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', fontsize=18, 
                     frameon=True, framealpha=0.9, fancybox=True, ncol=1)
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.08, top=0.92, wspace=0.22, hspace=0.3)
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{output_prefix}.svg'), format='svg', bbox_inches='tight')
    plt.show()
    plt.close(fig)