import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import argparse
from matplotlib import cm

config = None

def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def process_data(df1, df2, keys_to_interpret):
    """Process the data based on multiple keys to interpret."""
    results = {}

    for key in keys_to_interpret:
        # Filter the dataframes for the specific key
        vals1 = df1[df1['Key'] == key].sort_values('xVal')
        vals2 = df2[df2['Key'] == key].sort_values('xVal')

        if not vals1.empty and not vals2.empty:
            # Calculate the desired quantity
            geometric_mean = np.sqrt(vals1['Mean'] * vals2['Mean'])
            error = np.sqrt(vals1['StdErr'] * vals2['StdErr'])
            results[key] = (vals1, vals2, geometric_mean, error)
        else:
            print(f"Warning: No data found for key '{key}' in one of the files.")

    return results


def generate_colors(n):
    """Generate a color scheme with n shades."""
    colormap_list = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges']  # List of vibrant colormaps
    color_palette = []

    for i in range(n):
        cmap = cm.get_cmap(colormap_list[i % len(colormap_list)])  # Cycle through colormaps
        color_shades = [cmap(0.4), cmap(0.6), cmap(0.8)]  # Generate 3 progressively lighter shades
        color_palette.append(color_shades)

    return color_palette

def plot_results(results, config, ax, name, color, marker, linestyle):
    """Generate plots for each key and save them."""    
    
    ax1 = ax

    color_palette = generate_colors(len(results))  # Get unique color schemes for each key

    for i, (key, (biased, unbiased, resolution, error)) in enumerate(results.items()):
        # Extract colors for the current key
        color_shades = color_palette[i]

        # Scatter plots with thin line plots for each dataset
        sns.lineplot(x=biased['xVal'], y=resolution, color=color, linewidth=3, linestyle=linestyle, label=name, marker=marker, markersize=10)
        # ax1.plot(biased['xVal'], resolution, color=color, linewidth=3, linestyle=linestyle)
        # ax1.scatter(biased['xVal'], resolution, label=name,
        #             color=color, s=50, marker=marker)
        ax1.errorbar(biased['xVal'], resolution, yerr=error, fmt='.', capsize=5, color=color)

        # ax1.plot(biased['xVal'], biased['Mean'], color=color_shades[1], linewidth=1, linestyle='--')
        # ax1.scatter(biased['xVal'], biased['Mean'], label=f'$\\sigma$ biased',
        #             color=color_shades[1], s=50)

        # ax1.plot(unbiased['xVal'], unbiased['Mean'], color=color_shades[0], linewidth=1, linestyle='--')
        # ax1.scatter(unbiased['xVal'], unbiased['Mean'], label=f'$\\sigma$ unbiased',
        #             color=color_shades[0], s=50)

        # Add text labels above the lines, with the same color
        # x_max = biased['xVal'].max()  # Find the maximum xVal
        # ax1.text(x_max - x_max / 2.0, quantity.iloc[-1] + 0.35, f'Spatial resolution', color=color_shades[2], fontsize=fontsize, weight='bold')
        # ax1.text(x_max - x_max / 2.0, biased['Mean'].iloc[-1] + 0.35, f'$\\sigma$ biased', color=color_shades[1], fontsize=fontsize)
        # ax1.text(x_max - x_max / 2.0, unbiased['Mean'].iloc[-1] + 0.35, f'$\\sigma$ unbiased', color=color_shades[0], fontsize=fontsize)

    # Add legend and labels
    plt.legend()
    plt.grid()
    plt.xlabel(config.get('xlabel', ''))
    plt.ylabel(config.get('ylabel', ''))
    title = config.get('title', None)
    plt.title(title)
    xlim = config.get('xlim', None)
    if xlim:
        plt.xlim((xlim[0], xlim[1]))
    ylim = config.get('ylim', None)
    if ylim:
        plt.ylim((ylim[0], ylim[1]))

    
    if "x2label" in config and 'polynomial_coefficients' in config:
        ax2 = ax1.secondary_xaxis("top", functions=(primary_to_secondary, secondary_to_primary))
        ax2.set_xlabel(config["x2label"])

    # Save the plot to a file    
    # plt.show()

# Define the transformation function
def primary_to_secondary(x):
    coeffs = config["polynomial_coefficients"]
    return np.polyval(coeffs, np.asarray(x))  # Ensure input is an array

# Define an approximate inverse function
def secondary_to_primary(x2):
    coeffs = config["polynomial_coefficients"]

    if len(coeffs) == 2:  # If it's a linear function, invert directly
        a, b = coeffs
        return (np.asarray(x2) - b) / a

    x_values = np.linspace(0, 10, 1000)
    y_values = np.polyval(coeffs, x_values)
    return np.interp(x2, y_values, x_values)  # Interpolate inverse


def main():
    global config
    parser = argparse.ArgumentParser(description="Process two CSV files and generate a plot.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")

    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)


    # Load the CSV files specified in the config
    data_biased = config['data_biased']
    if not isinstance(data_biased, list):
        data_biased = [data_biased]

    data_unbiased = config['data_unbiased']
    if not isinstance(data_unbiased, list):
        data_biased = [data_unbiased]

    names = config['names']
    if not isinstance(names, list):
        names = [names]

    colors = config['colors']
    if not isinstance(colors, list):
        colors = [colors]

    fontsize = config.get('fontsize', None)

    if fontsize is not None:
        plt.rcParams.update({
            'font.size': fontsize,  # Base font size
            'axes.titlesize': fontsize,  # Title size
            'axes.labelsize': fontsize,  # X/Y label size
            'xtick.labelsize': fontsize,  # X tick label size
            'ytick.labelsize': fontsize,  # Y tick label size
            'legend.fontsize': fontsize,  # Legend text size
            'legend.title_fontsize': fontsize  # Legend title size
        })
    fig = plt.figure(figsize=(20, 12))
    ax = fig.subplots()

    for i, _ in enumerate(data_biased):

        print('\n\nnumber', i, '\n\n')
        df1 = pd.read_csv(data_biased[i])
        df2 = pd.read_csv(data_unbiased[i])
        name = names[i]
        color = colors[i]
        linestyle = config['linestyles'][i]
        marker = config['markers'][i]

        # Process the data
        results = process_data(df1, df2, config['keys_to_interpret'])

        # Plot the results and save the plot
        

        # plot_results(results, config, ax, name, color)
        plot_results(results=results, config=config, ax=ax, color=color, name=name, linestyle=linestyle, marker=marker)

    plt.savefig(config['output_plot'])
    plt.show()


if __name__ == "__main__":
    main()
