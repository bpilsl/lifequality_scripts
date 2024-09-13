import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import argparse
from matplotlib import cm


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
            quantity = np.sqrt(vals1['Mean'] * vals2['Mean'])
            results[key] = (vals1, vals2, quantity)
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


def plot_results(results, output_plot, config):
    """Generate plots for each key and save them."""
    plt.figure(figsize=(10, 6))

    color_palette = generate_colors(len(results))  # Get unique color schemes for each key

    for i, (key, (vals1, vals2, quantity)) in enumerate(results.items()):
        # Extract colors for the current key
        color_shades = color_palette[i]

        # Scatter plots with thin line plots for each dataset
        plt.plot(vals1['xVal'], quantity, color=color_shades[0], linewidth=1)
        plt.scatter(vals1['xVal'], quantity, label=f'Spatial resolution for {key.split("/")[-1]}',
                    color=color_shades[0], s=50)

        plt.plot(vals1['xVal'], vals1['Mean'], color=color_shades[1], linewidth=1)
        plt.scatter(vals1['xVal'], vals1['Mean'], label=f'$\\sigma$ biased for {key.split("/")[-1]}',
                    color=color_shades[1], s=50)

        plt.plot(vals2['xVal'], vals2['Mean'], color=color_shades[2], linewidth=1)
        plt.scatter(vals2['xVal'], vals2['Mean'], label=f'$\\sigma$ unbiased for {key.split("/")[-1]}',
                    color=color_shades[2], s=50)

        # Add text labels above the lines, with the same color
        x_max = vals1['xVal'].max()  # Find the maximum xVal
        plt.text(x_max - x_max / 15.0, quantity.iloc[-1] + 0.35, f'Spatial resolution for {key.split("/")[-1]}', color=color_shades[0], fontsize=10)
        plt.text(x_max - x_max / 15.0, vals1['Mean'].iloc[-1] + 0.35, f'$\\sigma$ biased for {key.split("/")[-1]}', color=color_shades[1], fontsize=10)
        plt.text(x_max - x_max / 15.0, vals2['Mean'].iloc[-1] + 0.35, f'$\\sigma$ unbiased for {key.split("/")[-1]}', color=color_shades[2], fontsize=10)

    # Add legend and labels
    plt.legend()
    plt.grid()
    plt.xlabel(config.get('xlabel', ''))
    plt.ylabel(config.get('ylabel', ''))
    plt.title('Comparison of Spatial Resolution and Sigma X')

    # Save the plot to a file
    plt.savefig(output_plot)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Process two CSV files and generate a plot.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")

    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Load the CSV files specified in the config
    df1 = pd.read_csv(config['data_biased'])
    df2 = pd.read_csv(config['data_unbiased'])

    # Process the data
    results = process_data(df1, df2, config['keys_to_interpret'])

    # Plot the results and save the plot
    plot_results(results, config['output_plot'], config)


if __name__ == "__main__":
    main()
