import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
import numpy as np
import argparse
import os.path
import re
import yaml
from sympy.physics.units import temperature


# Function to determine where the actual data starts
def skip_header_lines(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Detect the first line that starts with 'timestamp[s]' (i.e., the actual data header)
    for i, line in enumerate(lines):
        if line.startswith("timestamp[s]"):
            return i

def get_temperature(filename):
    match = re.search(r'_(neg)*(\d+)_', filename)
    temperature = np.nan
    if match:
        sign = +1
        if match.group(1):
            sign = -1
        temperature = sign * float(match.group(2))
    return temperature

# Function to process the data and generate the plot
def plot_iv_curve(file_path, first, **kwargs):
    # Determine the number of header lines to skip
    skip_lines = skip_header_lines(file_path)
    
    voltage_smu = []
    current_smu = []
    

    # Load the data from the file, skipping the non-CSV-compliant header
    data = pd.read_csv(file_path, delim_whitespace=True, skiprows=skip_lines)
    f = open(file_path)
    for i, line in enumerate(f):
        if i <= skip_lines:
            continue
        splitted = line.split('\t')
        voltage_smu.append(abs(float(splitted[1])))
        current_smu.append(abs(float(splitted[3])))
        


    # Plotting
    # Normalize the values between 0 and 1 for the colormap
    norm = Normalize(vmin=kwargs['min_temperature'], vmax=kwargs['max_temperature'])  # Normalizing to the range -20 to 0
    cmap = matplotlib.colormaps['plasma']
    color = cmap(norm(kwargs['temperature']))

    # Scatter plot
    # plt.scatter(voltage_smu, current_smu, color=color, label=f'IV at {temperature} $^\circ$')

    label = kwargs['sensor']
    fmt = kwargs['sensor_fmt_map'][kwargs['sensor']]
    # print(temperature)
    # if np.isnan(kwargs['temperature']):
    #     label += 'not annealed'
    #     color = 'black'
    # else:
    #     label += f'annealed at {kwargs["temperature"]}$^\circ C$'

    if first:
        fig, ax = plt.subplots(figsize=(8, 6))
        # Add a color bar to show the color scale representing temperatures
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # ScalarMappable needs an array
        cbar = fig.colorbar(sm, ax=ax)  # Use the ax argument to link the colorbar to the plot
        cbar.set_label('Temperature ($^\circ$C)')  # Label for the colorbar
        # plt.colorbar(sm, label='Temperature ($^\circ$C)')  # Show the colorbar with temperature in °C

    # Line plot
    plt.plot(voltage_smu, current_smu, color=color, label=label, linestyle=fmt)


    # Log scale for y-axis
    plt.yscale('log')
    # Access the y_range and ensure it's converted to floats (if necessary)
    y_range = [float(i) for i in kwargs['y_range']]  # Ensure float conversion
    x_range = [float(i) for i in kwargs['x_range']]
    plt.ylim(y_range)
    plt.xlim(x_range)

    # Labels and title
    plt.xlabel('Reverse Bias Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title(kwargs['title'])
    plt.legend()

    # Show plot
    plt.grid(True, which="both", ls="--")
    output_file = 'figs/' + os.path.basename(file_path) + '.png'
    #plt.savefig(output_file)

# Main function to handle command-line arguments
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot IV Curve from file(s)")
    parser.add_argument("config", help="yaml style config file")


    # Get the file path from command-line arguments
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)


    # Call the plot function with the provided file
    first = True
    for m in config['measurements']:
        t = m['temperature']
        file = m['file']
        sensor = m['sensor']
        plot_iv_curve(file, first, temperature=t, sensor=sensor, **config)
        first = False
    plt.show()

if __name__ == "__main__":
    main()

