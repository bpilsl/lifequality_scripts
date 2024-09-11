import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os.path

# Function to determine where the actual data starts
def skip_header_lines(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Detect the first line that starts with 'timestamp[s]' (i.e., the actual data header)
    for i, line in enumerate(lines):
        if line.startswith("timestamp[s]"):
            return i

# Function to process the data and generate the plot
def plot_iv_curve(file_path):
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
        
        
    print('U = ', voltage_smu, '\n I = ', current_smu)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Scatter plot
    plt.scatter(voltage_smu, current_smu, color='blue', label='Scatter plot')

    # Line plot
    plt.plot(voltage_smu, current_smu, color='red', label='Line plot')

    # Log scale for y-axis
    #plt.yscale('log')

    # Labels and title
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('IV Curve: Voltage vs Current (Log Scale)')
    plt.legend()

    # Show plot
    plt.grid(True, which="both", ls="--")
    output_file = 'figs/' + os.path.basename(file_path) + '.png'
    plt.savefig(output_file)
    #plt.show()

# Main function to handle command-line arguments
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot IV Curve from file")
    parser.add_argument("input_file", help="Path to the input file")

    # Get the file path from command-line arguments
    args = parser.parse_args()

    # Call the plot function with the provided file
    plot_iv_curve(args.input_file)

if __name__ == "__main__":
    main()

