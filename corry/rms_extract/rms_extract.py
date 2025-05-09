import uproot
import numpy as np
import yaml
import matplotlib.pyplot as plt
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Calculate truncated RMS and plot histogram from a ROOT file.")
parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
args = parser.parse_args()

# Load the configuration from the YAML file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

root_file = config["root_file"]
histogram_name = config["histogram_name"]
lower_quantile = config["lower_quantile"]
upper_quantile = config["upper_quantile"]
output_plot = config["output_plot"]
xlims = config["x_range"]

# Open the ROOT file and get the histogram
file = uproot.open(root_file)
hist = file[histogram_name]

# Extract bin contents (residuals) and bin edges
bin_contents = hist.values()
bin_edges = hist.axis().edges()

# Reconstruct the distribution
# To avoid type issues, use list comprehension and avoid np.repeat
values = np.concatenate([
    np.full(int(count), (bin_edges[i] + bin_edges[i+1]) / 2)
    for i, count in enumerate(bin_contents)
])

# Determine the quantiles
lower_q_value = np.percentile(values, lower_quantile)
upper_q_value = np.percentile(values, upper_quantile)

# Filter residuals within the quantiles
truncated_values = values[(values >= lower_q_value) & (values <= upper_q_value)]

# Calculate RMS
truncated_rms = np.sqrt(np.mean(truncated_values**2))

print(f'lower q ={lower_q_value} for {lower_quantile}th Percenile')
print(f'lower q ={upper_q_value} for {upper_quantile}th Percenile')
print(f'Truncated RMS: {truncated_rms}')

# Plot the histogram and quantiles
fontsize = 28
plt.rcParams.update({
    'font.size': fontsize,             # Base font size
    'axes.titlesize': fontsize,        # Title size
    'axes.labelsize': fontsize,        # X/Y label size
    'xtick.labelsize': fontsize,       # X tick label size
    'ytick.labelsize': fontsize,       # Y tick label size
    'legend.fontsize': fontsize,       # Legend text size
    'legend.title_fontsize': fontsize  # Legend title size
})

plt.figure(figsize=(16, 9))
plt.hist(values, bins=bin_edges, alpha=0.7, label='Residuals', edgecolor='black')

# Add lines for quantiles
plt.axvline(x=lower_q_value, color='r', linestyle='--', linewidth=2, label=f'{lower_quantile}th Percentile')
plt.axvline(x=upper_q_value, color='g', linestyle='--', linewidth=2, label=f'{upper_quantile}th Percentile')

plt.title(f'"{histogram_name.split("/")[-1]}" with Quantiles')
plt.xlabel('Residual Value')
plt.ylabel('Counts')
plt.legend()
plt.xlim(xlims)
plt.grid(True)
plt.savefig(output_plot)
plt.show()

