import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse
import os

# Defining the Gaussian function to fit the data
def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Fit data to Gaussian function and plot the result.')
parser.add_argument('data_file', type=str, help='Path to the data file.')
parser.add_argument('-s', '--save_plot', type=str, default=None, help='Path to save the plot (optional).')
args = parser.parse_args()

# Opening the data file specified as a command-line argument
data_file = open(args.data_file)

# Initializing empty lists to store data points
x = []
y = []

# Reading data from the file and appending to the respective lists
for line in data_file:
    split = line.split(',')
    x.append(float(split[0].strip()))
    y.append(float(split[1].strip()))

# Converting lists to numpy arrays for further processing
x = np.asarray(x)
y = np.asarray(y)

# Stacking the arrays together to create a 2D array
data = np.vstack((x, y))

# Sorting the data based on the first row (x values)
ind = np.argsort(data[0, :])
x = data[0, ind]
y = data[1, ind]

# Fitting the data to the Gaussian function using curve_fit
parameters, covariance = curve_fit(gauss, x, y)

# Generating the fitted curve using the obtained parameters
fit_y = gauss(x, parameters[0], parameters[1], parameters[2], parameters[3])

x0_fit = parameters[2]
sigma_fit = parameters[3]
print(parameters)
print(covariance)

SMALL_SIZE = 18
MEDIUM_SIZE = 24
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Plotting the data and the fitted curve
plt.figure(figsize=(12, 8))  # Set the figure size (adjust the values as needed)
plt.plot(x, y, 'b+', label='data')
plt.plot(x, fit_y, 'r-', label='fit')
plt.legend()

# Remove the file extension from the label in the plot title
file_name_without_ext = os.path.splitext(args.save_plot)[0] if args.save_plot else "Plot"
plt.title(f'Efficiencies for different time_shifts ({file_name_without_ext})')

plt.figtext(.15, .8, f'μ = {"%.2f" % x0_fit}\n σ = {"%.2f" % sigma_fit}')
plt.xlabel('t [μs]')
plt.ylabel('ε [%]')
plt.grid()

# Saving the plot if the file path is provided
if args.save_plot:
    plt.savefig(args.save_plot, dpi=300)  # Set the dpi (dots per inch) value as needed
else:
    # Maximize the plot window to full screen before displaying the plot
    plt.get_current_fig_manager().window.state('zoomed')
    plt.show()

