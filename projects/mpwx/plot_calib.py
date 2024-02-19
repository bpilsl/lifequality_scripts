import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import warnings
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from scipy.stats import norm
import pandas as pd

sensor_dim = (64, 64)
nBins = 100


def s_curve_stats(file, output_dict):
    # Define the sigmoidal function (logistic function)
    def sigmoid(x, L, x0, k, b):
        y = L / (1 + np.exp(-k * (x - x0))) + b
        return y

    def vt_from_s(y, L, x0, k, b):
        return - 1 / k * np.log(L / (y - b) - 1) + x0

    pixel_data = []
    current_pixel = {}

    with open(file, 'r') as f:
        for line in f:
            if line.startswith('#! Pixel'):
                if current_pixel:
                    pixel_data.append(current_pixel.copy())
                    current_pixel = {}
                _, pixel_info = line.split('#! Pixel')
                current_pixel['Pixel'] = pixel_info.strip()
                current_pixel['Voltage'] = []
                current_pixel['Hits'] = []
                row, col = pixel_info.strip().split(':')
                row = int(row)
                col = int(col)
                current_pixel['Index'] = (row, col)
            elif line.startswith('0.') or line.startswith('1.'):
                # Extract voltage and hits information for each pixel
                try:
                    voltage, hits, _ = line.split()
                    current_pixel['Voltage'].append(float(voltage))
                    current_pixel['Hits'].append(int(hits))
                except:
                    print('broken line ', line)
                    continue

    # Append the last pixel's data
    if current_pixel:
        pixel_data.append(current_pixel.copy())

    for pixel in pixel_data:
        x_data = np.array(pixel['Voltage'])
        y_data = np.array(pixel['Hits'])
        try:
            p0 = [max(y_data), np.median(x_data), 1, min(y_data)]  # this is a mandatory initial guess
            popt, _ = curve_fit(sigmoid, x_data, y_data, p0, maxfev=100000)
            output_dict['vt50'][pixel['Index'][0], pixel['Index'][1]] = vt_from_s(50, *popt)
        except RuntimeWarning:
            pass
        except RuntimeError as rte:
            print(rte)
        except OptimizeWarning as ow:
            print('optimization failed for', pixel["Pixel"], ow)


def parse_tdac(file, output_dict):
    with open(file, 'r') as f:
        row = 0
        for line in f:
            if line.startswith('#') or len(line) == 0 or len(line.strip()) == 0:
                # comment or empty or just whitespace line
                continue

            splitted = line.split(' ')
            if len(splitted) != sensor_dim[0] + 1:
                print('invalid line with ', len(splitted), ' entries in hitmap')
                return
            row = int(splitted[0])  # first number in line indicates row number of pixels
            if row >= sensor_dim[0]:
                print('invalid line', line)
                continue
            for col in range(1, sensor_dim[1] + 1):
                # each line contains number of hits for column index of pixel in current row,
                # column index corresponds to position in line
                # eg : 39 1 0 2 4 7 0 1 0 0 13 1 186 3 1 0 2 0 2 4 0 0 7 14 4 3 3 1 ...
                # corresponds to hits in row 39, pixel 39:00 got 1 hit, 39:01 got 0 hits, 39:02 got 2 hits, ...

                tdac = int(splitted[col])
                output_dict['tdac'][row][col - 1] = tdac


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)


def v_to_q(v):
    # Q = C * U
    # with C ~ 2.8fF and electron charge
    # a charge of 16.56 ke-/mV is evaluated
    return v * 16.568

def q_to_v(q):
    return q / 16.568


data = {'tdac': [], 'vt50': []}
for key in data:
    data[key] = np.zeros(sensor_dim)

s_curve_stats(sys.argv[1], data)
parse_tdac(sys.argv[2], data)
data['tdac'] = data['tdac'].flatten()
data['vt50'] = data['vt50'].flatten()

# filter not scanned pixels
data['tdac'] = data['tdac'][data['vt50'] > 0]
data['vt50'] = data['vt50'][data['vt50'] > 0] * 1000  #convert to mV
df = pd.DataFrame(data)

tdac_seperated = []
labels = []

for tdac_value in range(16):
    labels.append(tdac_value)
    # create list of VT50 for each trimDAC value

    tdac_seperated.append(data['vt50'][data['tdac'] == tdac_value])

    # ax.hist(subset_df['vt50'], bins=10, stacked=False, alpha=0.7, label=f'tdac={tdac_value}', histtype='bar')

fig, ax = plt.subplots()

cmap = plt.get_cmap('magma')
colors = cmap(np.linspace(1/16, 1, 16))
# create a stacked histogram, where VT50 will be binned but, stacked on top of each other, grouped by the used
# trimDAC values
ax.hist(tdac_seperated, stacked=True, label=labels, histtype='bar', alpha=0.7, bins=nBins, color=colors)

hist, bin_edges = np.histogram(df['vt50'], bins=nBins)
# Calculate bin centers
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Use curve_fit to fit the Gaussian function to the histogram data
initial_guess = [1.0, max(df['vt50']), np.std(df['vt50'])]
params, covariance = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)
amplitude, mean, stddev = params
stddev = abs(stddev)
x_fit = np.linspace(min(df['vt50']), max(df['vt50']), 1000)
ax.plot(x_fit, gaussian(x_fit, amplitude, mean, stddev), '--', label='Fit', color='black')

# box with statistics
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
stats = (f'$\\mu$ = {mean :.1f}mV $\\approx$ {v_to_q(mean):.0f} $e^-$ \n$\\sigma$ = {stddev:.1f}mV '
         f'$\\approx$ {v_to_q(stddev):.0f} $e^-$')

ax.text(0.75, 0.95, stats, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


secax = ax.secondary_xaxis('top', functions=(v_to_q, q_to_v))
secax.set_xlabel('Threshold ($e^-$)')
plt.xlabel('VT50 [mV]')
plt.ylabel('Counts')
plt.title('VT50 grouped by TrimDAC')
plt.legend()
plt.grid()
plt.show()
